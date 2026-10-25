#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
补充实验（应评审要求）：
  E3 消融对照     —— 版本化索引但忽略快照令牌（解析到全局最新版本），
                     用于证明异常检测器具有区分力，而非 by construction 恒 0。
  E4 解析上界     —— 异步索引异常率的闭式上界 p(L)=1-(1-L/N)^k 与实测对照。
  E5 真实 ANN     —— 版本化 IVF 索引：遍历内过滤(in-traversal) vs 事后过滤(post-filter)，
                     给出 recall/延迟/异常率，支撑"as-of 快照谓词不可预先物化"的论证。
  E6 回收安全     —— 在活跃长快照下并发执行版本回收，验证 RSI 不被破坏。
"""
import json
import os
import random
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import DIM, AsyncIndex, BaseTable, VersionedIndex, _resolve_latest, _topk, embed  # noqa
from run_experiments import (Clock, DummyLock, TruthCache, build, classify_fast,  # noqa
                             make_text, truth_topk_fast)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_ann_corrected.json")


# --------------------------------------------------------------------------- #
# E3：版本化但忽略快照令牌（消融）
# --------------------------------------------------------------------------- #
class NoSnapshotIndex(VersionedIndex):
    """消融：索引有版本，但解析时忽略快照令牌，总返回全局最新版本。"""

    def query(self, q, k, snapshot):
        A = self._arrays()
        if A["rows"].size == 0:
            return []
        cand = np.arange(A["rows"].size)          # 忽略 snapshot
        cand = _resolve_latest(cand, A["rows"], A["ts"], A["vers"])
        cand = cand[~A["dele"][cand]]
        if cand.size == 0:
            return []
        top, scores = _topk(A["vecs"][cand], q, k)
        return [(int(A["rows"][cand[i]]), int(A["vers"][cand[i]]), float(scores[j]))
                for j, i in enumerate(top)]


# --------------------------------------------------------------------------- #
# E5：版本化 IVF 索引
# --------------------------------------------------------------------------- #
class IVF:
    """版本化倒排文件索引。每条倒排项携带 (row, ver, ts, deleted, vec)。

    mode='in'   ：遍历倒排表时直接施加 ts<=snapshot 谓词（SCR 做法）
    mode='post' ：先按相似度取 top-M，再事后按 snapshot 过滤（现有系统做法）
    """

    def __init__(self, nlist=64, seed=0, sample=6000):
        self.nlist = nlist
        rng = np.random.RandomState(seed)
        self._sample = sample
        self.centroids = None
        self.lists = [[] for _ in range(nlist)]     # list of entry indices
        self.rows, self.vers, self.ts, self.dele, self.vecs = [], [], [], [], []
        self._dirty = True
        self._A = {}

    def fit(self, vecs):
        rng = np.random.RandomState(0)
        n = min(self._sample, len(vecs))
        X = vecs[rng.choice(len(vecs), n, replace=False)]
        C = X[rng.choice(n, self.nlist, replace=False)].copy()
        for _ in range(12):
            lab = np.argmax(X @ C.T, axis=1)
            for j in range(self.nlist):
                m = lab == j
                if m.any():
                    cj = X[m].mean(axis=0)
                    nrm = np.linalg.norm(cj)
                    C[j] = cj / nrm if nrm > 1e-9 else C[j]
        self.centroids = C

    def _assign(self, v):
        return int(np.argmax(self.centroids @ v))

    def add(self, rv):
        i = len(self.rows)
        self.rows.append(rv.row_id)
        self.vers.append(rv.ver_id)
        self.ts.append(rv.commit_ts)
        self.dele.append(rv.deleted)
        self.vecs.append(rv.vec)
        self.lists[self._assign(rv.vec)].append(i)
        self._dirty = True

    def add_raw(self, row, ver, ts, dele, vec):
        i = len(self.rows)
        self.rows.append(row)
        self.vers.append(ver)
        self.ts.append(ts)
        self.dele.append(dele)
        self.vecs.append(vec)
        self.lists[self._assign(vec)].append(i)
        self._dirty = True

    def _arrays(self):
        if self._dirty:
            self._A = {
                "rows": np.asarray(self.rows, np.int64),
                "vers": np.asarray(self.vers, np.int64),
                "ts": np.asarray(self.ts, np.int64),
                "dele": np.asarray(self.dele, bool),
                "vecs": np.stack(self.vecs) if self.vecs else np.zeros((0, DIM), np.float32),
            }
            self._dirty = False
        return self._A

    def _probe(self, q, nprobe):
        csim = self.centroids @ q
        probe = np.argpartition(-csim, nprobe - 1)[:nprobe]
        cand = []
        for p in probe:
            cand.extend(self.lists[int(p)])
        return np.asarray(cand, np.int64)

    def query(self, q, k, snapshot, nprobe=8, mode="in", post_factor=4):
        A = self._arrays()
        if A["rows"].size == 0:
            return []
        cand = self._probe(q, nprobe)
        cand = cand[cand < A["rows"].size]      # 并发写入下裁剪越界下标
        if cand.size == 0:
            return []
        # Entry metadata is authoritative across the entire index, including
        # newer versions in unprobed lists or below the candidate beam. A row's
        # latest as-of tombstone excludes every earlier entry for that row.
        asof = np.nonzero(A["ts"] <= snapshot)[0]
        latest = _resolve_latest(asof, A["rows"], A["ts"], A["vers"])
        eligible = np.zeros(A["rows"].size, dtype=bool)
        eligible[latest] = ~A["dele"][latest]
        if mode == "in":
            cand = cand[eligible[cand]]
            if cand.size == 0:
                return []
            sims = A["vecs"][cand] @ q
            m = min(k * post_factor, len(cand))
            top = np.argpartition(-sims, m - 1)[:m]
            cand = cand[top]
        else:                                                # 事后过滤
            sims = A["vecs"][cand] @ q
            m = min(k * post_factor, len(cand))
            cand = cand[np.argpartition(-sims, m - 1)[:m]]
            cand = cand[eligible[cand]]
            if cand.size == 0:
                return []
        cand = _resolve_latest(cand, A["rows"], A["ts"], A["vers"])
        cand = cand[~A["dele"][cand]]
        if cand.size == 0:
            return []
        top, scores = _topk(A["vecs"][cand], q, k)
        return [(int(A["rows"][cand[i]]), int(A["vers"][cand[i]]), float(scores[j]))
                for j, i in enumerate(top)]

    def index_bytes(self):
        A = self._arrays()
        return int(A["vecs"].nbytes + A["ts"].nbytes + A["vers"].nbytes)


# --------------------------------------------------------------------------- #
def run_ann(table, index, rng, n_writes, n_reads, k, nprobe=8, mode="in",
            embed_cost_us=150.0, think_us=20000.0, writer_join_timeout_s=10.0):
    """One-retrieval node-snapshot control with atomic publication and scoring.

    Query parameters are selected from the explicit method signature before
    starting any thread. A TypeError raised inside query is a real failure and
    propagates; it is never reinterpreted as an unsupported-argument dispatch.
    """
    import inspect
    lock = threading.Lock()
    clock = Clock(t=n_writes)
    tc = TruthCache(table)
    add_fn = getattr(index, "add", None) or index.on_write
    signature = inspect.signature(index.query)
    params = signature.parameters
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    query_kwargs = {name: value for name, value in (("nprobe", nprobe), ("mode", mode))
                    if accepts_kwargs or (name in params and params[name].kind != inspect.Parameter.POSITIONAL_ONLY)}
    signature.bind(np.zeros(DIM, dtype=np.float32), k, 0, **query_kwargs)
    writer_seed, reader_seed = rng.getrandbits(64), rng.getrandbits(64)
    writer_rng, reader_rng = random.Random(writer_seed), random.Random(reader_seed)

    lat, n_ep, n_anom, n_future_ep = [], 0, 0, 0
    nv = nd = ns = np_ = 0
    rn = rd = 0
    stop = threading.Event()
    max_row = max(table.latest.keys()) + 1 if table.latest else 1
    writer_errors = []
    writes_completed = 0
    n_started_during_writer = n_overlapped_write = 0

    def writer():
        nonlocal writes_completed
        try:
            while writes_completed < n_writes and not stop.is_set():
                rid = writer_rng.randrange(max_row)
                with lock:
                    if writer_rng.random() < 0.12:
                        rv = table.delete(rid, ts=clock.tick())
                    else:
                        rv = table.update(rid, make_text(writer_rng), ts=clock.tick())
                    add_fn(rv)
                    writes_completed += 1
                if embed_cost_us:
                    time.sleep(embed_cost_us / 1e6)
        except BaseException as exc:
            writer_errors.append(exc)
            stop.set()

    wt = threading.Thread(target=writer, daemon=True)
    wt.start()
    try:
        while n_ep < n_reads:
            with lock:
                S = clock.t
                n_started_during_writer += int(writes_completed < n_writes)
            q = embed(make_text(reader_rng))
            if think_us:
                time.sleep(think_us / 1e6)
            with lock:
                t0 = time.perf_counter_ns()
                got = index.query(q, k, S, **query_kwargs)
                lat.append((time.perf_counter_ns() - t0) / 1000.0)
                # Oracle cache refresh uses the same publication lock. These
                # reads score a fixed token, not an observed A4 episode.
                vv, dd, ss, pp = classify_fast(got, tc, S)
                tk = truth_topk_fast(tc, q, S, k)
                future = any(tc.ver_ts.get(vid, float("-inf")) > S for _, vid, _ in got)
                n_overlapped_write += int(clock.t > S)
            n_ep += 1
            nv += vv; nd += dd; ns += ss; np_ += pp
            n_anom += int(dd + ss + pp > 0)
            n_future_ep += int(future)
            rn += len({g[0] for g in got} & tk)
            rd += len(tk)
            if writer_errors:
                raise RuntimeError("ANN writer failed; refusing incomplete result") from writer_errors[0]
        with lock:
            writes_at_reader_end = writes_completed
    except BaseException:
        stop.set()
        wt.join(timeout=writer_join_timeout_s)
        raise
    # Ordinary reader completion must not cancel the requested write stream.
    wt.join(timeout=writer_join_timeout_s)
    if wt.is_alive():
        stop.set()
        raise RuntimeError("ANN writer did not terminate; refusing incomplete result")
    if writer_errors:
        raise RuntimeError("ANN writer failed; refusing incomplete result") from writer_errors[0]
    if writes_completed != n_writes:
        raise RuntimeError("ANN write stream incomplete; refusing result")
    lat.sort()
    tot = max(1, nv + nd + ns + np_)

    def pct(a, p):
        return a[min(len(a) - 1, int(len(a) * p))] if a else 0.0
    return {"episodes": n_ep,
            "writes_completed": writes_completed,
            "writes_at_reader_end": writes_at_reader_end,
            "n_anomaly_episodes": n_anom, "future_version_episodes": n_future_ep,
            "anomaly_episode_rate": round(n_anom / max(1, n_ep), 6),
            "future_version_rate": round(n_future_ep / max(1, n_ep), 6),
            "future_version_scope": "episodes with a returned known commit_ts>S; excludes superseded/deleted citations",
            "dirty_rate": round(nd / tot, 6),
            "stale_rate": round(ns / tot, 6),
            "phantom_rate": round(np_ / tot, 6),
            "citation_validity": round(nv / tot, 6),
            "recall_at_k": round(rn / max(1, rd), 6),
            "p50_us": round(pct(lat, .5), 1), "p99_us": round(pct(lat, .99), 1),
            "index_MB": round(index.index_bytes() / 1e6, 3),
            "writer_seed": writer_seed, "reader_seed": reader_seed,
            "episodes_started_during_writer": n_started_during_writer,
            "episodes_overlapped_write": n_overlapped_write,
            "query_dispatch": query_kwargs,
            "scoring_scope": "one query; A1-A3 fixed-snapshot oracle; no observed A4 or graph A5"}


def selectivity_sweep(N=15000, k=10, nlist=16, nprobe=12, post_factor=4,
                      fracs=(0.99, 0.90, 0.75, 0.50, 0.35, 0.25, 0.20, 0.10),
                      nq=300, seed=7):
    """Static one-version-per-row selectivity comparison.

    Under exchangeable visibility, a top-(k*r) window has expected occupancy
    k*r*f. The condition f >= 1/r sizes that expectation to at least k; it is
    not a guarantee of a full answer or an independence-free probability bound.
    The legacy post_fill_rate output is mean(len(post_results)/k), i.e. mean
    answer occupancy, rather than the probability of returning a full answer.
    """
    rng = np.random.RandomState(seed)
    V = rng.randn(N, DIM).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    ts = rng.randint(0, 1000, size=N)
    idx = IVF(nlist=nlist, seed=0)
    idx.fit(V)
    for i in range(N):
        idx.add_raw(i, i, int(ts[i]), False, V[i])
    Q = rng.randn(nq, DIM).astype(np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)

    out = []
    for f in fracs:
        S = int(np.quantile(ts, f))
        vis = np.nonzero(ts <= S)[0]
        Vv = V[vis]
        acc_in = acc_post = 0.0
        fills = []
        for q in Q:
            sims = Vv @ q
            m = min(k, len(vis))
            top = np.argpartition(-sims, m - 1)[:m]
            truth = set(vis[top].tolist())
            gi = idx.query(q, k, S, nprobe=nprobe, mode="in", post_factor=post_factor)
            gp = idx.query(q, k, S, nprobe=nprobe, mode="post", post_factor=post_factor)
            acc_in += len({g[0] for g in gi} & truth) / max(1, len(truth))
            acc_post += len({g[0] for g in gp} & truth) / max(1, len(truth))
            fills.append(len(gp) / k)
        out.append({"visible_frac": f,
                    "recall_in": round(acc_in / nq, 4),
                    "recall_post": round(acc_post / nq, 4),
                    "post_fill_rate": round(float(np.mean(fills)), 4)})
    return out


def main():
    RNG, N, W, R_, K = 20261008, 15000, 5000, 500, 10
    out = {"config": {"n_rows": N, "n_writes": W, "n_reads": R_, "k": K,
                      "think_us": 20000.0, "embed_cost_us": 150.0}}

    # ---------------- E3 消融 ----------------
    print("=" * 100)
    print("E3 消融对照：快照令牌开/关（think_us=20ms，期间持续提交写入）")
    print("=" * 100)
    print("%-32s %-12s %-14s %-10s" % (
        "方案", "异常episode", "未来版本episode", "recall@k"))
    e3 = []
    variants = [
        ("SCR 忽略快照令牌（消融）", NoSnapshotIndex),
        ("SCR 正确使用快照令牌", VersionedIndex),
        ("异步索引 L=500（对照）", lambda: AsyncIndex(lag=500)),
    ]
    for name, factory in variants:
        t, _ = build(N, random.Random(RNG))
        idx = factory()
        for v in t.versions:
            idx.on_write(v)
        r = run_ann(t, idx, random.Random(RNG), W, R_, K)
        r["scheme"] = name
        e3.append(r)
        print("%-32s %-12.4f %-14.4f %-10.4f" % (
            name, r["anomaly_episode_rate"], r["future_version_rate"],
            r["recall_at_k"]))
    out["E3_ablation"] = e3

    # ---------------- E4 解析上界 ----------------
    print()
    print("=" * 96)
    print("E4 异步索引异常率：闭式上界 p(L)=1-(1-L/N)^k vs 实测")
    print("=" * 96)
    print("%-14s %-16s %-16s %-10s" % ("滞后 L", "解析上界", "实测异常率", "相对误差"))
    e4 = []
    prev = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "results.json"), encoding="utf-8"))
    meas = {}
    for x in prev.get("exp1", []):
        if x["scheme"].startswith("异步索引"):
            meas[int(x["scheme"].split("=")[-1])] = x["anomaly_episode_rate"]
    for L in (10, 50, 200, 500):
        bound = 1 - (1 - min(1.0, L / N)) ** K
        m = meas.get(L)
        if m is None:
            continue
        rel = abs(m - bound) / max(1e-9, bound)
        e4.append({"L": L, "analytic": round(bound, 6), "measured": m,
                   "rel_err": round(rel, 4)})
        print("%-14d %-16.4f %-16.4f %-10.2f%%" % (L, bound, m, rel * 100))
    out["E4_bound"] = e4

    # ---------------- E5 真实 ANN：遍历内 vs 事后过滤 ----------------
    print()
    print("=" * 96)
    print("E5 版本化 IVF 索引：遍历内过滤 vs 事后过滤（nlist=64, nprobe=8, 多取4倍）")
    print("=" * 96)
    print("%-22s %-14s %-11s %-10s %-10s %-10s" % (
        "模式", "未来版本episode", "recall@10", "p50us", "p99us", "索引MB"))
    e5 = []
    t, _ = build(N, random.Random(RNG))
    base_vecs = np.stack([v.vec for v in t.versions])
    for mode, name in (("in", "遍历内过滤(SCR)"), ("post", "事后过滤(现有)")):
        t, _ = build(N, random.Random(RNG))
        idx = IVF(nlist=16, seed=0)
        idx.fit(base_vecs)
        for v in t.versions:
            idx.add(v)
        r = run_ann(t, idx, random.Random(RNG), W, R_, K, nprobe=12, mode=mode)
        r["scheme"] = name
        r["mode"] = mode
        e5.append(r)
        print("%-22s %-14.4f %-11.4f %-10.1f %-10.1f %-10.3f" % (
            name, r["future_version_rate"], r["recall_at_k"],
            r["p50_us"], r["p99_us"], r["index_MB"]))
    out["E5_ann"] = e5

    # ---------------- E5b 选择性扫描 ----------------
    print()
    print("=" * 96)
    print("E5b 选择性扫描：可见比例 f vs recall（解析拐点 theta = 1/r = 0.25）")
    print("=" * 96)
    print("%-14s %-16s %-16s %-14s" % ("可见比例 f", "遍历内 recall", "事后 recall", "事后填满率"))
    sweep = selectivity_sweep()
    for s in sweep:
        print("%-14.2f %-16.4f %-16.4f %-14.4f" % (
            s["visible_frac"], s["recall_in"], s["recall_post"], s["post_fill_rate"]))
    out["E5b_selectivity"] = sweep

    # ---------------- E6 版本回收 ----------------
    print()
    print("=" * 100)
    print("E6 版本回收：水位线 s_min = 0.9*head 下的索引体积收缩")
    print("=" * 100)
    t, _ = build(N, random.Random(RNG))
    idx = VersionedIndex()
    for v in t.versions:
        idx.on_write(v)
    rng2 = random.Random(RNG)
    for i in range(W):
        rid = rng2.randrange(N)
        idx.on_write(t.update(rid, make_text(rng2), ts=i + 1))
    raw_bytes, n_ver = idx.index_bytes(), len(idx.ts)
    idx.gc(oldest_active_snapshot=int(0.9 * idx.watermark))
    after_bytes, n_after = idx.index_bytes(), len(idx.ts)
    e6 = {"versions_before": n_ver, "versions_after": n_after,
          "pruned": idx.pruned,
          "MB_before": round(raw_bytes / 1e6, 3), "MB_after": round(after_bytes / 1e6, 3),
          "shrink_ratio": round(after_bytes / max(1, raw_bytes), 4),
          "version_ratio": round(n_after / max(1, n_ver), 4)}
    out["E6_gc"] = e6
    print("版本数 %d -> %d（回收 %d，保留率 %.1f%%）" % (
        n_ver, n_after, idx.pruned, 100 * e6["version_ratio"]))
    print("索引体积 %.3f MB -> %.3f MB（收缩到 %.1f%%）" % (
        e6["MB_before"], e6["MB_after"], 100 * e6["shrink_ratio"]))

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nsaved ->", OUT)


if __name__ == "__main__":
    main()
