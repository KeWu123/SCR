#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SCR 原型实验：跨存储读异常率 / 召回 / 延迟 / 锁等待 / 索引体积。

运行：python run_experiments.py  ->  results.json + 控制台表格
"""
import json
import os
import random
import statistics
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import AsyncIndex, BaseTable, DIM, LockIndex, VersionedIndex, embed  # noqa

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
         "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho",
         "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega"]

sys.setrecursionlimit(10000)


def make_text(rng, n_tok=12):
    return " ".join(rng.choice(WORDS) for _ in range(n_tok))


class Clock:
    def __init__(self, t=0):
        self.t = t
        self.lock = threading.Lock()

    def tick(self):
        with self.lock:
            self.t += 1
            return self.t


class DummyLock:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# --------------------------------------------------------------------------- #
class TruthCache:
    """向量化真值：ts<=T 的每行最新版本。"""

    def __init__(self, table):
        self.table = table
        self.n = 0
        self.ts = np.zeros(0, np.int64)
        self.row = np.zeros(0, np.int64)
        self.ver = np.zeros(0, np.int64)
        self.dele = np.zeros(0, bool)
        self.vec = np.zeros((0, DIM), np.float32)
        self.ver_ts = {}          # ver_id -> commit_ts（增量维护）

    def sync(self):
        V = self.table.versions
        m = len(V)
        if m > self.n:
            new = V[self.n:m]
            self.ts = np.concatenate([self.ts, np.array([v.commit_ts for v in new], np.int64)])
            self.row = np.concatenate([self.row, np.array([v.row_id for v in new], np.int64)])
            self.ver = np.concatenate([self.ver, np.array([v.ver_id for v in new], np.int64)])
            self.dele = np.concatenate([self.dele, np.array([v.deleted for v in new], bool)])
            self.vec = np.concatenate([self.vec, np.stack([v.vec for v in new])])
            for v in new:
                self.ver_ts[v.ver_id] = v.commit_ts
            self.n = m

    def truth_idx(self, T):
        self.sync()
        cand = np.nonzero(self.ts <= T)[0]
        if len(cand) == 0:
            return cand
        r = self.row[cand]
        t = self.ts[cand]
        order = np.lexsort((t, r))
        cand = cand[order]
        r = r[order]
        last = np.append(r[1:] != r[:-1], True)
        sel = cand[last]
        return sel[~self.dele[sel]]


def classify_fast(retrieved, tc: TruthCache, ts):
    sel = tc.truth_idx(ts)
    truth_ver = dict(zip(tc.row[sel].tolist(), tc.ver[sel].tolist()))
    ver_ts = dict(zip(tc.ver.tolist(), tc.ts.tolist()))
    n_dirty = n_stale = n_phantom = n_valid = 0
    for rid, vid, _ in retrieved:
        cts = ver_ts.get(vid)
        if cts is None or cts > ts:
            n_dirty += 1
        elif rid not in truth_ver:
            n_phantom += 1
        elif truth_ver[rid] != vid:
            n_stale += 1
        else:
            n_valid += 1
    return n_valid, n_dirty, n_stale, n_phantom


def truth_topk_fast(tc: TruthCache, q, ts, k):
    sel = tc.truth_idx(ts)
    if len(sel) == 0:
        return set()
    sims = tc.vec[sel] @ q
    k = min(k, len(sel))
    top = np.argpartition(-sims, k - 1)[:k]
    return set(tc.row[sel[top]].tolist())


# --------------------------------------------------------------------------- #
def build(n_rows, rng):
    t0 = time.perf_counter()
    table = BaseTable()
    for r in range(n_rows):
        table.insert(r, make_text(rng), ts=0)
    return table, (time.perf_counter() - t0) * 1000


def attach(index, table):
    for v in table.versions:
        index.on_write(v)


def run_workload(table, index, rng, n_writes, n_reads, k, mode="exact", delta=0,
                 lock=None, embed_cost_us=150.0, think_us=20000.0,
                 writer_join_timeout_s=30.0):
    """Single-query flat-index control with atomic publication and snapshot capture.

    The writer and reader have independent random streams.  The supplied lock
    must serialize table/index publication, Begin, query, and oracle refresh.
    Artificial embedding delay is writer pacing outside this critical section;
    query latency excludes oracle scoring and is not a graph-engine latency.
    A successful run completes every requested write and propagates failures.
    """
    if n_writes < 0 or n_reads < 0 or k <= 0 or delta < 0:
        raise ValueError("nonnegative counts/delta and positive k required")
    if mode not in ("exact", "bounded"):
        raise ValueError("unknown snapshot mode")
    if embed_cost_us < 0 or think_us < 0 or writer_join_timeout_s <= 0:
        raise ValueError("nonnegative pacing and positive join timeout required")
    if isinstance(lock, DummyLock):
        raise ValueError("DummyLock cannot protect a concurrent workload")
    publication_lock = threading.RLock() if lock is None else lock
    writer_seed, query_seed = rng.getrandbits(64), rng.getrandbits(64)
    writer_rng, query_rng = random.Random(writer_seed), random.Random(query_seed)
    # The head is the last successfully published commit, not an allocated ID.
    initial_head = max((v.commit_ts for v in table.versions), default=0)
    clock = Clock(t=initial_head)
    tc = TruthCache(table)
    lat, wait, snapshots, begin_heads, effective_lags = [], [], [], [], []
    n_ep = n_anom = n_started_during_writer = n_overlapped_write = 0
    n_valid = n_dirty = n_stale = n_phantom = 0
    recall_num = recall_den = writes_completed = 0
    writer_errors = []
    cancel = threading.Event()
    max_row = max((v.row_id for v in table.versions), default=0) + 1
    writer_seconds = 0.0
    started = time.perf_counter()

    def fail_if_writer_failed():
        # Called while publication_lock is held, or after writer join.
        if writer_errors:
            raise RuntimeError("workload writer failed; no scores published") from writer_errors[0]

    def writer():
        nonlocal writes_completed, writer_seconds
        writer_started = time.perf_counter()
        try:
            for _ in range(n_writes):
                if cancel.is_set():
                    break
                rid = writer_rng.randrange(max_row)
                deleted = writer_rng.random() < 0.12
                text = None if deleted else make_text(writer_rng)
                with publication_lock:
                    if cancel.is_set():
                        break
                    # Failed publication poisons the run before releasing the
                    # mutex: Begin must never accept a partially applied write.
                    try:
                        commit_ts = clock.t + 1
                        rv = (table.delete(rid, ts=commit_ts) if deleted else
                              table.update(rid, text, ts=commit_ts))
                        index.on_write(rv)
                        clock.tick()
                        writes_completed += 1
                    except BaseException as exc:
                        writer_errors.append(exc)
                        cancel.set()
                        return
                if embed_cost_us:
                    time.sleep(embed_cost_us / 1e6)
        except BaseException as exc:
            with publication_lock:
                writer_errors.append(exc)
                cancel.set()
        finally:
            writer_seconds = time.perf_counter() - writer_started

    wt = threading.Thread(target=writer, name="control-writer", daemon=True)
    wt.start()
    try:
        while n_ep < n_reads:
            with publication_lock:                 # atomic Begin
                fail_if_writer_failed()
                S_now = clock.t
                writes_at_begin = writes_completed
            S = max(initial_head, S_now - delta) if mode == "bounded" else S_now
            q = embed(make_text(query_rng))
            if think_us:
                time.sleep(think_us / 1e6)
            w0 = time.perf_counter_ns()
            with publication_lock:
                fail_if_writer_failed()
                w1 = time.perf_counter_ns()
                got = index.query(q, k, S)
                dt = (time.perf_counter_ns() - w1) / 1000.0
                writes_at_query = writes_completed
                # TruthCache snapshots and refresh are protected by the same
                # publication mutex; both scores use the captured cut S.
                vv, dd, ss, pp = classify_fast(got, tc, S)
                tk = truth_topk_fast(tc, q, S, k)
            wait.append((w1 - w0) / 1000.0)
            lat.append(dt)
            snapshots.append(S)
            begin_heads.append(S_now)
            effective_lags.append(S_now - S)
            n_started_during_writer += int(writes_at_begin < n_writes)
            n_overlapped_write += int(writes_at_query > writes_at_begin)
            n_ep += 1
            n_valid += vv
            n_dirty += dd
            n_stale += ss
            n_phantom += pp
            n_anom += int(dd + ss + pp > 0)
            recall_num += len({g[0] for g in got} & tk)
            recall_den += len(tk)
        with publication_lock:
            fail_if_writer_failed()
            writes_at_reader_end = writes_completed
        reader_seconds = time.perf_counter() - started
    except BaseException:
        cancel.set()
        wt.join(timeout=writer_join_timeout_s)
        raise
    # Normal reader completion must not truncate the requested write stream.
    wt.join(timeout=writer_join_timeout_s)
    if wt.is_alive():
        cancel.set()
        raise RuntimeError("writer did not finish before join timeout; no scores published")
    fail_if_writer_failed()
    if writes_completed != n_writes:
        raise RuntimeError("incomplete write stream; no scores published")
    total_seconds = time.perf_counter() - started
    lat.sort()
    wait.sort()

    def pct(a, p):
        return a[min(len(a) - 1, int(len(a) * p))] if a else 0.0

    def bounds(a):
        return [min(a), max(a)] if a else None

    tot = max(1, n_valid + n_dirty + n_stale + n_phantom)
    return {
        "episodes": n_ep,
        "anomaly_episode_rate": round(n_anom / max(1, n_ep), 6),
        "dirty_rate": round(n_dirty / tot, 6),
        "stale_rate": round(n_stale / tot, 6),
        "phantom_rate": round(n_phantom / tot, 6),
        "citation_validity": round(n_valid / tot, 6),
        "recall_at_k": round(recall_num / max(1, recall_den), 6),
        "p50_us": round(pct(lat, 0.50), 1),
        "p99_us": round(pct(lat, 0.99), 1),
        "lock_wait_p50_us": round(pct(wait, 0.50), 1),
        "lock_wait_p99_us": round(pct(wait, 0.99), 1),
        "index_MB": round(index.index_bytes() / 1e6, 3),
        "anomalous_episodes": n_anom,
        "valid_citations": n_valid,
        "dirty_citations": n_dirty,
        "stale_citations": n_stale,
        "phantom_citations": n_phantom,
        "recall_numerator": recall_num,
        "recall_denominator": recall_den,
        "writes_requested": n_writes,
        "writes_completed": writes_completed,
        "writes_at_reader_end": writes_at_reader_end,
        "episodes_started_during_writer": n_started_during_writer,
        "episodes_overlapped_write": n_overlapped_write,
        "writer_rng_seed": writer_seed,
        "query_rng_seed": query_seed,
        "initial_published_head": initial_head,
        "final_published_head": clock.t,
        "snapshot_range": bounds(snapshots),
        "begin_head_range": bounds(begin_heads),
        "effective_delta_range": bounds(effective_lags),
        "writer_seconds": round(writer_seconds, 6),
        "reader_seconds": round(reader_seconds, 6),
        "workload_seconds": round(total_seconds, 6),
    }


HEAD = "%-22s %-11s %-8s %-8s %-8s %-10s %-9s %-9s %-9s %-8s" % (
    "方案", "异常episode", "脏读", "陈旧", "幻读", "引用有效性", "recall@k",
    "p50us", "p99us", "索引MB")


def row(name, r):
    return "%-22s %-12.4f %-8.4f %-8.4f %-8.4f %-11.4f %-9.4f %-9.1f %-9.1f %-8.3f" % (
        name, r["anomaly_episode_rate"], r["dirty_rate"], r["stale_rate"],
        r["phantom_rate"], r["citation_validity"], r["recall_at_k"],
        r["p50_us"], r["p99_us"], r["index_MB"])


def main():
    RNG, N_ROWS, N_WRITES, N_READS, K = 20261008, 15000, 5000, 1200, 10
    res = {"config": {"n_rows": N_ROWS, "n_writes": N_WRITES, "n_reads": N_READS,
                      "k": K, "embed_cost_us": 150.0, "think_us": 20000.0}}
    print("=" * 118)
    print("实验一：异步索引（滞后 L） vs 全局锁 vs SCR（本文）")
    print("=" * 118)
    print(HEAD)
    print("-" * 118)
    exp1 = []
    for lag in (0, 10, 50, 200, 500):
        t, _ = build(N_ROWS, random.Random(RNG))
        idx = AsyncIndex(lag=lag)
        attach(idx, t)
        r = run_workload(t, idx, random.Random(RNG), N_WRITES, N_READS, K)
        r["scheme"] = "异步索引 L=%d" % lag
        exp1.append(r)
        print(row(r["scheme"], r))

    t, _ = build(N_ROWS, random.Random(RNG))
    idx = LockIndex()
    attach(idx, t)
    r = run_workload(t, idx, random.Random(RNG), N_WRITES, N_READS, K,
                     lock=threading.Lock())
    r["scheme"] = "全局锁"
    exp1.append(r)
    print(row(r["scheme"], r))

    for gcl in (False, True):
        t, _ = build(N_ROWS, random.Random(RNG))
        idx = VersionedIndex(gc_enabled=gcl)
        attach(idx, t)
        r = run_workload(t, idx, random.Random(RNG), N_WRITES, N_READS, K)
        raw = idx.index_bytes()
        if gcl:
            idx.gc(oldest_active_snapshot=0)
            r["pruned_versions"] = idx.pruned
            r["index_MB_after_gc"] = round(idx.index_bytes() / 1e6, 3)
            r["gc_shrink_ratio"] = round(idx.index_bytes() / max(1, raw), 4)
        else:
            r["pruned_versions"] = 0
        r["scheme"] = "SCR-exact 回收=%s" % ("开" if gcl else "关")
        exp1.append(r)
        print(row(r["scheme"], r))
    res["exp1"] = exp1

    print()
    print("=" * 118)
    print("实验二：有界陈旧 —— 陈旧与一致性正交（异常率对 δ 不敏感）")
    print("=" * 118)
    print(HEAD)
    print("-" * 118)
    exp2 = []
    for delta in (0, 10, 50, 200, 500, 1000):
        t, _ = build(N_ROWS, random.Random(RNG))
        idx = VersionedIndex()
        attach(idx, t)
        r = run_workload(t, idx, random.Random(RNG), N_WRITES, N_READS, K,
                         mode="bounded", delta=delta)
        r["scheme"] = "SCR-bounded δ=%d" % delta
        r["delta"] = delta
        exp2.append(r)
        print(row(r["scheme"], r))
    res["exp2"] = exp2

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("\nsaved ->", OUT)


if __name__ == "__main__":
    main()
