#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Cross-store, multi-retrieval node-version episode driver.

Each episode combines a relational row read with two retrieval calls:

    1. S <- Begin()                       (snapshot token)
    2. r <- pick a row;  v_tab <- Read(r, S)      (a REAL relational read)
    3. q <- v_tab.vec                     (query derived from that row read)
    4. deliberate for `think_us`          (writes commit during this window)
    5. K1 <- Retrieve(q, k, S)            (first retrieval)
    6. K2 <- Retrieve(q, k, S)            (second, later tool call)
    7. validate

Anomalies classified
--------------------
  A1/A2/A3  as before, but scored over K1 and K2 (node-level, index vs truth).
   Observed-read conflict (not a general no-single-state certificate):
         The episode holds two observations of row r -- its own relational read
         (v_tab) and the index's entry for r, if r is returned.  A conflict fires when
        they disagree: the index returns r at a version other than the one the
        episode read, or returns r when the episode's read says r is deleted.
        This is the genuinely cross-store condition, and it is evaluated against
        the episode's OWN read -- not against an oracle.
   Retrieval disagreement (not an adjacency-level A5 certificate):
        K1 != K2 for the same query and the same token.  In an exact index this
        can only come from the index having moved between the two calls, i.e.
        from non-pinned structure/visibility.
  magnitude:
        |K1 symmetric-difference K2|, so that prevalence and magnitude are
        reported separately.  An anomaly that moves one near-duplicate is not
        the same event as one that moves all ten.

Run: python run_episodes_v3.py (writes results_episodes_v3_corrected.json).
The original results_episodes_v3.json is preserved as a historical artifact.
"""
from __future__ import annotations

import json
import os
import random
import threading
import time
from bisect import bisect_right

import numpy as np

from engine import AsyncIndex, BaseTable, LockIndex, VersionedIndex, embed
from run_experiments import (Clock, TruthCache, build, classify_fast,
                             make_text, truth_topk_fast)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_episodes_v3_corrected.json")


class RealLock:
    """Serialises the prototype's own data-structure mutation.

    This is NOT a design lock on the read path: it models the atomicity a real
    index provides when it publishes a new version.  Without it, a Python-level
    race between the writer's append and the reader's array rebuild manufactures
    anomalies for every scheme, including the one that cannot have them.
    """

    def __init__(self):
        self._l = threading.Lock()

    def __enter__(self):
        self._l.acquire()
        return self

    def __exit__(self, *a):
        self._l.release()
        return False


class DummyLock:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class RowIndex:
    """Per-row sorted (ts, version) list so an as-of read is O(log n).

    The writer keeps appending versions while episodes run, so `refresh` must be
    called before each read to include newly appended versions. Commit timestamps increase
    monotonically, so appending keeps each row's list sorted.
    """

    def __init__(self, table: BaseTable):
        self.by_row = {}
        self.ts_cache = {}
        self._seen = 0
        self.refresh(table)

    def refresh(self, table: BaseTable):
        vers = table.versions
        end = len(vers)
        for i in range(self._seen, end):
            v = vers[i]
            lst = self.by_row.setdefault(v.row_id, [])
            lst.append(v)
            if len(lst) >= 2 and lst[-2].commit_ts > v.commit_ts:
                lst.sort(key=lambda x: x.commit_ts)
            self.ts_cache[v.row_id] = [x.commit_ts for x in lst]
        self._seen = end

    def read_at(self, r, ts):
        lst = self.by_row.get(r)
        if not lst:
            return None
        i = bisect_right(self.ts_cache[r], ts) - 1
        if i < 0:
            return None
        v = lst[i]
        return None if v.deleted else v


def run_episode_workload(table, index, rng, n_writes, n_reads, k, mode="exact",
                         delta=0, lock=None, embed_cost_us=150.0, think_us=20000.0,
                         max_row=None, np_seed=0, between_calls_us=1000.0):
    """`rng` is a random.Random (as in the v1 driver); `np_seed` seeds the
    NumPy generator used only for the small query perturbation."""
    lock = lock or RealLock()
    clock = Clock(t=n_writes)
    tc = TruthCache(table)
    max_row = max_row or (max(table.latest.keys()) + 1 if table.latest else 1)
    nprng = np.random.default_rng(np_seed)
    writer_seed, reader_seed = rng.getrandbits(64), rng.getrandbits(64)
    writer_rng, reader_rng = random.Random(writer_seed), random.Random(reader_seed)

    lat, wait = [], []
    n_ep = 0
    n_a1 = n_a2 = n_a3 = n_a4 = n_a5 = 0
    n_anom_any = 0
    n_torn_only = 0                       # A4 with no A1/A2/A3: the pure case
    diffs = []                            # |K1 xor K2|
    jac = []
    rec1_num = rec1_den = 0
    a4_detail = {"resolved": 0, "deleted": 0, "absent": 0}
    stop = threading.Event()
    writer_errors = []
    n_valid = n_dirty = n_superseded = n_phantom = 0
    n_invalid_bundle = 0
    n_overlapping = n_concurrent = 0
    commits_completed = 0
    witnesses = []

    def writer():
        nonlocal commits_completed
        c = 0
        try:
            while c < n_writes and not stop.is_set():
                rid = writer_rng.randrange(max_row)
                with lock:
                    if writer_rng.random() < 0.12:
                        rv = table.delete(rid, ts=clock.tick())
                    else:
                        rv = table.update(rid, make_text(writer_rng), ts=clock.tick())
                    index.on_write(rv)
                    commits_completed += 1
                c += 1
                # Pace the next write outside the atomic publication section;
                # otherwise an unfair lock can starve Begin until all writes finish.
                if embed_cost_us:
                    time.sleep(embed_cost_us / 1e6)
        except BaseException as exc:
            writer_errors.append(exc)

    wt = threading.Thread(target=writer, daemon=True)
    wt.start()

    try:
        with lock:
            ri = RowIndex(table)
        while n_ep < n_reads:
            # Begin and the first relational observation use one consistent cache.
            # The writer cannot commit between refresh, token capture, and Read.
            with lock:
                S_now = clock.t
                S = max(0, S_now - delta) if (mode == "bounded" and delta) else S_now
                ri.refresh(table)
                r = reader_rng.randrange(max_row)
                v_tab = ri.read_at(r, S)
                n_concurrent += int(commits_completed < n_writes)

            # ---- step 3: the query derives from what the episode read --------- #
            if v_tab is not None:
                q = v_tab.vec.copy()
            else:
                q = embed(make_text(reader_rng))
            q = q + 0.02 * nprng.normal(size=q.shape).astype(np.float32)
            q /= np.linalg.norm(q)

            if think_us:
                time.sleep(think_us / 1e6)        # writes commit during this window

            # ---- step 5/6: two retrievals in the same episode ----------------- #
            retrievals = []
            for call in range(2):
                if call and between_calls_us:
                    time.sleep(between_calls_us / 1e6)
                w0 = time.perf_counter_ns()
                with lock:
                    w1 = time.perf_counter_ns()
                    retrievals.append(index.query(q, k, S))
                    lat.append((time.perf_counter_ns() - w1) / 1000.0)
                wait.append((w1 - w0) / 1000.0)
            K1, K2 = retrievals
            n_ep += 1

            # ---- A1/A2/A3 over both result sets ------------------------------ #
            # These are explicitly oracle scores, outside the observed episode.
            with lock:
                d1 = classify_fast(K1, tc, S)
                d2 = classify_fast(K2, tc, S)
                tk = truth_topk_fast(tc, q, S, k)
                n_overlapping += int(clock.t > S_now)
            n_valid += d1[0] + d2[0]
            n_dirty += d1[1] + d2[1]
            n_superseded += d1[2] + d2[2]
            n_phantom += d1[3] + d2[3]
            a1 = d1[1] + d2[1]
            a2 = d1[2] + d2[2]
            a3 = d1[3] + d2[3]
            n_a1 += int(a1 > 0)
            n_a2 += int(a2 > 0)
            n_a3 += int(a3 > 0)

            # Compare only the relation row the episode actually observed. A fixed-
            # snapshot conflict does not establish that no alternative state fits.
            torn = False
            n_disagree = 0
            for rid_, vid_, _ in set((x[0], x[1], None) for x in list(K1) + list(K2)):
                if rid_ != r:
                    continue
                if v_tab is None or v_tab.ver_id != vid_:
                    n_disagree += 1
                    if v_tab is None:
                        a4_detail["deleted"] += 1
                    else:
                        a4_detail["resolved"] += 1
            if n_disagree:
                torn = True
                n_a4 += 1

            # Exact-index disagreement compares provenance, including versions.
            s1 = {x[0] for x in K1}
            s2 = {x[0] for x in K2}
            c1 = {(x[0], x[1]) for x in K1}
            c2 = {(x[0], x[1]) for x in K2}
            if c1 != c2:
                n_a5 += 1
            diffs.append(len(s1 ^ s2))
            uni = len(s1 | s2)
            jac.append(len(s1 & s2) / uni if uni else 1.0)

            node_anom = (a1 + a2 + a3) > 0
            invalid_bundle = node_anom or c1 != c2
            n_invalid_bundle += int(invalid_bundle)
            if node_anom or torn or c1 != c2:
                n_anom_any += 1
            if torn and not node_anom and c1 == c2:
                n_torn_only += 1
            if invalid_bundle and len(witnesses) < 6:
                witnesses.append({"episode": n_ep, "snapshot": S, "observed_row": r,
                                  "observed_version": None if v_tab is None else v_tab.ver_id,
                                  "citations_1": sorted(c1), "citations_2": sorted(c2),
                                  "a1": a1 > 0, "a2": a2 > 0, "a3": a3 > 0,
                                  "observed_read_conflict": torn,
                                  "interpretation": "fixed-snapshot violation; no no-state certificate"})
            rec1_num += len(s1 & tk)
            rec1_den += len(tk)

    except BaseException:
        # Reader failure must stop background mutation before the error escapes.
        # Preserve the reader exception; never produce scores for a failed run.
        stop.set()
        wt.join(timeout=10)
        if wt.is_alive():
            raise RuntimeError("reader failed and writer did not stop; result is incomplete")
        raise

    # Normal reader completion must not cancel the requested writer workload.
    # The figure inputs record complete trajectories, including a short-reader
    # run whose remaining commits occur after the last episode.
    writes_at_reader_end = commits_completed
    wt.join(timeout=10)
    if wt.is_alive():
        stop.set()
        wt.join(timeout=10)
        raise RuntimeError("writer did not stop; result is incomplete")
    if writer_errors:
        raise RuntimeError("writer failed; result is incomplete") from writer_errors[0]
    if commits_completed != n_writes:
        raise RuntimeError("writer trajectory incomplete; refuse to score")
    lat.sort()
    wait.sort()

    def pct(a, p):
        return a[min(len(a) - 1, int(len(a) * p))] if a else 0.0

    n = max(1, n_ep)
    return {
        "episodes": n_ep,
        "anomaly_episode_rate": round(n_anom_any / n, 6),
        "a1_dirty_episode_rate": round(n_a1 / n, 6),
        "a2_superseded_episode_rate": round(n_a2 / n, 6),
        "a3_phantom_episode_rate": round(n_a3 / n, 6),
        "observed_read_conflict_rate": round(n_a4 / n, 6),
        "observed_read_conflict_only_rate": round(n_torn_only / n, 6),
        "observed_read_conflict_detail": a4_detail,
        "citation_disagreement_rate": round(n_a5 / n, 6),
        "bundle_invalidation_rate": round(n_invalid_bundle / n, 6),
        "n_invalid_bundles": n_invalid_bundle,
        "anomaly_count": n_anom_any,
        "a1_count": n_a1, "a2_count": n_a2, "a3_count": n_a3,
        "citation_validity": round(n_valid / max(1, n_valid + n_dirty + n_superseded + n_phantom), 6),
        "writer_seed": writer_seed, "reader_seed": reader_seed,
        "writes_completed": commits_completed,
        "writes_requested": n_writes,
        "writes_at_reader_end": writes_at_reader_end,
        "episodes_started_during_writer": n_concurrent,
        "episodes_overlapped_write": n_overlapping,
        "between_calls_us": between_calls_us,
        "witnesses": witnesses,
        "mean_symmetric_difference": round(float(np.mean(diffs)), 6),
        "max_symmetric_difference": int(np.max(diffs)) if diffs else 0,
        "frac_disagreement_gt1": round(float(np.mean([d > 1 for d in diffs])), 6),
        "mean_jaccard_k1_k2": round(float(np.mean(jac)), 6),
        "min_jaccard_k1_k2": round(float(np.min(jac)), 6),
        "recall_at_k": round(rec1_num / max(1, rec1_den), 6),
        "p50_us": round(pct(lat, 0.5), 1),
        "p99_us": round(pct(lat, 0.99), 1),
        "scheme": "unnamed",
    }


def main(output_path=None):
    t0 = time.time()
    N = 15000
    n_writes, n_reads, k = 5000, 1200, 10
    seeds = [20261008, 20261009, 20261010]
    schemes = [
        ("async50", "async", 50),
        ("async200", "async", 200),
        ("async500", "async", 500),
        ("lock", "lock", None),
        ("scr", "scr", None),
    ]
    runs = []
    for scheme, kind, lag in schemes:
        for sd in seeds:
            rng = random.Random(sd)
            table, _ = build(N, rng)
            max_row = N
            if kind == "async":
                idx = AsyncIndex(lag=lag)
                for r in range(N):
                    idx.on_write(table.versions[r])
                lk = RealLock()
            elif kind == "lock":
                idx = LockIndex()
                for r in range(N):
                    idx.on_write(table.versions[r])
                lk = RealLock()
            else:
                idx = VersionedIndex(gc_enabled=True)
                for r in range(N):
                    idx.on_write(table.versions[r])
                lk = RealLock()
            res = run_episode_workload(table, idx, rng, n_writes, n_reads, k,
                                       mode="exact", lock=lk, max_row=max_row,
                                       np_seed=sd)
            res["scheme"] = scheme
            res["seed"] = sd
            runs.append(res)
            print("%-9s seed=%d  any=%.4f  observed_conflict=%.4f  citation_change=%.4f  "
                  "magsym=%.2f  recall=%.4f" %
                  (scheme, sd, res["anomaly_episode_rate"],
                   res["observed_read_conflict_rate"], res["citation_disagreement_rate"],
                   res["mean_symmetric_difference"], res["recall_at_k"]), flush=True)

    agg = {}
    for scheme, _, _ in schemes:
        rs = [r for r in runs if r["scheme"] == scheme]
        keys = ["anomaly_episode_rate", "a1_dirty_episode_rate",
                "a2_superseded_episode_rate", "a3_phantom_episode_rate",
                 "observed_read_conflict_rate", "observed_read_conflict_only_rate",
                 "citation_disagreement_rate", "bundle_invalidation_rate",
                 "citation_validity", "mean_symmetric_difference",
                 "frac_disagreement_gt1", "recall_at_k", "p50_us", "p99_us"]
        agg[scheme] = {}
        for kk in keys:
            v = [r[kk] for r in rs]
            agg[scheme][kk] = {"mean": round(float(np.mean(v)), 6),
                               "std": round(float(np.std(v, ddof=1)), 6),
                               "values": v}
    out = {"generated_by": "proto/run_episodes_v3.py",
           "protocol": "locked Begin+relation read -> query from that observed row "
                       "-> 20ms deliberation -> retrieval -> 1ms tool gap -> retrieval; "
                       "A1-A3 and bundle validity are fixed-snapshot oracle scores; "
                       "observed-read conflict uses only the observed row and is not a no-state certificate",
           "config": {"n_rows": N, "n_writes": n_writes, "n_reads": n_reads,
                      "k": k, "seeds": seeds, "think_us": 20000.0,
                      "between_calls_us": 1000.0, "latency_scope": "one retrieval call",
                      "std_ddof": 1, "rng_streams": "separate deterministic writer and reader streams"},
           "aggregate": agg, "runs": runs, "runtime_s": round(time.time() - t0, 1)}
    output_path = output_path or OUT
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("\nsaved ->", output_path, "(%.1fs)" % out["runtime_s"])


if __name__ == "__main__":
    main()
