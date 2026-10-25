#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E8 -- multi-seed replication of the v1 main sweep (reuses the v1 driver).

For each seed: builds the 15k-row table, runs five schemes through the exact
v1 workload (concurrent writer, 20 ms deliberation per episode, k=10):
async L in {50, 200, 500}, global lock, SCR-exact.
Output: paper/exp/results/results_v2_multiseed.json
"""
from __future__ import annotations

import json
import os
import random
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.abspath(os.path.join(HERE, "..", "..", "proto"))
sys.path.insert(0, PROTO)

from engine import AsyncIndex, LockIndex, VersionedIndex  # noqa: E402
import run_experiments as R  # noqa: E402

SEEDS = (20261008, 20261009, 20261010)
N_ROWS, N_WRITES, N_READS, K = 15000, 5000, 1200, 10
SCHEMES = ("async50", "async200", "async500", "lock", "scr")
OUT = os.path.abspath(os.path.join(HERE, "..", "results", "results_v2_multiseed.json"))


def run_one(seed, scheme):
    t, _ = R.build(N_ROWS, random.Random(seed))
    lock = None
    if scheme.startswith("async"):
        idx = AsyncIndex(lag=int(scheme[5:]))
    elif scheme == "lock":
        idx = LockIndex()
        lock = threading.Lock()
    else:
        idx = VersionedIndex()
    R.attach(idx, t)
    r = R.run_workload(t, idx, random.Random(seed), N_WRITES, N_READS, K, lock=lock)
    r["scheme"] = scheme
    r["seed"] = seed
    return r


def main():
    t0 = time.time()
    runs = []
    for seed in SEEDS:
        for scheme in SCHEMES:
            r = run_one(seed, scheme)
            runs.append(r)
            print("seed=%d %-10s anom=%.4f recall=%.4f p50=%.1fus" % (
                seed, scheme, r["anomaly_episode_rate"], r["recall_at_k"], r["p50_us"]),
                flush=True)
    agg = {}
    for scheme in SCHEMES:
        vals = [r for r in runs if r["scheme"] == scheme]
        agg[scheme] = {}
        for key in ("anomaly_episode_rate", "recall_at_k", "citation_validity", "p50_us"):
            a = np.asarray([v[key] for v in vals], dtype=float)
            agg[scheme][key] = {"mean": round(float(a.mean()), 6),
                                "std": round(float(a.std(ddof=0)), 6),
                                "values": [float(x) for x in a]}
    out = {"generated_by": "paper/exp/skeleton/e8_multiseed.py",
           "config": {"n_rows": N_ROWS, "n_writes": N_WRITES, "n_reads": N_READS,
                      "k": K, "seeds": list(SEEDS), "think_us": 20000.0,
                      "embed_cost_us": 150.0},
           "runs": runs, "aggregate": agg,
           "runtime_s": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("\nsaved ->", OUT, "(%.1fs)" % out["runtime_s"])


if __name__ == "__main__":
    main()
