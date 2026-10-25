#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E9 driver: tear rate vs. compaction blast radius x beam width x link repair.

Writes ../results/results_v2_e9.json (merged into results_v2.json by run_all.py).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

import experiments as E

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "results", "results_v2_e9.json"))

SEEDS = [11, 22]
FRACS = [0.05, 0.10, 0.20, 0.40]
EFS = [16, 24, 48]
DIVS = [True, False]
EPISODES = 200


def agg(rows, key):
    v = [r[key] for r in rows]
    return {"mean": round(float(np.mean(v)), 6), "std": round(float(np.std(v, ddof=1)), 6), "values": v}


def main():
    t0 = time.time()
    runs = []
    for div in DIVS:
        for ef in EFS:
            for frac in FRACS:
                group = []
                for sd in SEEDS:
                    r = E.e9_run(sd, frac, n_episodes=EPISODES, ef=ef,
                                 rebuild_diverse=div)
                    runs.append(r)
                    group.append(r)
                    print("  frac=%.2f ef=%-3d div=%-5s seed=%-3d tear=%.4f jac=%.5f"
                          % (frac, ef, div, sd, r["tear_rate"], r["jaccard_mean"]),
                          flush=True)
                agg_row = {
                    "merge_frac": frac, "ef": ef, "rebuild_diverse": bool(div),
                    "tear_rate": agg(group, "tear_rate"),
                    "jaccard_mean": agg(group, "jaccard_mean"),
                    "straddle_rate": agg(group, "straddle_rate"),
                    "filter_violations_total": int(sum(r["filter_violations"] for r in group)),
                    "episodes_total": int(sum(r["episodes"] for r in group)),
                    "seeds": SEEDS,
                }
                AGG.append(agg_row)
    out = {
        "generated_by": "run_e9.py",
        "config": {"seeds": SEEDS, "fracs": FRACS, "efs": EFS,
                   "rebuild_diverse": DIVS, "episodes_per_run": EPISODES},
        "aggregate": AGG,
        "runs": runs,
        "runtime_s": round(time.time() - t0, 1),
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("saved ->", OUT, "(%.1fs)" % out["runtime_s"])


AGG = []

if __name__ == "__main__":
    main()
