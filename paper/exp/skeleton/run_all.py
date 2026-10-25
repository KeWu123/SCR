#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run E6 (ef x rebuild-strength sweep + headline) and E7, dump results_v2.json.

  python run_all.py               # full run
  SCR_QUICK=1 python run_all.py   # fast smoke run (small sizes)
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from experiments import e6_run, e7_run

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "results", "results_v2.json"))
QUICK = os.environ.get("SCR_QUICK") == "1"


def ms(xs):
    a = np.asarray(xs, dtype=float)
    return round(float(a.mean()), 6), round(float(a.std(ddof=0)), 6)


def main():
    t0 = time.time()
    if QUICK:
        e6_seeds, sweep_efs = (11, 22), (24, 48)
        e6_kw, e7_kw, e7_seeds = dict(N=1200, n_episodes=80), dict(N=2000, nq=40), (7, 8)
        head_extra = []
    else:
        e6_seeds, sweep_efs = (11, 22), (24, 32, 48, 64)
        e6_kw, e7_kw, e7_seeds = {}, {}, (7, 8)
        head_extra = [33]                       # third headline seed beyond the sweep

    # ---------------- E6 sweep: ef x rebuild strength ----------------
    print("=" * 96)
    print("E6 sweep -- structural tear vs beam width ef x rebuild strength")
    print("=" * 96)
    sweep = []
    for seed in e6_seeds:
        for div in (True, False):
            for ef in sweep_efs:
                r = e6_run(seed, ef=ef, rebuild_diverse=div, **e6_kw)
                sweep.append(r)
                print("seed=%d ef=%2d diverse=%-5s tear=%.4f str=%.3f rec=%.3f (%.0fs)" % (
                    seed, ef, div, r["native_tear_rate_set"], r["straddle_rate"],
                    r["native_recall_r1"], time.time() - t0), flush=True)

    sweep_rows = []
    for div in (True, False):
        for ef in sweep_efs:
            rs = [r for r in sweep if r["ef"] == ef and r["rebuild_diverse"] == div]
            m, s = ms([r["native_tear_rate_set"] for r in rs])
            m2, s2 = ms([r["native_tear_given_straddle"] for r in rs])
            m3, s3 = ms([r["native_recall_r1"] for r in rs])
            sweep_rows.append({"ef": ef, "rebuild_diverse": div,
                               "tear_rate": m, "tear_rate_std": s,
                               "tear_given_straddle": m2, "tear_given_straddle_std": s2,
                               "recall": m3, "recall_std": s3,
                               "seeds": [r["seed"] for r in rs]})
    print("\nE6 sweep aggregate:")
    for row in sweep_rows:
        print("  ef=%2d diverse=%-5s tear=%.4f+/-%.4f  tear|straddle=%.4f  recall=%.3f" % (
            row["ef"], row["rebuild_diverse"], row["tear_rate"], row["tear_rate_std"],
            row["tear_given_straddle"], row["recall"]))

    # ---------------- E6 headline: ef=48, diverse=True ----------------
    headline_runs = [r for r in sweep if r["ef"] == 48 and r["rebuild_diverse"]]
    for seed in head_extra:
        r = e6_run(seed, ef=48, rebuild_diverse=True, **e6_kw)
        headline_runs.append(r)
        print("headline seed=%d tear=%.4f rec=%.3f (%.0fs)" % (
            seed, r["native_tear_rate_set"], r["native_recall_r1"], time.time() - t0),
            flush=True)
    e6_agg = {"seeds": [r["seed"] for r in headline_runs]}
    for key in ("straddle_rate", "native_tear_rate_set", "native_tear_rate_ord",
                "native_tear_given_straddle", "native_tear_given_nostraddle",
                "native_jaccard_mean", "native_jaccard_min",
                "native_recall_r1", "scr_recall", "retained_MB_native",
                "retained_MB_scr", "retained_MB_fullcopy_modeled"):
        m, s = ms([r[key] for r in headline_runs])
        e6_agg[key] = {"mean": m, "std": s, "values": [r[key] for r in headline_runs]}
    print("\nE6 headline (ef=48, diverse rebuild, %d seeds):" % len(headline_runs))
    for key, v in e6_agg.items():
        if isinstance(v, dict):
            print("  %-30s %.4f +/- %.4f %s" % (key, v["mean"], v["std"], v["values"]))

    # ---------------- E7 ----------------
    print()
    print("=" * 96)
    print("E7 -- navigability vs visible fraction f")
    print("=" * 96)
    e7 = [e7_run(s, **e7_kw) for s in e7_seeds]
    fs = [row["visible_frac"] for row in e7[0]["rows"]]
    e7_rows = []
    for i, f in enumerate(fs):
        row = {"visible_frac": f, "visible_nodes": e7[0]["rows"][i]["visible_nodes"]}
        for key in ("recall_all", "recall_through", "recall_post", "recall_strict",
                    "post_fill_rate", "visible_reach_frac"):
            m, s = ms([e7[j]["rows"][i][key] for j in range(len(e7))])
            row[key] = m
            row[key + "_std"] = s
        e7_rows.append(row)
    print("\n%-6s %-9s %-9s %-9s %-9s %-9s %-9s" % (
        "f", "all", "through", "post", "strict", "postfill", "reach"))
    for row in e7_rows:
        print("%-6.2f %-9.3f %-9.3f %-9.3f %-9.3f %-9.3f %-9.3f" % (
            row["visible_frac"], row["recall_all"], row["recall_through"],
            row["recall_post"], row["recall_strict"], row["post_fill_rate"],
            row["visible_reach_frac"]))

    out = {
        "generated_by": "paper/exp/skeleton/run_all.py",
        "quick": QUICK,
        "config": {"e6_sweep_seeds": list(e6_seeds), "e6_sweep_efs": list(sweep_efs),
                   "e6_headline_seeds": e6_agg["seeds"], "e7_seeds": list(e7_seeds),
                   "e6_kw": e6_kw, "e7_kw": e7_kw},
        "E6": {"sweep": sweep_rows, "sweep_runs": sweep,
               "headline_runs": headline_runs, "headline_aggregate": e6_agg},
        "E7": {"runs": e7, "aggregate": e7_rows},
        "runtime_s": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("\nsaved ->", OUT, "(%.1fs)" % out["runtime_s"])


if __name__ == "__main__":
    main()
