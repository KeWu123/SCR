"""pgvector recall-confounding decomposition (LanceDB-parity check).

Purpose
-------
The paper's pgvector headline (matched tear 0.465 vs same-order floor 0.068)
and its LanceDB counterpart are both "any-difference" (A4) rates.  For LanceDB
we already showed that a mean-recall-matched pair (|dr| <= 0.004) is NOT
query-matched, so part of the tear is mechanically forced by per-query recall
dispersion.  pgvector never recorded per-query recall, so the same
decomposition could not be done there.  This script supplies it.

What it does
------------
1. Re-runs the pgvector same-order rebuild floor (3 seeds x 3 pass-pairs,
   200 episodes each) using the SAME protocol as engine_matched2_pg.py, now
   with per-query recalls recorded (reuses M.run_pg_floor, which passes
   `exact` into M.compare).  The floor raw top-10 were never stored, so this
   step is a genuine re-measurement and also tests run-to-run determinism.
2. Re-derives the pgvector effort sweeps (27 cells) AND the matched /
   pollution-control cells from the raw top-10 stored in
   results_engine_multiseed.json -- the SAME source the paper's aggregates
   come from -- adding per-query recalls.  Their aggregate tear is therefore
   identical to the paper by construction (asserted below); only the
   per-query fields are new.
3. Decomposes every cell into the equal-recall subset (recall_A == recall_B
   for that query) and the unequal-recall subset, exactly as the LanceDB
   study does.

Output
------
results_engine_matched2_pg_decomp.json  (NEW file; nothing pre-existing is
written or overwritten).

Raw top-10 keys note: the ef sweep's ef=64 answer lives under
`A_idorder_rebuild_efc64_ef64`, because `A_idorder_efc64_ef64` is later
OVERWRITTEN by the efc sweep's efc=64 rebuild.  Verified by tear reproduction.
"""
import json
import os
import time
import datetime
import importlib.util

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NEW = os.path.join(HERE, "results_engine_matched2_pg_decomp.json")
OLD = os.path.join(HERE, "results_engine_matched2.json")
MS = os.path.join(HERE, "results_engine_multiseed.json")

SEEDS = [20261008, 20261009, 20261010]
EF_SWEEP = [64, 128, 256, 512]
EFC_SWEEP = [64, 96, 128, 192, 256]

_spec = importlib.util.spec_from_file_location(
    "emp", os.path.join(HERE, "engine_matched2_pg.py"))
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)


def log(msg):
    print("[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), msg), flush=True)


# --------------------------------------------------------------------------- #
def decompose(per_episode):
    """Split episodes by whether the two sides have equal recall for that query.

    `tear` is the any-difference rate (set(top10_A) != set(top10_B)); it is the
    same quantity the paper reports.  In the unequal-recall subset it is
    mechanically 1.0 (equal recall is necessary for equal sets), which the
    script asserts.
    """
    n = len(per_episode)
    eq = [e for e in per_episode
          if round(e["recall_A_vs_exact"], 6) == round(e["recall_B_vs_exact"], 6)]
    ne = [e for e in per_episode
          if round(e["recall_A_vs_exact"], 6) != round(e["recall_B_vs_exact"], 6)]

    def tear(sub):
        return (sum(1 for e in sub if not e["set_equal"]) / len(sub)) if sub else None

    tear_all = sum(1 for e in per_episode if not e["set_equal"]) / n
    tear_eq = tear(eq)
    tear_ne = tear(ne)
    if ne:
        assert tear_ne == 1.0, "unequal recall must force set inequality"
    ra = float(np.mean([e["recall_A_vs_exact"] for e in per_episode]))
    rb = float(np.mean([e["recall_B_vs_exact"] for e in per_episode]))
    return {
        "n": n,
        "tear_all": round(tear_all, 4),
        "n_eq": len(eq),
        "tear_eq": (round(tear_eq, 4) if tear_eq is not None else None),
        "n_ne": len(ne),
        "tear_ne": (round(tear_ne, 4) if tear_ne is not None else None),
        "frac_unequal": round(len(ne) / n, 4),
        "recall_A_mean": round(ra, 4),
        "recall_B_mean": round(rb, 4),
        "recall_gap_mean": round(abs(ra - rb), 4),
    }


def _rows_ok(cell):
    return all("recall_A_vs_exact" in e and "recall_B_vs_exact" in e
               for e in cell["per_episode"])


# --------------------------------------------------------------------------- #
def derive_from_multiseed():
    """Build the 27 sweep cells + matched/pollution controls from stored raw."""
    ms = json.load(open(MS, encoding="utf-8"))
    cells = {}
    checks = []
    for seed in SEEDS:
        g = ms["cells"]["G_s%d" % seed]
        R = g["raw_top10"]
        _, _, exact = M.make_data(seed)
        B = R["B_perm_order_efc64_ef64"]

        # ---- ef sweep (27-cell family, part 1) ---------------------------- #
        for ef in EF_SWEEP:
            # ef=64 uses the CLUSTER-back rebuild key (see module docstring).
            akey = ("A_idorder_rebuild_efc64_ef64" if ef == 64
                    else "A_idorder_efc64_ef%d" % ef)
            A = R[akey]
            c = M.compare(A, B, exact)
            stored = g["recall_matched_query_effort"]["by_ef"][str(ef)]
            checks.append((seed, "ef%d" % ef, c["tear_rate_set_ineq"],
                           stored["tear_rate_set_ineq"]))
            nm = "sweep_pg_s%d_efsearch_ef%d" % (seed, ef)
            cells[nm] = {
                "cell": nm,
                "note": ("pgvector search-effort sweep, per-query-recall"
                         " decomposition. A = %s, B = perm-order efc=64 at ef=64."
                         % akey),
                "sweep_axis": "ef_search", "ef_search": ef,
                "raw_A_key": akey, "raw_B_key": "B_perm_order_efc64_ef64",
                "recall_A_vs_exact": round(
                    float(np.mean([e["recall_A_vs_exact"] for e in c["per_episode"]])), 4),
                "tear_rate_set_ineq": c["tear_rate_set_ineq"],
                "mean_symdiff": c["mean_symdiff"],
                "episodes": c["episodes"],
                "per_episode": c["per_episode"],
            }

        # ---- efc sweep (27-cell family, part 2) --------------------------- #
        for efc in EFC_SWEEP:
            akey = "A_idorder_efc%d_ef64" % efc
            A = R[akey]
            c = M.compare(A, B, exact)
            stored = g["efc_sweep_generationA"]["sweep"][str(efc)]
            checks.append((seed, "efc%d" % efc, c["tear_rate_set_ineq"],
                           stored["tear_rate_set_ineq_vs_B"]))
            nm = "sweep_pg_s%d_buildeffort_efc%d" % (seed, efc)
            cells[nm] = {
                "cell": nm,
                "note": ("pgvector build-effort sweep, per-query-recall"
                         " decomposition. A = %s, B = perm-order efc=64." % akey),
                "sweep_axis": "ef_construction", "efc": efc,
                "raw_A_key": akey, "raw_B_key": "B_perm_order_efc64_ef64",
                "recall_A_vs_exact": round(
                    float(np.mean([e["recall_A_vs_exact"] for e in c["per_episode"]])), 4),
                "tear_rate_set_ineq": c["tear_rate_set_ineq"],
                "mean_symdiff": c["mean_symdiff"],
                "episodes": c["episodes"],
                "per_episode": c["per_episode"],
            }

        # ---- matched headline cell (paper's 0.465) ------------------------ #
        A_clean = R["A_clean_id_order_efc64_ef64"]
        c = M.compare(A_clean, B, exact)
        stored = g["A_vs_B_clean"]["tear_rate_set_ineq"]
        checks.append((seed, "matched", c["tear_rate_set_ineq"], stored))
        nm = "matched_pg_s%d" % seed
        cells[nm] = {
            "cell": nm,
            "note": ("pgvector recall-matched headline: clean id-order A vs "
                     "perm-order B, both efc=64 ef=64, gap 0.001-0.0025."),
            "raw_A_key": "A_clean_id_order_efc64_ef64",
            "raw_B_key": "B_perm_order_efc64_ef64",
            "recall_A_vs_exact": round(
                float(np.mean([e["recall_A_vs_exact"] for e in c["per_episode"]])), 4),
            "tear_rate_set_ineq": c["tear_rate_set_ineq"],
            "mean_symdiff": c["mean_symdiff"],
            "episodes": c["episodes"],
            "per_episode": c["per_episode"],
        }

        # ---- pollution control (paper's 0.870) ---------------------------- #
        A_leg = R["A_legacy_polluted_ef64"]
        c = M.compare(A_clean, A_leg, exact)
        stored = g["pollution_only_clean_vs_legacy_A"]["tear_rate_set_ineq"]
        checks.append((seed, "polluted", c["tear_rate_set_ineq"], stored))
        nm = "polluted_pg_s%d" % seed
        cells[nm] = {
            "cell": nm,
            "note": ("pgvector polluted-index control: clean vs legacy "
                     "(dead-tuple) A, same heap order, unmatched recall."),
            "raw_A_key": "A_clean_id_order_efc64_ef64",
            "raw_B_key": "A_legacy_polluted_ef64",
            "recall_A_vs_exact": round(
                float(np.mean([e["recall_A_vs_exact"] for e in c["per_episode"]])), 4),
            "tear_rate_set_ineq": c["tear_rate_set_ineq"],
            "mean_symdiff": c["mean_symdiff"],
            "episodes": c["episodes"],
            "per_episode": c["per_episode"],
        }
    return cells, checks


# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    old = json.load(open(OLD, encoding="utf-8"))

    log("=== re-running pgvector same-order rebuild floor (fresh, per-query recall)")
    floor_cells, pg_runs = M.run_pg_floor()
    log("floor done: %d cells" % len(floor_cells))

    log("=== deriving sweeps + matched/pollution controls from multiseed raw top-10")
    derived_cells, checks = derive_from_multiseed()

    cells = {}
    cells.update(floor_cells)
    cells.update(derived_cells)

    # ---- consistency: derived tear must equal the stored aggregate ------ #
    bad = []
    for seed, tag, new_t, old_t in checks:
        if abs(new_t - old_t) > 1e-9:
            bad.append((seed, tag, new_t, old_t))
    assert not bad, "derived aggregate tear drifted from stored: %s" % bad
    log("derived-aggregate reproduction: %d/%d exact (all match)"
        % (len(checks), len(checks)))

    # ---- decomposition ------------------------------------------------- #
    per_cell = {}
    for k, v in cells.items():
        if "per_episode" in v and _rows_ok(v):
            per_cell[k] = decompose(v["per_episode"])

    def agg(prefix, seeds=None):
        vals = [per_cell[k] for k in per_cell
                if k.startswith(prefix)] if seeds is None else \
               [per_cell["%s_s%d" % (prefix, s)] for s in seeds
                if "%s_s%d" % (prefix, s) in per_cell]
        return vals

    def family_summary(keys):
        d = [per_cell[k] for k in keys if k in per_cell]
        if not d:
            return None
        return {
            "n_cells": len(d),
            "n_episodes_total": sum(x["n"] for x in d),
            "tear_all_mean": round(float(np.mean([x["tear_all"] for x in d])), 4),
            "tear_eq_mean": round(float(np.mean(
                [x["tear_eq"] for x in d if x["tear_eq"] is not None])), 4),
            "tear_eq_per_cell": [x["tear_eq"] for x in d],
            "frac_unequal_mean": round(float(np.mean(
                [x["frac_unequal"] for x in d])), 4),
            "frac_unequal_min": min(x["frac_unequal"] for x in d),
            "frac_unequal_max": max(x["frac_unequal"] for x in d),
            "tear_all_per_cell": [x["tear_all"] for x in d],
        }

    # family/level x seed aggregation
    agg_out = {}
    agg_out["floor_pg_parallel"] = family_summary(
        ["floor_pg_s%d_%s" % (s, t) for s in SEEDS
         for t in ("pass1_vs_pass2", "pass2_vs_pass3")])
    agg_out["floor_pg_serial"] = family_summary(
        ["floor_pg_s%d_pass4_vs_pass5" % s for s in SEEDS])
    agg_out["floor_pg_all"] = family_summary(
        ["floor_pg_s%d_%s" % (s, t) for s in SEEDS
         for t in ("pass1_vs_pass2", "pass2_vs_pass3", "pass4_vs_pass5")])
    agg_out["matched"] = family_summary(["matched_pg_s%d" % s for s in SEEDS])
    agg_out["polluted"] = family_summary(["polluted_pg_s%d" % s for s in SEEDS])
    for ef in EF_SWEEP:
        agg_out["efsearch_ef%d" % ef] = family_summary(
            ["sweep_pg_s%d_efsearch_ef%d" % (s, ef) for s in SEEDS])
    for efc in EFC_SWEEP:
        agg_out["buildeffort_efc%d" % efc] = family_summary(
            ["sweep_pg_s%d_buildeffort_efc%d" % (s, efc) for s in SEEDS])
    agg_out["sweep_ef_ef64"] = agg_out["efsearch_ef64"]
    agg_out["sweep_efc_efc64"] = agg_out["buildeffort_efc64"]

    # ---- consistency table: new (fresh floor) vs old -------------------- #
    cons = []
    for k in sorted(cells):
        if k.startswith("floor_pg_") and k in old["cells"]:
            nv, ov = cells[k], old["cells"][k]
            cons.append({
                "cell": k,
                "tear_new": nv["tear_rate_set_ineq"],
                "tear_old": ov["tear_rate_set_ineq"],
                "d_tear": round(nv["tear_rate_set_ineq"] - ov["tear_rate_set_ineq"], 4),
                "recall_1_new": nv["recall_1_vs_exact"],
                "recall_1_old": ov["recall_1_vs_exact"],
                "recall_2_new": nv["recall_2_vs_exact"],
                "recall_2_old": ov["recall_2_vs_exact"],
            })
    for k in sorted(derived_cells):
        if k in old["cells"]:
            nv, ov = derived_cells[k], old["cells"][k]
            cons.append({
                "cell": k,
                "tear_new": nv["tear_rate_set_ineq"],
                "tear_old": ov["tear_rate_set_ineq"],
                "d_tear": round(nv["tear_rate_set_ineq"] - ov["tear_rate_set_ineq"], 4),
                "recall_1_new": nv.get("recall_A_vs_exact"),
                "recall_1_old": ov.get("recall_A_vs_exact"),
                "recall_2_new": None,
                "recall_2_old": None,
            })

    # ---- run-to-run dispersion of the floor rerun ----------------------- #
    def te(keys):
        return [cells[k]["tear_rate_set_ineq"] for k in keys if k in cells]
    floor_disp = {
        "parallel_by_seed": {
            str(s): round(float(np.mean(te(["floor_pg_s%d_pass1_vs_pass2" % s,
                                            "floor_pg_s%d_pass2_vs_pass3" % s]))), 4)
            for s in SEEDS},
        "serial_by_seed": {
            str(s): cells["floor_pg_s%d_pass4_vs_pass5" % s]["tear_rate_set_ineq"]
            for s in SEEDS},
        "parallel_mean": round(float(np.mean(te(
            ["floor_pg_s%d_%s" % (s, t) for s in SEEDS
             for t in ("pass1_vs_pass2", "pass2_vs_pass3")]))), 4),
        "serial_mean": round(float(np.mean(te(
            ["floor_pg_s%d_pass4_vs_pass5" % s for s in SEEDS]))), 4),
    }

    # ---- meta ----------------------------------------------------------- #
    meta = {
        "purpose": ("pgvector per-query recall decomposition, LanceDB parity. "
                    "Records recall_A_vs_exact / recall_B_vs_exact per query "
                    "and splits each cell into the equal-recall and "
                    "unequal-recall subsets."),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script_changes": [
            "engine_matched2_pg.py: added per_query_recall(); compare() now "
            "takes an optional `exact` and appends recall_A_vs_exact / "
            "recall_B_vs_exact to every per-episode record (omitted when "
            "exact is None; the only call site, run_pg_floor, now passes exact). "
            "No other field or semantic changed.",
        ],
        "rerun": {
            "floor": ("genuine re-measurement on PostgreSQL 17.4 + pgvector "
                      "0.8.0, port 5439; 3 seeds x 3 pass-pairs x 200 episodes; "
                      "same protocol as engine_matched2_pg.py."),
            "sweeps_and_controls": ("re-derived from the stored raw top-10 in "
                                    "results_engine_multiseed.json (the source of "
                                    "the paper's aggregates); aggregate tear "
                                    "reproduced exactly, only per-query fields "
                                    "are new. See raw_top10-key note."),
            "seeds": SEEDS,
            "wall_seconds": None,
        },
        "raw_top10_key_note": ("The ef-sweep ef=64 answer is read from "
                               "A_idorder_rebuild_efc64_ef64: "
                               "A_idorder_efc64_ef64 is later overwritten by the "
                               "efc-sweep efc=64 rebuild, so it no longer holds the "
                               "ef-sweep's ef=64 list.  Verified by tear reproduction."),
        "decomposition_convention": (
            "For each query, eq := (recall_A_vs_exact == recall_B_vs_exact). "
            "tear_all = any-difference rate over all 200 episodes (the paper's "
            "quantity); tear_eq = same over the eq subset; tear_ne = same over "
            "the ne subset (mechanically 1.0); frac_unequal = n_ne / n."),
        "consistency_new_vs_old": cons,
        "floor_run_to_run": floor_disp,
        "old_source": os.path.basename(OLD),
        "multiseed_source": os.path.basename(MS),
    }
    meta["rerun"]["wall_seconds"] = round(time.time() - t0, 1)

    doc = {
        "meta": meta,
        "cells": cells,
        "aggregate": M.build_aggregate(cells),
        "decomposition": {"per_cell": per_cell, "aggregate": agg_out},
        "pg_floor_runs": pg_runs,
    }
    json.dump(doc, open(NEW, "w", encoding="utf-8"), indent=1)
    log("wrote %s (%d cells, %d decomposed)"
        % (NEW, len(cells), len(per_cell)))

    # ---- console summary ------------------------------------------------ #
    print("\n== floor: new vs old ==")
    for r in cons:
        if r["cell"].startswith("floor_pg_"):
            print("  %-38s new=%.4f old=%.4f d=%+.4f  r1 %.4f->%.4f r2 %.4f->%.4f"
                  % (r["cell"], r["tear_new"], r["tear_old"], r["d_tear"],
                     r["recall_1_old"], r["recall_1_new"],
                     r["recall_2_old"], r["recall_2_new"]))
    print("\n== decomposition (mean over 3 seeds) ==")
    for key in ("matched", "floor_pg_all", "floor_pg_parallel", "floor_pg_serial",
                "polluted", "efsearch_ef64", "buildeffort_efc64",
                "efsearch_ef128", "efsearch_ef256"):
        s = agg_out.get(key)
        if s:
            print("  %-20s n=%3d tear_all=%.4f tear_eq=%.4f frac_unequal=%.4f (%.4f-%.4f)"
                  % (key, s["n_cells"], s["tear_all_mean"], s["tear_eq_mean"],
                     s["frac_unequal_mean"], s["frac_unequal_min"],
                     s["frac_unequal_max"]))
    print("\nwall = %.1f s" % meta["rerun"]["wall_seconds"])


if __name__ == "__main__":
    main()
