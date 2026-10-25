#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Merge the two halves of the recall-matching study into one artifact.

The pgvector and LanceDB runs use independent knob grids. The merge requires
disjoint cell names and records the composition of the resulting artifact.

It also restores provenance that a later in-place `_save()` had dropped from the
LanceDB half (`floor_lance_runs`, `floor_lance_method`) and records the grid the
same-order floor actually used, so that every knob written into a floor cell can
be found in the grid the harness documents.

Run:  python merge_matched2.py
Writes: results_engine_matched2.json (96 cells) - the artifact the paper cites.
"""
from __future__ import annotations

import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PG = os.path.join(HERE, "results_engine_matched2_pg_part.json")
LANCE = os.path.join(HERE, "results_engine_matched2_lance_part.json")
OUT = os.path.join(HERE, "results_engine_matched2.json")

SEEDS = ("s20261008", "s20261009", "s20261010")
TARGETS = ("R09", "R095", "R099", "default")


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def tear_of(cell):
    """Rate recomputed from the cell's own raw records (never trusted)."""
    raw = cell["per_episode"]
    return sum(1 for r in raw if not r["set_equal"]) / float(len(raw))


def main():
    pg = load(PG)
    ln = load(LANCE)

    pg_cells, ln_cells = pg["cells"], ln["cells"]
    overlap = sorted(set(pg_cells) & set(ln_cells))
    if overlap:
        raise SystemExit(
            "refusing to merge: %d cell names appear in both halves: %s"
            % (len(overlap), overlap[:10]))

    # ---- restore the floor provenance the LanceDB half had lost ---------- #
    floor_runs = []
    for fam in ("hnsw", "ivfpq"):
        for seed in SEEDS:
            for tgt in TARGETS:
                key = "match2_%s_%s_floor_lance_%s" % (fam, seed, tgt)
                cell = ln_cells.get(key)
                if not cell:
                    continue
                m = cell.get("matched") or {}
                floor_runs.append({
                    "cell": key,
                    "family": fam,
                    "seed": int(seed[1:]),
                    "target": tgt,
                    "tear": tear_of(cell),
                    "recall_1": cell.get("recall_A_vs_exact"),
                    "recall_2": cell.get("recall_B_vs_exact"),
                    "gap": m.get("gap"),
                    "knob_1": m.get("kA"),
                    "knob_2": m.get("kB"),
                    "matched_le_0p004": m.get("reached_le_0p004"),
                })

    knobs_hnsw = sorted({json.loads(r["knob_1"])["ef"] for r in floor_runs
                         if r["family"] == "hnsw" and r["knob_1"]})
    knobs_ivfpq = sorted({json.loads(r["knob_1"])["refine_factor"]
                          for r in floor_runs
                          if r["family"] == "ivfpq" and r["knob_1"]})

    floor_method = (
        "Same-order rebuild floor.  table.optimize() is idempotent on unchanged "
        "data (a second call returns the same index_uuid), so the floor cannot "
        "be taken from two optimize() calls.  It is measured instead between "
        "rebuild #1, the index that optimize() itself produced, and rebuild #2, "
        "a second genuine rebuild issued through the engine's own index-build "
        "path create_index(replace=True) - the routine optimize() invokes.  Both "
        "rebuilds serve the same data in the same row order, so the pair's "
        "recall gap is the builder's own run-to-run variation, and any answer "
        "difference is structure-only.  Knobs are chosen by the same pick_pair "
        "rule as the matched rows (min |gap| inside a +/-0.01 window of the "
        "target)."
    )

    floor_note = (
        "Two independent same-order floor measurements exist and are BOTH kept. "
        "(i) cells match2_<fam>_s<seed>_floor_lance_* - the measurement this "
        "artifact is built around (R=0.90: HnswFlat 0.740/0.695/0.745, IvfPq "
        "0.880/0.825/0.810), written with the finer knob grid recorded in "
        "floor_lance_grid_used.  (ii) meta.runs[*].floor_summary - an earlier "
        "run of the SAME procedure on a coarser grid (R=0.90: HnswFlat 0.850 "
        "(s20261008) / 0.855 (s20261009), IvfPq 0.800 (s20261008)); runs whose "
        "floor_summary is {} were not re-measured.  They are the same quantity "
        "measured twice, and on every seed where both exist the two meas-"
        "urements STRADDLE that seed's matched tear (HnswFlat 0.740 < 0.847 < "
        "0.850; IvfPq 0.800 < 0.805 < 0.880).  This spread - not the query-paired "
        "difference - is what bounds the build-level noise, and it is as large "
        "as the matched-tear minus same-order-floor difference; the paper "
        "therefore reports this floor as a reference band beside every LanceDB "
        "rate instead of quoting a rewrite-specific ratio for this engine."
    )

    dev_extra = [
        "Recall alignment of the floor cells is weaker than that of the matched "
        "rows: 5 of 9 floor pairs per family reach the 0.004 gap (max 0.0205 for "
        "HnswFlat, 0.018 for IvfPq), against 9 of 9 (HnswFlat) and 8 of 9 "
        "(IvfPq, max 0.017) for the matched rows.  Floor tears are reported with "
        "their own gap rather than pooled as if equally matched.",
        "meta.runs[*].floor_summary records a default-effort row whose two sides "
        "do NOT have the same recall (e.g. HnswFlat s20261008: 0.481 vs 0.930), "
        "because the rebuilt index answers the default search at a different "
        "operating point.  That row is therefore not comparable with any "
        "recall-matched row and is never used as evidence of separability.",
    ]

    lance_meta = dict(ln["meta"])
    lance_meta["deviations"] = list(lance_meta.get("deviations") or []) + dev_extra
    lance_meta["floor_lance_method"] = floor_method
    lance_meta["floor_lance_grid_used"] = {
        "note": ("Extended grid actually used for the floor search; the earlier "
                 "grid in floor_lance2.py's GRID has been extended to match so "
                 "that every recorded knob is inside it."),
        "hnsw_ef": knobs_hnsw,
        "ivfpq_refine_factor": knobs_ivfpq,
        "nprobes": "default (inert for both post-optimize index kinds)",
    }
    lance_meta["floor_lance_runs"] = floor_runs

    out = {
        "meta": {
            "purpose": ("Shipped-engine rebuild tear: pgvector 0.8.0 (recall-"
                        "matched rows, the same-order rebuild floor over pass "
                        "pairs, and the efc/ef effort sweeps) and LanceDB 0.40.0 "
                        "(two index families, recall-matched rows, same-order "
                        "rebuild floors, controls).  Merged from the two halves "
                        "named under composition."),
            "composition": {
                "pg_half": os.path.basename(PG),
                "pg_cells": len(pg_cells),
                "lance_half": os.path.basename(LANCE),
                "lance_cells": len(ln_cells),
                "merged_cells": len(pg_cells) + len(ln_cells),
                "cell_name_overlap": overlap,
                "merged_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "merged_by": "merge_matched2.py",
            },
            "pg_meta": pg["meta"],
            "lance_meta": lance_meta,
            # mirrors for tools that scan meta for a runs list
            "runs": ln["meta"].get("runs") or [],
            "floor_definition_note": floor_note,
            "floor_lance_method": floor_method,
            "floor_lance_grid_used": lance_meta["floor_lance_grid_used"],
            "floor_lance_runs": floor_runs,
            "deviations": [("pg half: see meta.pg_meta.deviations; lance half: "
                            "see meta.lance_meta.deviations")],
        },
        "cells": dict(pg_cells, **ln_cells),
    }

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False)

    # ---- self-check: every recorded knob must be inside the harness grid --- #
    assert out["meta"]["composition"]["cell_name_overlap"] == []
    assert len(out["cells"]) == 96

    def harness_grid():
        """Read GRID out of floor_lance2.py without importing it (importing
        would run the engine)."""
        path = os.path.join(HERE, "floor_lance2.py")
        if not os.path.exists(path):
            return None
        src = open(path, encoding="utf-8").read()
        i = src.find("GRID = {")
        if i < 0:
            return None
        j, depth = src.index("{", i), 0
        k = j
        while k < len(src):
            if src[k] == "{":
                depth += 1
            elif src[k] == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        ns = {}
        exec("GRID = " + src[j:k + 1], ns)          # noqa: S102 - local file
        return ns["GRID"]

    grid = harness_grid()
    if grid:
        have_h = {g.get("ef") for g in grid.get("hnsw", [])}
        have_i = {g.get("refine_factor") for g in grid.get("ivfpq", [])}
        miss_h = sorted(set(knobs_hnsw) - have_h)
        miss_i = sorted(set(knobs_ivfpq) - have_i)
        print("hnsw knobs missing from harness grid : %s" % (miss_h or "none"))
        print("ivfpq knobs missing from harness grid: %s" % (miss_i or "none"))
        if miss_h or miss_i:
            raise SystemExit(
                "harness grid does not cover the recorded floor knobs: "
                "hnsw %s / ivfpq %s" % (miss_h, miss_i))
    else:
        print("harness grid: floor_lance2.py GRID not found (skipped)")

    print("merged cells      : %d" % len(out["cells"]))
    print("name overlap      : %d" % len(overlap))
    print("floor cells       : %d" % len(floor_runs))
    print("hnsw floor knobs  : %s" % knobs_hnsw)
    print("ivfpq floor knobs : %s" % knobs_ivfpq)
    print("written           : %s" % OUT)


if __name__ == "__main__":
    main()
