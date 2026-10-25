# -*- coding: utf-8 -*-
"""Merge per-(seed,kind) partial outputs of floor_disjoint.py into one file."""
import json, os, sys, glob

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("FD_OUT", os.path.join(HERE, "results_engine_floor_disjoint.json"))
PARTS = sys.argv[1:] or sorted(glob.glob(os.path.join(HERE, "_fd_part_*.json")))

cells = {}
runs = []
op_runs = []
meta0 = None
for p in PARTS:
    d = json.load(open(p, encoding="utf-8"))
    if meta0 is None:
        meta0 = d["meta"]
    cells.update(d["cells"])
    runs.extend(d["meta"]["runs"])
    op_runs.extend(d["order_preservation"]["runs"])
    print("merged", p, "->", len(d["cells"]), "cells")

runs.sort(key=lambda r: (r["seed"], r["index_kind"]))
op_runs.sort(key=lambda r: (r["seed"], r["kind"]))
meta0["runs"] = runs
out = {
    "meta": meta0,
    "cells": cells,
    "by_family_level": {},
    "disjoint_ratio": {},
    "order_preservation": {
        "method": ("Read the physical id sequence from the underlying lance dataset "
                   "(table.to_lance().to_table(['id'])) at four points: after create_index (genA), after "
                   "optimize() (genB), after rebuild#2 (genR2), after rebuild#3 (genR3). Compare each later "
                   "sequence to the genA sequence by exact equality, longest common prefix and sha256."),
        "runs": op_runs,
        "all_preserved": all(r["build_vs_optimize"]["identical"] and r["build_vs_R2"]["identical"]
                             and r["build_vs_R3"]["identical"] for r in op_runs),
    },
    "caveats": {},
}
json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("WROTE", OUT, "| cells:", len(cells), "| all_preserved:", out["order_preservation"]["all_preserved"])
