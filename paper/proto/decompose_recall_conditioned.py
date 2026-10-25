# -*- coding: utf-8 -*-
"""recall-conditioned tear decomposition for the shipped-engine LanceDB experiments.

READS ONLY. Writes exactly one NEW file:
  paper/proto/results_engine_recall_conditioned.json
Never touches any pre-existing JSON.
"""
import json, os, statistics

PROTO = os.path.dirname(os.path.abspath(__file__))
MATCH2 = os.path.join(PROTO, "results_engine_matched2.json")
PIN = os.path.join(PROTO, "results_engine_second_pin.json")
OUT = os.path.join(PROTO, "results_engine_recall_conditioned.json")


def load(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def eq(a, b, tol=1e-9):
    return abs(a - b) <= tol


def decompose(per_episode):
    """per_episode: list of dicts with set_equal, recall_A_vs_exact, recall_B_vs_exact."""
    n = len(per_episode)
    n_tear = 0
    n_eq = 0
    n_eq_tear = 0
    n_ne = 0
    n_ne_tear = 0
    abs_deltas = []
    for e in per_episode:
        ra = float(e["recall_A_vs_exact"])
        rb = float(e["recall_B_vs_exact"])
        teared = not bool(e["set_equal"])
        d = abs(ra - rb)
        abs_deltas.append(d)
        if teared:
            n_tear += 1
        if eq(ra, rb):
            n_eq += 1
            if teared:
                n_eq_tear += 1
        else:
            n_ne += 1
            if teared:
                n_ne_tear += 1
    return {
        "n": n,
        "n_tear": n_tear,
        "tear_all": (n_tear / n) if n else None,
        "n_eq": n_eq,
        "n_eq_tear": n_eq_tear,
        "tear_eq": (n_eq_tear / n_eq) if n_eq else None,
        "n_ne": n_ne,
        "n_ne_tear": n_ne_tear,
        "tear_ne": (n_ne_tear / n_ne) if n_ne else None,
        "frac_unequal": (n_ne / n) if n else None,
        "mean_abs_drecall": (sum(abs_deltas) / n) if n else None,
        "max_abs_drecall": max(abs_deltas) if abs_deltas else None,
    }


# ---------------------------------------------------------------- matched2
d2 = load(MATCH2)
cells2 = d2["cells"]

LANCE = [k for k in cells2 if k.startswith("match2_hnsw_") or k.startswith("match2_ivfpq_")]
assert len(LANCE) == 60, len(LANCE)

cell_out = {}
mech_violations = []          # cells where tear_ne != 1.0 despite n_ne>0
mech_implication_violations = []  # episodes where recall differs but set_equal True
tear_all_mismatch = []

for name in sorted(LANCE):
    c = cells2[name]
    pe = c["per_episode"]
    dec = decompose(pe)
    dec["cell"] = name
    dec["family"] = "hnsw" if name.startswith("match2_hnsw_") else "ivfpq"
    dec["recall_A_vs_exact_reported"] = c.get("recall_A_vs_exact")
    m = c.get("matched") or {}
    dec["matched_gap"] = m.get("gap")
    dec["matched_recall_A"] = m.get("recall_A")
    dec["matched_recall_B"] = m.get("recall_B")
    dec["reached_le_0p004"] = m.get("reached_le_0p004")
    dec["knob_A"] = m.get("kA")
    dec["knob_B"] = m.get("kB")
    dec["recall_B_vs_exact_reported"] = c.get("recall_B_vs_exact")
    dec["tear_rate_reported"] = c.get("tear_rate_any_difference")
    dec["note"] = c.get("note")
    dec["is_floor"] = "floor_lance" in name
    # classify suffix (floor_lance first: 'floor_lance_default' also ends with '_default')
    if "floor_lance" in name:
        sub = "F_" + name.split("floor_lance_")[1]
    elif name.endswith("_default"):
        sub = "default"
    elif name.endswith("samegen_control_A"):
        sub = "samegen_control_A"
    elif name.endswith("samegen_control_B"):
        sub = "samegen_control_B"
    elif name.endswith("recallmatched_R09"):
        sub = "R09"
    elif name.endswith("recallmatched_R095"):
        sub = "R095"
    elif name.endswith("recallmatched_R099"):
        sub = "R099"
    else:
        sub = "UNKNOWN:" + name
    dec["level"] = sub
    # seed
    try:
        dec["seed"] = int([t for t in name.split("_") if t.startswith("s2026")][0][1:])
    except Exception:
        dec["seed"] = None
    dec["target_R"] = c.get("target_R")
    # consistency check vs reported tear rate
    if dec["tear_all"] is not None and c.get("tear_rate_any_difference") is not None:
        if not eq(dec["tear_all"], c["tear_rate_any_difference"], 1e-6):
            tear_all_mismatch.append({"cell": name, "computed": dec["tear_all"],
                                      "reported": c["tear_rate_any_difference"]})
    # mechanical checks
    if dec["n_ne"] > 0 and dec["tear_ne"] is not None and not eq(dec["tear_ne"], 1.0):
        mech_violations.append({"cell": name, "tear_ne": dec["tear_ne"], "n_ne": dec["n_ne"]})
    for e in pe:
        if not eq(float(e["recall_A_vs_exact"]), float(e["recall_B_vs_exact"])) and bool(e["set_equal"]):
            mech_implication_violations.append({"cell": name, "q": e.get("q"),
                                                "ra": e["recall_A_vs_exact"], "rb": e["recall_B_vs_exact"]})
    cell_out[name] = dec

# ------------------------------------------------- by_family_level (cross-seed)
def aggregate(keys):
    rows = [cell_out[k] for k in keys]
    def mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None
    def rng(vals):
        vals = [v for v in vals if v is not None]
        return [min(vals), max(vals)] if vals else None
    return {
        "n_cells": len(rows),
        "n_seeds": len(set(r["seed"] for r in rows)),
        "n_eq_total": sum(r["n_eq"] for r in rows),
        "n_ne_total": sum(r["n_ne"] for r in rows),
        "n_total": sum(r["n"] for r in rows),
        "tear_all_mean": mean([r["tear_all"] for r in rows]),
        "tear_all_range": rng([r["tear_all"] for r in rows]),
        "tear_eq_mean": mean([r["tear_eq"] for r in rows]),
        "tear_eq_range": rng([r["tear_eq"] for r in rows]),
        "tear_ne_mean": mean([r["tear_ne"] for r in rows]),
        "tear_ne_range": rng([r["tear_ne"] for r in rows]),
        "frac_unequal_mean": mean([r["frac_unequal"] for r in rows]),
        "frac_unequal_range": rng([r["frac_unequal"] for r in rows]),
        "mean_abs_drecall_mean": mean([r["mean_abs_drecall"] for r in rows]),
        "mean_abs_drecall_range": rng([r["mean_abs_drecall"] for r in rows]),
        "max_abs_drecall_max": max([r["max_abs_drecall"] for r in rows if r["max_abs_drecall"] is not None]),
        "tear_eq_per_cell": [r["tear_eq"] for r in rows],
        "tear_all_per_cell": [r["tear_all"] for r in rows],
        "frac_unequal_per_cell": [r["frac_unequal"] for r in rows],
        "matched_gap_mean": mean([r["matched_gap"] for r in rows]),
        "matched_gap_range": rng([r["matched_gap"] for r in rows]),
        "matched_gap_per_cell": [r["matched_gap"] for r in rows],
        "matched_recall_A_mean": mean([r["matched_recall_A"] for r in rows]),
        "matched_recall_B_mean": mean([r["matched_recall_B"] for r in rows]),
        "reached_le_0p004_per_cell": [r["reached_le_0p004"] for r in rows],
        "n_seeds_reached_le_0p004": sum(1 for r in rows if r["reached_le_0p004"]),
    }


LEVELS = ["default", "samegen_control_A", "samegen_control_B",
          "R09", "R095", "R099",
          "F_default", "F_R09", "F_R095", "F_R099"]
FAMILIES = ["hnsw", "ivfpq"]

by_family_level = {}
for fam in FAMILIES:
    for lv in LEVELS:
        keys = [k for k in LANCE if cell_out[k]["family"] == fam and cell_out[k]["level"] == lv]
        if keys:
            by_family_level[f"{fam}::{lv}"] = aggregate(keys)

# ---------------------------------------------------------------- pin file
dp = load(PIN)
cellsp = dp["cells"]
pin_out = {}
pin_available = {}
for name in sorted(cellsp):
    c = cellsp[name]
    pe = c.get("per_episode")
    has_rec = bool(pe) and all("recall_A_vs_exact" in e and "recall_B_vs_exact" in e for e in pe)
    pin_available[name] = has_rec
    if has_rec:
        dec = decompose(pe)
        dec["cell"] = name
        dec["recall_A_vs_exact_reported"] = c.get("recall_A_vs_exact")
        dec["recall_B_vs_exact_reported"] = c.get("recall_B_vs_exact")
        dec["tear_rate_reported"] = c.get("tear_rate_any_difference")
        dec["note"] = c.get("note")
        pin_out[name] = dec

# pin aggregate per family/kind
pin_agg = {}
def pkey(name):
    fam = name.split("_")[0]
    rest = name[len(fam) + 1:]
    # rest = s2026xxxx_<kind>
    parts = rest.split("_", 1)
    kind = parts[1] if len(parts) > 1 else rest
    return fam, kind
pinkinds = {}
for name in pin_out:
    fam, kind = pkey(name)
    pinkinds.setdefault((fam, kind), []).append(name)
for (fam, kind), keys in sorted(pinkinds.items()):
    rows = [pin_out[k] for k in keys]
    def mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None
    pin_agg[f"{fam}::{kind}"] = {
        "n_cells": len(rows),
        "n_seeds": len(rows),
        "n_eq_total": sum(r["n_eq"] for r in rows),
        "n_ne_total": sum(r["n_ne"] for r in rows),
        "n_total": sum(r["n"] for r in rows),
        "tear_all_mean": mean([r["tear_all"] for r in rows]),
        "tear_eq_mean": mean([r["tear_eq"] for r in rows]),
        "tear_eq_per_cell": [r["tear_eq"] for r in rows],
        "tear_ne_mean": mean([r["tear_ne"] for r in rows]),
        "frac_unequal_mean": mean([r["frac_unequal"] for r in rows]),
        "mean_abs_drecall_mean": mean([r["mean_abs_drecall"] for r in rows]),
        "max_abs_drecall_max": max([r["max_abs_drecall"] for r in rows]),
        "recall_A_mean": mean([r["recall_A_vs_exact_reported"] for r in rows]),
        "recall_B_mean": mean([r["recall_B_vs_exact_reported"] for r in rows]),
    }

# pin mechanical check
pin_mech_viol = []
pin_mech_implication = []
for name, dec in pin_out.items():
    if dec["n_ne"] > 0 and dec["tear_ne"] is not None and not eq(dec["tear_ne"], 1.0):
        pin_mech_viol.append({"cell": name, "tear_ne": dec["tear_ne"], "n_ne": dec["n_ne"]})
    for e in cellsp[name]["per_episode"]:
        if not eq(float(e["recall_A_vs_exact"]), float(e["recall_B_vs_exact"])) and bool(e["set_equal"]):
            pin_mech_implication.append({"cell": name, "q": e.get("q")})

# ---------------------------------------------------------------- write out
out = {
    "meta": {
        "purpose": "Recall-conditioned decomposition of the shipped-engine LanceDB tear. "
                   "Splits every episode into 'both indexes returned the same per-query recall' "
                   "(recall_A_vs_exact == recall_B_vs_exact) and 'different per-query recall'. "
                   "Only the equal-recall subset is a confound-free 'same quality, different answer' rate; "
                   "the unequal subset tears mechanically because recall = |top10 n exact10| / 10.",
        "sources_readonly": {
            "matched2": MATCH2,
            "pin": PIN,
        },
        "field_semantics": {
            "tear_all": "fraction of episodes with set_equal == False (= the paper's any-difference tear).",
            "n_eq": "episodes with recall_A_vs_exact == recall_B_vs_exact (per-query recall equal).",
            "tear_eq": "tear_all restricted to the n_eq subset; the CONFOUND-FREE tear.",
            "n_ne": "episodes with recall_A_vs_exact != recall_B_vs_exact.",
            "tear_ne": "tear_all restricted to the n_ne subset; mechanically 1.0 (see mechanical_check).",
            "frac_unequal": "n_ne / n.",
            "mean_abs_drecall": "mean over episodes of |recall_A_vs_exact - recall_B_vs_exact|.",
            "max_abs_drecall": "max over episodes of |recall_A_vs_exact - recall_B_vs_exact|.",
        },
        "recall_equality_tol": 1e-9,
        "recall_quantum_note": "k=10 so every recall value is a multiple of 0.1; equality is exact and unambiguous.",
        "cell_naming": {
            "matched2_lance_cells": "match2_{hnsw|ivfpq}_s{seed}_<level>; 60 cells = 2 families x 3 seeds x 10 levels.",
            "levels": {
                "default": "genA default search -> optimize() -> genB default search (unequal effort).",
                "samegen_control_A/B": "same generation searched twice, no maintenance (must be 0).",
                "R09/R095/R099": "recall-matched pair (fine knob search, window +/-0.01) at target R=0.90/0.95/0.99.",
                "F_default/F_R09/F_R095/F_R099": "floor_lance: same-order rebuild (same data, same insertion order, "
                                                 "index rebuilt) at default or at the given target R.",
            },
        },
    },
    "cells": cell_out,
    "by_family_level": by_family_level,
    "mechanical_check": {
        "claim": "per-query recall differs => top-10 sets differ (tear == 1.0); i.e. tear_ne == 1.0 for every cell.",
        "scope_matched2_lance_cells": len(LANCE),
        "n_cells_with_n_ne_gt0": sum(1 for r in cell_out.values() if r["n_ne"] > 0),
        "n_cells_violating_tear_ne_is_1": len(mech_violations),
        "violations": mech_violations,
        "n_episodes_recall_differs_but_set_equal": len(mech_implication_violations),
        "implication_violation_examples": mech_implication_violations[:20],
        "tear_all_vs_reported_mismatches": tear_all_mismatch,
        "all_passed": (len(mech_violations) == 0 and len(tear_all_mismatch) == 0),
    },
    "pin_alignment": {
        "claim": "Both sides of the pinned/unpinned comparison carry per-episode recall, so the "
                 "same conditioning applies to the pin experiment.",
        "per_episode_recall_available": all(pin_available.values()) and len(pin_available) > 0,
        "cells_with_per_episode_recall": sum(1 for v in pin_available.values() if v),
        "cells_total": len(pin_available),
        "aggregate_by_family_kind": pin_agg,
        "mechanical_check": {
            "n_cells_violating_tear_ne_is_1": len(pin_mech_viol),
            "violations": pin_mech_viol,
            "n_episodes_recall_differs_but_set_equal": len(pin_mech_implication),
        },
    },
    "caveats": {
        "matched_gap_not_always_le_0p004": "The 'matched' rows are aligned by minimizing the pair gap under a "
            "window around the target, but the 3-seed worst seed can exceed 0.004 (e.g. ivfpq R0.90: gaps "
            "0.002/0.017/0.003, so 1/3 seeds miss the target). The floor rows routinely miss it too "
            "(hnsw F_R0.95 gaps up to 0.0205). Report per-cell gaps, not just the mean.",
        "lancedb_matched_vs_floor_share_genB": "LanceDB matched compares genA (fresh create_index) vs genB "
            "(post-optimize rebuild); the floor compares genB vs genR2 (create_index(replace=True)). The two "
            "comparisons SHARE the genB build, so their tear estimates are correlated (the floor is NOT an "
            "independent baseline). The builds available in the data are only {genA, genB, genR2}; no pair is "
            "independent of both comparisons, so a fully independent floor would require a fresh rebuild "
            "(e.g. genR3), i.e. an engine re-run.",
        "lancedb_has_no_permutation": "Unlike pgvector (matched = id-order vs permuted-order), the LanceDB "
            "rewrite (optimize()) does NOT reorder rows: genA and genB are both same-data, same-order builds. "
            "So the LanceDB 'matched' comparison is itself a same-order rebuild comparison, which is the "
            "structural reason its tear does not separate from the floor.",
        "pgvector_floor_is_independent": "pgvector's floor (id-order CREATE INDEX vs id-order REINDEX) shares "
            "no build with its matched pair (id-order vs permuted) -- the two estimators are constructed "
            "differently across engines; the matched/floor RATIO is comparable but the absolute construction is not.",
    },
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print("WROTE", OUT)
print("mech violations (matched2):", len(mech_violations))
print("tear_all vs reported mismatches:", len(tear_all_mismatch))
print("pin per-episode recall available:", out["pin_alignment"]["per_episode_recall_available"])
