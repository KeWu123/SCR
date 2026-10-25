# -*- coding: utf-8 -*-
"""Post-process results_engine_floor_disjoint.json: add the equal-recall
decomposition, cross-seed aggregates, the matched/disjoint ratio (with the
shared-genB ratio as control) and caveats. Rewrites ONLY that new file."""
import json, os, collections

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("FD_OUT", os.path.join(HERE, "results_engine_floor_disjoint.json"))


def eq(a, b, tol=1e-9):
    return abs(a - b) <= tol


def decompose(per_episode):
    n = len(per_episode)
    n_tear = n_eq = n_eq_tear = n_ne = n_ne_tear = 0
    deltas = []
    for e in per_episode:
        ra = float(e["recall_A_vs_exact"]); rb = float(e["recall_B_vs_exact"])
        teared = not bool(e["set_equal"])
        deltas.append(abs(ra - rb))
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
        "n": n, "tear_all": n_tear / n if n else None,
        "n_eq": n_eq, "tear_eq": (n_eq_tear / n_eq) if n_eq else None,
        "n_ne": n_ne, "tear_ne": (n_ne_tear / n_ne) if n_ne else None,
        "frac_unequal": n_ne / n if n else None,
        "mean_abs_drecall": sum(deltas) / n if n else None,
        "max_abs_drecall": max(deltas) if deltas else None,
    }


def parse_cell(name):
    # {kind}_s{seed}_{comparison}_{level}
    parts = name.split("_")
    kind = parts[0]
    seed = int(parts[1][1:])
    comparison = parts[2]
    comparison = {"disjointfloor": "disjoint_floor", "sharedfloor": "shared_floor"}.get(comparison, comparison)
    level = "_".join(parts[3:])
    return kind, seed, comparison, level


d = json.load(open(OUT, encoding="utf-8"))
cells = d["cells"]

dec = {}
for name, c in cells.items():
    kind, seed, comparison, level = parse_cell(name)
    x = decompose(c["per_episode"])
    x.update({"cell": name, "family": kind, "seed": seed,
              "comparison": comparison, "level": level,
              "gap": (c.get("pair") or {}).get("gap"),
              "recall_A": (c.get("pair") or {}).get("recall_A"),
              "recall_B": (c.get("pair") or {}).get("recall_B"),
              "reached_le_0p004": (c.get("pair") or {}).get("reached_le_0p004"),
              "tear_all_reported": c.get("tear_rate_any_difference")})
    # consistency
    x["tear_all_matches_reported"] = (x["tear_all"] is not None
                                      and eq(round(x["tear_all"], 4), c.get("tear_rate_any_difference", -1)))
    dec[name] = x

d["decomposition"] = dec

# consistency / mechanical check
bad = [n for n, x in dec.items() if not x["tear_all_matches_reported"]]
mech_viol = [{"cell": n, "tear_ne": x["tear_ne"]} for n, x in dec.items()
             if x["n_ne"] > 0 and x["tear_ne"] is not None and not eq(x["tear_ne"], 1.0)]
d["mechanical_check"] = {
    "tear_all_vs_reported_mismatches": bad,
    "n_cells_violating_tear_ne_is_1": len(mech_viol),
    "violations": mech_viol,
    "all_passed": (len(bad) == 0 and len(mech_viol) == 0),
}

# ---------------- by family x comparison x level ----------------
def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None

def rng(v):
    v = [x for x in v if x is not None]
    return [min(v), max(v)] if v else None

bfl = {}
LEVELS = ["default", "R09", "R095", "R099"]
COMPARISONS = ["matched", "disjoint_floor", "shared_floor"]
for fam in ["hnsw", "ivfpq"]:
    for comp in COMPARISONS:
        for lv in LEVELS:
            keys = [k for k, x in dec.items() if x["family"] == fam and x["comparison"] == comp and x["level"] == lv]
            if not keys:
                continue
            rows = [dec[k] for k in keys]
            gaps = [r["gap"] for r in rows]
            bfl[f"{fam}::{comp}::{lv}"] = {
                "n_cells": len(rows), "n_seeds": len(set(r["seed"] for r in rows)),
                "n_total": sum(r["n"] for r in rows),
                "n_eq_total": sum(r["n_eq"] for r in rows),
                "tear_all_mean": mean([r["tear_all"] for r in rows]),
                "tear_eq_mean": mean([r["tear_eq"] for r in rows]),
                "tear_eq_range": rng([r["tear_eq"] for r in rows]),
                "tear_eq_per_seed": [r["tear_eq"] for r in rows],
                "frac_unequal_mean": mean([r["frac_unequal"] for r in rows]),
                "frac_unequal_range": rng([r["frac_unequal"] for r in rows]),
                "mean_abs_drecall_mean": mean([r["mean_abs_drecall"] for r in rows]),
                "gap_mean": mean(gaps), "gap_range": rng(gaps),
                "reached_le_0p004_per_seed": [r["reached_le_0p004"] for r in rows],
                "n_seeds_reached_le_0p004": sum(1 for r in rows if r["reached_le_0p004"]),
                "recall_A_mean": mean([r["recall_A"] for r in rows]),
                "recall_B_mean": mean([r["recall_B"] for r in rows]),
            }
d["by_family_level"] = bfl

# ---------------- disjoint ratio ----------------
# per (family, level): ratio of tear_eq means AND per-seed ratio range.
ratios = {}
for fam in ["hnsw", "ivfpq"]:
    for lv in LEVELS:
        def tm(comp):
            keys = [k for k, x in dec.items() if x["family"] == fam and x["comparison"] == comp and x["level"] == lv]
            return {x["seed"]: x["tear_eq"] for x in (dec[k] for k in keys)}
        m = tm("matched"); dj = tm("disjoint_floor"); sh = tm("shared_floor")
        seeds = sorted(set(m) & set(dj) & set(sh))

        def ratio(a, b, s):
            if a[s] is None or b[s] is None or b[s] == 0:
                return None
            return a[s] / b[s]
        r_disj = [ratio(m, dj, s) for s in seeds]
        r_shared = [ratio(m, sh, s) for s in seeds]
        mm = mean([m[s] for s in seeds]); mdj = mean([dj[s] for s in seeds]); msh = mean([sh[s] for s in seeds])
        ratios[f"{fam}::{lv}"] = {
            "tear_eq_matched_mean": mm,
            "tear_eq_disjoint_floor_mean": mdj,
            "tear_eq_shared_floor_mean": msh,
            "ratio_matched_over_disjoint_mean_of_seed_ratios": mean(r_disj),
            "ratio_matched_over_disjoint_range": rng(r_disj),
            "ratio_matched_over_disjoint_per_seed": r_disj,
            "ratio_of_means_matched_over_disjoint": (mm / mdj) if (mm is not None and mdj) else None,
            "ratio_matched_over_shared_mean_of_seed_ratios": mean(r_shared),
            "ratio_matched_over_shared_range": rng(r_shared),
            "ratio_matched_over_shared_per_seed": r_shared,
            "ratio_of_means_matched_over_shared": (mm / msh) if (mm is not None and msh) else None,
            "note_divzero": ("disjoint floor tear_eq = 0 on >=1 seed; ratio undefined there" if any(v is None for v in r_disj) else None),
        }
d["disjoint_ratio"] = ratios

# ---------------- caveats ----------------
d["caveats"] = {
    "isolated_floor_construction": "disjoint_floor = tear(genR2, genR3) shares NO build with matched (genA, genB); this "
        "is the estimate isomorphic to pgvector's independent-rebuild floor. shared_floor = tear(genB, genR2) is the "
        "matched2 floor and shares genB with matched (correlated); it is reported only as a control.",
    "recall_level_dependence": "tear_eq depends strongly on the operating recall level; compare only at matching targets. "
        "The matched row and the two floor rows are all aligned by pick_pair to the same target R over each side's grid.",
    "gap_can_miss": "the 3-seed worst case can miss |gap|<=0.004; per-seed reached flags and gap ranges are reported.",
    "row_order_measured": "order_preservation is MEASURED by comparing the physical id sequence (table.to_lance().to_table(['id'])) "
        "before/after optimize() and after each rebuild; see order_preservation.all_preserved.",
    "grids_used": "HnswFlat used the dense grids (gridA=168, gridB=141). IvfPq used a trimmed grid (gridA=80, gridB=12) "
        "because the dense ivfpq grid (~120-224 settings) pushed one seed past the single-command budget; the two families "
        "therefore differ in grid density, which only affects how close each target is bracketed (report the per-row gap). "
        "Note the matched2 data itself was produced with grids that differ from the current script file (matched cells: "
        "hnsw gridA=216/gridB=101, ivfpq gridA=120/gridB=70), so the matched rows here are internally consistent with THIS "
        "run's disjoint floor rather than byte-identical to the earlier matched rows.",
}
json.dump(d, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("WROTE", OUT)
print("mech check:", d["mechanical_check"]["all_passed"], "| bad:", bad[:3], "| viol:", len(mech_viol))
for k, v in d["disjoint_ratio"].items():
    print(k, "tear_eq matched/dj/share =", v["tear_eq_matched_mean"], v["tear_eq_disjoint_floor_mean"], v["tear_eq_shared_floor_mean"],
          "| ratio(dj)=", v["ratio_matched_over_disjoint_mean_of_seed_ratios"],
          "| ratio(shared)=", v["ratio_matched_over_shared_mean_of_seed_ratios"])
