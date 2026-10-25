# -*- coding: utf-8 -*-
"""Disjoint floor for LanceDB: a floor built from two rebuilds that share NO build
with the recall-matched pair, so the matched/floor ratio is isomorphic to pgvector's.

Protocol identical to engine_matched2_lance.py (N=20000, d=64, L2, 200 queries,
k=10, seeds 20261008/9/10, HnswFlat & IvfPq, 8 fragments), plus a third
independent rebuild genR3.

Builds per (seed, kind):
  genA  = table right after create_index           (searched over grid A)
  genB  = after table.optimize()                    (searched over grid B)
  genR2 = create_index(replace=True)                (searched over grid A)
  genR3 = create_index(replace=True) AGAIN (NEW)    (searched over grid A)
Comparisons:
  matched        = (genA, genB)          -- as in matched2
  disjoint_floor = (genR2, genR3)        -- NEW: no build shared with matched
  shared_floor   = (genB, genR2)         -- reproduces the matched2 floor (shares genB)

Also measures whether optimize()/rebuilds perturb the physical row order (id
sequence before vs after), so the "LanceDB has no permutation" fact is MEASURED.

Writes ONLY a new file: results_engine_floor_disjoint.json
"""
import os, sys, json, time, tempfile, datetime, hashlib
import numpy as np
import lancedb
import lancedb.index as li
import pyarrow as pa

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("FD_OUT", os.path.join(HERE, "results_engine_floor_disjoint.json"))

SEEDS = [20261008, 20261009, 20261010]
N = int(os.environ.get("FD_N", 20000))
D = 64
NQ = int(os.environ.get("FD_NQ", 200))
K = 10
NB = int(os.environ.get("FD_NB", 8))
TARGETS = [0.90, 0.95, 0.99]
GAP_OK = 0.004
_SMALL = os.environ.get("FD_SMALL") == "1"

HNSW_NPROBES_A = [9, 10, 11, 12, 13, 14, 15, 16]
HNSW_EF_A = list(range(10, 31))
HNSW_EF_B = list(range(10, 61)) + list(range(62, 601, 6))
IVFPQ_NPROBES_A = list(range(1, 17))
# Post-optimize IVF-PQ uses its own full probing ladder, as the original
# experiment wrappers did. Keep it explicit so this driver runs standalone.
IVFPQ_NPROBES_B = list(range(1, 17))
IVFPQ_RF_A = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 50]
IVFPQ_RF_B = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 23, 26,
              30, 35, 40, 50, 60, 75, 100]
if _SMALL:
    HNSW_NPROBES_A = [9, 12]
    HNSW_EF_A = [20, 40]
    HNSW_EF_B = [20, 40]
    IVFPQ_NPROBES_A = [1, 4]
    IVFPQ_NPROBES_B = [1, 4]
    IVFPQ_RF_A = [4, 10]
    IVFPQ_RF_B = [4, 10]
if os.environ.get("FD_IVFPQ_SMALL") == "1":
    # matched2's ivfpq gridA was ~120 settings (~200s per grid pass); 3 passes +
    # build would exceed a single 600s command budget, so trim to ~90 settings.
    # rf is kept DENSE (15 levels, incl. 5/7/9) because placement of both the A
    # and B curves near the target recall is limited by the rf ladder, not nprobes.
    IVFPQ_NPROBES_A = [1, 2, 4, 8, 12, 16]
    IVFPQ_RF_A = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 50]
    IVFPQ_RF_B = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 50, 75, 100]


def log(msg):
    print("[%s] %s" % (datetime.datetime.now().strftime('%H:%M:%S'), msg), flush=True)


def make_data(seed):
    rng = np.random.default_rng(seed)
    xb = rng.random((N, D), dtype=np.float32)
    qb = rng.random((NQ, D), dtype=np.float32)
    x2 = (xb * xb).sum(axis=1)[None, :]
    q2 = (qb * qb).sum(axis=1)[:, None]
    dd = q2 + x2 - 2.0 * (qb @ xb.T)
    exact = np.argsort(dd, axis=1)[:, :K]
    return xb, qb, exact


def config_for(kind):
    if kind == "hnsw":
        return li.HnswFlat(distance_type="l2", num_partitions=16, m=20, ef_construction=150)
    if kind == "ivfpq":
        return li.IvfPq(distance_type="l2", num_partitions=16, num_sub_vectors=16)
    raise ValueError(kind)


def build_table(dbdir, xb, kind, name="v"):
    db = lancedb.connect(dbdir)
    sch = pa.schema([pa.field("id", pa.int64()),
                     pa.field("vector", pa.list_(pa.float32(), D))])
    t = db.create_table(name, schema=sch)
    pb = N // NB
    for b in range(NB):
        ids = np.arange(b * pb, (b + 1) * pb)
        t.add(pa.table({"id": pa.array(ids, type=pa.int64()),
                        "vector": pa.array(xb[ids].tolist(),
                                           type=pa.list_(pa.float32(), D))}))
    t.create_index("vector", config=config_for(kind), replace=True)
    return t


def grid_a(kind):
    if kind == "hnsw":
        return [{"nprobes": p, "ef": e} for p in HNSW_NPROBES_A for e in HNSW_EF_A]
    return [{"nprobes": p, "refine_factor": r}
            for p in IVFPQ_NPROBES_A for r in IVFPQ_RF_A]


def grid_b(kind):
    if kind == "hnsw":
        return [{"nprobes": None, "ef": e} for e in HNSW_EF_B]
    return [{"nprobes": p, "refine_factor": r}
            for p in IVFPQ_NPROBES_B for r in IVFPQ_RF_B]


def pkey(params):
    return json.dumps(params, sort_keys=True)


def search_all(t, qb, params=None):
    q = t.search(qb).metric("l2").limit(K)
    if params:
        if params.get("ef") is not None:
            q = q.ef(int(params["ef"]))
        if params.get("nprobes") is not None:
            q = q.nprobes(int(params["nprobes"]))
        if params.get("refine_factor") is not None:
            q = q.refine_factor(int(params["refine_factor"]))
    tb = q.to_arrow().select(["query_index", "id"])
    qi = tb.column("query_index").to_numpy()
    ids = tb.column("id").to_numpy()
    out = [[] for _ in range(NQ)]
    for a, b in zip(qi, ids):
        out[int(a)].append(int(b))
    return out


def recall_of(res, exact):
    return float(np.mean([len(set(res[i]) & set(exact[i])) / K for i in range(NQ)]))


def cell_record(A, B, exact, name, note, extra=None):
    per = []
    for i in range(NQ):
        sa, sb = set(A[i]), set(B[i])
        per.append({
            "q": i,
            "set_equal": bool(sa == sb),
            "ordered_equal": bool(list(A[i]) == list(B[i])),
            "symdiff": int(len(sa ^ sb)),
            "recall_A_vs_exact": round(len(sa & set(exact[i])) / K, 4),
            "recall_B_vs_exact": round(len(sb & set(exact[i])) / K, 4),
        })
    n = len(per)
    syms = [p["symdiff"] for p in per]
    out = {
        "cell": name, "note": note, "n_episodes": n,
        "tear_rate_any_difference": round(sum(1 for p in per if not p["set_equal"]) / n, 4),
        "mean_symdiff": round(float(np.mean(syms)), 4),
        "max_symdiff": int(np.max(syms)),
        "n_episodes_with_symdiff_gt0": int(sum(1 for s in syms if s > 0)),
        "ordered_equal_rate": round(sum(1 for p in per if p["ordered_equal"]) / n, 4),
        "recall_A_vs_exact": round(recall_of(A, exact), 4),
        "recall_B_vs_exact": round(recall_of(B, exact), 4),
        "per_episode": per,
    }
    if extra:
        out.update(extra)
    return out


def index_uuid(t):
    try:
        return t.list_indices()[0].index_uuid
    except Exception:
        return None


def pick_pair(curveA, curveB, target, w=0.01):
    ca = [k for k in curveA if abs(curveA[k] - target) <= w]
    cb = [k for k in curveB if abs(curveB[k] - target) <= w]
    if ca and cb:
        pairs = [(abs(curveA[ka] - curveB[kb]),
                  abs((curveA[ka] + curveB[kb]) / 2.0 - target), ka, kb)
                 for ka in ca for kb in cb]
        ok = [p for p in pairs if p[0] <= GAP_OK]
        if ok:
            g, m, ka, kb = min(ok, key=lambda p: (p[1], p[0]))
            reached = True
        else:
            g, m, ka, kb = min(pairs, key=lambda p: (p[0], p[1]))
            reached = False
        return {"window": w, "placeable": True, "kA": ka, "kB": kb,
                "recall_A": round(curveA[ka], 4), "recall_B": round(curveB[kb], 4),
                "gap": round(g, 4), "reached_le_0p004": bool(reached),
                "dist_to_target": round(m, 4)}
    allp = [(abs(curveA[ka] - curveB[kb]),
             abs((curveA[ka] + curveB[kb]) / 2.0 - target), ka, kb)
            for ka in curveA for kb in curveB]
    ok = [p for p in allp if p[0] <= GAP_OK]
    if ok:
        g, m, ka, kb = min(ok, key=lambda p: (p[1], p[0]))
        return {"window": w, "placeable": False, "kA": ka, "kB": kb,
                "recall_A": round(curveA[ka], 4), "recall_B": round(curveB[kb], 4),
                "gap": round(g, 4), "reached_le_0p004": True,
                "dist_to_target": round(m, 4), "note": "no in-window pair; genuine <=0.004 match elsewhere"}
    g, m, ka, kb = min(allp, key=lambda p: (p[0], p[1]))
    return {"window": w, "placeable": False, "kA": ka, "kB": kb,
            "recall_A": round(curveA[ka], 4), "recall_B": round(curveB[kb], 4),
            "gap": round(g, 4), "reached_le_0p004": False,
            "dist_to_target": round(m, 4), "note": "no in-window pair and tightest gap > 0.004"}


# --------------------------------------------------------- order measurement
def id_order(t):
    """physical id sequence, read from the underlying lance dataset."""
    try:
        tb = t.to_lance().to_table(columns=["id"])
        return tb.column("id").to_numpy()
    except Exception:
        arr = t.to_arrow().column("id").to_numpy()
        return arr


def order_sig(ids):
    b = np.ascontiguousarray(ids.astype(np.int64)).tobytes()
    return hashlib.sha256(b).hexdigest()


def order_compare(a, b):
    a = np.asarray(a); b = np.asarray(b)
    n = min(len(a), len(b))
    same_len = (len(a) == len(b))
    ident = same_len and bool(np.array_equal(a, b))
    lcp = 0
    for i in range(n):
        if a[i] == b[i]:
            lcp += 1
        else:
            break
    return {"identical": ident, "len_a": int(len(a)), "len_b": int(len(b)),
            "longest_common_prefix": int(lcp),
            "first_diff_index": (None if ident else int(lcp)),
            "sha256_a": order_sig(a), "sha256_b": order_sig(b)}


# --------------------------------------------------------- main
def run_seed_index(seed, kind):
    log("seed=%d kind=%s: data" % (seed, kind))
    xb, qb, exact = make_data(seed)
    dbdir = os.path.join(tempfile.mkdtemp(prefix="disjoint_"), "run_%d_%s" % (seed, kind))
    log("seed=%d kind=%s: build table + index" % (seed, kind))
    tbl = build_table(dbdir, xb, kind)
    uuid_a = index_uuid(tbl)
    ids_build = id_order(tbl)

    gridA = grid_a(kind); gridB = grid_b(kind)
    log("seed=%d kind=%s: grids A=%d B=%d" % (seed, kind, len(gridA), len(gridB)))

    A_default = search_all(tbl, qb)
    curveA = {}; resA = {}
    t0 = time.time()
    for p in gridA:
        r = search_all(tbl, qb, p); k = pkey(p); resA[k] = r; curveA[k] = recall_of(r, exact)
    log("  A grid %.1fs" % (time.time() - t0))

    log("seed=%d kind=%s: table.optimize()" % (seed, kind))
    tbl.optimize()
    uuid_b = index_uuid(tbl)
    ids_after_opt = id_order(tbl)
    ord_build_vs_opt = order_compare(ids_build, ids_after_opt)

    B_default = search_all(tbl, qb)
    curveB = {}; resB = {}
    t0 = time.time()
    for p in gridB:
        r = search_all(tbl, qb, p); k = pkey(p); resB[k] = r; curveB[k] = recall_of(r, exact)
    log("  B grid %.1fs" % (time.time() - t0))

    log("seed=%d kind=%s: rebuild genR2" % (seed, kind))
    tbl.create_index("vector", config=config_for(kind), replace=True)
    uuid_r2 = index_uuid(tbl)
    ids_after_r2 = id_order(tbl)
    ord_build_vs_r2 = order_compare(ids_build, ids_after_r2)
    R2_default = search_all(tbl, qb)
    curveR2 = {}; resR2 = {}
    t0 = time.time()
    for p in gridA:
        r = search_all(tbl, qb, p); k = pkey(p); resR2[k] = r; curveR2[k] = recall_of(r, exact)
    log("  R2 grid %.1fs" % (time.time() - t0))

    log("seed=%d kind=%s: rebuild genR3 (independent)" % (seed, kind))
    tbl.create_index("vector", config=config_for(kind), replace=True)
    uuid_r3 = index_uuid(tbl)
    ids_after_r3 = id_order(tbl)
    ord_build_vs_r3 = order_compare(ids_build, ids_after_r3)
    R3_default = search_all(tbl, qb)
    curveR3 = {}; resR3 = {}
    t0 = time.time()
    for p in gridA:
        r = search_all(tbl, qb, p); k = pkey(p); resR3[k] = r; curveR3[k] = recall_of(r, exact)
    log("  R3 grid %.1fs" % (time.time() - t0))

    cells = {}
    pfx = "%s_s%d" % (kind, seed)

    # default rows (engine default search on both sides)
    cells[pfx + "_matched_default"] = cell_record(
        A_default, B_default, exact, pfx + "_matched_default", "genA default vs genB default")
    cells[pfx + "_disjointfloor_default"] = cell_record(
        R2_default, R3_default, exact, pfx + "_disjointfloor_default",
        "genR2 default vs genR3 default (independent rebuilds, no permutation)")
    cells[pfx + "_sharedfloor_default"] = cell_record(
        B_default, R2_default, exact, pfx + "_sharedfloor_default",
        "genB default vs genR2 default (as in matched2 floor; shares genB with matched)")

    matched_summary = {}; disjoint_summary = {}; shared_summary = {}
    for R in TARGETS:
        tag = "R%s" % str(R).replace(".", "")
        # matched (genA, genB)
        prm = pick_pair(curveA, curveB, R, 0.01)
        nm = "%s_matched_%s" % (pfx, tag)
        cells[nm] = cell_record(resA[prm["kA"]], resB[prm["kB"]], exact, nm,
                                "matched (genA vs genB) target R=%.2f; gap=%.4f" % (R, prm["gap"]),
                                extra={"target_R": R, "pair": prm, "comparison": "matched"})
        matched_summary[tag] = prm
        # disjoint floor (genR2, genR3)
        prd = pick_pair(curveR2, curveR3, R, 0.01)
        nmd = "%s_disjointfloor_%s" % (pfx, tag)
        cells[nmd] = cell_record(resR2[prd["kA"]], resR3[prd["kB"]], exact, nmd,
                                 "disjoint floor (genR2 vs genR3) target R=%.2f; gap=%.4f" % (R, prd["gap"]),
                                 extra={"target_R": R, "pair": prd, "comparison": "disjoint_floor"})
        disjoint_summary[tag] = prd
        # shared-genB floor (genB, genR2) -> reproduces matched2 floor
        prs = pick_pair(curveB, curveR2, R, 0.01)
        nms = "%s_sharedfloor_%s" % (pfx, tag)
        cells[nms] = cell_record(resB[prs["kA"]], resR2[prs["kB"]], exact, nms,
                                 "shared-genB floor (genB vs genR2) target R=%.2f; gap=%.4f" % (R, prs["gap"]),
                                 extra={"target_R": R, "pair": prs, "comparison": "shared_floor"})
        shared_summary[tag] = prs

    meta_cur = {
        "seed": seed, "index_kind": kind,
        "uuid_genA": uuid_a, "uuid_genB": uuid_b,
        "uuid_genR2": uuid_r2, "uuid_genR3": uuid_r3,
        "genR2_equals_genR3": (uuid_r2 == uuid_r3),
        "order_preservation": {
            "build_vs_optimize": ord_build_vs_opt,
            "build_vs_R2": ord_build_vs_r2,
            "build_vs_R3": ord_build_vs_r3,
        },
        "matched_summary": matched_summary,
        "disjoint_summary": disjoint_summary,
        "shared_summary": shared_summary,
        "gridA_size": len(gridA), "gridB_size": len(gridB),
    }
    log("seed=%d kind=%s DONE" % (seed, kind))
    return cells, meta_cur


def main():
    args = set(sys.argv[1:])
    seeds = [s for s in SEEDS if str(s) in args] or SEEDS
    kinds = [k for k in ("hnsw", "ivfpq") if k in args] or ["hnsw", "ivfpq"]
    cells_all = {}; runs = []
    for seed in seeds:
        for kind in kinds:
            c, m = run_seed_index(seed, kind)
            cells_all.update(c); runs.append(m)
    out = {
        "meta": {
            "engine": "LanceDB 0.40.0",
            "purpose": "Disjoint-rebuild floor for LanceDB: floor = tear(genR2, genR3) which shares NO "
                       "build with the recall-matched pair (genA, genB). Also reproduces the shared-genB "
                       "floor (genB, genR2) and MEASURES whether optimize()/rebuilds perturb the row order.",
            "protocol": {"N": N, "d": D, "metric": "L2", "n_queries": NQ, "k": K,
                         "seeds": SEEDS, "fragments_at_build": NB, "targets": TARGETS,
                         "index_hnsw": "HnswFlat(m=20, ef_construction=150, num_partitions=16)",
                         "index_ivfpq": "IvfPq(num_partitions=16, num_sub_vectors=16)"},
            "constructions": {
                "matched": "(genA, genB): genA=fresh create_index, genB=post-optimize rebuild. Shares genB with shared_floor.",
                "disjoint_floor": "(genR2, genR3): two further create_index(replace=True) rebuilds. Shares NO build with matched.",
                "shared_floor": "(genB, genR2): reproduces the matched2 floor. Shares genB with matched.",
            },
            "recall_equality_tol": 1e-9,
            "runs": runs,
        },
        "cells": {k: v for k, v in cells_all.items()},
        "by_family_level": {},
        "disjoint_ratio": {},
        "order_preservation": {},
        "caveats": {},
    }
    # accumulate order_preservation summary
    op = []
    for m in runs:
        op.append({"seed": m["seed"], "kind": m["index_kind"],
                   "genR2_equals_genR3": m["genR2_equals_genR3"],
                   "uuid_genA": m["uuid_genA"], "uuid_genB": m["uuid_genB"],
                   "uuid_genR2": m["uuid_genR2"], "uuid_genR3": m["uuid_genR3"],
                   **m["order_preservation"]})
    out["order_preservation"] = {
        "method": "Read the physical id sequence from the underlying lance dataset "
                  "(table.to_lance().to_table(['id'])) at four points: after create_index (genA), after "
                  "optimize() (genB), after rebuild#2 (genR2), after rebuild#3 (genR3). Compare each later "
                  "sequence to the genA sequence by exact equality, longest common prefix and sha256.",
        "runs": op,
        "all_preserved": all(r["build_vs_optimize"]["identical"] and r["build_vs_R2"]["identical"]
                             and r["build_vs_R3"]["identical"] for r in op),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log("WROTE %s" % OUT)
    return out


if __name__ == "__main__":
    main()
