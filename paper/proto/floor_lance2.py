"""LanceDB same-order rebuild non-determinism floor, measured on the current
3-seed / 200-episode protocol.

For each (seed, index_kind) TWO independent tables are created over the SAME,
non-reordered data (identical insertion order, identical index parameters),
each is passed through the engine's own maintenance path table.optimize(), and
the two generations' top-10 sets are compared.

Why two fresh tables instead of optimize() twice on one table:
  table.optimize() is idempotent on unchanged data -- a second call leaves the
  index_uuid unchanged (verified), so it cannot supply the second rebuild.
  Two independent builds through the same code path that optimize() executes do.

The floor is reported at the engine default search setting and recall-matched
(the pair-selection rule in engine_matched2_lance.py) at target recall 0.90 / 0.95 / 0.99.

This script REPLACES every `match2_*_floor_lance_*` cell in the LanceDB part
file so the whole floor is measured one way.
"""
import os, json, time, tempfile
import numpy as np
import lancedb, lancedb.index as li, pyarrow as pa

HERE = os.path.dirname(os.path.abspath(__file__))
PART = os.path.join(HERE, "results_engine_matched2_lance_part.json")

SEEDS = [20261008, 20261009, 20261010]
N, D, NQ, K = 20000, 64, 200, 10
NB = 8
TARGETS = [0.90, 0.95, 0.99]
GAP_OK = 0.004

# Matching grids.  Both tables are post-optimize rebuilds, where nprobes is
# INERT for both index kinds (verified: recall range identical for every
# nprobes value), so the grids are dense 1-D ladders: ef for HnswFlat,
# refine_factor for IvfPq.
GRID = {
    # The hnsw ladder is extended to cover the ef values recorded in the floor
    # cells of results_engine_matched2.json (90, 120, 160, 250, 310): those
    # cells were produced on this extended ladder, so the grid must contain
    # them, otherwise the knobs written into the artifact cannot be reproduced
    # from this script.  merge_matched2.py asserts this inclusion.
    "hnsw": [{"nprobes": None, "ef": e}
             for e in (10, 12, 14, 16, 18, 20, 24, 28, 34, 40, 50, 64, 80, 90,
                       100, 120, 130, 160, 170, 220, 250, 290, 310, 380, 500)],
    "ivfpq": [{"nprobes": None, "refine_factor": r}
              for r in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20,
                        25, 30, 40, 50, 75, 100)],
}


def pick_pair(curveA, curveB, target, w=0.01):
    """Prefer an in-window pair with gap<=GAP_OK closest
    to the target; else smallest in-window gap; if nothing is in-window, the
    gap<=GAP_OK pair closest to the target (flagged); else tightest gap."""
    ca = [k for k in curveA if abs(curveA[k] - target) <= w]
    cb = [k for k in curveB if abs(curveB[k] - target) <= w]
    if ca and cb:
        pairs = [(abs(curveA[ka] - curveB[kb]),
                  abs((curveA[ka] + curveB[kb]) / 2.0 - target), ka, kb)
                 for ka in ca for kb in cb]
        ok = [p for p in pairs if p[0] <= GAP_OK]
        if ok:
            g, m, ka, kb = min(ok, key=lambda p: (p[1], p[0])); reached = True
        else:
            g, m, ka, kb = min(pairs, key=lambda p: (p[0], p[1])); reached = False
        return {"window": w, "placeable": True, "kA": ka, "kB": kb,
                "recall_A": round(curveA[ka], 4),
                "recall_B": round(curveB[kb], 4),
                "gap": round(g, 4), "reached_le_0p004": bool(reached),
                "dist_to_target": round(m, 4)}
    allp = [(abs(curveA[ka] - curveB[kb]),
             abs((curveA[ka] + curveB[kb]) / 2.0 - target), ka, kb)
            for ka in curveA for kb in curveB]
    ok = [p for p in allp if p[0] <= GAP_OK]
    if ok:
        g, m, ka, kb = min(ok, key=lambda p: (p[1], p[0]))
        return {"window": w, "placeable": False, "kA": ka, "kB": kb,
                "recall_A": round(curveA[ka], 4),
                "recall_B": round(curveB[kb], 4),
                "gap": round(g, 4), "reached_le_0p004": True,
                "dist_to_target": round(m, 4),
                "note": ("no knob setting lands within +/-%.3f of the target; "
                         "genuine <=0.004 match at a recall further from it" % w)}
    g, m, ka, kb = min(allp, key=lambda p: (p[0], p[1]))
    return {"window": w, "placeable": False, "kA": ka, "kB": kb,
            "recall_A": round(curveA[ka], 4), "recall_B": round(curveB[kb], 4),
            "gap": round(g, 4), "reached_le_0p004": False,
            "dist_to_target": round(m, 4),
            "note": "no pair reaches gap<=0.004; tightest gap reported"}


def config_for(kind):
    if kind == "hnsw":
        return li.HnswFlat(distance_type="l2", num_partitions=16, m=20,
                           ef_construction=150)
    return li.IvfPq(distance_type="l2", num_partitions=16, num_sub_vectors=16)


def build(xb, kind, tag):
    dbdir = os.path.join(tempfile.mkdtemp(prefix="floorlance_"), tag)
    db = lancedb.connect(dbdir)
    sch = pa.schema([pa.field("id", pa.int64()),
                     pa.field("vector", pa.list_(pa.float32(), D))])
    t = db.create_table("v", schema=sch)
    for b in range(NB):
        ids = np.arange(b * (N // NB), (b + 1) * (N // NB))
        t.add(pa.table({"id": pa.array(ids, type=pa.int64()),
                        "vector": pa.array(xb[ids].tolist(),
                                           type=pa.list_(pa.float32(), D))}))
    t.create_index("vector", config=config_for(kind), replace=True)
    t.optimize()
    return t


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


def rec(r, exact):
    return float(np.mean([len(set(r[i]) & set(exact[i])) / K for i in range(NQ)]))


def pkey(p):
    return json.dumps(p, sort_keys=True)


def cell_record(A, B, exact, name, note, extra=None):
    per = []
    for i in range(NQ):
        sa, sb = set(A[i]), set(B[i])
        per.append({"q": i, "set_equal": bool(sa == sb),
                    "ordered_equal": bool(list(A[i]) == list(B[i])),
                    "symdiff": int(len(sa ^ sb)),
                    "recall_A_vs_exact": round(len(sa & set(exact[i])) / K, 4),
                    "recall_B_vs_exact": round(len(sb & set(exact[i])) / K, 4)})
    n = len(per); syms = [p["symdiff"] for p in per]
    out = {"cell": name, "note": note, "n_episodes": n,
           "tear_rate_any_difference": round(sum(1 for p in per if not p["set_equal"]) / n, 4),
           "mean_symdiff": round(float(np.mean(syms)), 4),
           "max_symdiff": int(np.max(syms)),
           "n_episodes_with_symdiff_gt0": int(sum(1 for s in syms if s > 0)),
           "ordered_equal_rate": round(sum(1 for p in per if p["ordered_equal"]) / n, 4),
           "recall_A_vs_exact": round(rec(A, exact), 4),
           "recall_B_vs_exact": round(rec(B, exact), 4),
           "per_episode": per}
    if extra:
        out.update(extra)
    return out


def main():
    doc = json.load(open(PART, encoding="utf-8"))
    cells = doc["cells"]
    for k in [k for k in cells if "_floor_lance_" in k]:
        del cells[k]
    runs = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        xb = rng.random((N, D), dtype=np.float32)
        qb = rng.random((NQ, D), dtype=np.float32)
        x2 = (xb * xb).sum(1)[None, :]; q2 = (qb * qb).sum(1)[:, None]
        exact = np.argsort(q2 + x2 - 2.0 * (qb @ xb.T), axis=1)[:, :K]
        for kind in ("hnsw", "ivfpq"):
            print("[%s] seed=%d kind=%s: two fresh builds + optimize"
                  % (time.strftime("%H:%M:%S"), seed, kind), flush=True)
            t1 = build(xb, kind, "a_%d_%s" % (seed, kind))
            t2 = build(xb, kind, "b_%d_%s" % (seed, kind))
            grid = GRID[kind]
            c1, r1, c2, r2 = {}, {}, {}, {}
            for p in grid:
                k = pkey(p)
                r1[k] = search_all(t1, qb, p); c1[k] = rec(r1[k], exact)
                r2[k] = search_all(t2, qb, p); c2[k] = rec(r2[k], exact)
            d1 = search_all(t1, qb); d2 = search_all(t2, qb)
            pfx = "match2_%s_s%d" % (kind, seed)
            cells[pfx + "_floor_lance_default"] = cell_record(
                d1, d2, exact, pfx + "_floor_lance_default",
                "LanceDB same-order rebuild floor at the engine default search "
                "setting: two independent builds of the SAME non-reordered data, "
                "each passed through table.optimize()")
            fs = {"default": {"tear": cells[pfx + "_floor_lance_default"]["tear_rate_any_difference"],
                              "recall_1": cells[pfx + "_floor_lance_default"]["recall_A_vs_exact"],
                              "recall_2": cells[pfx + "_floor_lance_default"]["recall_B_vs_exact"]}}
            for R in TARGETS:
                tag = "R%s" % str(R).replace(".", "")
                pr = pick_pair(c1, c2, R, 0.01)
                pr2 = pick_pair(c1, c2, R, 0.02)
                nm = "%s_floor_lance_%s" % (pfx, tag)
                cells[nm] = cell_record(
                    r1[pr["kA"]], r2[pr["kB"]], exact, nm,
                    "LanceDB same-order rebuild floor at target R=%.2f (window "
                    "+/-0.01): rebuild#1 %s -> %.4f, rebuild#2 %s -> %.4f" % (
                        R, pr["kA"], pr["recall_A"], pr["kB"], pr["recall_B"]),
                    extra={"target_R": R, "matched": pr, "matched_w0p02": pr2,
                           "floor_kind": "same-order rebuild",
                           "gap": pr["gap"]})
                fs[tag] = {"tear": cells[nm]["tear_rate_any_difference"],
                           "gap": pr["gap"], "gap_w0p02": pr2["gap"],
                           "recall_1": pr["recall_A"], "recall_2": pr["recall_B"]}
            runs.append({"seed": seed, "index_kind": kind, "grid_size": len(grid),
                         "floor_summary": fs})
            print("   floor default=%.3f R09=%.3f R095=%.3f R099=%.3f" % (
                fs["default"]["tear"], fs["R09"]["tear"], fs["R095"]["tear"],
                fs["R099"]["tear"]), flush=True)
            json.dump(doc, open(PART, "w", encoding="utf-8"), indent=1)
    doc["meta"]["floor_lance_runs"] = runs
    doc["meta"]["floor_lance_method"] = (
        "two independent fresh builds of the SAME non-reordered data, each passed "
        "through table.optimize(); optimize() itself is idempotent on unchanged "
        "data so it cannot provide the second rebuild.")
    json.dump(doc, open(PART, "w", encoding="utf-8"), indent=1)
    print("floor done; replaced all _floor_lance_ cells")


if __name__ == "__main__":
    main()
