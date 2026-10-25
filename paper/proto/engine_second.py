"""ENG2: second shipped engine's OWN maintenance/compaction path.

Engine: LanceDB 0.40.0 (Rust-backed, pyarrow-backed table format).
Maintenance op: table.optimize()  -- the engine's own VACUUM-like path; it
compacts fragments AND rewrites the vector index (observed: new index_uuid,
new created_at, different size_bytes).  No vector is added or removed.

Protocol matches the paper's engine experiment:
  N=20000, d=64, L2, 200 queries, k=10, seeds 20261008/9/10.
  gen A = search right after create_index
  (engine maintenance: table.optimize())
  gen B = search after the rewrite
  tear (any-difference) = set(top10_A) != set(top10_B)
  same-generation control = search the SAME generation twice (must be 0)
  recall vs exact top-10 for every generation
  recall-matched variant: search each generation through its own effort knob
  so both hit the same recall, then measure the tear (closes the
  "generation B is just a worse index" objection).

Two index types are measured:
  IvfHnswFlat (created via lancedb.index.HnswFlat) -- LanceDB's HNSW, the
     closest analogue to the pgvector HNSW cell.
  IvfPq (created via lancedb.index.IvfPq) -- closest analogue to the paper's
     own FAISS IVF-PQ prototype.

Raw per_episode records are stored for every cell.
"""
import os, sys, json, time, shutil, datetime, tempfile
import numpy as np
import lancedb
import lancedb.index as li
import pyarrow as pa

TMPROOT = tempfile.mkdtemp(prefix="lance_engine_second_")

SEEDS = [20261008, 20261009, 20261010]
N, D, NQ, K = 20000, 64, 200, 10
NB = 8                      # fragments the engine has to compact
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_engine_second.json")

EF_GRID = [10, 12, 15, 18, 22, 30, 40, 50, 75, 100, 150, 200, 300, 500, 1000, 2000]
REFINE_GRID = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 100, 200]
MATCH_TARGETS = [0.90, 0.95, 0.99]


def log(msg):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def make_data(seed):
    rng = np.random.default_rng(seed)
    xb = rng.random((N, D), dtype=np.float32)
    qb = rng.random((NQ, D), dtype=np.float32)
    # squared L2 via matmul (memory-light, exact same ordering as direct L2)
    x2 = (xb * xb).sum(axis=1)[None, :]
    q2 = (qb * qb).sum(axis=1)[:, None]
    d = q2 + x2 - 2.0 * (qb @ xb.T)          # (NQ, N)
    exact = np.argsort(d, axis=1)[:, :K]
    return xb, qb, exact


def build_table(dbdir, xb, index_kind):
    db = lancedb.connect(dbdir)
    sch = pa.schema([pa.field("id", pa.int64()),
                     pa.field("vector", pa.list_(pa.float32(), D))])
    t = db.create_table("v", schema=sch)
    pb = N // NB
    for b in range(NB):
        ids = np.arange(b * pb, (b + 1) * pb)
        t.add(pa.table({"id": pa.array(ids, type=pa.int64()),
                        "vector": pa.array(xb[ids].tolist(), type=pa.list_(pa.float32(), D))}))
    if index_kind == "hnsw":
        t.create_index("vector", config=li.HnswFlat(distance_type="l2", num_partitions=16,
                                                    m=20, ef_construction=150), replace=True)
    elif index_kind == "ivfpq":
        t.create_index("vector", config=li.IvfPq(distance_type="l2", num_partitions=16,
                                                 num_sub_vectors=16), replace=True)
    else:
        raise ValueError(index_kind)
    return t


def index_meta(t):
    try:
        inds = t.list_indices()
    except Exception as e:
        return {"error": str(e)}
    out = []
    for ix in inds:
        out.append({
            "name": ix.name,
            "index_type": ix.index_type,
            "index_uuid": ix.index_uuid,
            "created_at": str(ix.created_at),
            "size_bytes": int(ix.size_bytes),
            "num_indexed_rows": int(ix.num_indexed_rows),
            "num_unindexed_rows": int(ix.num_unindexed_rows),
            "num_segments": int(ix.num_segments),
            "index_details": json.loads(json.dumps(ix.index_details, default=str)),
        })
    return out


def search_all(t, qb, knob=None, val=None):
    out = []
    for i in range(NQ):
        q = t.search(qb[i]).metric("l2").limit(K)
        if knob is not None:
            q = getattr(q, knob)(val)
        rows = q.to_list()
        out.append([r["id"] for r in rows])
    return out


def recall_of(res, exact):
    return float(np.mean([len(set(res[i]) & set(exact[i])) / K for i in range(NQ)]))


def cell_record(A, B, exact, name, note):
    per = []
    for i in range(NQ):
        sa, sb = set(A[i]), set(B[i])
        sym = len(sa ^ sb)
        per.append({
            "q": i,
            "set_equal": bool(sa == sb),
            "ordered_equal": bool(list(A[i]) == list(B[i])),
            "symdiff": int(sym),
            "recall_A_vs_exact": round(len(sa & set(exact[i])) / K, 4),
            "recall_B_vs_exact": round(len(sb & set(exact[i])) / K, 4),
        })
    n = len(per)
    tear = sum(1 for p in per if not p["set_equal"]) / n
    syms = [p["symdiff"] for p in per]
    return {
        "cell": name,
        "note": note,
        "n_episodes": n,
        "tear_rate_any_difference": round(tear, 4),
        "mean_symdiff": round(float(np.mean(syms)), 4),
        "max_symdiff": int(np.max(syms)),
        "n_episodes_with_symdiff_gt0": int(sum(1 for s in syms if s > 0)),
        "ordered_equal_rate": round(sum(1 for p in per if p["ordered_equal"]) / n, 4),
        "recall_A_vs_exact": round(recall_of(A, exact), 4),
        "recall_B_vs_exact": round(recall_of(B, exact), 4),
        "per_episode": per,
    }


def nearest(grid, curve, target):
    return min(grid, key=lambda g: abs(curve[g] - target))


def run_seed_index(seed, index_kind):
    log(f"seed={seed} index={index_kind}: data")
    xb, qb, exact = make_data(seed)
    dbdir = os.path.join(TMPROOT, f"run_{seed}_{index_kind}")
    log(f"seed={seed} index={index_kind}: build table + index")
    t0 = time.time()
    tbl = build_table(dbdir, xb, index_kind)
    build_s = time.time() - t0
    rows_before = tbl.count_rows()
    meta_before = index_meta(tbl)

    knob = "ef" if index_kind == "hnsw" else "refine_factor"
    grid = EF_GRID if index_kind == "hnsw" else REFINE_GRID

    log(f"seed={seed} index={index_kind}: search generation A")
    A_default = search_all(tbl, qb)
    A_ctrl = search_all(tbl, qb)                     # same-generation control
    A_grid = {}
    for g in grid:
        A_grid[g] = search_all(tbl, qb, knob, g)
        log(f"   A {knob}={g} recall={recall_of(A_grid[g], exact):.3f}")

    log(f"seed={seed} index={index_kind}: table.optimize()")
    t1 = time.time()
    tbl.optimize()
    optimize_s = time.time() - t1
    rows_after = tbl.count_rows()
    meta_after = index_meta(tbl)

    log(f"seed={seed} index={index_kind}: search generation B")
    B_default = search_all(tbl, qb)
    B_ctrl = search_all(tbl, qb)
    B_grid = {}
    for g in grid:
        B_grid[g] = search_all(tbl, qb, knob, g)
        log(f"   B {knob}={g} recall={recall_of(B_grid[g], exact):.3f}")

    cells = {}
    pfx = f"{index_kind}_s{seed}"
    cells[f"{pfx}_default"] = cell_record(
        A_default, B_default, exact, f"{pfx}_default",
        "gen A default search -> table.optimize() -> gen B default search")
    cells[f"{pfx}_samegen_control_A"] = cell_record(
        A_default, A_ctrl, exact, f"{pfx}_samegen_control_A",
        "search generation A twice, no maintenance in between (must be 0)")
    cells[f"{pfx}_samegen_control_B"] = cell_record(
        B_default, B_ctrl, exact, f"{pfx}_samegen_control_B",
        "search generation B twice, no maintenance in between (must be 0)")

    curveA = {g: recall_of(A_grid[g], exact) for g in grid}
    curveB = {g: recall_of(B_grid[g], exact) for g in grid}
    for R in MATCH_TARGETS:
        gA = nearest(grid, curveA, R)
        gB = nearest(grid, curveB, R)
        nm = f"{pfx}_recallmatched_R{str(R).replace('.','')}"
        cells[nm] = cell_record(
            A_grid[gA], B_grid[gB], exact, nm,
            f"recall-matched: A {knob}={gA} (recall {curveA[gA]:.3f}) vs "
            f"B {knob}={gB} (recall {curveB[gB]:.3f}); target R={R}")
        cells[nm]["matched"] = {f"A_{knob}": gA, f"B_{knob}": gB,
                                "recall_A": round(curveA[gA], 4),
                                "recall_B": round(curveB[gB], 4),
                                "target_R": R, "knob": knob}

    cur = {
        "seed": seed,
        "index_kind": index_kind,
        "index_type_reported_by_engine": meta_before[0]["index_type"] if meta_before and "index_type" in meta_before[0] else None,
        "knob": knob,
        "build_s": round(build_s, 2),
        "optimize_s": round(optimize_s, 2),
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_preserved": rows_before == rows_after == N,
        "index_meta_before": meta_before,
        "index_meta_after": meta_after,
        "index_rewritten": bool(meta_before and meta_after and
                                meta_before[0]["index_uuid"] != meta_after[0]["index_uuid"]),
        "recall_curve_A": {str(g): round(curveA[g], 4) for g in grid},
        "recall_curve_B": {str(g): round(curveB[g], 4) for g in grid},
    }
    log(f"seed={seed} index={index_kind}: DONE "
        f"default tear={cells[f'{pfx}_default']['tear_rate_any_difference']} "
        f"recA={cells[f'{pfx}_default']['recall_A_vs_exact']} "
        f"recB={cells[f'{pfx}_default']['recall_B_vs_exact']}")
    return cells, cur


def main():
    all_cells = {}
    meta_runs = []
    for seed in SEEDS:
        for kind in ("hnsw", "ivfpq"):
            cells, cur = run_seed_index(seed, kind)
            all_cells.update(cells)
            meta_runs.append(cur)
            # incremental save so nothing is lost
            _save(all_cells, meta_runs)

    # aggregate across seeds per (index_kind, cell_suffix)
    agg = {}
    for k, v in all_cells.items():
        if "per_episode" not in v:
            continue
        # suffix = everything after the seed part
        parts = k.split("_s", 1)
        kind = parts[0]
        rest = parts[1] if len(parts) > 1 else k
        # rest starts with seednumber_cellsuffix
        suffix = rest.split("_", 1)[1] if "_" in rest else rest
        key = f"{kind}::{suffix}"
        agg.setdefault(key, []).append(v)
    agg_summary = {}
    for key, vs in agg.items():
        agg_summary[key] = {
            "n_seeds": len(vs),
            "tear_rate_any_difference_mean": round(float(np.mean([x["tear_rate_any_difference"] for x in vs])), 4),
            "tear_rate_any_difference_per_seed": [x["tear_rate_any_difference"] for x in vs],
            "mean_symdiff_mean": round(float(np.mean([x["mean_symdiff"] for x in vs])), 4),
            "mean_symdiff_per_seed": [x["mean_symdiff"] for x in vs],
            "max_symdiff_max": int(max(x["max_symdiff"] for x in vs)),
            "recall_A_mean": round(float(np.mean([x["recall_A_vs_exact"] for x in vs])), 4),
            "recall_B_mean": round(float(np.mean([x["recall_B_vs_exact"] for x in vs])), 4),
            "recall_A_per_seed": [x["recall_A_vs_exact"] for x in vs],
            "recall_B_per_seed": [x["recall_B_vs_exact"] for x in vs],
        }
    _save(all_cells, meta_runs, agg_summary)
    log("ALL DONE")
    # console headline
    for key, s in agg_summary.items():
        log(f"{key}: tear={s['tear_rate_any_difference_mean']} sym={s['mean_symdiff_mean']} "
            f"rA={s['recall_A_mean']} rB={s['recall_B_mean']}")


def _save(cells, runs, agg=None):
    doc = {
        "meta": {
            "engine": "LanceDB 0.40.0 (lancedb python package; Rust lance engine)",
            "engine_import": f"lancedb {lancedb.__version__}",
            "table_format": "Lance (columnar, fragment-based) with pyarrow 25.0.1",
            "maintenance_api_call": "table.optimize()  # defaults: cleanup_older_than=7d, delete_unverified=False, retrain deprecated",
            "maintenance_semantics": ("LanceDB's VACUUM-like path: compacts fragments AND rewrites the "
                                      "vector index. Observed: index_uuid and created_at change, size_bytes changes, "
                                      "rows stay 20000. No vector is deleted."),
            "search_api": "table.search(q).metric('l2').limit(10)[.<knob>(v)].to_list()",
            "index_types": {
                "hnsw": "lancedb.index.HnswFlat(distance_type='l2', num_partitions=16, m=20, ef_construction=150); engine reports it as IvfHnswFlat",
                "ivfpq": "lancedb.index.IvfPq(distance_type='l2', num_partitions=16, num_sub_vectors=16); engine reports it as IvfPq",
            },
            "protocol": {
                "N": N, "d": D, "metric": "L2", "n_queries": NQ, "k": K,
                "fragments_at_build": NB,
                "data": "default_rng(seed).random((20000,64)) then same stream (200,64), float32",
                "exact_reference": "numpy full pairwise L2, argsort top-10",
                "episode": "top10(genA, q) vs top10(genB, q); genA = after create_index, genB = after table.optimize()",
                "tear_def": "set(top10_before) != set(top10_after)  (ANY-DIFFERENCE convention: these engines expose no HNSW adjacency, so a structural tear cannot be certified)",
                "seeds": SEEDS,
                "cells_per_seed_index": "default + same-generation controls (A and B) + recall-matched R in {0.90,0.95,0.99}",
                "recall_matched_knob": "HnswFlat -> ef (>=k); IvfPq -> refine_factor",
            },
            "adjacency_exposed": False,
            "deviations": [
                "LanceDB exposes no HNSW adjacency -> the paper's structural-tear (A5) claim is NOT certified here; only the answer-level (A4) any-difference tear is measured.",
                "Data is inserted as 8 appends (fragments) and the index is built over those fragments, then table.optimize() compacts them; the same tear also appears with 1 fragment, so the effect is the index rewrite, not compaction per se.",
                "The engine's default search (no explicit knob) is used for the headline cell; the recall-matched cell adjusts the engine's own effort knob.",
                "Uniform random data as in the paper's engine experiment; no vector is removed by optimize().",
            ],
            "limitations": [
                "Second engine is LanceDB (embedded library-backed system), not a server-side multi-tenant vector DB; it ships its own on-disk index + compaction path.",
                "Recall-matched control is at R in {0.90,0.95,0.99}; the tear is reported at each.",
            ],
            "runs": runs,
        },
        "cells": cells,
        "aggregate": agg if agg is not None else {},
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)


if __name__ == "__main__":
    main()
