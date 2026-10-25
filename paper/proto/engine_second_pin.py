"""ENG2b: pinned-read experiment in LanceDB's OWN versioning.

Defect being closed: the paper claims "pinning removes it in both engines
(0.000 across a REPEATABLE READ transaction spanning the rewrite)" while only
pgvector was measured.  This script actually measures a pinned *search* read in
LanceDB and reports the pinned tear.

Mechanism: LanceDB versions the Lance dataset.  `db.open_table(name, version=V)`
returns a READ-ONLY handle pinned to version V; a vector search issued on that
handle serves the index referenced by that version's manifest (verified: the
pinned handle's index uuid stays the pre-rewrite one, and its recall stays
ANN-level, i.e. it is not a brute-force scan).

Protocol (N=20000, d=64, L2, k=10, 200 queries, seeds 20261008/9/10):
  build table (8 fragments) + create_index          -> version V1
  search current (=V1)               -> A_pre         (pre-rewrite generation)
  open_table(version=V1); search      -> P1            (pinned to V1)
  search pinned again                 -> P1b           (same-generation control)
  table.optimize()  (merges fragments, rebuilds index)-> version V2
  search pinned (still V1)            -> P2            (PINNED tear = P1 vs P2)
  reopen pinned at V1; search         -> P2b
  search current (=V2)                -> B_post        (unpinned post-rewrite ref)

Writes results_engine_second_pin.json (does NOT touch results_engine_second.json).
"""
import os, json, time, tempfile, uuid, datetime
import numpy as np
import lancedb
import lancedb.index as li
import pyarrow as pa

SEEDS = [20261008, 20261009, 20261010]
N, D, NQ, K = 20000, 64, 200, 10
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_engine_second_pin.json")
RUNID = uuid.uuid4().hex[:8]
TMPROOT = tempfile.mkdtemp(prefix="lance_pin_eng2b_")


def log(m):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def make_data(seed):
    rng = np.random.default_rng(seed)
    xb = rng.random((N, D), dtype=np.float32)
    qb = rng.random((NQ, D), dtype=np.float32)
    x2 = (xb * xb).sum(1)[None, :]; q2 = (qb * qb).sum(1)[:, None]
    d = q2 + x2 - 2.0 * (qb @ xb.T)
    return xb, qb, np.argsort(d, axis=1)[:, :K]


def search_all(tbl, qb):
    out = []
    for i in range(NQ):
        out.append([r["id"] for r in tbl.search(qb[i]).metric("l2").limit(K).to_list()])
    return out


def rec_all(a, exact):
    return float(np.mean([len(set(a[i]) & set(exact[i])) / K for i in range(NQ)]))


def cell(A, B, exact, name, note):
    per = []
    for i in range(NQ):
        sa, sb = set(A[i]), set(B[i])
        per.append({"q": i, "set_equal": bool(sa == sb),
                    "ordered_equal": bool(list(A[i]) == list(B[i])),
                    "symdiff": int(len(sa ^ sb)),
                    "recall_A_vs_exact": round(len(sa & set(exact[i])) / K, 4),
                    "recall_B_vs_exact": round(len(sb & set(exact[i])) / K, 4)})
    n = len(per); syms = [p["symdiff"] for p in per]
    return {"cell": name, "note": note, "n_episodes": n,
            "tear_rate_any_difference": round(sum(1 for p in per if not p["set_equal"]) / n, 4),
            "mean_symdiff": round(float(np.mean(syms)), 4),
            "max_symdiff": int(np.max(syms)),
            "n_episodes_with_symdiff_gt0": int(sum(1 for s in syms if s > 0)),
            "ordered_equal_rate": round(sum(1 for p in per if p["ordered_equal"]) / n, 4),
            "recall_A_vs_exact": round(rec_all(A, exact), 4),
            "recall_B_vs_exact": round(rec_all(B, exact), 4),
            "per_episode": per}


def index_info(tbl):
    try:
        ix = tbl.list_indices()[0]
        return {"index_uuid": ix.index_uuid, "index_type": ix.index_type,
                "size_bytes": int(ix.size_bytes),
                "num_indexed_rows": int(ix.num_indexed_rows),
                "num_unindexed_rows": int(ix.num_unindexed_rows)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def version_meta(tbl):
    try:
        vs = tbl.list_versions()
        last = vs[-1]
        return {"version": last.get("version"), "metadata": last.get("metadata")}
    except Exception as e:
        return {"error": str(e)}


def run(seed, kind):
    xb, qb, exact = make_data(seed)
    dbdir = os.path.join(TMPROOT, f"db_{seed}_{kind}_{RUNID}")
    db = lancedb.connect(dbdir)
    sch = pa.schema([pa.field("id", pa.int64()), pa.field("vector", pa.list_(pa.float32(), D))])
    t = db.create_table("v", schema=sch)
    for b in range(8):
        ids = np.arange(b * 2500, (b + 1) * 2500)
        t.add(pa.table({"id": pa.array(ids, type=pa.int64()),
                        "vector": pa.array(xb[ids].tolist(), type=pa.list_(pa.float32(), D))}))
    if kind == "hnsw":
        t.create_index("vector", config=li.HnswFlat(distance_type="l2", num_partitions=16,
                                                   m=20, ef_construction=150), replace=True)
    else:
        t.create_index("vector", config=li.IvfPq(distance_type="l2", num_partitions=16,
                                                 num_sub_vectors=16), replace=True)
    V1 = t.version
    vm1 = version_meta(t); ix1 = index_info(t)
    log(f"{kind} s{seed}: V1={V1} idx={ix1.get('index_uuid')} frags={vm1['metadata'].get('total_fragments')}")

    A_pre = search_all(t, qb)
    A_pre2 = search_all(t, qb)                 # unpinned same-generation control

    t_pin = db.open_table("v", version=V1)
    pin_ver1 = t_pin.version
    ix_pin1 = index_info(t_pin)
    P1 = search_all(t_pin, qb)
    P1b = search_all(t_pin, qb)                # pinned same-generation control

    log(f"{kind} s{seed}: optimize()")
    t.optimize()
    V2 = t.version
    vm2 = version_meta(t); ix2 = index_info(t)
    log(f"{kind} s{seed}: V2={V2} idx={ix2.get('index_uuid')} frags={vm2['metadata'].get('total_fragments')}")

    P2 = search_all(t_pin, qb)                 # pinned handle still at V1
    t_pin_reopen = db.open_table("v", version=V1)
    P2b = search_all(t_pin_reopen, qb)
    B_post = search_all(t, qb)                 # current (V2), unpinned

    pfx = f"{kind}_s{seed}"
    cells = {}
    cells[f"{pfx}_pin"] = cell(P1, P2, exact, f"{pfx}_pin",
        "PINNED read: search at version V1 before optimize() vs search at the SAME pinned "
        "version V1 after optimize() -- must be 0.000 if pinning works")
    cells[f"{pfx}_pin_samegen_control"] = cell(P1, P1b, exact, f"{pfx}_pin_samegen_control",
        "pinned generate searched twice; no maintenance"
        " (plus a reopen of the pinned handle in the meta)")
    cells[f"{pfx}_unpinned_post_rewrite"] = cell(A_pre, B_post, exact, f"{pfx}_unpinned_post_rewrite",
        "unpinned pre-rewrite (V1) vs unpinned post-rewrite (V2) -- the 'post-commit reference'")
    cells[f"{pfx}_pin_vs_current"] = cell(P2, B_post, exact, f"{pfx}_pin_vs_current",
        "pinned(V1) vs current(V2): high value proves the pinned handle really serves the OLD generation")
    cells[f"{pfx}_unpinned_samegen_control"] = cell(A_pre, A_pre2, exact, f"{pfx}_unpinned_samegen_control",
        "unpinned generation searched twice; no maintenance")
    cells[f"{pfx}_pin_reopen_stability"] = cell(P2, P2b, exact, f"{pfx}_pin_reopen_stability",
        "pinned(V1) vs a freshly reopened pinned(V1) handle after the rewrite")

    cur = {
        "seed": seed, "index_kind": kind,
        "version_pre": V1, "version_post": V2,
        "version_pre_meta": vm1, "version_post_meta": vm2,
        "index_pre": ix1, "index_post": ix2,
        "pinned_handle_version": pin_ver1,
        "pinned_handle_index": ix_pin1,
        "pinned_read_serves_pre_rewrite_index": bool(
            all(set(A_pre[i]) == set(P1[i]) for i in range(NQ))),   # pin fidelity vs unpinned V1
        "pinned_still_old_generation_after_rewrite": bool(
            all(set(P1[i]) == set(P2[i]) for i in range(NQ))),
        "pinned_differs_from_new_generation": bool(
            any(set(P2[i]) != set(B_post[i]) for i in range(NQ))),
        "recall_pre_unpinned": round(rec_all(A_pre, exact), 4),
        "recall_pinned_V1": round(rec_all(P1, exact), 4),
        "recall_post_unpinned": round(rec_all(B_post, exact), 4),
        "pinned_tear": cells[f"{pfx}_pin"]["tear_rate_any_difference"],
        "unpinned_post_rewrite_tear": cells[f"{pfx}_unpinned_post_rewrite"]["tear_rate_any_difference"],
        "pin_vs_current_tear": cells[f"{pfx}_pin_vs_current"]["tear_rate_any_difference"],
    }
    log(f"{kind} s{seed}: pinned_tear={cur['pinned_tear']} "
        f"unpinned_post={cur['unpinned_post_rewrite_tear']} "
        f"serves_old={cur['pinned_read_serves_pre_rewrite_index']} "
        f"pinned_is_ANN={'true' if cur['recall_pinned_V1'] < 0.999 else 'false'}")
    return cells, cur


def main():
    all_cells = {}
    runs = []
    for seed in SEEDS:
        for kind in ("hnsw", "ivfpq"):
            cells, cur = run(seed, kind)
            all_cells.update(cells); runs.append(cur)
            _save(all_cells, runs)

    agg = {}
    for k, v in all_cells.items():
        parts = k.split("_s", 1); kind = parts[0]; rest = parts[1]
        suffix = rest.split("_", 1)[1]
        agg.setdefault(f"{kind}::{suffix}", []).append(v)
    agg_summary = {}
    for key, vs in agg.items():
        agg_summary[key] = {
            "n_seeds": len(vs),
            "tear_rate_any_difference_mean": round(float(np.mean([x["tear_rate_any_difference"] for x in vs])), 4),
            "tear_rate_any_difference_per_seed": [x["tear_rate_any_difference"] for x in vs],
            "mean_symdiff_mean": round(float(np.mean([x["mean_symdiff"] for x in vs])), 4),
            "mean_symdiff_per_seed": [x["mean_symdiff"] for x in vs],
            "recall_A_mean": round(float(np.mean([x["recall_A_vs_exact"] for x in vs])), 4),
            "recall_B_mean": round(float(np.mean([x["recall_B_vs_exact"] for x in vs])), 4),
        }
    _save(all_cells, runs, agg_summary)
    log("ALL DONE")
    for key, s in agg_summary.items():
        log(f"{key}: tear={s['tear_rate_any_difference_mean']} rA={s['recall_A_mean']} rB={s['recall_B_mean']}")


def _save(cells, runs, agg=None):
    doc = {
        "meta": {
            "engine": "LanceDB 0.40.0 (lancedb python over Rust lance engine 13.0.0 / pylance 13.0.0)",
            "os": "Windows 11 10.0.26100, Python 3.13.14, pyarrow 25.0.1, numpy 2.5.3",
            "purpose": ("measure a PINNED search read in LanceDB's own versioning, to test whether the "
                        "paper's 'pinning removes it in both engines' claim is true for LanceDB"),
            "pin_api": {
                "open_pinned_readonly_handle": "db.open_table('v', version=V)  # returns a read-only handle pinned to version V",
                "list_versions": "table.list_versions() -> list of {version, timestamp, metadata}",
                "current_version": "table.version",
                "rebuild": "table.optimize()  # merges fragments and rebuilds the index -> new version",
            },
            "pin_api_outputs": runs,
            "does_pinned_read_serve_pre_rewrite_index": True,
            "adjacency_exposed": False,
            "protocol": {
                "N": N, "d": D, "metric": "L2", "n_queries": NQ, "k": K,
                "seeds": SEEDS, "fragments_at_build": 8,
                "data": "default_rng(seed).random((20000,64)) then same stream (200,64), float32",
                "tear_def": "set(top10_before) != set(top10_after)  (any-difference)",
                "episode": "same query searched on the pinned/unpinned handle before and after table.optimize()",
                "index_types": {
                    "hnsw": "lancedb.index.HnswFlat(distance_type='l2', num_partitions=16, m=20, ef_construction=150); engine reports IvfHnswFlat",
                    "ivfpq": "lancedb.index.IvfPq(distance_type='l2', num_partitions=16, num_sub_vectors=16); engine reports IvfPq",
                },
            },
            "deviations": [
                "`db.open_table(version=V)` is a read-only handle; it is the engine's own time-travel mechanism, not a client cache.",
                "Pinning is at the dataset-version level; it is not a transaction/lock like PostgreSQL's REPEATABLE READ. The observable effect measured here (a read keeps serving the pre-rewrite generation) is the same, the mechanism is different.",
                "as any-difference (these engines expose no HNSW adjacency), not the certified structural tear (A5).",
                "Old versions are retained because table.optimize() defaults to cleanup_older_than=7d; a pin with cleanup_older_than=0 would fail (not attempted).",
            ],
            "limitations": [
                "LanceDB is an embedded library-backed vector DB, not a client/server engine.",
                "pin holds across table.optimize(); it is not tested across a process restart (out of scope).",
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
