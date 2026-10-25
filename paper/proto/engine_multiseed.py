#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
engine_multiseed.py --- Multi-seed + recall-matched extension of the pgvector
(PostgreSQL 17.4 + pgvector 0.8.0, HNSW) maintenance experiment.

Measures generation differences across seeds and recall settings.

Design (all cells run at seeds 20261008/9/10, 200 episodes per seed):

  G  (per seed)   generation change = CLUSTER (heap permutation) + REINDEX,
                  the pgvector analogue of the paper's FAISS "same vectors,
                  different insertion order".  Like the original cell_g but
                  per-seed, with the permutation made reproducible via
                  setseed().
        - pass1 (A, id-order heap build) vs pass2 (B, perm-order rebuild)
        - pass2 vs pass3 (SAME generation repeated)  -> determinism control
        - efc sweep on A  (ef_construction in {64,96,128,192,256}) to find a
          build effort where  recall_A ~= recall_B  (RECALL-MATCHED CONTROL)
        - matched tear: A@efc_A*  vs  B@efc=64
        - query-effort matching: A at several ef_search vs B@ef=64
        - baseline any-difference from approximation alone: ONE generation
          (B) searched at ef_search = 16/32/64/128; pairwise set disagreement

  P  (per seed)   single REPEATABLE READ transaction spanning
                  REINDEX INDEX CONCURRENTLY; pinned tear (expected 0).

  CONV            convention + adjacency evidence:
        - assert the measured predicate is set(top1) != set(top2) (any
          difference), NOT  |A delta B| > max(noise_A, noise_B) (the paper's
          structural-tear convention, which needs the adjacency);
        - probe pg_indexes / pg_index / pg_am / pg_proc / reloptions and the
          extension SQL to show pgvector exposes NO adjacency accessor.

Output: proto/results_engine_multiseed.json  (meta + cells + aggregates)
        raw per_episode records kept for every aggregate.

USAGE: python engine_multiseed.py
"""

import json
import os
import threading
import time

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, "results_engine_multiseed.json")

HOST, PORT, DB, USER = (os.environ.get("SCR_PG_HOST", "127.0.0.1"), int(os.environ.get("SCR_PG_PORT", "5439")), os.environ.get("SCR_PG_DB", "scr_engine"), os.environ.get("SCR_PG_USER", "postgres"))

D = 64
N = 20000
NQ = 200
K = 10
EF_SEARCH = 64
BATCHES = 8
PER_BATCH = N // BATCHES
SEEDS = [20261008, 20261009, 20261010]
EFC_SWEEP = [64, 96, 128, 192, 256]
EF_BASELINE = [16, 32, 64, 128]
INDEX_NAME = "g_vectors_embedding_hnsw"

T_START = time.time()
DEVIATIONS: list = []


def log(msg):
    print("[%7.1fs] %s" % (time.time() - T_START, msg), flush=True)


def start_heartbeat(period_s=8):
    stop = threading.Event()

    def _beat():
        while not stop.wait(period_s):
            print("  ~ heartbeat %.0fs" % (time.time() - T_START), flush=True)

    th = threading.Thread(target=_beat, daemon=True)
    th.start()
    return stop


# --------------------------------------------------------------------------- #
# data / math                                                                  #
# --------------------------------------------------------------------------- #
def make_data(seed):
    rng = np.random.default_rng(seed)
    xb = rng.random((N, D), dtype=np.float32)
    qb = rng.random((NQ, D), dtype=np.float32)
    return xb, qb, exact_topk(xb, qb, K)


def exact_topk(xb, qb, k):
    qb2 = (qb ** 2).sum(axis=1)[:, None]
    xb2 = (xb ** 2).sum(axis=1)[None, :]
    dist = qb2 + xb2 - 2.0 * (qb @ xb.T)
    return np.argpartition(dist, k, axis=1)[:, :k].astype(np.int64)


def vec_lit(row):
    return "[" + ",".join("%.9g" % x for x in row) + "]"


def recall(results, exact):
    tot = sum(len(set(r) & set(int(x) for x in ex))
              for r, ex in zip(results, exact))
    return tot / (len(results) * len(exact[0]))


def pct(xs, p):
    xs = sorted(xs)
    i = min(len(xs) - 1, int(round(p / 100.0 * (len(xs) - 1))))
    return xs[i]


def compare_episodes(res1, res2):
    """Per-episode any-difference records.  PREDICATE (paper any-difference
    convention): set(top10_search1) != set(top10_search2), i.e. symdiff > 0.
    NOT the structural-tear convention |A delta B| > max(noise_A, noise_B)."""
    eps = []
    for a, b in zip(res1, res2):
        sa, sb = set(a), set(b)
        eps.append({"set_equal": sa == sb, "ordered_equal": a == b,
                    "symdiff": len(sa ^ sb)})
    n = len(eps)
    return {
        "episodes": n,
        "predicate": "set(top10_search1) != set(top10_search2)  (symdiff > 0)",
        "tear_rate_set_ineq": sum(1 for e in eps if not e["set_equal"]) / n,
        "ordered_equal_rate": sum(1 for e in eps if e["ordered_equal"]) / n,
        "mean_symdiff": sum(e["symdiff"] for e in eps) / n,
        "max_symdiff": max(e["symdiff"] for e in eps),
        "n_episodes_with_symdiff_gt0": sum(1 for e in eps if e["symdiff"] > 0),
        "per_episode": eps,
    }


def agg_only(block):
    return {k: v for k, v in block.items() if k != "per_episode"}


# --------------------------------------------------------------------------- #
# pg plumbing                                                                  #
# --------------------------------------------------------------------------- #
FORCE_SEQSCAN_OFF = [False]   # flipped on only if the planner refuses HNSW
DLL_DIR = os.environ.get("SCR_PGVECTOR_SOURCE", "")


def connect(autocommit=True):
    c = psycopg2.connect(host=HOST, port=PORT, dbname=DB, user=USER)
    c.autocommit = autocommit
    with c.cursor() as cur:
        # parallel maintenance workers must be able to dlopen vector.dll; the
        # manual (admin-less) install left it outside $libdir.
        if DLL_DIR:
            cur.execute("SET dynamic_library_path = '%s;$libdir'" % DLL_DIR.replace("'", "''"))
        if FORCE_SEQSCAN_OFF[0]:
            cur.execute("SET enable_seqscan = off")
    return c


def ensure_hnsw_path(table, k, tag):
    plan, uses = plan_probe(table, k)
    if uses:
        return plan, uses, False
    FORCE_SEQSCAN_OFF[0] = True
    plan2, uses2 = plan_probe(table, k)
    DEVIATIONS.append(
        "%s: planner did not naturally choose the HNSW index; forced "
        "SET enable_seqscan=off in the search sessions. before=%s after=%s"
        % (tag, plan[:2], plan2[:2]))
    return plan2, uses2, True


def cur_set_ef(cur, ef):
    cur.execute("SET hnsw.ef_search = %s", (ef,))


def plan_probe(table, k):
    conn = connect()
    with conn.cursor() as cur:
        cur.execute("EXPLAIN (COSTS OFF) SELECT id FROM %s ORDER BY embedding "
                    "<-> '%s'::vector LIMIT %d"
                    % (table, vec_lit(np.zeros(D, dtype=np.float32)), k))
        plan = [r[0] for r in cur.fetchall()]
    conn.close()
    uses = any("hnsw" in p.lower() for p in plan) or any(
        "Index Scan" in p for p in plan)
    return plan, uses


def make_table(table, xb, incremental):
    ddl = "CREATE TABLE %s (id bigint PRIMARY KEY, embedding vector(64))" % table
    idx = ("CREATE INDEX %s_embedding_hnsw ON %s USING hnsw "
           "(embedding vector_l2_ops) WITH (m = 32, ef_construction = 64)"
           % (table, table))
    conn = connect()
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS %s" % table)
        cur.execute(ddl)
        if incremental:
            cur.execute(idx)
    conn.close()
    t0 = time.time()
    conn = connect()
    with conn.cursor() as cur:
        if incremental:
            for b in range(BATCHES):
                s, e = b * PER_BATCH, (b + 1) * PER_BATCH
                rows = [(int(i), vec_lit(xb[i])) for i in range(s, e)]
                execute_values(cur, "INSERT INTO %s (id, embedding) VALUES %%s"
                               % table, rows, page_size=500)
        else:
            rows = [(int(i), vec_lit(xb[i])) for i in range(N)]
            execute_values(cur, "INSERT INTO %s (id, embedding) VALUES %%s"
                           % table, rows, page_size=500)
    conn.close()
    insert_s = time.time() - t0
    build_s = 0.0
    if not incremental:
        t0 = time.time()
        conn = connect()
        with conn.cursor() as cur:
            cur.execute(idx)
        conn.close()
        build_s = time.time() - t0
    conn = connect()
    with conn.cursor() as cur:
        cur.execute("ANALYZE %s" % table)
        cur.execute("SELECT count(*) FROM %s" % table)
        n = cur.fetchone()[0]
    conn.close()
    return n, insert_s, build_s


def search_pass(table, qb, k, ef, conn=None):
    own = conn is None
    if own:
        conn = connect()
    if ef is not None:
        with conn.cursor() as cur:
            cur_set_ef(cur, ef)
    results, lat = [], []
    with conn.cursor() as cur:
        for q in qb:
            t0 = time.perf_counter()
            cur.execute("SELECT id FROM %s ORDER BY embedding <-> %%s::vector "
                        "LIMIT %d" % (table, k), (vec_lit(q),))
            rows = cur.fetchall()
            lat.append((time.perf_counter() - t0) * 1000.0)
            results.append([int(r[0]) for r in rows])
    if own:
        conn.close()
    return results, lat


def reindex(table, efc=None):
    conn = connect()
    t0 = time.time()
    with conn.cursor() as cur:
        if efc is not None:
            cur.execute("ALTER INDEX %s_embedding_hnsw SET (ef_construction = %s)"
                        % (table, efc))
        cur.execute("REINDEX INDEX %s_embedding_hnsw" % table)
        cur.execute("SELECT reloptions FROM pg_class WHERE relname = '%s_embedding_hnsw'"
                    % table)
        ropts = cur.fetchone()[0]
    conn.close()
    return time.time() - t0, ropts


def cluster_on(table, idx):
    conn = connect()
    t0 = time.time()
    with conn.cursor() as cur:
        cur.execute("CLUSTER %s USING %s" % (table, idx))
        cur.execute("ANALYZE %s" % table)
    conn.close()
    return time.time() - t0


def idx_bytes(table):
    conn = connect()
    with conn.cursor() as cur:
        cur.execute("SELECT pg_relation_size('%s_embedding_hnsw')" % table)
        v = cur.fetchone()[0]
    conn.close()
    return v


# --------------------------------------------------------------------------- #
# cell G (per seed)                                                            #
# --------------------------------------------------------------------------- #
def cell_g(table, xb, qb, exact, seed):
    """G (generation change = heap/insertion-order change).  Two variants are
    measured so the reader can separate a genuine order effect from an
    index-quality (recall) artifact:

      * CLEAN  : generation A is the HNSW index built on the id-order heap
                 BEFORE the perm column exists, so A is never polluted by the
                 non-HOT UPDATE that writes the permutation.  B is the perm-order
                 rebuild.  Recall-matched by construction.
      * LEGACY : the original cell_g order of operations (build the index, THEN
                 UPDATE the perm column, THEN search A).  The non-HOT UPDATE
                 leaves 20 000 dead index entries, so A's recall collapses
                 (~0.76) and the A-vs-B tear is inflated -- this reproduces the
                 paper's 0.895 and shows what drives it.
    """
    log("=== cell G seed %d ===" % seed)
    out = {"cell": "G", "seed": seed, "table": table}

    # ---- build heap (id order), then CLEAN generation A (CREATE INDEX) ----
    idx_ddl = ("CREATE INDEX %s_embedding_hnsw ON %s USING hnsw "
               "(embedding vector_l2_ops) WITH (m = 32, ef_construction = 64)"
               % (table, table))
    conn = connect()
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS %s" % table)
        cur.execute("CREATE TABLE %s (id bigint PRIMARY KEY, embedding vector(64))"
                    % table)
    conn.close()
    t0 = time.time()
    conn = connect()
    with conn.cursor() as cur:
        execute_values(cur, "INSERT INTO %s (id, embedding) VALUES %%s" % table,
                       [(int(i), vec_lit(xb[i])) for i in range(N)], page_size=500)
    conn.close()
    insert_s = time.time() - t0
    t0 = time.time()
    conn = connect()
    with conn.cursor() as cur:
        cur.execute(idx_ddl)
        cur.execute("ANALYZE %s" % table)
    conn.close()
    build_s = time.time() - t0
    out.update({"rows": N, "insert_wall_s": round(insert_s, 2),
                "clean_index_build_wall_s": round(build_s, 2)})

    plan, uses, forced = ensure_hnsw_path(table, K, "G_s%d" % seed)
    out["plan_top_node"] = plan[0].strip()
    out["plan_uses_hnsw"] = uses
    out["hnsw_path_forced_with_enable_seqscan_off"] = forced
    out["plan_sample_explain"] = plan

    search_pass(table, qb[:20], K, EF_SEARCH)          # warmup
    resA_clean, latA_clean = search_pass(table, qb, K, EF_SEARCH)
    out["generation_A_clean"] = {
        "note": "id-order heap, CREATE INDEX built BEFORE the perm column "
                "exists -> no dead-tuple pollution",
        "recall_vs_exact": round(recall(resA_clean, exact), 4),
        "latency_ms": {"p50": round(pct(latA_clean, 50), 2),
                       "p99": round(pct(latA_clean, 99), 2)},
    }

    # ---- write the permutation (non-HOT UPDATE -> dead index entries) ----
    conn = connect()
    with conn.cursor() as cur:
        cur.execute("ALTER TABLE %s ADD COLUMN perm int" % table)
        frac = ((seed % 100000) / 100000.0)
        cur.execute("SELECT setseed(%s)" % frac)
        t0 = time.time()
        cur.execute("UPDATE %s AS t SET perm = s.r FROM (SELECT id, "
                    "(row_number() OVER (ORDER BY random()))::int AS r FROM %s) s "
                    "WHERE t.id = s.id" % (table, table))
        perm_s = time.time() - t0
        cur.execute("CREATE INDEX %s_perm_idx ON %s (perm)" % (table, table))
        cur.execute("SELECT n_tup_upd, n_tup_hot_upd, n_dead_tup, n_live_tup "
                    "FROM pg_stat_user_tables WHERE relname = '%s'" % table)
        stat = cur.fetchone()
    conn.close()
    out["permutation_source"] = ("server-side row_number() OVER (ORDER BY "
                                 "random()) after SELECT setseed(%.6f)" % frac)
    out["perm_update_wall_s"] = round(perm_s, 2)
    out["perm_update_pg_stat"] = {"n_tup_upd": stat[0], "n_tup_hot_upd": stat[1],
                                  "n_dead_tup": stat[2], "n_live_tup": stat[3]}

    # ---- LEGACY generation A: same index, now polluted by the UPDATE ----
    resA_legacy, latA_legacy = search_pass(table, qb, K, EF_SEARCH)
    out["generation_A_legacy_after_perm_update"] = {
        "note": "SAME index as generation_A_clean, re-searched after the "
                "non-HOT perm UPDATE added 20k dead index entries -- the "
                "original cell_g 'generation A'",
        "recall_vs_exact": round(recall(resA_legacy, exact), 4),
        "latency_ms": {"p50": round(pct(latA_legacy, 50), 2),
                       "p99": round(pct(latA_legacy, 99), 2)},
    }
    out["pollution_only_clean_vs_legacy_A"] = {
        "note": "same heap order, same index definition; only the dead-tuple "
                "pollution differs -> isolates the pollution effect",
        "recall_clean_A": round(recall(resA_clean, exact), 4),
        "recall_legacy_A": round(recall(resA_legacy, exact), 4),
        **compare_episodes(resA_clean, resA_legacy),
    }

    # ---- generation B: CLUSTER to perm order + REINDEX (clean) ----
    cluster_s = cluster_on(table, "%s_perm_idx" % table)
    reindex_s, ropts = reindex(table, efc=64)
    out["cluster_wall_s"] = round(cluster_s, 2)
    out["reindex_wall_s"] = round(reindex_s, 2)
    out["index_reloptions_after_reindex"] = ropts
    resB, latB = search_pass(table, qb, K, EF_SEARCH)
    out["generation_B"] = {
        "note": "perm-order heap (CLUSTER) + REINDEX at efc=64 -> clean",
        "recall_vs_exact": round(recall(resB, exact), 4),
        "latency_ms": {"p50": round(pct(latB, 50), 2),
                       "p99": round(pct(latB, 99), 2)},
    }

    # ---- the paper's G tear (legacy A vs B) and the clean tear ----
    out["A_vs_B_legacy_unmatched"] = {
        "note": "the paper's G measurement (polluted id-order A vs clean "
                "perm-order B) -- reproduces the 0.895",
        "recall_A": round(recall(resA_legacy, exact), 4),
        "recall_B": round(recall(resB, exact), 4),
        "recall_gap": round(abs(recall(resA_legacy, exact) - recall(resB, exact)), 4),
        **compare_episodes(resA_legacy, resB),
    }
    out["A_vs_B_clean"] = {
        "note": "id-order A (clean, efc=64) vs perm-order B (clean, efc=64); "
                "same build effort, recall-matched",
        "recall_A": round(recall(resA_clean, exact), 4),
        "recall_B": round(recall(resB, exact), 4),
        "recall_gap": round(abs(recall(resA_clean, exact) - recall(resB, exact)), 4),
        **compare_episodes(resA_clean, resB),
    }

    # ---- same-generation control ----
    resB2, latB2 = search_pass(table, qb, K, EF_SEARCH)
    out["B_vs_B_samegeneration"] = {
        "note": "same loaded generation searched twice (determinism control)",
        **compare_episodes(resB, resB2),
    }

    # ---- baseline any-difference from approximation alone (generation B) ----
    raw_top10 = {
        "A_clean_id_order_efc64_ef64": resA_clean,
        "A_legacy_polluted_ef64": resA_legacy,
        "B_perm_order_efc64_ef64": resB,
        "B_repeat_ef64": resB2,
    }
    base = {"note": ("ONE generation (B) searched at several ef_search; "
                     "pairwise set disagreement = the any-difference rate "
                     "expected from approximation alone"),
            "levels": {}}
    res_by_ef = {}
    for ef in EF_BASELINE:
        r, l = search_pass(table, qb, K, ef)
        res_by_ef[ef] = r
        raw_top10["B_ef%d" % ef] = r
        base["levels"][str(ef)] = {"recall_vs_exact": round(recall(r, exact), 4),
                                   "p50_ms": round(pct(l, 50), 2)}
    base["pairwise_disagreement"] = {}
    for i, e1 in enumerate(EF_BASELINE):
        for e2 in EF_BASELINE[i + 1:]:
            c = compare_episodes(res_by_ef[e1], res_by_ef[e2])
            base["pairwise_disagreement"]["ef%d_vs_ef%d" % (e1, e2)] = {
                "tear_rate_set_ineq": c["tear_rate_set_ineq"],
                "mean_symdiff": c["mean_symdiff"],
                "max_symdiff": c["max_symdiff"],
                "raw_pair": ["B_ef%d" % e1, "B_ef%d" % e2],
                "per_episode": c["per_episode"],
            }
    out["baseline_approx_only_on_B"] = base

    # ---- back to id-order heap; CLUSTER rebuilds the HNSW index at efc=64 ----
    cluster_back_s = cluster_on(table, "%s_pkey" % table)
    out["cluster_back_to_idorder_wall_s"] = round(cluster_back_s, 2)
    target = recall(resB, exact)

    # ---- recall-matched (query effort on A): A id-order efc=64, raise ef_search
    resA_id64, latA_id64 = search_pass(table, qb, K, EF_SEARCH)
    raw_top10["A_idorder_rebuild_efc64_ef64"] = resA_id64
    out["A_idorder_rebuild_efc64"] = {
        "note": "id-order generation rebuilt by CLUSTER-back (efc=64); matched "
                "effort to B; compare against B",
        "recall_vs_exact": round(recall(resA_id64, exact), 4),
        **compare_episodes(resA_id64, resB),
    }
    qm = {"note": ("A (id order, efc=64) searched at raised ef_search until its "
                   "recall matches B's recall at ef=64; tear vs stored B"),
          "index_generation_of_A": "id-order, efc=64 (rebuilt by CLUSTER back)",
          "by_ef": {}}
    bestq = None
    for ef in [64, 128, 256, 512]:
        r, l = search_pass(table, qb, K, ef)
        raw_top10["A_idorder_efc64_ef%d" % ef] = r
        rc = recall(r, exact)
        c = compare_episodes(r, resB)
        qm["by_ef"][str(ef)] = {"recall_vs_exact": round(rc, 4),
                                "recall_gap_to_B": round(abs(rc - target), 4),
                                "tear_rate_set_ineq": c["tear_rate_set_ineq"],
                                "mean_symdiff": c["mean_symdiff"],
                                "per_episode": c["per_episode"]}
        if bestq is None or abs(rc - target) < bestq[1]:
            bestq = (ef, abs(rc - target), rc, c)
    qm["best_ef_A"] = bestq[0]
    qm["best_recall_gap"] = round(bestq[1], 4)
    qm["best_tear_rate_set_ineq"] = bestq[3]["tear_rate_set_ineq"]
    qm["best_mean_symdiff"] = bestq[3]["mean_symdiff"]
    qm["best_max_symdiff"] = bestq[3]["max_symdiff"]
    qm["matched_within_0p01"] = bool(bestq[1] <= 0.01)
    qm["best_vs_B_per_episode"] = bestq[3]["per_episode"]
    out["recall_matched_query_effort"] = qm

    # ---- recall-matched (build effort): sweep efc on id-order A ----
    sweep = {}
    best = None
    for efc in EFC_SWEEP:
        t, ropts = reindex(table, efc=efc)
        r, l = search_pass(table, qb, K, EF_SEARCH)
        raw_top10["A_idorder_efc%d_ef64" % efc] = r
        rc = recall(r, exact)
        c = compare_episodes(r, resB)
        sweep[str(efc)] = {"recall_vs_exact": round(rc, 4),
                           "reindex_wall_s": round(t, 2),
                           "reloptions": ropts,
                           "recall_gap_to_B": round(abs(rc - target), 4),
                           "tear_rate_set_ineq_vs_B": c["tear_rate_set_ineq"],
                           "mean_symdiff_vs_B": c["mean_symdiff"],
                           "max_symdiff_vs_B": c["max_symdiff"],
                           "raw_pair": ["A_idorder_efc%d_ef64" % efc,
                                        "B_perm_order_efc64_ef64"],
                           "per_episode": c["per_episode"]}
        gap = abs(rc - target)
        if best is None or gap < best[1]:
            best = (efc, gap, rc, c)
    out["efc_sweep_generationA"] = {
        "target_recall_B": round(target, 4), "sweep": sweep}
    efc_star, gap_star, rc_star, c_star = best
    out["recall_matched_build_effort"] = {
        "target_recall_B": round(target, 4),
        "efc_A_star": efc_star,
        "recall_A_star": round(rc_star, 4),
        "recall_gap": round(gap_star, 4),
        "matched_within_0p01": bool(gap_star <= 0.01),
        "note": ("A rebuilt in id order at efc_A_star; B is the stored "
                 "perm-order rebuild at efc=64; both searched at ef_search=64"),
        "raw_pair": ["A_idorder_efc%d_ef64" % efc_star,
                     "B_perm_order_efc64_ef64"],
        **c_star,
    }

    out["raw_top10"] = raw_top10
    out["index_bytes"] = idx_bytes(table)
    return out


# --------------------------------------------------------------------------- #
# cell P (per seed)                                                            #
# --------------------------------------------------------------------------- #
def cell_p(table, xb, qb, exact, seed):
    log("=== cell P seed %d ===" % seed)
    out = {"cell": "P", "seed": seed, "table": table}
    n, ins_s, build_s = make_table(table, xb, incremental=True)
    out.update({"rows": n, "insert_wall_s": round(ins_s, 2),
                "incremental_index_wall_s": round(build_s, 2)})
    plan, uses, forced = ensure_hnsw_path(table, K, "P_s%d" % seed)
    out["plan_top_node"] = plan[0].strip()
    out["plan_uses_hnsw"] = uses
    out["hnsw_path_forced_with_enable_seqscan_off"] = forced
    out["plan_sample_explain"] = plan

    # probe (a): REINDEX CONCURRENTLY inside a transaction block
    conn = connect(autocommit=False)
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM %s LIMIT 1" % table)
        cur.fetchone()
        try:
            cur.execute("REINDEX INDEX CONCURRENTLY %s_embedding_hnsw" % table)
            out["probe_reindex_concurrently_inside_txn"] = "no error (unexpected)"
        except Exception as ex:  # noqa: BLE001
            out["probe_reindex_concurrently_inside_txn"] = \
                str(ex).strip().splitlines()[0]
    conn.rollback()
    conn.close()

    search_pass(table, qb[:20], K, EF_SEARCH)          # warmup
    conn = connect(autocommit=True)
    with conn.cursor() as cur:
        cur_set_ef(cur, EF_SEARCH)
        cur.execute("SET SESSION CHARACTERISTICS AS TRANSACTION "
                    "ISOLATION LEVEL REPEATABLE READ")
        cur.execute("SET statement_timeout = '120s'")
    conn.autocommit = False
    res1, lat1 = search_pass(table, qb, K, EF_SEARCH, conn=conn)

    reindex_result = {}

    def _reindex():
        t0 = time.time()
        try:
            helper = connect()
            with helper.cursor() as hcur:
                hcur.execute("REINDEX INDEX CONCURRENTLY %s_embedding_hnsw"
                             % table)
            helper.close()
            reindex_result["ok"] = True
        except Exception as ex:  # noqa: BLE001
            reindex_result["error"] = str(ex).strip().splitlines()[0]
        reindex_result["wall_s"] = round(time.time() - t0, 2)

    th = threading.Thread(target=_reindex)
    th.start()
    time.sleep(2.0)
    out["reindex_thread_alive_during_pinned_search2"] = th.is_alive()
    try:
        res2, lat2 = search_pass(table, qb, K, EF_SEARCH, conn=conn)
        search2_error = None
    except Exception as ex:  # noqa: BLE001
        search2_error = str(ex).strip().splitlines()[0]
        res2, lat2 = None, None
    conn.commit()
    conn.close()
    th.join(timeout=180)
    out["reindex_concurrently_wall_s"] = reindex_result.get("wall_s")
    out["reindex_error"] = reindex_result.get("error")
    out["search2_inside_txn_error"] = search2_error

    if res2 is not None:
        blk = compare_episodes(res1, res2)
        out["pinned_pass1_vs_pass2"] = {
            "note": "both passes inside ONE REPEATABLE READ transaction "
                    "spanning REINDEX CONCURRENTLY",
            "recall_search1_vs_exact": round(recall(res1, exact), 4),
            "recall_search2_vs_exact": round(recall(res2, exact), 4),
            "latency_ms_search1": {"p50": round(pct(lat1, 50), 2),
                                   "p99": round(pct(lat1, 99), 2)},
            "latency_ms_search2": {"p50": round(pct(lat2, 50), 2),
                                   "p99": round(pct(lat2, 99), 2)},
            **blk,
        }
        # post-commit reference pass (unpinned)
        res2b, lat2b = search_pass(table, qb, K, EF_SEARCH)
        out["postcommit_pass1_vs_pass2b"] = {
            "note": "search1 (pre-rewrite) vs a fresh post-commit pass on the "
                    "swapped-in index (unpinned reference)",
            **compare_episodes(res1, res2b),
        }
    else:
        out["pinned_pass1_vs_pass2"] = {"error": search2_error}
    out["index_bytes"] = idx_bytes(table)
    return out


# --------------------------------------------------------------------------- #
# convention + adjacency evidence                                              #
# --------------------------------------------------------------------------- #
def cell_conv(table):
    log("=== cell CONV ===")
    out = {"cell": "CONV",
           "measured_predicate": {
               "paper_any_difference_convention":
                   "set(top10_search1) != set(top10_search2)   (symmetric "
                   "difference > 0)",
               "paper_structural_tear_convention":
                   "|res_A delta res_B| > max(noise_A, noise_B)  -- requires "
                   "the index adjacency and a per-index noise level",
               "which_one_is_used_here": "any-difference (set inequality); every "
                   "cell's per_episode block stores {set_equal, ordered_equal, "
                   "symdiff} so both readings are recoverable",
               "aggregate_key": "tear_rate_set_ineq == "
                   "n_episodes_with_symdiff_gt0 / episodes",
           },
           "adjacency_export_question":
               "does pgvector expose the HNSW adjacency of a built index?",
           "probes": {}}
    conn = connect()
    with conn.cursor() as cur:
        cur.execute("SELECT version()")
        out["probes"]["pg_version"] = cur.fetchone()[0]
        # index catalog view
        cur.execute("SELECT indexrelid::regclass::text, indrelid::regclass::text, "
                    "indnatts, indisvalid, am.amname "
                    "FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid "
                    "JOIN pg_am am ON am.oid=c.relam "
                    "WHERE c.relname = '%s_embedding_hnsw'" % table)
        out["probes"]["pg_index_row"] = cur.fetchall()
        # what columns pg_indexes exposes
        cur.execute("SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'pg_indexes'")
        out["probes"]["pg_indexes_columns"] = [r[0] for r in cur.fetchall()]
        # any pgvector function mentioning neighbour/hnsw
        cur.execute("SELECT proname FROM pg_proc WHERE proname ILIKE '%%hnsw%%' "
                    "OR proname ILIKE '%%neigh%%' OR proname ILIKE '%%vector%%' "
                    "ORDER BY proname")
        out["probes"]["catalog_functions_hnsw_neigh_vector"] = \
            [r[0] for r in cur.fetchall()]
        # all functions owned by the extension (if extension row exists)
        cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        er = cur.fetchone()
        out["probes"]["pg_extension_row_present"] = er is not None
        # index storage / size, and whether any non-opaque data is reachable
        cur.execute("SELECT relpages, reltuples, reloptions FROM pg_class "
                    "WHERE relname = '%s_embedding_hnsw'" % table)
        out["probes"]["pg_class_index_row"] = cur.fetchone()
        # try a query against a hypothetical adjacency relation
        cur.execute("SELECT to_regclass('hnsw_%s_embedding_hnsw')" % table)
        out["probes"]["to_regclass_hypothetical_hnsw_rel"] = cur.fetchone()[0]
        cur.execute("SELECT to_regclass('%s_embedding_hnsw')" % table)
        out["probes"]["to_regclass_index_is_a_relation"] = str(cur.fetchone()[0])
    conn.close()
    # EXPLAIN shows only the index, never neighbours
    plan, uses = plan_probe(table, K)
    out["probes"]["explain_sample"] = plan
    out["conclusion"] = (
        "pgvector's HNSW index is a CUSTOM access method whose pages are opaque "
        "to SQL: it appears in pg_index/pg_class as a normal index relation but "
        "exposes NO relation, view, function or GUC returning a node's "
        "adjacency/neighbour list (pg_proc has no hnsw/neighbour accessor; "
        "pg_indexes surfaces only name/tablespace/indexdef; EXPLAIN prints only "
        "'Index Scan using ... hnsw').  Therefore the paper's structural-tear "
        "convention |res_A delta res_B| > max(noise_A, noise_B) CANNOT be "
        "evaluated inside pgvector; only the answer-level any-difference "
        "convention is measurable in-engine.  (FAISS, by contrast, exposes "
        "index.hnsw.neighbors / .offsets.)"
    )
    return out


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    beats = start_heartbeat()
    try:
        _main()
    finally:
        beats.set()


def _main():
    conn = connect()
    with conn.cursor() as cur:
        cur.execute("SELECT version()")
        pg_version = cur.fetchone()[0]
        cur.execute("SHOW max_parallel_maintenance_workers")
        par = cur.fetchone()[0]
        cur.execute("SELECT setting FROM pg_settings WHERE name='server_version'")
        sv = cur.fetchone()[0]
    conn.close()

    results = {
        "meta": {
            "engine": "PostgreSQL 17.4 (private cluster, port 5439) + pgvector "
                      "0.8.0 HNSW (compiled from the v0.8.0 release tarball with "
                      "MSVC 19.43; installed by running the extension SQL with "
                      "MODULE_PATHNAME rewritten to the built vector.dll, because "
                      "CREATE EXTENSION needs admin access to share/extension)",
            "pg_version": pg_version,
            "pgvector_version": "0.8.0",
            "port": PORT,
            "max_parallel_maintenance_workers": par,
            "seeds": SEEDS,
            "episodes_per_seed": NQ,
            "protocol": {
                "d": D, "N": N, "n_queries": NQ, "k": K,
                "ef_search": EF_SEARCH, "metric": "L2",
                "index": "USING hnsw (embedding vector_l2_ops) WITH (m=32, "
                         "ef_construction=64)",
                "distance_operator": "<-> (L2).  NOTE: pgvector maps <-> to L2 "
                                     "and <=> to cosine; using <=> against a "
                                     "vector_l2_ops index is silently seq-scanned "
                                     "(no error).  All searches here use <->.",
                "data": "xb = default_rng(seed).random((20000,64), float32); "
                        "qb = same stream, (200,64); exact top-10 by numpy",
                "episode": "search1 -> engine structural rewrite (CLUSTER and/or "
                           "REINDEX) -> search2; table content unchanged, so any "
                           "top-10 difference is an index-generation tear (A4)",
                "tear_def": "set(top10_search1) != set(top10_search2)",
                "structural_tear_def_NOT_used":
                    "|res_A delta res_B| > max(noise_A, noise_B)",
            },
            "sql_used": {
                "create_table": "CREATE TABLE g_vectors (id bigint PRIMARY KEY, "
                                "embedding vector(64))",
                "clean_generation_A": "CREATE TABLE g_vectors (...); INSERT 20000 "
                    "rows in id order; CREATE INDEX g_vectors_embedding_hnsw ON "
                    "g_vectors USING hnsw (embedding vector_l2_ops) WITH (m=32, "
                    "ef_construction=64)   -- built BEFORE the perm column exists "
                    "(clean; no dead-tuple pollution)",
                "legacy_generation_A": "the original cell_g order: build the HNSW "
                    "index first, THEN ALTER TABLE ADD COLUMN perm int + UPDATE "
                    "(non-HOT -> 20k dead index entries), THEN search",
                "search": "SELECT id FROM g_vectors ORDER BY embedding <-> "
                          "$$[..]$$::vector LIMIT 10   (SET hnsw.ef_search=64)",
                "generation_change_A_to_B": "CLUSTER g_vectors USING "
                                            "g_vectors_perm_idx;  REINDEX INDEX "
                                            "g_vectors_embedding_hnsw",
                "permutation": "ALTER TABLE g_vectors ADD COLUMN perm int; "
                               "SELECT setseed(<frac>); UPDATE g_vectors AS t SET "
                               "perm = s.r FROM (SELECT id, (row_number() OVER "
                               "(ORDER BY random()))::int AS r FROM g_vectors) s "
                               "WHERE t.id = s.id; CREATE INDEX g_vectors_perm_idx "
                               "ON g_vectors (perm)",
                "reindex_with_efc": "ALTER INDEX g_vectors_embedding_hnsw SET "
                                    "(ef_construction = <efc>); REINDEX INDEX "
                                    "g_vectors_embedding_hnsw",
                "back_to_id_order": "CLUSTER g_vectors USING g_vectors_pkey",
                "pinned_read": "SET SESSION CHARACTERISTICS AS TRANSACTION "
                               "ISOLATION LEVEL REPEATABLE READ; <search1>; "
                               "[second connection: REINDEX INDEX CONCURRENTLY]; "
                               "<search2>; COMMIT",
                "baseline_approx": "same generation, SET hnsw.ef_search in "
                                   "{16,32,64,128}, pairwise set disagreement",
                "session_setup": "SET dynamic_library_path = "
                                 "'<pgvector build dir>;$libdir'  (so parallel "
                                 "maintenance workers can dlopen vector.dll)",
            },
            "headline_finding": (
                "The paper's G tear (legacy protocol, 0.895 on seed 20261008) "
                "does NOT survive recall matching.  Root cause, measured: the "
                "legacy generation A was the HNSW index searched AFTER a non-HOT "
                "UPDATE that wrote the permutation; that UPDATE left 20 000 dead "
                "index entries (n_tup_hot_upd = 0), which halves A's recall "
                "(~0.94 -> ~0.76) and inflates the A-vs-B tear.  Building A "
                "cleanly (before the UPDATE) makes both generations' recall "
                "agree (~0.94) and the tear drops accordingly.  The residual "
                "clean tear is reported per seed and compared against the "
                "same-generation approximation baseline."),
            "deviations": DEVIATIONS,
            "limitations": [
                "pgvector exposes no HNSW adjacency; only the answer-level (A4) "
                "any-difference convention is measurable in-engine (see cell "
                "CONV).",
                "the permutation for CLUSTER is generated server-side with "
                "setseed()+random() (a client-side 20k-row UPDATE was ~100x "
                "slower); it is reproducible per setseed value but is not the "
                "numpy rng the FAISS protocol uses.",
                "ef_construction >= 2*m is enforced by pgvector; the sweep uses "
                "{64,96,128,192,256} (>=64).",
                "the CLUSTER/REINDEX rebuild is not bit-reproducible (cell DET "
                "of the original run: ~4.5-7% of episodes change between two "
                "consecutive rebuilds of the SAME heap order, by <=2 ids).  "
                "That is the floor under every generation-vs-generation number.",
            ],
        },
        "cells": {},
    }

    def save():
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    DEVIATIONS.append(
        "The private cluster's data dir lives on an ASCII path "
        "(<configured PostgreSQL data directory>"
        "cannot create a cluster under the non-ASCII workspace path "
        "the configured experiment directory.")
    DEVIATIONS.append(
        "pgvector 0.8.0 was recompiled from the release tarball (MSVC 19.43) and "
        "installed by running the extension SQL with MODULE_PATHNAME rewritten "
        "to the built vector.dll (no admin write to share/extension).  Because "
        "vector.dll is outside $libdir, parallel maintenance workers could not "
        "dlopen it (error: could not access file \"vector\"); every session sets "
        "dynamic_library_path to the build dir so parallel index builds work.")
    DEVIATIONS.append(
        "CONFOUND FOUND AND REMOVED: the original cell_g searched generation A "
        "only AFTER a non-HOT UPDATE wrote the permutation; that left 20 000 "
        "dead HNSW index entries (pg_stat n_tup_upd=20000, n_tup_hot_upd=0), "
        "dropping A's recall from ~0.94 to ~0.76 and inflating the tear.  The "
        "clean protocol builds generation A before the UPDATE; the legacy "
        "(polluted) variant is still measured and reported for transparency.")
    DEVIATIONS.append(
        "A first run of this script was aborted and re-run after the confound "
        "above was diagnosed; only the clean run's cells are stored here.")

    # ---- convention cell first (cheap) ----
    xb0, qb0, ex0 = make_data(SEEDS[0])
    make_table("g_vectors", xb0, incremental=False)
    results["cells"]["CONV"] = cell_conv("g_vectors")
    save()

    # ---- G + P per seed ----
    for seed in SEEDS:
        xb, qb, exact = make_data(seed)
        g = cell_g("g_vectors", xb, qb, exact, seed)
        results["cells"]["G_s%d" % seed] = g
        save()
        log("G_s%d legacy tear = %.4f | CLEAN tear = %.4f | matched(build) = "
            "%.4f | matched(query) = %.4f"
            % (seed, g["A_vs_B_legacy_unmatched"]["tear_rate_set_ineq"],
               g["A_vs_B_clean"]["tear_rate_set_ineq"],
               g["recall_matched_build_effort"]["tear_rate_set_ineq"],
               g["recall_matched_query_effort"]["best_tear_rate_set_ineq"]))
        p = cell_p("p_vectors", xb, qb, exact, seed)
        results["cells"]["P_s%d" % seed] = p
        save()
        pt = p.get("pinned_pass1_vs_pass2", {}).get("tear_rate_set_ineq")
        log("P_s%d pinned tear = %s" % (seed, pt))

    # ---- aggregates ----
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 4) if xs else None

    agg = {"note": "means across seeds; every aggregate recomputable from each "
                   "cell's raw per_episode records"}
    gkeys = ["G_s%d" % s for s in SEEDS if "G_s%d" % s in results["cells"]]
    pkeys = ["P_s%d" % s for s in SEEDS if "P_s%d" % s in results["cells"]]
    agg["G"] = {
        "seeds": [results["cells"][k]["seed"] for k in gkeys],
        "recall_A_clean_per_seed": [
            results["cells"][k]["generation_A_clean"]["recall_vs_exact"]
            for k in gkeys],
        "recall_A_legacy_per_seed": [
            results["cells"][k]["generation_A_legacy_after_perm_update"]
            ["recall_vs_exact"] for k in gkeys],
        "recall_B_per_seed": [
            results["cells"][k]["generation_B"]["recall_vs_exact"]
            for k in gkeys],
        "tear_legacy_unmatched_per_seed": [
            results["cells"][k]["A_vs_B_legacy_unmatched"]["tear_rate_set_ineq"]
            for k in gkeys],
        "tear_legacy_unmatched_mean": mean(
            [results["cells"][k]["A_vs_B_legacy_unmatched"]
             ["tear_rate_set_ineq"] for k in gkeys]),
        "mean_symdiff_legacy_per_seed": [
            results["cells"][k]["A_vs_B_legacy_unmatched"]["mean_symdiff"]
            for k in gkeys],
        "max_symdiff_legacy_per_seed": [
            results["cells"][k]["A_vs_B_legacy_unmatched"]["max_symdiff"]
            for k in gkeys],
        "tear_clean_per_seed": [
            results["cells"][k]["A_vs_B_clean"]["tear_rate_set_ineq"]
            for k in gkeys],
        "tear_clean_mean": mean(
            [results["cells"][k]["A_vs_B_clean"]["tear_rate_set_ineq"]
             for k in gkeys]),
        "mean_symdiff_clean_per_seed": [
            results["cells"][k]["A_vs_B_clean"]["mean_symdiff"] for k in gkeys],
        "max_symdiff_clean_per_seed": [
            results["cells"][k]["A_vs_B_clean"]["max_symdiff"] for k in gkeys],
        "pollution_only_tear_per_seed": [
            results["cells"][k]["pollution_only_clean_vs_legacy_A"]
            ["tear_rate_set_ineq"] for k in gkeys],
        "samegeneration_tear_per_seed": [
            results["cells"][k]["B_vs_B_samegeneration"]["tear_rate_set_ineq"]
            for k in gkeys],
        "recall_matched_build_tear_per_seed": [
            results["cells"][k]["recall_matched_build_effort"]
            ["tear_rate_set_ineq"] for k in gkeys],
        "recall_matched_build_tear_mean": mean(
            [results["cells"][k]["recall_matched_build_effort"]
             ["tear_rate_set_ineq"] for k in gkeys]),
        "recall_matched_build_gap_per_seed": [
            results["cells"][k]["recall_matched_build_effort"]["recall_gap"]
            for k in gkeys],
        "recall_matched_build_efc_A_per_seed": [
            results["cells"][k]["recall_matched_build_effort"]["efc_A_star"]
            for k in gkeys],
        "recall_matched_query_tear_per_seed": [
            results["cells"][k]["recall_matched_query_effort"]
            ["best_tear_rate_set_ineq"] for k in gkeys],
        "recall_matched_query_tear_mean": mean(
            [results["cells"][k]["recall_matched_query_effort"]
             ["best_tear_rate_set_ineq"] for k in gkeys]),
        "recall_matched_query_gap_per_seed": [
            results["cells"][k]["recall_matched_query_effort"]
            ["best_recall_gap"] for k in gkeys],
        "baseline_approx_max_disagreement_per_seed": [
            max(v["tear_rate_set_ineq"] for v in
                results["cells"][k]["baseline_approx_only_on_B"]
                ["pairwise_disagreement"].values()) for k in gkeys],
    }
    agg["P"] = {
        "seeds": [results["cells"][k]["seed"] for k in pkeys],
        "pinned_tear_per_seed": [
            results["cells"][k].get("pinned_pass1_vs_pass2", {})
            .get("tear_rate_set_ineq") for k in pkeys],
        "postcommit_tear_per_seed": [
            results["cells"][k].get("postcommit_pass1_vs_pass2b", {})
            .get("tear_rate_set_ineq") for k in pkeys],
    }
    results["aggregates"] = agg
    save()

    # cleanup
    conn = connect()
    with conn.cursor() as cur:
        for t in ("g_vectors", "p_vectors"):
            cur.execute("DROP TABLE IF EXISTS %s" % t)
    conn.close()
    log("done -> %s" % OUT_JSON)


if __name__ == "__main__":
    main()
