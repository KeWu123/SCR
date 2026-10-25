#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
engine_deletion.py --- Run the paper's §6.7 two-retrieval protocol INSIDE a shipped
engine's OWN structural-rewrite (REINDEX) and deletion paths: PostgreSQL 17.4 +
pgvector 0.8.0 (HNSW). Deployment constraints are recorded in the result
metadata; the experiment does not evaluate Milvus.

PROTOCOL (answer level, A4)
---------------------------
Episode = (search1(q), search2(q)) for the same query q, with the ENGINE's own
structural rewrite (REINDEX) between the two passes.  The table content is
unchanged by REINDEX, so any top-10 difference between the two retrievals is a
cross-generation index tear.

Cells
-----
  U1 (unpinned, bulk->bulk):   index built by bulk CREATE INDEX; search1 ->
      plain REINDEX -> search2 -> REINDEX -> search3.  Pass 2v3 tests whether
      pgvector's bulk rebuild (SeedRandom(42), hnswbuild.c:1100) is a pure
      function of heap order; pass 1v2 is the literal §6.7 unpinned measurement.
  U2 (unpinned, incremental->bulk):  index built INCREMENTALLY (CREATE INDEX on
      empty table, then 8 batches x 2500 rows, mirroring the Milvus 8-segment
      protocol) -> search1 -> plain REINDEX (bulk rebuild) -> search2.  This is
      the true cross-generation tear: insert-path graph vs bulk-build graph.
  G  (heap/insertion-order generation change):  the pgvector analogue of the
      paper's FAISS protocol "same vectors, different insertion order".  Bulk
      HNSW index over the id-order heap -> search1 -> CLUSTER on a random
      permutation btree (reorders the heap and rebuilds the HNSW graph) ->
      REINDEX -> search2 (-> search3, same generation, as a determinism check).
  P  (pinned, snapshot spanning the rewrite):  BEGIN ISOLATION LEVEL REPEATABLE
      READ; search1 (inside txn); REINDEX CONCURRENTLY from a second connection;
      search2 (SAME txn); commit.  Tests whether an open MVCC snapshot pins the
      index generation.  Plus two lock probes:
        (a) REINDEX CONCURRENTLY inside the txn  -> expected error (recorded);
        (b) plain REINDEX against an open reader -> lock_timeout (quiesce evidence).
  D  (deletion, policy effects only, NOT a tear criterion): DELETE a random 10%
      of ids -> search (ghost ids must be 0; recall vs the EXACT top-10 over the
      surviving set) -> REINDEX -> search again.

Anti-ceiling probes (ef_search=16, k=100) are appended automatically if the U2
tear is 0.

SCOPE: pgvector does not expose the HNSW adjacency of a built index,
so the structural tear (A5) is NOT measurable inside the engine; this experiment
reaches the answer level (A4) only.

Data config matches the paper's FAISS experiment for comparability:
  d=64, L2, N=20000;  xb = default_rng(seed).random((20000,64), float32);
  qb = same rng stream, .random((200,64), float32);  exact top-10 via numpy.

USAGE
-----
    python engine_deletion.py [--cells U1,U2,P,D] [--seeds 20261008] [--append]
"""

import argparse
import json
import os
import sys
import threading
import time

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, "results_engine_deletion.json")

D = 64
N = 20000
NQ = 200
BATCHES = 8
PER_BATCH = N // BATCHES
SEED_MAIN = 20261008
K = 10
EF_SEARCH = 64
HNSW_WITH = "WITH (m = 32, ef_construction = 64)"

API_QUIRKS = [
    "pgvector requires ef_construction >= 2*m; the comparison's FAISS-style "
    "(m=32, ef_construction=40) is rejected with InvalidParameterValue. "
    "Used ef_construction=64 (the minimum for m=32); recorded as a deviation "
    "from the paper's FAISS index config.",
]

DEVIATIONS = []


def log(msg):
    print("[%7.1fs] %s" % (time.time() - T_START, msg), flush=True)


def start_heartbeat(period_s=8):
    """Keep stdout alive: the execution wrapper treats long silence as a hang and
    kills the process, so emit a heartbeat while long DB work is in flight."""
    stop = threading.Event()

    def _beat():
        while not stop.wait(period_s):
            print("  ~ heartbeat %.0fs" % (time.time() - T_START), flush=True)

    th = threading.Thread(target=_beat, daemon=True)
    th.start()
    return stop


T_START = time.time()


# --------------------------------------------------------------------------- #
# data                                                                         #
# --------------------------------------------------------------------------- #
def make_data(seed):
    rng = np.random.default_rng(seed)
    xb = rng.random((N, D), dtype=np.float32)
    qb = rng.random((NQ, D), dtype=np.float32)
    qb2 = (qb ** 2).sum(axis=1)[:, None]
    xb2 = (xb ** 2).sum(axis=1)[None, :]
    dist = qb2 + xb2 - 2.0 * (qb @ xb.T)
    exact = np.argpartition(dist, K, axis=1)[:, :K]
    return xb, qb, exact


def exact_topk(xb, qb, k, dead=None):
    keep = np.ones(xb.shape[0], dtype=bool) if dead is None else ~dead
    sub = xb[keep]
    qb2 = (qb ** 2).sum(axis=1)[:, None]
    xb2 = (sub ** 2).sum(axis=1)[None, :]
    dist = qb2 + xb2 - 2.0 * (qb @ sub.T)
    # positions inside `sub` -> original row ids
    return np.flatnonzero(keep)[np.argpartition(dist, k, axis=1)[:, :k]]


def vec_lit(row):
    return "[" + ",".join("%.9g" % x for x in row) + "]"


# --------------------------------------------------------------------------- #
# pg plumbing                                                                  #
# --------------------------------------------------------------------------- #
FORCE_SEQSCAN_OFF = [False]   # flipped on only if the planner refuses the HNSW path


def connect(autocommit=True):
    conn = psycopg2.connect(host=os.environ.get("SCR_PG_HOST", "127.0.0.1"),
                            port=int(os.environ.get("SCR_PG_PORT", "5439")),
                            dbname=os.environ.get("SCR_PG_DB", "scr_engine"),
                            user=os.environ.get("SCR_PG_USER", "postgres"))
    conn.autocommit = autocommit
    with conn.cursor() as cur:
        cur.execute("SET hnsw.ef_search = %s", (EF_SEARCH,))
        if FORCE_SEQSCAN_OFF[0]:
            cur.execute("SET enable_seqscan = off")
    return conn


def explain_query(table, k, query_vector):
    """Full EXPLAIN texts for the literal and the parameterized query forms."""
    lit = vec_lit(query_vector)
    out = {"literal": None, "parameterized": None, "analyze_literal": None}
    conn = connect()
    with conn.cursor() as cur:
        cur.execute("SET hnsw.ef_search = %s", (EF_SEARCH,))
        if FORCE_SEQSCAN_OFF[0]:
            cur.execute("SET enable_seqscan = off")
        cur.execute("EXPLAIN (COSTS OFF) SELECT id FROM %s ORDER BY embedding "
                    "<-> '%s'::vector LIMIT %d" % (table, lit, k))
        out["literal"] = [r[0] for r in cur.fetchall()]
        cur.execute("PREPARE _sp(vector) AS SELECT id FROM %s ORDER BY "
                    "embedding <-> $1 LIMIT %d" % (table, k))
        cur.execute("EXPLAIN (COSTS OFF) EXECUTE _sp('%s'::vector)" % lit)
        out["parameterized"] = [r[0] for r in cur.fetchall()]
        cur.execute("EXPLAIN (ANALYZE, BUFFERS, COSTS OFF) SELECT id FROM %s "
                    "ORDER BY embedding <-> '%s'::vector LIMIT %d"
                    % (table, lit, k))
        out["analyze_literal"] = [r[0] for r in cur.fetchall()]
    conn.close()
    return out


def plan_uses_hnsw(table, k):
    conn = connect()
    with conn.cursor() as cur:
        if FORCE_SEQSCAN_OFF[0]:
            cur.execute("SET enable_seqscan = off")
        cur.execute("EXPLAIN (COSTS OFF) SELECT id FROM %s "
                    "ORDER BY embedding <-> '%s'::vector LIMIT %d"
                    % (table, vec_lit(np.zeros(D, dtype=np.float32)), k))
        plan = [r[0] for r in cur.fetchall()]
    conn.close()
    return plan, any("hnsw" in p.lower() or "Index Scan" in p for p in plan)


def ensure_hnsw_path(table, k, cell_tag):
    """The engine must be exercised on its HNSW path.  Verify the natural plan;
    only if the planner refuses it, force enable_seqscan=off and record that."""
    plan, ok = plan_uses_hnsw(table, k)
    if not ok:
        FORCE_SEQSCAN_OFF[0] = True
        plan2, ok2 = plan_uses_hnsw(table, k)
        API_QUIRKS.append(
            "%s: planner did not choose the HNSW index naturally; forced "
            "SET enable_seqscan=off in the search sessions (recorded per the "
            "protocol). plan before: %s | after: %s"
            % (cell_tag, plan[:3], plan2[:3]))
        return plan2, ok2, True
    return plan, ok, False


def make_table(table, xb, incremental):
    drop = "DROP TABLE IF EXISTS %s" % table
    ddl = ("CREATE TABLE %s (id bigint PRIMARY KEY, embedding vector(64))"
           % table)
    idx = ("CREATE INDEX %s_embedding_hnsw ON %s USING hnsw "
           "(embedding vector_l2_ops) %s" % (table, table, HNSW_WITH))
    conn = connect()
    with conn.cursor() as cur:
        cur.execute(drop)
        cur.execute(ddl)
        if incremental:
            cur.execute(idx)          # empty index; rows inserted afterwards
    conn.close()
    t0 = time.time()
    conn = connect()
    with conn.cursor() as cur:
        if incremental:
            for b in range(BATCHES):  # 8 batches x 2500, mirroring Milvus flushes
                s, e = b * PER_BATCH, (b + 1) * PER_BATCH
                rows = [(int(i), vec_lit(xb[i])) for i in range(s, e)]
                execute_values(cur, "INSERT INTO %s (id, embedding) VALUES %%s"
                               % table, rows, page_size=500)
                log("  %s: batch %d/%d inserted (%.0f rows)"
                    % (table, b + 1, BATCHES, e))
        else:
            rows = [(int(i), vec_lit(xb[i])) for i in range(N)]
            execute_values(cur, "INSERT INTO %s (id, embedding) VALUES %%s"
                           % table, rows, page_size=500)
            log("  %s: %d rows inserted" % (table, N))
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
        n_rows = cur.fetchone()[0]
        cur.execute("SELECT pg_relation_size('%s_embedding_hnsw')" % table)
        idx_bytes = cur.fetchone()[0]
    conn.close()
    return n_rows, idx_bytes, insert_s, build_s


def search_pass(table, qb, k, ef=None, conn=None, in_txn=False):
    """200 timed point queries; returns (ids per query, latencies ms)."""
    own = conn is None
    if own:
        conn = connect()
    if ef is not None:
        with conn.cursor() as cur:
            cur.execute("SET hnsw.ef_search = %s", (ef,))
    results, lat = [], []
    with conn.cursor() as cur:
        for i, q in enumerate(qb):
            t0 = time.perf_counter()
            cur.execute("SELECT id FROM %s ORDER BY embedding <-> %%s::vector "
                        "LIMIT %d" % (table, k), (vec_lit(q),))
            rows = cur.fetchall()
            lat.append((time.perf_counter() - t0) * 1000.0)
            results.append([int(r[0]) for r in rows])
            if (i + 1) % 50 == 0:
                log("  %s: search pass %d/%d queries" % (table, i + 1, len(qb)))
    if own:
        conn.close()
    return results, lat


def reindex(table, concurrently=False):
    stmt = "REINDEX INDEX %s %s_embedding_hnsw" % (
        "CONCURRENTLY" if concurrently else "", table)
    t0 = time.time()
    conn = connect()
    with conn.cursor() as cur:
        cur.execute(stmt)
    conn.close()
    return time.time() - t0


def idx_size(table):
    conn = connect()
    with conn.cursor() as cur:
        cur.execute("SELECT pg_relation_size('%s_embedding_hnsw')" % table)
        v = cur.fetchone()[0]
    conn.close()
    return v


def pct(xs, p):
    xs = sorted(xs)
    i = min(len(xs) - 1, int(round(p / 100.0 * (len(xs) - 1))))
    return xs[i]


def compare_episodes(res1, res2):
    eps = []
    for a, b in zip(res1, res2):
        sa, sb = set(a), set(b)
        eps.append({"set_equal": sa == sb, "ordered_equal": a == b,
                    "symdiff": len(sa ^ sb)})
    n = len(eps)
    return {
        "episodes": n,
        "tear_rate_set_ineq": sum(1 for e in eps if not e["set_equal"]) / n,
        "ordered_equal_rate": sum(1 for e in eps if e["ordered_equal"]) / n,
        "mean_symdiff": sum(e["symdiff"] for e in eps) / n,
        "max_symdiff": max(e["symdiff"] for e in eps),
        "n_episodes_with_symdiff_gt0": sum(1 for e in eps if e["symdiff"] > 0),
        "per_episode": eps,
    }


def recall_vs_exact(results, exact):
    tot = sum(len(set(r) & set(int(x) for x in ex)) for r, ex in zip(results, exact))
    return tot / (len(results) * len(exact[0]))


def tear_block(res1, res2, exact, lat1, lat2, extra=None):
    blk = {
        "recall_search1_vs_exact": round(recall_vs_exact(res1, exact), 4),
        "recall_search2_vs_exact": round(recall_vs_exact(res2, exact), 4),
        "latency_ms_search1": {"p50": round(pct(lat1, 50), 2),
                               "p99": round(pct(lat1, 99), 2)},
        "latency_ms_search2": {"p50": round(pct(lat2, 50), 2),
                               "p99": round(pct(lat2, 99), 2)},
        **compare_episodes(res1, res2),
    }
    if extra:
        blk.update(extra)
    return blk


# --------------------------------------------------------------------------- #
# cells                                                                        #
# --------------------------------------------------------------------------- #
def cell_u1(table, xb, qb, exact):
    log("cell U1 (bulk build -> plain REINDEX x2)")
    n_rows, idx_b, insert_s, build_s = make_table(table, xb, incremental=False)
    plan, uses_hnsw, forced = ensure_hnsw_path(table, K, "U1")
    log("plan uses hnsw: %s (forced=%s) | %s" % (uses_hnsw, forced, plan[0].strip()))
    plan_sample = explain_query(table, K, qb[0])
    search_pass(table, qb[:20], K)                      # warmup
    res1, lat1 = search_pass(table, qb, K)
    t_reindex1 = reindex(table, concurrently=False)
    res2, lat2 = search_pass(table, qb, K)
    t_reindex2 = reindex(table, concurrently=False)
    res3, lat3 = search_pass(table, qb, K)
    out = {
        "cell": "U1",
        "table": table,
        "index_construction": "bulk CREATE INDEX (insert-then-index)",
        "reindex_variant": "plain REINDEX (no concurrent readers; autocommit)",
        "rows": n_rows, "index_bytes_before": idx_b,
        "index_bytes_after": idx_size(table),
        "insert_wall_s": round(insert_s, 2),
        "index_build_wall_s": round(build_s, 2),
        "reindex1_wall_s": round(t_reindex1, 2),
        "reindex2_wall_s": round(t_reindex2, 2),
        "plan_top_node": plan[0].strip(),
        "plan_uses_hnsw": uses_hnsw,
        "hnsw_path_forced_with_enable_seqscan_off": forced,
        "plan_sample_explain": plan_sample,
        "pass1_vs_pass2": tear_block(res1, res2, exact, lat1, lat2),
        "pass2_vs_pass3": tear_block(res2, res3, exact, lat2, lat3),
        "recall_search3_vs_exact": round(recall_vs_exact(res3, exact), 4),
        "latency_ms_search3": {"p50": round(pct(lat3, 50), 2),
                               "p99": round(pct(lat3, 99), 2)},
    }
    return out


def cell_u2(table, xb, qb, exact, ef=EF_SEARCH, k=K, tag="U2"):
    log("cell %s (incremental index -> plain REINDEX)" % tag)
    n_rows, idx_b, insert_s, build_s = make_table(table, xb, incremental=True)
    plan, uses_hnsw, forced = ensure_hnsw_path(table, k, tag)
    plan_sample = explain_query(table, k, qb[0])
    if ef != EF_SEARCH or k != K:
        search_pass(table, qb[:20], k, ef=ef)           # warmup with probe params
    else:
        search_pass(table, qb[:20], K)
    res1, lat1 = search_pass(table, qb, k, ef=ef)
    t_reindex = reindex(table, concurrently=False)
    res2, lat2 = search_pass(table, qb, k, ef=ef)
    out = {
        "cell": tag,
        "table": table,
        "index_construction": ("incremental (CREATE INDEX on empty table, "
                               "8 batches x 2500 inserts, mirroring Milvus "
                               "8-segment flush protocol)"),
        "reindex_variant": "plain REINDEX (bulk rebuild; autocommit)",
        "ef_search": ef, "k": k,
        "rows": n_rows, "index_bytes_before": idx_b,
        "index_bytes_after": idx_size(table),
        "insert_wall_s": round(insert_s, 2),
        "incremental_index_wall_s": round(build_s, 2),
        "reindex_wall_s": round(t_reindex, 2),
        "plan_top_node": plan[0].strip(),
        "plan_uses_hnsw": uses_hnsw,
        "hnsw_path_forced_with_enable_seqscan_off": forced,
        "plan_sample_explain": plan_sample,
        **tear_block(res1, res2, exact, lat1, lat2),
    }
    return out


def cell_g(table, xb, qb, exact, seed):
    """G (heap-order / insertion-order generation change): the pgvector analogue
    of the paper's FAISS protocol "same vectors, different insertion order".
    Build the HNSW index on the id-heap-order heap, then reorder the heap with
    CLUSTER (which rebuilds every index on the table, hence the HNSW graph) and
    search again."""
    log("cell G (CLUSTER heap reorder -> HNSW rebuilt on new insertion order)")
    n_rows, idx_b, insert_s, build_s = make_table(table, xb, incremental=False)
    # deterministic-but-random permutation, generated SERVER-SIDE in one statement
    # (a client-driven 20k-row UPDATE was 100x slower and starved under autovacuum)
    conn = connect()
    with conn.cursor() as cur:
        cur.execute("ALTER TABLE %s ADD COLUMN perm int" % table)
        t0 = time.time()
        cur.execute("UPDATE %s AS t SET perm = s.r FROM (SELECT id, "
                    "(row_number() OVER (ORDER BY random()))::int AS r FROM %s) "
                    "s WHERE t.id = s.id" % (table, table))
        perm_s = time.time() - t0
        cur.execute("CREATE INDEX %s_perm_idx ON %s (perm)" % (table, table))
    conn.close()
    plan, uses_hnsw, forced = ensure_hnsw_path(table, K, "G")
    plan_sample_before = explain_query(table, K, qb[0])
    search_pass(table, qb[:20], K)                      # warmup
    res1, lat1 = search_pass(table, qb, K)              # generation A: id order
    t0 = time.time()
    conn = connect()
    with conn.cursor() as cur:
        cur.execute("CLUSTER %s USING %s_perm_idx" % (table, table))
        cur.execute("ANALYZE %s" % table)
    conn.close()
    cluster_s = time.time() - t0
    t_reindex = reindex(table, concurrently=False)
    res2, lat2 = search_pass(table, qb, K)              # generation B: perm order
    res3, lat3 = search_pass(table, qb, K)              # repeat, same generation
    out = {
        "cell": "G",
        "table": table,
        "index_construction": ("bulk CREATE INDEX on the heap in id order; then "
                               "CLUSTER USING <perm btree> (rewrites the heap "
                               "in a random permutation and rebuilds every "
                               "index, incl. HNSW) -> REINDEX"),
        "generation_change": ("generation A = HNSW over heap in id order; "
                              "generation B = HNSW bulk-rebuilt over heap in "
                              "random-permutation order (the pgvector analogue "
                              "of the paper's FAISS 'same vectors, different "
                              "insertion order' protocol)"),
        "rows": n_rows, "index_bytes_before": idx_b,
        "index_bytes_after": idx_size(table),
        "insert_wall_s": round(insert_s, 2),
        "index_build_wall_s": round(build_s, 2),
        "perm_update_wall_s": round(perm_s, 2),
        "cluster_wall_s": round(cluster_s, 2),
        "reindex_wall_s": round(t_reindex, 2),
        "plan_top_node": plan[0].strip(),
        "plan_uses_hnsw": uses_hnsw,
        "hnsw_path_forced_with_enable_seqscan_off": forced,
        "plan_sample_explain": plan_sample_before,
        "pass1_vs_pass2": tear_block(res1, res2, exact, lat1, lat2),
        "pass2_vs_pass3_samegeneration": tear_block(res2, res3, exact, lat2, lat3),
    }
    return out


def cell_det(table, xb, qb, exact):
    """Determinism probe: is a bulk HNSW rebuild a pure function of heap order?
    Repeat REINDEX twice with the default parallel maintenance workers and then
    with max_parallel_maintenance_workers = 0 (serial build)."""
    log("cell DET (bulk-rebuild determinism, parallel vs serial)")
    out = {"cell": "DET", "table": table,
           "question": ("is a bulk HNSW rebuild a pure function of heap order? "
                        "compare two consecutive REINDEXes under the default "
                        "parallel maintenance workers and under a serial build")}
    n_rows, idx_b, insert_s, build_s = make_table(table, xb, incremental=False)
    plan, uses_hnsw, forced = ensure_hnsw_path(table, K, "DET")
    out["plan_uses_hnsw"] = uses_hnsw

    def two_reindex_runs(tag):
        conn = connect()
        with conn.cursor() as cur:
            cur.execute(tag)                    # GUC for the REINDEX below
        conn.close()
        search_pass(table, qb[:20], K)
        r1, l1 = search_pass(table, qb, K)
        t0 = time.time()
        conn = connect()
        with conn.cursor() as cur:
            cur.execute(tag)
            cur.execute("REINDEX INDEX %s_embedding_hnsw" % table)
        conn.close()
        w1 = time.time() - t0
        r2, l2 = search_pass(table, qb, K)
        t0 = time.time()
        conn = connect()
        with conn.cursor() as cur:
            cur.execute(tag)
            cur.execute("REINDEX INDEX %s_embedding_hnsw" % table)
        conn.close()
        w2 = time.time() - t0
        r3, l3 = search_pass(table, qb, K)
        return {
            "reindex1_wall_s": round(w1, 2), "reindex2_wall_s": round(w2, 2),
            "pass1_vs_pass2": tear_block(r1, r2, exact, l1, l2),
            "pass2_vs_pass3": tear_block(r2, r3, exact, l2, l3),
        }

    out["parallel_default"] = two_reindex_runs("RESET max_parallel_maintenance_workers")
    out["serial"] = two_reindex_runs("SET max_parallel_maintenance_workers = 0")
    conn = connect()
    with conn.cursor() as cur:
        cur.execute("RESET max_parallel_maintenance_workers")
    conn.close()
    return out


def cell_p(table, xb, qb, exact):
    log("cell P (REPEATABLE READ snapshot spanning REINDEX CONCURRENTLY)")
    n_rows, idx_b, insert_s, build_s = make_table(table, xb, incremental=True)
    plan, uses_hnsw, forced = ensure_hnsw_path(table, K, "P")
    plan_sample = explain_query(table, K, qb[0])

    # (a) probe: REINDEX CONCURRENTLY inside a transaction block
    probe_inside_txn = None
    conn = connect(autocommit=False)
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM %s LIMIT 1" % table)
        cur.fetchone()
        try:
            cur.execute("REINDEX INDEX CONCURRENTLY %s_embedding_hnsw" % table)
            probe_inside_txn = "no error (unexpected)"
        except Exception as ex:  # noqa: BLE001
            probe_inside_txn = str(ex).strip().splitlines()[0]
    conn.rollback()
    conn.close()
    log("probe(a) REINDEX CONCURRENTLY inside txn -> %s" % probe_inside_txn)

    # (b) probe: plain REINDEX blocks on an open reader (quiesce evidence)
    probe_quiesce = None
    conn = connect(autocommit=False)            # reader txn
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM %s ORDER BY embedding <-> '%s'::vector "
                    "LIMIT 1" % (table, vec_lit(qb[0])))
        cur.fetchone()                          # holds AccessShare till commit
        helper = connect()
        t0 = time.time()
        try:
            with helper.cursor() as hcur:
                hcur.execute("SET lock_timeout = '3s'")
                hcur.execute("REINDEX INDEX %s_embedding_hnsw" % table)
            probe_quiesce = ("no block (unexpected) after %.1fs" % (time.time() - t0))
        except Exception as ex:  # noqa: BLE001
            probe_quiesce = ("blocked then failed after %.1fs: %s"
                             % (time.time() - t0,
                                str(ex).strip().splitlines()[0]))
        helper.close()
    conn.rollback()
    conn.close()
    log("probe(b) plain REINDEX vs open reader -> %s" % probe_quiesce)

    # main P flow: one REPEATABLE READ transaction spanning a concurrent reindex
    search_pass(table, qb[:20], K)              # warmup
    conn = psycopg2.connect(host=os.environ.get("SCR_PG_HOST", "127.0.0.1"),
                            port=int(os.environ.get("SCR_PG_PORT", "5439")),
                            dbname=os.environ.get("SCR_PG_DB", "scr_engine"),
                            user=os.environ.get("SCR_PG_USER", "postgres"))
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("SET hnsw.ef_search = %s", (EF_SEARCH,))
        if FORCE_SEQSCAN_OFF[0]:
            cur.execute("SET enable_seqscan = off")
        cur.execute("SET SESSION CHARACTERISTICS AS TRANSACTION "
                    "ISOLATION LEVEL REPEATABLE READ")
        cur.execute("SET statement_timeout = '60s'")
    conn.autocommit = False     # everything below runs inside ONE RR transaction
    res1, lat1 = search_pass(table, qb, K, conn=conn)

    # structural rewrite from a second connection while the RR txn stays open
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

    import threading
    th = threading.Thread(target=_reindex)
    th.start()
    time.sleep(2.0)
    reindex_alive_during_search2 = th.is_alive()
    try:
        res2, lat2 = search_pass(table, qb, K, conn=conn)   # same RR txn
        search2_error = None
    except Exception as ex:  # noqa: BLE001
        search2_error = str(ex).strip().splitlines()[0]
        res2, lat2 = None, None
    conn.commit()
    conn.close()
    th.join(timeout=120)

    # post-commit reference pass on a fresh connection (new index generation)
    res2b, lat2b = search_pass(table, qb, K)

    pinned = (tear_block(res1, res2, exact, lat1, lat2,
                         {"note": "both passes inside one REPEATABLE READ "
                                  "transaction spanning the rewrite"})
              if res2 is not None else {"error": search2_error})
    out = {
        "cell": "P",
        "table": table,
        "index_construction": "incremental (same as U2)",
        "pinning_mechanism": ("single REPEATABLE READ transaction spans the "
                              "structural rewrite; both search passes read the "
                              "same MVCC snapshot"),
        "reindex_variant": "REINDEX CONCURRENTLY (only variant runnable "
                           "alongside an open reader)",
        "rows": n_rows,
        "index_bytes_before": idx_b,
        "index_bytes_after": idx_size(table),
        "plan_top_node": plan[0].strip(),
        "plan_uses_hnsw": uses_hnsw,
        "hnsw_path_forced_with_enable_seqscan_off": forced,
        "plan_sample_explain": plan_sample,
        "probe_reindex_concurrently_inside_txn": probe_inside_txn,
        "probe_plain_reindex_vs_open_reader": probe_quiesce,
        "reindex_concurrently_wall_s": reindex_result.get("wall_s"),
        "reindex_error": reindex_result.get("error"),
        "reindex_thread_alive_during_pinned_search2":
            reindex_alive_during_search2,
        "search2_inside_txn_error": search2_error,
        "pinned_pass1_vs_pass2": pinned,
        "postcommit_pass1_vs_pass2b": tear_block(
            res1, res2b, exact, lat1, lat2b,
            {"note": "reference: search1 (pre-rewrite) vs a fresh post-commit "
                     "pass on the swapped-in index"}),
    }
    return out


def cell_d(table, xb, qb, seed):
    log("cell D (delete 10% -> search -> REINDEX -> search)")
    rng = np.random.default_rng(seed + 1)
    deleted = sorted(int(i) for i in rng.choice(N, size=N // 10, replace=False))
    deleted_set = set(deleted)
    dead = np.zeros(N, dtype=bool)
    dead[deleted] = True
    exact_full = exact_topk(xb, qb, K)
    exact_surv = exact_topk(xb, qb, K, dead=dead)
    n_rows, idx_b, insert_s, build_s = make_table(table, xb, incremental=False)
    plan, uses_hnsw, forced = ensure_hnsw_path(table, K, "D")
    plan_sample = explain_query(table, K, qb[0])
    search_pass(table, qb[:20], K)
    # baseline (pre-delete) recall for reference
    res0, lat0 = search_pass(table, qb, K)
    t0 = time.time()
    conn = connect()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM %s WHERE id = ANY(%%s)" % table, (deleted,))
    conn.close()
    delete_s = time.time() - t0
    res1, lat1 = search_pass(table, qb, K)      # dead tuples still in index
    t_re = reindex(table, concurrently=False)   # structural rewrite post-delete
    res2, lat2 = search_pass(table, qb, K)
    ghost1 = sum(1 for r in res1 for i in r if i in deleted_set)
    ghost2 = sum(1 for r in res2 for i in r if i in deleted_set)
    out = {
        "cell": "D",
        "table": table,
        "index_construction": "bulk CREATE INDEX",
        "n_deleted": len(deleted),
        "delete_wall_s": round(delete_s, 2),
        "reindex_wall_s": round(t_re, 2),
        "ghost_deleted_ids_post_delete_search": ghost1,
        "ghost_deleted_ids_post_reindex_search": ghost2,
        "recall_predelete_vs_full_exact": round(
            recall_vs_exact(res0, exact_full), 4),
        "recall_postdelete_vs_surviving_exact": round(
            recall_vs_exact(res1, exact_surv), 4),
        "recall_postreindex_vs_surviving_exact": round(
            recall_vs_exact(res2, exact_surv), 4),
        "latency_ms_predelete": {"p50": round(pct(lat0, 50), 2),
                                 "p99": round(pct(lat0, 99), 2)},
        "latency_ms_postdelete": {"p50": round(pct(lat1, 50), 2),
                                  "p99": round(pct(lat1, 99), 2)},
        "latency_ms_postreindex": {"p50": round(pct(lat2, 50), 2),
                                   "p99": round(pct(lat2, 99), 2)},
        "index_bytes_before": idx_b,
        "index_bytes_after": idx_size(table),
        "plan_top_node": plan[0].strip(),
        "plan_uses_hnsw": uses_hnsw,
        "hnsw_path_forced_with_enable_seqscan_off": forced,
        "plan_sample_explain": plan_sample,
    }
    return out


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    global T_START
    T_START = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="U1,U2,P,D")
    ap.add_argument("--seeds", default=str(SEED_MAIN))
    ap.add_argument("--append", action="store_true",
                    help="merge into the existing JSON instead of overwriting")
    args = ap.parse_args()

    beats = start_heartbeat()   # keep stdout warm for the execution wrapper
    try:
        _main(args)
    finally:
        beats.set()


def _main(args):
    # connectivity + engine fingerprint
    conn = connect()
    with conn.cursor() as cur:
        cur.execute("SELECT version()")
        pg_version = cur.fetchone()[0]
        cur.execute("SELECT extversion FROM pg_extension "
                    "WHERE extname = 'vector'")
        row = cur.fetchone()
        extversion = row[0] if row else "manual install (no pg_extension row)"
        cur.execute("SHOW max_parallel_maintenance_workers")
        par_workers = cur.fetchone()[0]
    conn.close()

    results = {
        "meta": {
            "engine": "PostgreSQL 17.4 (local) + pgvector 0.8.0 HNSW "
                      "(compiled from source with MSVC 19.43, "
                      "Windows PostgreSQL build)",
            "pg_version": pg_version,
            "pgvector_version": extversion,
            "port": 5439,
            "max_parallel_maintenance_workers": par_workers,
            "protocol": {
                "d": D, "N": N, "n_queries": NQ, "batches": BATCHES,
                "per_batch": PER_BATCH, "k": K, "ef_search": EF_SEARCH,
                "index": HNSW_WITH, "metric": "L2",
                "distance_operators": ("search uses <-> (L2), matching the "
                                       "vector_l2_ops index opclass; note the "
                                       "cosine operator <=> was used in the "
                                       "first (discarded) run and silently "
                                       "forced a Seq Scan -- see "
                                       "resolution_of_plan_check"),
                "data": "default_rng(seed).random((20000,64)) then same "
                        "stream (200,64), float32",
                "episode": "search1 -> engine structural rewrite (REINDEX "
                           "and/or CLUSTER) -> search2; table content "
                           "unchanged, so any top-10 difference is an "
                           "index-generation tear (A4)",
                "tear_def": "set(top10_search1) != set(top10_search2)",
            },
            "resolution_of_plan_check": {
                "symptom": ("first U1 run: plan_uses_hnsw=False but "
                            "recall_vs_exact=0.584 (not ~1.0) -- mutually "
                            "exclusive if the query had really been a seq scan"),
                "root_cause": ("the search SQL used the COSINE operator <=> "
                               "while the index was built with the L2 opclass "
                               "vector_l2_ops; pgvector does NOT reject the "
                               "mismatch, the planner silently falls back to "
                               "Seq Scan + Sort by cosine distance, whose "
                               "ranking differs from the L2 ranking the recall "
                               "reference uses -> apparent recall = 0.584"),
                "discriminating_evidence": {
                    "plan_literal_l2": ("Index Scan using "
                                        "u1_vectors_embedding_hnsw (after the "
                                        "fix to <->): HNSW path confirmed"),
                    "plan_literal_cosine": ("Limit -> Sort -> Seq Scan on "
                                            "u1_vectors (opclass/operator "
                                            "mismatch, no index)"),
                    "ef_search_sensitivity": ("with <=> the same top-10 came "
                                              "back for ef_search in "
                                              "{10,40,64,1000} (no index "
                                              "involved); the plan probe was "
                                              "right, the old queries never "
                                              "touched the graph"),
                    "distance_scale_check": ("PG <=> returned ~0.12 for L2 "
                                             "true distances ~5.6: cosine "
                                             "distance on all-positive "
                                             "vectors, confirming the metric "
                                             "mismatch"),
                    "recall_after_fix": ("literal-form recall@10 recomputed "
                                         "with <-> rises from 0.584 to the "
                                         "values recorded in the rerun cells"),
                },
                "action_taken": ("all search SQL switched to <-> (L2); the "
                                 "old U1 numbers were discarded and U1 rerun; "
                                 "each cell records plan_sample_explain and "
                                 "hnsw_path_forced_with_enable_seqscan_off"),
            },
            "api_quirks": API_QUIRKS,
            "deviations": DEVIATIONS,
            "limitations": [
                "pgvector does not expose the HNSW adjacency of a built index; "
                "structural tear (A5) NOT measurable in-engine; answer level "
                "(A4) only.",
                "Milvus (first-choice engine) unreachable in this sandbox: "
                "Docker daemon could not start (see report); pgvector fallback "
                "executed on a local PostgreSQL 17.4 instance instead of the "
                "planned docker pgvector:pg16 image.",
            ],
        },
        "cells": {},
    }
    if args.append and os.path.exists(OUT_JSON):
        with open(OUT_JSON, encoding="utf-8") as f:
            old = json.load(f)
        new_cells = results["cells"]
        meta = results["meta"]
        results = old
        results["cells"].update(new_cells)
        results["meta"].update(meta)
        results["meta"]["api_quirks"] = (API_QUIRKS
                                         + results["meta"].get("api_quirks", []))
        results["meta"]["deviations"] = (DEVIATIONS
                                         + results["meta"].get("deviations", []))

    def save():
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    DEVIATIONS.append(
        "Docker Desktop backend (com.docker.backend.exe) stalls at "
        "'loading settings from default-admin provider' (first-run privileged "
        "service missing; com.docker.service not installed; wsl.exe blocked by "
        "sandbox security policy); docker_engine pipe never appeared after "
        "8+ min. Milvus unreachable -> pgvector fallback on local PG 17.4.")
    DEVIATIONS.append(
        "pgvector installed by compiling v0.8.0 from source (MSVC nmake) and "
        "loading its SQL manually with MODULE_PATHNAME rewritten to the built "
        "vector.dll, because CREATE EXTENSION needs admin-only write access to "
        "share/extension. The engine code is the shipped pgvector 0.8.0.")
    DEVIATIONS.append(
        "First U1 run searched with the cosine operator <=> against a "
        "vector_l2_ops index; pgvector silently fell back to Seq Scan (no "
        "error) and the L2-based recall reference made it look like a 0.584 "
        "'recall'. All searches now use <-> (L2); U1 was rerun and the old "
        "numbers discarded. Details in resolution_of_plan_check.")

    seeds = [int(s) for s in args.seeds.split(",")]
    for seed in seeds:
        xb, qb, exact = make_data(seed)
        run = args.cells.split(",") if seed == seeds[0] else "U2"
        for cell in run:
            cell = cell.strip()
            key = "%s_s%d" % (cell, seed)
            log("=== cell %s ===" % key)
            if cell == "U1":
                out = cell_u1("u1_vectors", xb, qb, exact)
            elif cell == "U2":
                out = cell_u2("u2_vectors", xb, qb, exact)
            elif cell == "G":
                out = cell_g("g_vectors", xb, qb, exact, seed)
            elif cell == "DET":
                out = cell_det("det_vectors", xb, qb, exact)
            elif cell == "P":
                out = cell_p("p_vectors", xb, qb, exact)
            elif cell == "D":
                out = cell_d("d_vectors", xb, qb, seed)
            else:
                DEVIATIONS.append("unknown cell %s skipped" % cell)
                continue
            results["cells"][key] = out
            save()
            t = (out.get("pass1_vs_pass2", {}).get("tear_rate_set_ineq")
                 or out.get("tear_rate_set_ineq"))
            log("%s tear rate = %s" % (key, t))
            # anti-ceiling probes if the incremental->bulk tear is 0
            if cell == "U2" and t == 0.0:
                for tag, ef, k in (("U2_ef16", 16, 10), ("U2_k100", 200, 100)):
                    log("=== conditional probe %s (U2 tear was 0) ===" % tag)
                    out2 = cell_u2("%s_vectors" % tag.lower(), xb, qb,
                                   exact, ef=ef, k=k, tag=tag)
                    results["cells"]["%s_s%d" % (tag, seed)] = out2
                    save()

    # cleanup experiment tables
    conn = connect()
    with conn.cursor() as cur:
        for t in ("u1_vectors", "u2_vectors", "g_vectors", "p_vectors",
                  "d_vectors", "det_vectors", "u2_ef16_vectors",
                  "u2_k100_vectors", "probe_inc", "t_empty"):
            cur.execute("DROP TABLE IF EXISTS %s" % t)
    conn.close()
    log("all cells done -> %s" % OUT_JSON)


if __name__ == "__main__":
    main()
