"""pgvector rebuild controls, search/build effort sweeps and result merging.

Same-order rebuild non-determinism is measured in the
        CURRENT 3-seed / 200-episode protocol (port 5439, PostgreSQL 17.4 +
        pgvector 0.8.0).  The SAME generation is rebuilt twice with an identical
        id-order heap and identical build parameters (no permutation), and the
        two top-10 sets are compared.
Extract the per-setting answer-disagreement rate of the pgvector search-effort sweep
        (ef 64/128/256/512) and build-effort sweep (efc 64/96/128/192/256) from
        results_engine_multiseed.json, storing raw per_episode with each cell.
Merge: combine with the LanceDB part into results_engine_matched2.json.
"""
import os, sys, json, time, subprocess, datetime
import numpy as np
import psycopg2
from psycopg2.extras import execute_values

HERE = os.path.dirname(os.path.abspath(__file__))
LANCE_PART = os.path.join(HERE, "results_engine_matched2_lance_part.json")
MULTISEED = os.path.join(HERE, "results_engine_multiseed.json")
OUT = os.path.join(HERE, "results_engine_matched2.json")

HOST, PORT, DB, USER = (os.environ.get("SCR_PG_HOST", "127.0.0.1"), int(os.environ.get("SCR_PG_PORT", "5439")), os.environ.get("SCR_PG_DB", "scr_engine"), os.environ.get("SCR_PG_USER", "postgres"))
PGBIN = os.environ.get("SCR_PGBIN", "")
PGDATA = os.environ.get("SCR_PGDATA", "")
PGLOG = os.environ.get("SCR_PGLOG", "")
DLL_DIR = os.environ.get("SCR_PGVECTOR_SOURCE", "")

N, D, NQ, K = 20000, 64, 200, 10
SEEDS = [20261008, 20261009, 20261010]
EF_SEARCH = 64
M, EFC = 32, 64

EF_SWEEP = [64, 128, 256, 512]
EFC_SWEEP = [64, 96, 128, 192, 256]


def log(msg):
    print("[%s] %s" % (datetime.datetime.now().strftime('%H:%M:%S'), msg), flush=True)


# --------------------------------------------------------------------------- #
def _raw_connect():
    return psycopg2.connect(host=HOST, port=PORT, dbname=DB, user=USER)


def connect(autocommit=True, seqscan_off=False):
    c = _raw_connect()
    c.autocommit = autocommit
    with c.cursor() as cur:
        if DLL_DIR:
            cur.execute("SET dynamic_library_path = '%s;$libdir'" % DLL_DIR.replace("'", "''"))
        if seqscan_off:
            cur.execute("SET enable_seqscan = off")
    return c


def ensure_pg():
    try:
        c = connect(); c.close(); return
    except Exception:
        if not all((PGBIN, PGDATA, PGLOG)):
            raise RuntimeError("PostgreSQL is unavailable; configure SCR_PGBIN, SCR_PGDATA and SCR_PGLOG to start an experiment server.") from None
        log("pg not reachable; starting cluster")
        subprocess.run([os.path.join(PGBIN, "pg_ctl.exe"), "-D", PGDATA,
                        "-l", PGLOG, "start"],
                       capture_output=True, text=True)
        for _ in range(30):
            time.sleep(1)
            try:
                c = connect(); c.close(); return
            except Exception:
                pass
        raise RuntimeError("could not start PostgreSQL on port %d" % PORT)


def vec_lit(row):
    return "[" + ",".join("%.9g" % x for x in row) + "]"


def make_data(seed):
    rng = np.random.default_rng(seed)
    xb = rng.random((N, D), dtype=np.float32)
    qb = rng.random((NQ, D), dtype=np.float32)
    q2 = (qb ** 2).sum(axis=1)[:, None]
    x2 = (xb ** 2).sum(axis=1)[None, :]
    dist = q2 + x2 - 2.0 * (qb @ xb.T)
    exact = np.argpartition(dist, K, axis=1)[:, :K].astype(np.int64)
    return xb, qb, exact


def recall(results, exact):
    tot = sum(len(set(r) & set(int(x) for x in ex)) for r, ex in zip(results, exact))
    return tot / (len(results) * len(exact[0]))


def per_query_recall(top, exact):
    """|top10(q)  intersect  exact10(q)| / 10, one value per query."""
    return [len(set(int(x) for x in t) & set(int(x) for x in ex)) / float(len(ex))
            for t, ex in zip(top, exact)]


def compare(res1, res2, exact=None):
    eps = []
    # When `exact` is supplied, each per-episode record also carries the
    # per-query recall of BOTH sides against exact (the same field names the
    # LanceDB half stores), enabling a like-for-like recall-confounding
    # decomposition.  When exact is None the records are unchanged.
    ra = per_query_recall(res1, exact) if exact is not None else None
    rb = per_query_recall(res2, exact) if exact is not None else None
    for i, (a, b) in enumerate(zip(res1, res2)):
        sa, sb = set(a), set(b)
        rec = {"set_equal": sa == sb, "ordered_equal": list(a) == list(b),
               "symdiff": len(sa ^ sb)}
        if exact is not None:
            rec["recall_A_vs_exact"] = round(ra[i], 6)
            rec["recall_B_vs_exact"] = round(rb[i], 6)
        eps.append(rec)
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


def plan_probe(table):
    with connect(seqscan_off=True) as c, c.cursor() as cur:
        cur.execute("EXPLAIN (COSTS OFF) SELECT id FROM %s ORDER BY embedding "
                    "<-> '%s'::vector LIMIT %d"
                    % (table, vec_lit(np.zeros(D, dtype=np.float32)), K))
        plan = [r[0] for r in cur.fetchall()]
    return plan


def search_pass(table, qb, ef, seqscan_off=False):
    conn = connect(seqscan_off=seqscan_off)
    res, lat = [], []
    with conn.cursor() as cur:
        cur.execute("SET hnsw.ef_search = %s", (ef,))
        for q in qb:
            t0 = time.perf_counter()
            cur.execute("SELECT id FROM %s ORDER BY embedding <-> %%s::vector "
                        "LIMIT %d" % (table, K), (vec_lit(q),))
            rows = cur.fetchall()
            lat.append((time.perf_counter() - t0) * 1000.0)
            res.append([int(r[0]) for r in rows])
    conn.close()
    return res, lat


def heap_fingerprint(table):
    conn = connect()
    with conn.cursor() as cur:
        cur.execute("SELECT id, ctid FROM %s ORDER BY ctid LIMIT 8" % table)
        head = [(int(a), str(b)) for a, b in cur.fetchall()]
        cur.execute("SELECT md5(string_agg(id::text, ',' ORDER BY ctid)) FROM %s" % table)
        digest = cur.fetchone()[0]
    conn.close()
    return {"first8_by_ctid": head, "md5_ids_in_heap_order": digest}


def build_table(table, xb):
    conn = connect()
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS %s" % table)
        cur.execute("CREATE TABLE %s (id bigint PRIMARY KEY, embedding vector(%d))"
                    % (table, D))
    conn.close()
    t0 = time.time()
    conn = connect()
    with conn.cursor() as cur:
        rows = [(int(i), vec_lit(xb[i])) for i in range(N)]
        execute_values(cur, "INSERT INTO %s (id, embedding) VALUES %%s" % table,
                       rows, page_size=500)
    conn.close()
    insert_s = time.time() - t0
    t0 = time.time()
    conn = connect()
    with conn.cursor() as cur:
        cur.execute("CREATE INDEX %s_hnsw ON %s USING hnsw (embedding vector_l2_ops) "
                    "WITH (m = %d, ef_construction = %d)" % (table, table, M, EFC))
    conn.close()
    build_s = time.time() - t0
    conn = connect()
    with conn.cursor() as cur:
        cur.execute("ANALYZE %s" % table)
    conn.close()
    return insert_s, build_s


def reindex(table, serial=False):
    conn = connect()
    t0 = time.time()
    with conn.cursor() as cur:
        if serial:
            cur.execute("SET max_parallel_maintenance_workers = 0")
        cur.execute("REINDEX INDEX %s_hnsw" % table)
        cur.execute("SELECT reloptions FROM pg_class WHERE relname = '%s_hnsw'" % table)
        ropts = cur.fetchone()[0]
    conn.close()
    return time.time() - t0, ropts


# --------------------------------------------------------------------------- #
def run_pg_floor():
    ensure_pg()
    cells = {}
    runs = []
    for seed in SEEDS:
        table = "det2_vectors_s%d" % seed
        log("seed=%d: data + table" % seed)
        xb, qb, exact = make_data(seed)
        insert_s, build_s = build_table(table, xb)
        plan = plan_probe(table)
        uses = any("hnsw" in p.lower() for p in plan)
        if not uses:
            raise RuntimeError("planner did not use HNSW for %s: %s" % (table, plan))

        log("seed=%d: pass1 (CREATE INDEX state) + 2 REINDEXes (parallel)" % seed)
        p1, _ = search_pass(table, qb, EF_SEARCH)
        heap1 = heap_fingerprint(table)
        r2_s, ropts = reindex(table)
        p2, _ = search_pass(table, qb, EF_SEARCH)
        heap2 = heap_fingerprint(table)
        r3_s, _ = reindex(table)
        p3, _ = search_pass(table, qb, EF_SEARCH)

        log("seed=%d: 2 REINDEXes (serial maintenance workers=0)" % seed)
        r4_s, ropts_serial = reindex(table, serial=True)
        p4, _ = search_pass(table, qb, EF_SEARCH)
        r5_s, _ = reindex(table, serial=True)
        p5, _ = search_pass(table, qb, EF_SEARCH)

        pfx = "floor_pg_s%d" % seed
        for tag, (x, y), note in [
            ("pass1_vs_pass2", (p1, p2), "CREATE INDEX vs 1st REINDEX (parallel)"),
            ("pass2_vs_pass3", (p2, p3), "1st REINDEX vs 2nd REINDEX (parallel)"),
            ("pass4_vs_pass5", (p4, p5), "1st vs 2nd serial REINDEX (maintenance workers=0)"),
        ]:
            c = compare(x, y, exact)
            cells["%s_%s" % (pfx, tag)] = {
                "cell": "%s_%s" % (pfx, tag),
                "note": ("pgvector same-order rebuild floor: two rebuilds of the "
                         "SAME generation, identical id-order heap, identical "
                         "build parameters (m=%d, efc=%d), no permutation -- %s"
                         % (M, EFC, note)),
                "recall_1_vs_exact": round(recall(x, exact), 4),
                "recall_2_vs_exact": round(recall(y, exact), 4),
                "recall_gap": round(abs(recall(x, exact) - recall(y, exact)), 4),
                "ef_search": EF_SEARCH,
                **c,
            }
            cells["%s_%s" % (pfx, tag)]["per_episode"] = [
                dict(q=i, **{k: v for k, v in e.items()})
                for i, e in enumerate(c["per_episode"])]

        runs.append({
            "seed": seed, "table": table, "rows": N,
            "insert_wall_s": round(insert_s, 2), "create_index_wall_s": round(build_s, 2),
            "plan_top_node": plan[0] if plan else None, "plan_uses_hnsw": True,
            "reindex_wall_s": [round(r2_s, 2), round(r3_s, 2), round(r4_s, 2), round(r5_s, 2)],
            "index_reloptions_parallel": ropts,
            "index_reloptions_serial": ropts_serial,
            "heap_fingerprint_before": heap1,
            "heap_fingerprint_after_reindex": heap2,
            "heap_order_unchanged_by_reindex": heap1["md5_ids_in_heap_order"] == heap2["md5_ids_in_heap_order"],
            "pass_recalls": [round(recall(x, exact), 4) for x in (p1, p2, p3, p4, p5)],
        })
        log("seed=%d floor tears: p1p2=%.4f p2p3=%.4f p4p5=%.4f" % (
            seed,
            cells["%s_pass1_vs_pass2" % pfx]["tear_rate_set_ineq"],
            cells["%s_pass2_vs_pass3" % pfx]["tear_rate_set_ineq"],
            cells["%s_pass4_vs_pass5" % pfx]["tear_rate_set_ineq"]))
    return cells, runs


# --------------------------------------------------------------------------- #
def extract_task_c():
    doc = json.load(open(MULTISEED, encoding="utf-8"))
    cells = {}
    for seed in SEEDS:
        g = doc["cells"]["G_s%d" % seed]
        for ef in EF_SWEEP:
            v = g["recall_matched_query_effort"]["by_ef"][str(ef)]
            nm = "sweep_pg_s%d_efsearch_ef%d" % (seed, ef)
            cells[nm] = {
                "cell": nm,
                "note": ("pgvector query(search)-effort sweep: generation A "
                         "(id order, efc=%d) searched at ef_search=%d, compared "
                         "against the stored perm-order generation B searched at "
                         "ef_search=%d (B recall %.4f).  B is NOT re-tuned, so at "
                         "ef>=128 the two generations are no longer recall-matched."
                         % (EFC, ef, EF_SEARCH,
                            g["recall_matched_build_effort"]["target_recall_B"])),
                "sweep_axis": "ef_search", "ef_search": ef,
                "recall_A_vs_exact": v["recall_vs_exact"],
                "recall_gap_to_B": v["recall_gap_to_B"],
                "tear_rate_set_ineq": v["tear_rate_set_ineq"],
                "mean_symdiff": v["mean_symdiff"],
                "episodes": len(v["per_episode"]),
                "per_episode": v["per_episode"],
            }
        for efc in EFC_SWEEP:
            v = g["efc_sweep_generationA"]["sweep"][str(efc)]
            nm = "sweep_pg_s%d_buildeffort_efc%d" % (seed, efc)
            cells[nm] = {
                "cell": nm,
                "note": ("pgvector build-effort sweep: generation A rebuilt in id "
                         "order at ef_construction=%d, searched at ef_search=%d, "
                         "compared against the stored perm-order generation B "
                         "(efc=%d, ef_search=%d)." % (efc, EF_SEARCH, EFC, EF_SEARCH)),
                "sweep_axis": "ef_construction", "efc": efc,
                "recall_A_vs_exact": v["recall_vs_exact"],
                "recall_gap_to_B": v["recall_gap_to_B"],
                "tear_rate_set_ineq": v["tear_rate_set_ineq_vs_B"],
                "mean_symdiff": v["mean_symdiff_vs_B"],
                "reindex_wall_s": v["reindex_wall_s"],
                "episodes": len(v["per_episode"]),
                "per_episode": v["per_episode"],
            }
    return cells


# --------------------------------------------------------------------------- #
def main():
    argv = sys.argv
    pg_part = os.path.join(HERE, "results_engine_matched2_pg_part.json")

    # ---- mode: pgonly -> write pgvector rebuild and sweep results ---- #
    if "pgonly" in argv:
        cells = {}
        pg_cells, pg_runs = run_pg_floor()
        cells.update(pg_cells)
        tc = extract_task_c()
        cells.update(tc)
        log("sweep extraction produced %d cells" % len(tc))
        doc = {"meta": build_meta({}, pg_runs, True, True), "cells": cells,
               "aggregate": build_aggregate(cells)}
        json.dump(doc, open(pg_part, "w", encoding="utf-8"), indent=1)
        log("wrote CLEAN pg part %s with %d cells" % (pg_part, len(cells)))
        _headline(doc)
        return

    # ---- mode: merge -> authoritative merged file from both parts --------- #
    if "merge" in argv:
        lance = json.load(open(LANCE_PART, encoding="utf-8"))
        pp = json.load(open(pg_part, encoding="utf-8"))
        cells = {}
        cells.update(lance.get("cells", {}))
        n_lance = len(cells)
        cells.update(pp.get("cells", {}))
        assert not set(lance.get("cells", {})) & set(pp.get("cells", {})), \
            "cell-name collision between parts"
        log("merge: %d lance cells + %d pg cells" % (n_lance, len(pp.get("cells", {}))))
        doc = {"meta": build_meta(lance.get("meta", {}),
                                  pp.get("meta", {}).get("task_B", {}).get("runs", []),
                                  True, True),
               "cells": cells, "aggregate": build_aggregate(cells)}
        json.dump(doc, open(OUT, "w", encoding="utf-8"), indent=1)
        log("wrote %s with %d cells" % (OUT, len(cells)))
        _headline(doc)
        return

    # ---- full mode (default): run everything ------------------------------ #
    do_pg = "nopg" not in argv
    do_c = "noc" not in argv
    cells = {}
    if os.path.exists(LANCE_PART):
        lance = json.load(open(LANCE_PART, encoding="utf-8"))
        cells.update(lance.get("cells", {}))
        lance_meta = lance.get("meta", {})
    else:
        lance_meta = {}
        log("WARNING: lance part missing")
    pg_runs = []
    if do_pg:
        pg_cells, pg_runs = run_pg_floor()
        cells.update(pg_cells)
    if do_c:
        tc = extract_task_c()
        cells.update(tc)
        log("sweep extraction produced %d cells" % len(tc))
    doc = {"meta": build_meta(lance_meta, pg_runs, do_pg, do_c), "cells": cells,
           "aggregate": build_aggregate(cells)}
    json.dump(doc, open(OUT, "w", encoding="utf-8"), indent=1)
    log("wrote %s with %d cells" % (OUT, len(cells)))
    _headline(doc)


def build_aggregate(cells):
    buckets = {}
    for k, v in cells.items():
        if "per_episode" not in v:
            continue
        tear = v.get("tear_rate_any_difference", v.get("tear_rate_set_ineq"))
        if k.startswith("match2_"):
            parts = k.split("_")
            key = "%s::%s" % (parts[1], "_".join(parts[3:]))
        elif k.startswith("floor_pg_"):
            key = "floor_pg::" + k.split("_s", 1)[1].split("_", 1)[1]
        elif k.startswith("sweep_pg_"):
            key = "sweep_pg::" + k.split("_s", 1)[1].split("_", 1)[1]
            key = key.rsplit("_", 1)[0]
        else:
            key = k
        buckets.setdefault(key, []).append(tear)
    out = {}
    for key, tears in buckets.items():
        out[key] = {
            "n": len(tears),
            "tear_mean": round(float(np.mean(tears)), 4),
            "tear_per_seed": [round(t, 4) for t in tears],
            "tear_min": round(min(tears), 4),
            "tear_max": round(max(tears), 4),
        }
    return out


def build_meta(lance_meta, pg_runs, do_pg, do_c):
    meta = {
        "purpose": ("LanceDB recall matching to <=0.004 with a fine "
                    "knob-pair search; like-for-like same-order rebuild "
                    "non-determinism floors for pgvector and LanceDB in the "
                    "current 3-seed/200-episode protocol; per-setting "
                    "sweep values."),
        "engines": {
            "pgvector": "PostgreSQL 17.4 + pgvector 0.8.0 (private cluster, port 5439)",
            "lancedb": "LanceDB 0.40.0 (lancedb python package; Rust lance engine)",
        },
        "protocol": {
            "N": N, "d": D, "metric": "L2", "n_queries": NQ, "k": K,
            "seeds": SEEDS, "episodes_per_seed": NQ,
            "pgvector_build": "USING hnsw (embedding vector_l2_ops) WITH (m=%d, ef_construction=%d)" % (M, EFC),
            "pgvector_search": "SET hnsw.ef_search = %d" % EF_SEARCH,
            "operator": "embedding <-> query::vector  (the L2 operator for a vector_l2_ops index)",
            "lancedb_part": lance_meta,
        },
        "task_A": {
            "question": "Can LanceDB recall matching reach |recall_A-recall_B| <= 0.004?",
            "grid_and_rule": lance_meta.get("task_A_knob_grids"),
            "matching_rule": lance_meta.get("task_A_matching_rule"),
        },
        "task_B": {
            "pgvector": ("rebuild the SAME generation twice: identical id-order heap, "
                         "identical build parameters, no permutation, then compare the "
                         "two top-10 sets.  Runs: CREATE INDEX vs REINDEX, REINDEX vs "
                         "REINDEX (parallel), and 1st vs 2nd REINDEX with "
                         "max_parallel_maintenance_workers=0 (serial).  200 episodes."),
            "lancedb": ("two INDEPENDENT fresh builds of the SAME non-reordered data, "
                        "each passed through the identical pipeline (8 appends -> "
                        "create_index -> table.optimize()); table.optimize() is "
                        "idempotent on unchanged data (verified: index_uuid unchanged "
                        "on a second call) so it cannot itself supply the second "
                        "rebuild.  200 episodes per comparison."),
            "runs": pg_runs,
        },
        "task_C": {
            "source": "results_engine_multiseed.json cells G_s{seed}",
            "search_effort_sweep_ef": EF_SWEEP,
            "build_effort_sweep_efc": EFC_SWEEP,
        },
        "deviations": [
            "Sweep values are EXTRACTED from results_engine_multiseed.json (no new "
            "computation); raw per_episode is carried over unchanged.",
            "LanceDB's table.optimize() is idempotent on unchanged data (index_uuid "
            "unchanged on a second call), so the LanceDB same-order floor is measured "
            "as two independent fresh build->optimize pipelines over identical data.",
            "LanceDB exposes no HNSW adjacency: only the answer-level any-difference "
            "convention is measurable there.",
            "LanceDB post-optimize generation ignores the nprobes knob for BOTH index "
            "kinds (verified: identical recall range for every nprobes value), so "
            "generation B is matched over a dense refine_factor/ef ladder only.",
            "LanceDB knob granularity: refine_factor and nprobes are integers; near "
            "recall 0.90 the IvfPq recall ladder has ~0.04-0.05 steps for some seeds, "
            "so a +/-0.01-window match at 0.90 is not always placeable; cells where a "
            "genuine <=0.004 pair exists only at a recall further from the target are "
            "flagged placeable=false with the achieved recall recorded.",
        ],
        "limitations": [
            "pgvector floor uses the id-order (clean) heap, matching generation A of "
            "the paper's engine experiment.",
            "LanceDB has no server-side multi-tenant story; it is an embedded library.",
        ],
    }
    return meta


def _headline(doc):
    cells = doc["cells"]
    log("---- LanceDB recall matching ----")
    for k in sorted(cells):
        if k.startswith("match2_") and "_recallmatched_" in k:
            v = cells[k]
            m = v.get("matched", {})
            if m:
                log("  %-44s A=%.4f B=%.4f gap=%.4f tear=%.4f reach=%s"
                    % (k, v["recall_A_vs_exact"], v["recall_B_vs_exact"],
                       v.get("gap", abs(v["recall_A_vs_exact"] - v["recall_B_vs_exact"])),
                       v["tear_rate_any_difference"], m.get("reached_le_0p004")))
    log("---- Same-order rebuild controls ----")
    for k in sorted(cells):
        if k.startswith("floor_pg_"):
            v = cells[k]
            log("  %-34s tear=%.4f rA=%.4f rB=%.4f" % (
                k, v["tear_rate_set_ineq"], v["recall_1_vs_exact"], v["recall_2_vs_exact"]))
        if "_floor_lance_" in k:
            v = cells[k]
            log("  %-44s tear=%.4f rA=%.4f rB=%.4f" % (
                k, v["tear_rate_any_difference"], v["recall_A_vs_exact"], v["recall_B_vs_exact"]))
    log("---- Search and build effort sweeps ----")
    for k in sorted(cells):
        if k.startswith("sweep_pg_"):
            v = cells[k]
            log("  %-40s recall=%.4f tear=%.4f" % (
                k, v["recall_A_vs_exact"], v["tear_rate_set_ineq"]))


if __name__ == "__main__":
    main()
