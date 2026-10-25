#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SCR v2 experiments: E6 answer-change proxy + E7 navigability.

E6: measures returned-set changes across simulated region rebuilds of an
HNSW-style graph index. Visibility timestamps change but vectors remain fixed.
All visibility writes are committed strictly BEFORE the episode
window, so both retrievals of an episode see an IDENTICAL visible node set;
any result difference is therefore attributable to structure only.

E7: sweeps the visible fraction f and measures four predicate placements
(all / through / post / strict) plus visible-subgraph reachability.
"""
from __future__ import annotations

import numpy as np

from common import clustered_vectors, jaccard, unique_adj_bytes, wilson
from graph import NSG, merged_adjacency, reach_fraction, rebuild_region

MERGE_TIMES = [60, 72, 84, 96, 108, 120]
MERGE_PAIRS = [(0, 1), (2, 3), (1, 2), (3, 4), (0, 4), (1, 3)]


# --------------------------------------------------------------------------- #
# E6 -- structural tear
# --------------------------------------------------------------------------- #
def e6_run(seed, N=5000, dim=64, M=12, efC=64, n_episodes=300, k=10, ef=48,
           rebuild_diverse=True, spread=0.12):
    rng = np.random.default_rng(seed)
    V = clustered_vectors(N, dim, n_clusters=40, spread=spread, rng=rng)
    g = NSG(V, M=M, efC=efC, seed=seed)
    adj0 = g.build()

    # ---- pre-episode writes: 500 updates + 120 deletes, all committed at
    # ts in [1, 40], before the episode window: snapshot times are drawn from
    # [45, 126) and the second retrieval happens at most at tick 121.  All writes
    # therefore precede every retrieval, so row-level visibility is identical
    # across an episode and any difference between the two results is structural.
    row_ts = np.zeros(N, dtype=np.int64)
    row_del = np.zeros(N, dtype=bool)
    upd = rng.choice(N, size=500, replace=False)
    row_ts[upd] = rng.integers(1, 41, size=len(upd))
    dele = rng.choice(N, size=120, replace=False)
    row_del[dele] = True
    row_ts[dele] = rng.integers(1, 41, size=len(dele))

    # ---- maintenance schedule: 6 merges, destructive region rebuild,
    # generation published (and retained on the SCR side) per merge.
    segs = np.array_split(np.arange(N), 5)
    gens = [(0, adj0)]
    cur = adj0
    for t, (a, b) in zip(MERGE_TIMES, MERGE_PAIRS):
        region = np.concatenate([segs[a], segs[b]])
        cur = merged_adjacency(cur, rebuild_region(V, region, cur, M,
                                                   diverse=rebuild_diverse))
        gens.append((t, cur))

    def gen_at(t):
        out = gens[0]
        for gt, gadj in gens:
            if gt <= t:
                out = (gt, gadj)
        return out

    vis_cache = {}

    def visible(S):
        if S not in vis_cache:
            vis_cache[S] = (row_ts <= S) & ~row_del
        return vis_cache[S]

    def exact_truth(q, S):
        ids = np.nonzero(visible(S))[0]
        sims = V[ids] @ q
        top = np.argpartition(-sims, k - 1)[:k]
        return set(int(i) for i in ids[top])

    Ss = rng.integers(45, 126, size=n_episodes)   # some episodes straddle no merge
    torn_set = torn_ord = straddle = torn_gs = torn_gns = 0
    pinned_set_changes = pinned_order_changes = pinned_straddle_changes = 0
    jacs = []
    rec_native, rec_scr = [], []
    filter_viol = 0

    for S in Ss:
        S = int(S)
        nxt = [t for t in MERGE_TIMES if t > S]
        t2 = (min(nxt) + 1) if nxt else S + 1
        q = V[rng.integers(0, N)].copy()
        q = q + 0.05 * rng.normal(size=dim).astype(np.float32)
        q /= np.linalg.norm(q)
        vis = visible(S)
        _, adj1 = gen_at(S)
        _, adj2 = gen_at(t2)
        r1 = [i for _, i in g.search(adj1, q, k, ef, vis=vis, mode="through")]
        r2 = [i for _, i in g.search(adj2, q, k, ef, vis=vis, mode="through")]
        _, adjS = gen_at(S)                      # pinned generation for the episode
        p1 = [i for _, i in g.search(adjS, q, k, ef, vis=vis, mode="through")]
        p2 = [i for _, i in g.search(adjS, q, k, ef, vis=vis, mode="through")]

        straddle += int(adj1 is not adj2)
        pinned_changed = set(p1) != set(p2)
        pinned_set_changes += int(pinned_changed)
        pinned_order_changes += int(p1 != p2)
        pinned_straddle_changes += int(pinned_changed and adj1 is not adj2)
        if set(r1) != set(r2):
            torn_set += 1
            torn_gs += int(adj1 is not adj2)
        torn_ord += int(r1 != r2)
        if adj1 is adj2:
            torn_gns += int(set(r1) != set(r2))
        jacs.append(jaccard(r1, r2))
        filter_viol += sum(1 for i in (r1 + r2 + p1 + p2) if not vis[i])
        truth = exact_truth(q, S)
        rec_native.append(len(set(r1) & truth) / k)
        rec_scr.append(len(set(p1) & truth) / k)

    n = len(Ss)
    lo, hi = wilson(torn_set, n)
    lo_s, hi_s = wilson(torn_gs, max(1, straddle))
    return {
        "seed": seed, "N": N, "episodes": n, "k": k, "ef": ef,
        "rebuild_diverse": bool(rebuild_diverse),
        "straddle_rate": round(straddle / n, 6),
        "native_tear_rate_set": round(torn_set / n, 6),
        "native_tear_rate_ord": round(torn_ord / n, 6),
        "native_tear_rate_ci": [round(lo, 6), round(hi, 6)],
        "native_tear_given_straddle": round(torn_gs / max(1, straddle), 6),
        "native_tear_given_straddle_ci": [round(lo_s, 6), round(hi_s, 6)],
        "native_tear_given_nostraddle": round(torn_gns / max(1, n - straddle), 6),
        "native_jaccard_mean": round(float(np.mean(jacs)), 6),
        "native_jaccard_min": round(float(np.min(jacs)), 6),
        "scr_tear_rate": round(pinned_set_changes / n, 6),
        "scr_tear_given_straddle": round(pinned_straddle_changes / max(1, straddle), 6),
        "scr_answer_change_count": pinned_set_changes,
        "scr_order_change_count": pinned_order_changes,
        "straddle_count": straddle,
        "native_answer_change_count": torn_set,
        "metric_scope": "returned-set answer-change proxy; not an adjacency certificate",
        "write_scope": "pre-window visibility timestamp/deletion updates; static vectors",
        "filter_violations": int(filter_viol),
        "native_recall_r1": round(float(np.mean(rec_native)), 6),
        "scr_recall": round(float(np.mean(rec_scr)), 6),
        "retained_MB_native": round(unique_adj_bytes([cur]) / 1e6, 4),
        "retained_MB_scr": round(unique_adj_bytes([f for _, f in gens]) / 1e6, 4),
        "retained_MB_fullcopy_modeled": round(
            len(gens) * (N * dim * 4 + N * M * 4) / 1e6, 4),
        "generations": len(gens),
    }


# --------------------------------------------------------------------------- #
# E7 -- navigability under a shrinking visible fraction
# --------------------------------------------------------------------------- #
def e7_run(seed, N=8000, dim=64, M=12, efC=64, nq=150, k=10, ef_search=48, r=4,
           fs=(0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05), spread=0.12):
    rng = np.random.default_rng(seed)
    V = clustered_vectors(N, dim, n_clusters=40, spread=spread, rng=rng)
    g = NSG(V, M=M, efC=efC, seed=seed)
    adj = g.build()

    qs = []
    for _ in range(nq):
        q = V[rng.integers(0, N)].copy()
        q = q + 0.15 * rng.normal(size=dim).astype(np.float32)
        q /= np.linalg.norm(q)
        qs.append(q)

    rows = []
    for f in fs:
        nvis = max(k + 1, int(round(f * N)))
        vis = np.zeros(N, dtype=bool)
        vis[rng.choice(N, size=nvis, replace=False)] = True
        vid = np.nonzero(vis)[0]
        Vv = V[vid]
        rec = {"all": [], "through": [], "post": [], "strict": []}
        post_fill = 0
        for q in qs:
            sims = Vv @ q
            top = np.argpartition(-sims, k - 1)[:k]
            truth = set(int(i) for i in vid[top])
            out_all = g.search(adj, q, k, ef_search, vis=vis, mode="all")
            out_thr = g.search(adj, q, k, ef_search, vis=vis, mode="through")
            out_pst = g.search(adj, q, k, k * r, vis=vis, mode="post")
            out_str = g.search(adj, q, k, ef_search, vis=vis, mode="strict")
            rec["all"].append(len({i for _, i in out_all} & truth) / k)
            rec["through"].append(len({i for _, i in out_thr} & truth) / k)
            rec["post"].append(len({i for _, i in out_pst} & truth) / k)
            rec["strict"].append(len({i for _, i in out_str} & truth) / k)
            post_fill += int(len(out_pst) >= k)
        reach = []
        for q in qs[:5]:
            e = g._nearest_visible(q, vis)
            reach.append(reach_fraction(adj, vis, e))
        row = {
            "visible_frac": f,
            "visible_nodes": nvis,
            "recall_all": round(float(np.mean(rec["all"])), 6),
            "recall_through": round(float(np.mean(rec["through"])), 6),
            "recall_post": round(float(np.mean(rec["post"])), 6),
            "recall_strict": round(float(np.mean(rec["strict"])), 6),
            "post_fill_rate": round(post_fill / nq, 6),
            "visible_reach_frac": round(float(np.mean(reach)), 6),
        }
        rows.append(row)
        print("  f=%.2f  all=%.3f through=%.3f post=%.3f strict=%.3f reach=%.3f" % (
            f, row["recall_all"], row["recall_through"], row["recall_post"],
            row["recall_strict"], row["visible_reach_frac"]))
    return {"seed": seed, "N": N, "nq": nq, "ef_search": ef_search, "r": r,
            "strict_entry_scope": "exact nearest-visible vector oracle; cost not timed",
            "post_fill_metric": "fraction of queries returning at least k results",
            "rows": rows}


# --------------------------------------------------------------------------- #
# E9 -- tear rate vs. the fraction of the graph a compaction rewrites
# --------------------------------------------------------------------------- #
def e9_run(seed, merge_frac, N=5000, dim=64, M=12, efC=64, n_episodes=300, k=10,
           ef=48, rebuild_diverse=True, n_merges=6, spread=0.12):
    """Same protocol as e6_run, but the rewritten region covers `merge_frac` of N.

    Rationale: a compaction's blast radius is a deployment parameter (how many
    segments it merges).  E9 turns E6 from a point measurement into a curve.
    """
    rng = np.random.default_rng(seed)
    V = clustered_vectors(N, dim, n_clusters=40, spread=spread, rng=rng)
    g = NSG(V, M=M, efC=efC, seed=seed)
    adj0 = g.build()

    row_ts = np.zeros(N, dtype=np.int64)
    row_del = np.zeros(N, dtype=bool)
    upd = rng.choice(N, size=500, replace=False)
    row_ts[upd] = rng.integers(1, 41, size=len(upd))
    dele = rng.choice(N, size=120, replace=False)
    row_del[dele] = True
    row_ts[dele] = rng.integers(1, 41, size=len(dele))

    nreg = max(2, int(round(merge_frac * N)))
    times = [60, 72, 84, 96, 108, 120][:n_merges]
    gens = [(0, adj0)]
    cur = adj0
    for t in times:
        region = rng.choice(N, size=nreg, replace=False)
        cur = merged_adjacency(cur, rebuild_region(V, region, cur, M,
                                                   diverse=rebuild_diverse))
        gens.append((t, cur))

    def gen_at(t):
        out = gens[0]
        for gt, gadj in gens:
            if gt <= t:
                out = (gt, gadj)
        return out

    vis_cache = {}

    def visible(S):
        if S not in vis_cache:
            vis_cache[S] = (row_ts <= S) & ~row_del
        return vis_cache[S]

    def exact_truth(q, S):
        ids = np.nonzero(visible(S))[0]
        sims = V[ids] @ q
        top = np.argpartition(-sims, k - 1)[:k]
        return set(int(i) for i in ids[top])

    lo_t = min(times)
    Ss = rng.integers(lo_t - 15, max(times) + 12, size=n_episodes)
    torn_set = straddle = filt = pinned_set_changes = 0
    jacs = []
    for S in Ss:
        S = int(S)
        nxt = [t for t in times if t > S]
        t2 = (min(nxt) + 1) if nxt else S + 1
        q = V[rng.integers(0, N)].copy()
        q = q + 0.05 * rng.normal(size=dim).astype(np.float32)
        q /= np.linalg.norm(q)
        vis = visible(S)
        _, adj1 = gen_at(S)
        _, adj2 = gen_at(t2)
        r1 = [i for _, i in g.search(adj1, q, k, ef, vis=vis, mode="through")]
        r2 = [i for _, i in g.search(adj2, q, k, ef, vis=vis, mode="through")]
        pinned_second = [i for _, i in g.search(adj1, q, k, ef, vis=vis, mode="through")]
        pinned_set_changes += int(set(r1) != set(pinned_second))
        straddle += int(adj1 is not adj2)
        torn_set += int(set(r1) != set(r2))
        filt += sum(1 for i in (r1 + r2 + pinned_second) if not vis[i])
        jacs.append(jaccard(r1, r2))
    n = len(Ss)
    lo, hi = wilson(torn_set, n)
    return {
        "seed": seed, "merge_frac": merge_frac, "rewritten_nodes": nreg,
        "ef": ef, "rebuild_diverse": bool(rebuild_diverse),
        "episodes": n,
        "straddle_rate": round(straddle / n, 6),
        "tear_rate": round(torn_set / n, 6),
        "tear_rate_ci": [round(lo, 6), round(hi, 6)],
        "pinned_answer_change_count": pinned_set_changes,
        "pinned_answer_change_rate": round(pinned_set_changes / n, 6),
        "metric_scope": "returned-set answer-change proxy; no adjacency certificate",
        "jaccard_mean": round(float(np.mean(jacs)), 6),
        "filter_violations": int(filt),
    }
