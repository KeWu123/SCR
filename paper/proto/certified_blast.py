# -*- coding: utf-8 -*-
"""Certified blast: certified replay under graded rewrite fractions.

certified_tear.py established the whole-index-rebuild endpoint (certified
rate = 1.0000 at every beam width).  This script measures the CURVE between
the endpoints: for each rewrite fraction f, the rewritten generation B_f is
A with the level-0 adjacency of the nodes in S_f replaced by B's (remapped)
level-0 adjacency.

  - S_f = (rngS.permutation(N))[:round(f*N)], where
    rngS = np.random.default_rng(seed ^ 0x5EED) (one independent RNG and one
    permutation per seed); the S_f are nested prefixes.
  - Spliced level-0 adjacency of B_f: u in S_f -> nbB_shared[u];
    u not in S_f -> nbA_l0[u].
  - E_Bf = {(u, v): v in u's spliced level-0 row, v >= 0} (directed).
  - E_A  = A's all-level exported edge set (identical object to E_A in
    certified_tear.py); graph statistics are additionally reported against
    A's level-0-only edge set.
  - Replay: the same layer-0 ef-bounded beam as certified_tear.py, over the
    exported rows.  BOTH the A replay and the B_f replay start from A's
    entry point.  At f = 1.0 an extra B_f replay from B's entry point is
    recorded (*_B_entry) to align with certified_tear.py (~1.0000).

PREDICATE (per query):
  certified <=> (some B_f-walked edge outside E_A) AND
                (some A-walked edge outside E_Bf).

CONTROLS:
  f = 0.0 splices nothing (B_f == A), so certified must be EXACTLY 0
  (asserted at runtime; a violation means an implementation bug).
  f = 1.0 with B's entry point must reproduce the shipped whole-rebuild run.

Output: paper/proto/results_certified_blast.json
"""
import heapq
import json
import os
import time

import numpy as np
import faiss

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, "results_certified_blast.json")

N, D, M, EFC = 20000, 64, 32, 40
SEEDS = [20261008, 20261009, 20261010]
NQ, K = 200, 10
EFS = [16, 32, 64, 128]
FRACS = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00]


# ----------------------------------------------------------------- graphs -- #
def export_adjacency(idx):
    """All-level directed edges [(u, v)] of a built HNSW index, local ids."""
    h = idx.hnsw
    offs = faiss.vector_to_array(h.offsets).astype(np.int64)
    nbrs = faiss.vector_to_array(h.neighbors).astype(np.int64)
    n = offs.shape[0] - 1
    edges = []
    for u in range(n):
        for v in nbrs[offs[u]:offs[u + 1]]:
            if v >= 0:
                edges.append((u, int(v)))
    return edges, offs, nbrs, n


def edge_set_from_rows(rows):
    """Directed edge set {(u, v)} of an (N, slots) level-0 row array."""
    flat = rows.ravel()
    mask = flat >= 0
    us = np.repeat(np.arange(rows.shape[0], dtype=np.int64),
                   rows.shape[1])[mask]
    vs = flat[mask]
    return set(zip(us.tolist(), vs.tolist()))


def graph_stats(E_A, E_Bf):
    inter = len(E_A & E_Bf)
    union = len(E_A) + len(E_Bf) - inter
    return {
        "edges_A": len(E_A),
        "edges_Bf": len(E_Bf),
        "edges_intersection": inter,
        "jaccard": inter / union,
    }


# ------------------------------------------------------------------ replay -- #
def search(d2, rows, entry, ef):
    """HNSW layer-0 beam search (same beam as certified_tear.py).

    d2   : precomputed squared-L2 distances to the query for all N nodes.
    rows : (N, slots) level-0 neighbor rows (shared ids, -1 padded).
    Returns (top-K node list, walked-edge set).
    """
    visited = bytearray(N)
    cand = [(d2[entry], entry)]          # min-heap on distance
    res = [(-d2[entry], entry)]          # max-heap via negation
    visited[entry] = 1
    walked = set()
    heapq.heapify(cand)
    heapq.heapify(res)
    while cand:
        du, u = heapq.heappop(cand)
        if len(res) >= ef and du > -res[0][0]:
            break
        for v in rows[u]:
            v = int(v)
            if v < 0 or visited[v]:
                continue
            visited[v] = 1
            walked.add((u, v))
            dv = d2[v]
            if len(res) < ef:
                heapq.heappush(cand, (dv, v))
                heapq.heappush(res, (-dv, v))
            elif dv < -res[0][0]:
                heapq.heapreplace(res, (-dv, v))
                heapq.heappush(cand, (dv, v))
    topk = sorted((-nd, v) for nd, v in res)[:K]
    return [v for _, v in topk], walked


# --------------------------------------------------------------------- main -- #
def run_seed(seed):
    rng = np.random.default_rng(seed)
    xb = rng.random((N, D), dtype=np.float32)
    qb = rng.random((NQ, D), dtype=np.float32)

    idxA = faiss.IndexHNSWFlat(D, M)
    idxA.hnsw.efConstruction = EFC
    idxA.add(xb)
    perm = rng.permutation(N)
    idxB = faiss.IndexHNSWFlat(D, M)
    idxB.hnsw.efConstruction = EFC
    idxB.add(np.ascontiguousarray(xb[perm]))

    eA, offA, nbA, _ = export_adjacency(idxA)
    eB, offB, nbB, _ = export_adjacency(idxB)
    E_A_all = set(eA)                                            # as certified_tear.py
    E_B_all = set((int(perm[u]), int(perm[v])) for (u, v) in eB)  # shared ids

    slotsA = idxA.hnsw.nb_neighbors(0)
    slotsB = idxB.hnsw.nb_neighbors(0)
    if slotsA != slotsB:
        raise RuntimeError("level-0 slot mismatch: %d vs %d" % (slotsA, slotsB))

    # B's level-0 neighbor rows remapped to shared ids (as certified_tear.py)
    nbB_shared = np.full((N, slotsB), -1, dtype=np.int64)
    for j in range(N):
        row = nbB[offB[j]:offB[j] + slotsB]
        u_shared = int(perm[j])
        for t, v in enumerate(row):
            if v >= 0:
                nbB_shared[u_shared, t] = int(perm[v])
    nbA_l0 = np.full((N, slotsA), -1, dtype=np.int64)
    for u in range(N):
        row = nbA[offA[u]:offA[u] + slotsA]
        for t, v in enumerate(row):
            if v >= 0:
                nbA_l0[u, t] = int(v)

    E_A_l0 = edge_set_from_rows(nbA_l0)
    E_B_l0 = edge_set_from_rows(nbB_shared)

    entryA = int(getattr(idxA.hnsw, "entry_point", 0) or 0)
    entryB = int(perm[int(getattr(idxB.hnsw, "entry_point", 0) or 0)])

    # exact top-k and per-query distances (computed once per seed)
    exact = []
    d2_all = []
    for i in range(NQ):
        d2 = ((xb - qb[i]) ** 2).sum(1)
        d2_all.append(d2)
        exact.append(set(np.argpartition(d2, K)[:K].tolist()))

    rngS = np.random.default_rng(seed ^ 0x5EED)
    permS = rngS.permutation(N)

    per_f = {}
    graph_per_f = {}
    for f in FRACS:
        nf = int(round(f * N))
        inS = np.zeros(N, dtype=bool)
        if nf > 0:
            inS[permS[:nf]] = True
        nbSpliced = np.where(inS[:, None], nbB_shared, nbA_l0)
        E_Bf = edge_set_from_rows(nbSpliced)

        fk = "%.2f" % f
        graph_per_f[fk] = {
            "S_size": nf,
            "all_level": graph_stats(E_A_all, E_Bf),
            "level0": graph_stats(E_A_l0, E_Bf),
        }

        per_f[fk] = {}
        for ef in EFS:
            cert = agree = cert_agree = 0
            esc_Bf = esc_A = walked_cnt = touch = 0
            recA = recBf = 0
            cert_be = agree_be = 0
            esc_Bf_be = esc_A_be = 0
            for i in range(NQ):
                d2 = d2_all[i]
                topA, wA = search(d2, nbA_l0, entryA, ef)
                topBf, wBf = search(d2, nbSpliced, entryA, ef)
                walked_cnt += len(wA) + len(wBf)
                outBf = [e for e in wBf if e not in E_A_all]  # B_f-walked outside E_A
                outA = [e for e in wA if e not in E_Bf]       # A-walked outside E_Bf
                esc_Bf += len(outBf)
                esc_A += len(outA)
                certified = len(outBf) > 0 and len(outA) > 0
                cert += certified
                same = set(topA) == set(topBf)
                agree += same
                cert_agree += (certified and same)
                recA += len(set(topA) & exact[i]) / K
                recBf += len(set(topBf) & exact[i]) / K
                touch += bool(inS[entryA]) or any(bool(inS[v]) for (_, v) in wBf)
                if f == 1.0:
                    # extra alignment run: B_f replay from B's entry point
                    topB2, wB2 = search(d2, nbSpliced, entryB, ef)
                    o2b = [e for e in wB2 if e not in E_A_all]
                    o2a = [e for e in wA if e not in E_Bf]
                    esc_Bf_be += len(o2b)
                    esc_A_be += len(o2a)
                    cert_be += (len(o2b) > 0 and len(o2a) > 0)
                    agree_be += (set(topA) == set(topB2))
            m = {
                "certified_rate": cert / NQ,
                "answer_agree_rate": agree / NQ,
                "certified_given_agree": (cert_agree / agree) if agree else None,
                "certified_and_agree_rate": cert_agree / NQ,
                "mean_escape_edges_Bf_not_in_A": esc_Bf / NQ,
                "mean_escape_edges_A_not_in_Bf": esc_A / NQ,
                "mean_walked_edges_per_retrieval": walked_cnt / (2 * NQ),
                "replay_recall_A": recA / NQ,
                "replay_recall_Bf": recBf / NQ,
                "touch_S_rate": touch / NQ,
            }
            if f == 1.0:
                m["certified_rate_B_entry"] = cert_be / NQ
                m["answer_agree_rate_B_entry"] = agree_be / NQ
                m["mean_escape_edges_Bf_not_in_A_B_entry"] = esc_Bf_be / NQ
                m["mean_escape_edges_A_not_in_Bf_B_entry"] = esc_A_be / NQ
            per_f[fk]["%d" % ef] = m
            line = ("  seed %d f=%.2f ef=%3d cert=%.4f agree=%.4f "
                    "esc(Bf\\A)=%.1f esc(A\\Bf)=%.1f touchS=%.3f rec=%.3f/%.3f"
                    % (seed, f, ef, m["certified_rate"], m["answer_agree_rate"],
                       m["mean_escape_edges_Bf_not_in_A"],
                       m["mean_escape_edges_A_not_in_Bf"], m["touch_S_rate"],
                       m["replay_recall_A"], m["replay_recall_Bf"]))
            if f == 1.0:
                line += " | B-entry cert=%.4f" % m["certified_rate_B_entry"]
            print(line, flush=True)
            if f == 0.0 and cert != 0:
                raise RuntimeError(
                    "f=0.0 control violated at ef=%d: certified=%d/%d "
                    "(must be exactly 0)" % (ef, cert, NQ))

    inter_all = len(E_A_all & E_B_all)
    inter_l0 = len(E_A_l0 & E_B_l0)
    ctx = {
        "edges_A_all": len(E_A_all),
        "edges_B_all": len(E_B_all),
        "edges_A_level0": len(E_A_l0),
        "edges_B_level0": len(E_B_l0),
        "jaccard_A_B_all": inter_all / (len(E_A_all) + len(E_B_all) - inter_all),
        "jaccard_A_B_level0": inter_l0 / (len(E_A_l0) + len(E_B_l0) - inter_l0),
        "entry_A": entryA,
        "entry_B": entryB,
        "level0_slots": int(slotsA),
    }
    return per_f, graph_per_f, ctx


def main():
    t0 = time.time()
    all_seeds = {}
    graphs = {}
    ctxs = {}
    for seed in SEEDS:
        print("seed %d ..." % seed, flush=True)
        per_f, graph_per_f, ctx = run_seed(seed)
        all_seeds[seed] = per_f
        graphs[seed] = graph_per_f
        ctxs[seed] = ctx

    fkeys = ["%.2f" % f for f in FRACS]
    agg = {}
    for fk in fkeys:
        agg[fk] = {}
        for ef in EFS:
            ek = "%d" % ef
            agg[fk][ek] = {}
            for key in all_seeds[SEEDS[0]][fk][ek]:
                vals = [all_seeds[s][fk][ek][key] for s in SEEDS
                        if all_seeds[s][fk][ek][key] is not None]
                agg[fk][ek][key] = (sum(vals) / len(vals)) if vals else None

    graph_agg = {}
    for fk in fkeys:
        graph_agg[fk] = {"S_size": graphs[SEEDS[0]][fk]["S_size"]}
        for part in ("all_level", "level0"):
            graph_agg[fk][part] = {}
            for key in ("edges_A", "edges_Bf", "edges_intersection", "jaccard"):
                vals = [graphs[s][fk][part][key] for s in SEEDS]
                graph_agg[fk][part][key] = sum(vals) / len(vals)

    out = {
        "protocol": {
            "engine": "faiss-cpu %s" % faiss.__version__,
            "index": "IndexHNSWFlat",
            "N": N, "d": D, "M": M, "efConstruction": EFC,
            "k": K, "n_queries": NQ, "ef_values": EFS,
            "seeds": SEEDS,
            "data": "uniform [0,1)^d, L2",
            "rng_stream": "per seed, one np.random.default_rng(seed) supplies "
                          "xb, qb, then perm (insertion order of B) in that "
                          "order -- identical to certified_tear.py, so A, B "
                          "and the queries are bit-identical to the shipped "
                          "whole-rebuild run",
            "generations": "A: natural insertion order. B: same vectors "
                           "inserted in order perm = rng.permutation(N); B's "
                           "level-0 adjacency remapped to the shared id "
                           "space (nbB_shared), as in certified_tear.py",
            "rewrite_fractions": FRACS,
            "rewrite_set_sizes": {fk: int(round(f * N))
                                  for fk, f in zip(fkeys, FRACS)},
            "rewrite_set": "S_f = (rngS.permutation(N))[:round(f*N)] with "
                           "rngS = np.random.default_rng(seed ^ 0x5EED); one "
                           "independent RNG and one permutation per seed, so "
                           "the S_f are nested prefixes "
                           "(S_0.05 < S_0.10 < ... < S_1.0)",
            "splice_rule": "level-0 adjacency of B_f: u in S_f -> B's "
                           "remapped row nbB_shared[u]; u not in S_f -> A's "
                           "row nbA_l0[u]; upper levels are not part of B_f "
                           "(the spliced generation is defined by its "
                           "level-0 adjacency, which is also the only thing "
                           "the replay walks)",
            "E_A": "A's all-level exported edge set (identical object to E_A "
                   "in certified_tear.py); graph statistics are additionally "
                   "reported against A's level-0-only edge set",
            "E_Bf": "{(u, v): v in u's spliced level-0 row, v >= 0} "
                    "(directed)",
            "entry_points": "both the A replay and the B_f replay start from "
                            "A's entry point; at f=1.0 an additional B_f "
                            "replay from B's entry point is recorded "
                            "(*_B_entry) to align with certified_tear.py",
            "replay": "HNSW layer-0 ef-bounded beam search over the exported "
                      "(spliced) level-0 adjacency, every expanded-to-new "
                      "edge recorded -- same beam as certified_tear.py",
            "predicate": "certified <=> (some B_f-walked edge outside E_A) "
                         "AND (some A-walked edge outside E_Bf)",
            "touch_S": "fraction of episodes (queries) whose B_f replay "
                       "visited (entry point included) at least one node of "
                       "S_f; derived from the walked edges: a node is "
                       "visited iff it is the entry or the head of some "
                       "walked edge",
            "recall": "replay top-10 vs exact top-10 (np.argpartition on the "
                      "full distance vector)",
            "controls": "f=0.0: spliced graph identical to A, certified must "
                        "be exactly 0 (asserted at runtime); f=1.0 with B's "
                        "entry point: must reproduce certified_tear.py "
                        "(~1.0000)",
            "ef_grid_note": "full ef grid [16, 32, 64, 128] retained; "
                            "runtime fit the budget, no reduction applied",
        },
        "per_seed": {str(s): all_seeds[s] for s in SEEDS},
        "aggregate": agg,
        "graph_meta": {
            "per_seed": {str(s): graphs[s] for s in SEEDS},
            "aggregate": graph_agg,
            "context": {str(s): ctxs[s] for s in SEEDS},
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote", OUT_JSON)

    print("\nf      J_l0    cert@16 cert@32 cert@64 cert@128 agree@128")
    for fk in fkeys:
        print("%s  %.4f  %.4f  %.4f  %.4f  %.4f    %.4f"
              % (fk, graph_agg[fk]["level0"]["jaccard"],
                 agg[fk]["16"]["certified_rate"],
                 agg[fk]["32"]["certified_rate"],
                 agg[fk]["64"]["certified_rate"],
                 agg[fk]["128"]["certified_rate"],
                 agg[fk]["128"]["answer_agree_rate"]))
    print("f=1.00 B-entry certified: " + ", ".join(
        "ef%d=%.4f" % (ef, agg["1.00"]["%d" % ef]["certified_rate_B_entry"])
        for ef in EFS))


if __name__ == "__main__":
    main()
