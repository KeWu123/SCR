# -*- coding: utf-8 -*-
"""Lambda >= 3 generations: pairwise separation, pairwise certified tear, and
the triple-union residue of Prop. 3(b3).

Two assumptions carry the paper's lower bounds:
  (P1) RSI(ii) constrains TRAVERSED adjacency, not the returned top-k.
  (P2) a separating probe family exists (some probe separates any two
       admissible adjacency sets).
The two-generation witness does not establish pairwise beam-distinctness
for Lambda >= 3. This experiment measures that condition directly.

THIS SCRIPT measures, in real FAISS HNSW, for Lambda = 4 simultaneously
published generations over one node set:

  1. SEPARATION (Assumption P2, P1-facing): for every one of the 6 pairs,
     how often does a probe's TRAVERSAL differ, and how often does its
     returned top-k differ.  The pair is "beam-distinct" iff some probe
     separates its traversal.
  2. CERTIFIED TEAR, pairwise: union of the two walked edge sets embeds in
     no published generation (the literal A5 definition over Lambda
     published generations, not just the two that produced the walks).
  3. TRIPLE UNION (Prop. 3(b3) residue): union of three walked edge sets.
     The residue needs every PAIR union to embed in SOME published
     generation while the triple does not; we report both rates, so a
     reader can see whether the residue is exercised or dominated.

CONTROL.  Two traversals of ONE generation from different entry points:
their union always embeds in that generation, so the certified rate must
be 0.

CROSS-CHECK.  Generation 1 uses the SAME rng draw the two-generation study
used, so pair (0,1) must reproduce results_certified_tear.json
(certified 1.0000, answer agreement 0.5867 at ef=128).  If it does not,
this script and that one are not measuring the same object.

OUTPUT: paper/proto/results_certified_multigen.json
"""
import heapq
import itertools
import json
import os
import time

import numpy as np
import faiss

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, "results_certified_multigen.json")

N, D, M, EFC = 20000, 64, 32, 40
SEEDS = [20261008, 20261009, 20261010]
NQ, K = 200, 10
EFS = [16, 32, 64, 128]
LAM = 4                       # simultaneously published generations


# ------------------------------------------------------------------ build -- #
def build_generation(xb, perm):
    """Build HNSW over xb[perm]; return level-0 adjacency and all-level edge
    set, both in SHARED (natural-vector) ids, plus the engine entry point."""
    idx = faiss.IndexHNSWFlat(D, M)
    idx.hnsw.efConstruction = EFC
    idx.add(np.ascontiguousarray(xb[perm]))

    offs = faiss.vector_to_array(idx.hnsw.offsets).astype(np.int64)
    nbrs = faiss.vector_to_array(idx.hnsw.neighbors).astype(np.int64)
    n = offs.shape[0] - 1
    slots0 = idx.hnsw.nb_neighbors(0)

    l0 = np.full((n, slots0), -1, dtype=np.int64)
    all_edges = set()
    for u_local in range(n):
        u = int(perm[u_local])
        for t, v_local in enumerate(nbrs[offs[u_local]:offs[u_local] + slots0]):
            if v_local >= 0:
                v = int(perm[v_local])
                l0[u, t] = v
                all_edges.add((u, v))
        # higher levels: still part of the "published adjacency" superset
        for pos in range(offs[u_local] + slots0, offs[u_local + 1]):
            v_local = nbrs[pos]
            if v_local >= 0:
                all_edges.add((u, int(perm[v_local])))
    entry_local = int(getattr(idx.hnsw, "entry_point", 0) or 0)
    return l0, all_edges, int(perm[entry_local])


def beam_search(xb, q, d2, entry, l0, ef):
    """HNSW level-0 beam search over an exported adjacency.  Returns
    (top-k list, walked-edge set).  d2 is the precomputed squared distance
    vector of q to every node."""
    visited = bytearray(xb.shape[0])
    cand = [(d2[entry], entry)]
    res = [(-d2[entry], entry)]
    visited[entry] = 1
    heapq.heapify(cand)
    heapq.heapify(res)
    walked = set()
    n_slots = l0.shape[1]
    while cand:
        du, u = heapq.heappop(cand)
        if len(res) >= ef and du > -res[0][0]:
            break
        row = l0[u]
        for t in range(n_slots):
            v = int(row[t])
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


# -------------------------------------------------------------------- run -- #
def run_seed(seed):
    rng = np.random.default_rng(seed)
    xb = rng.random((N, D), dtype=np.float32)
    qb = rng.random((NQ, D), dtype=np.float32)

    perms = [np.arange(N)]                      # generation 0: natural order
    for _ in range(LAM - 1):                    # gen 1 == the 2-gen study's perm
        perms.append(rng.permutation(N))

    gens = []
    for g, pm in enumerate(perms):
        t0 = time.time()
        l0, edges, entry = build_generation(xb, pm)
        gens.append({"l0": l0, "edges": edges, "entry": entry})
        print("    gen %d built (%d all-level edges, %.1fs)"
              % (g, len(edges), time.time() - t0))

    pairs = list(itertools.combinations(range(LAM), 2))
    triples = list(itertools.combinations(range(LAM), 3))
    jac_l0 = {}
    jac_all = {}
    for i, j in pairs:
        # level-0 edge sets
        s_i = set()
        l0i = gens[i]["l0"]
        for u in range(N):
            for v in l0i[u]:
                if v >= 0:
                    s_i.add((u, int(v)))
        s_j = set()
        l0j = gens[j]["l0"]
        for u in range(N):
            for v in l0j[u]:
                if v >= 0:
                    s_j.add((u, int(v)))
        jac_l0["%d-%d" % (i, j)] = len(s_i & s_j) / len(s_i | s_j)
        ea, eb = gens[i]["edges"], gens[j]["edges"]
        jac_all["%d-%d" % (i, j)] = len(ea & eb) / len(ea | eb)

    per_ef = {}
    for ef in EFS:
        # separation and certification accumulators
        sep_trav = {p: 0 for p in pairs}
        sep_ans = {p: 0 for p in pairs}
        sep_trav_eq = {p: 0 for p in pairs}   # separation in BOTH directions
        cert_pair = {p: 0 for p in pairs}     # both walks escape each other
        cert_noemb = {p: 0 for p in pairs}    # union in NO published gen
        emb_some = {p: 0 for p in pairs}      # union embeds in >=1 published gen
        trip_noemb = {t: 0 for t in triples}
        trip_pairwise_embeds = {t: 0 for t in triples}
        trip_residue = {t: 0 for t in triples}
        ctl = 0
        agree = 0
        for qi in range(NQ):
            q = qb[qi]
            d2 = ((xb - q) ** 2).sum(1)
            walk = {}
            top = {}
            for g in range(LAM):
                top[g], walk[g] = beam_search(xb, q, d2, gens[g]["entry"],
                                              gens[g]["l0"], ef)
            for (i, j) in pairs:
                wi, wj = walk[i], walk[j]
                sep_trav[(i, j)] += (wi != wj)
                sep_trav_eq[(i, j)] += (wi != wj and wj != wi)
                sep_ans[(i, j)] += (set(top[i]) != set(top[j]))
                union = wi | wj
                escapes_i = any(e not in gens[i]["edges"] for e in wj)
                escapes_j = any(e not in gens[j]["edges"] for e in wi)
                cert_pair[(i, j)] += (escapes_i and escapes_j)
                emb = [g for g in range(LAM)
                       if union <= gens[g]["edges"]]
                cert_noemb[(i, j)] += (len(emb) == 0)
                emb_some[(i, j)] += (len(emb) > 0)
            for (i, j, k) in triples:
                u3 = walk[i] | walk[j] | walk[k]
                no_emb = all(not (u3 <= gens[g]["edges"]) for g in range(LAM))
                trip_noemb[(i, j, k)] += no_emb
                pr = all(any((walk[a] | walk[b]) <= gens[g]["edges"]
                             for g in range(LAM))
                         for (a, b) in itertools.combinations((i, j, k), 2))
                trip_pairwise_embeds[(i, j, k)] += pr
                trip_residue[(i, j, k)] += (pr and no_emb)
            agree += (set(top[0]) == set(top[1]))
            # control: two traversals of generation 0, different entry points
            e2 = int(rng.integers(N))
            if e2 == gens[0]["entry"]:
                e2 = (e2 + 1) % N
            _, w2 = beam_search(xb, q, d2, e2, gens[0]["l0"], ef)
            u = walk[0] | w2
            ctl += 0 if (u <= gens[0]["edges"]) else 1
        per_ef[ef] = {
            "separation_traversal": {("%d-%d" % p): sep_trav[p] / NQ
                                     for p in pairs},
            "separation_answer": {("%d-%d" % p): sep_ans[p] / NQ
                                  for p in pairs},
            "certified_pairwise": {("%d-%d" % p): cert_pair[p] / NQ
                                   for p in pairs},
            "certified_no_embedding": {("%d-%d" % p): cert_noemb[p] / NQ
                                       for p in pairs},
            "union_embeds_somewhere": {("%d-%d" % p): emb_some[p] / NQ
                                       for p in pairs},
            "triple_no_embedding": {("%d-%d-%d" % t): trip_noemb[t] / NQ
                                    for t in triples},
            "triple_pairwise_embedding_holds": {
                ("%d-%d-%d" % t): trip_pairwise_embeds[t] / NQ
                for t in triples},
            "triple_union_residue": {("%d-%d-%d" % t): trip_residue[t] / NQ
                                     for t in triples},
            "min_pairwise_separation_traversal": min(
                sep_trav[p] / NQ for p in pairs),
            "min_pairwise_separation_answer": min(
                sep_ans[p] / NQ for p in pairs),
            "pk_pair01_certified": cert_pair[(0, 1)] / NQ,
            "pk_pair01_answer_agree": agree / NQ,
            "control_one_generation_rate": ctl / NQ,
        }
        r = per_ef[ef]
        print("  seed %d ef=%3d  min sepr(trav)=%.4f sepr(ans)=%.4f "
              "cert(n,3 pairs)=%.4f..%.4f  cert(noembed)=%.4f..%.4f  "
              "tripNoemb<=%.4f tripPairEmb<=%.4f  ctl=%.4f"
              % (seed, ef, r["min_pairwise_separation_traversal"],
                 r["min_pairwise_separation_answer"],
                 min(r["certified_pairwise"].values()),
                 max(r["certified_pairwise"].values()),
                 min(r["certified_no_embedding"].values()),
                 max(r["certified_no_embedding"].values()),
                 max(r["triple_no_embedding"].values()),
                 max(r["triple_pairwise_embedding_holds"].values()),
                 r["control_one_generation_rate"]))
    return per_ef, {"jaccard_level0": jac_l0, "jaccard_all_levels": jac_all,
                    "edges": [len(g["edges"]) for g in gens]}


def main():
    t0 = time.time()
    all_seeds = {}
    meta = None
    for seed in SEEDS:
        print("seed %d ..." % seed)
        all_seeds[seed], meta = run_seed(seed)
    agg = {}
    for ef in EFS:
        agg[ef] = {}
        keys = all_seeds[SEEDS[0]][ef].keys()
        for key in keys:
            vals = [all_seeds[s][ef][key] for s in SEEDS]
            if isinstance(vals[0], dict):
                agg[ef][key] = {kk: sum(v[kk] for v in vals) / len(vals)
                                for kk in vals[0]}
            elif all(v is not None for v in vals):
                agg[ef][key] = sum(vals) / len(vals)
            else:
                agg[ef][key] = None
    out = {
        "protocol": {
            "engine": "faiss-cpu %s" % faiss.__version__,
            "index": "IndexHNSWFlat", "N": N, "d": D, "M": M,
            "efConstruction": EFC, "k": K, "n_queries": NQ,
            "ef_values": EFS, "seeds": SEEDS, "lambda": LAM,
            "generations": "Lambda independent insertion-order permutations of "
                           "one node set; ids remapped to shared space",
            "replay": "HNSW level-0 beam search over each generation's "
                      "exported adjacency, every expanded edge recorded",
            "separation": "same probe, same beam, different generation; "
                          "traversal separation = the two walked edge sets "
                          "differ",
            "certified_pairwise": "some B-walked edge outside A AND some "
                                  "A-walked edge outside B",
            "certified_no_embedding": "union of the two walked edge sets is "
                                      "not a subset of ANY of the Lambda "
                                      "published generations (literal A5)",
            "triple_union_residue": "every pairwise union embeds in some "
                                    "published generation while the triple "
                                    "union does not (Prop. 3(b3))",
            "control": "two traversals of generation 0 from different entry "
                       "points; must be 0",
        },
        "per_seed": {str(s): all_seeds[s] for s in SEEDS},
        "aggregate": agg,
        "cross_check": "pair 0-1 must reproduce results_certified_tear.json",
        "graph_meta": meta,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote", OUT_JSON)
    for ef in EFS:
        a = agg[ef]
        print("ef=%3d sepr_trav_min=%.4f cert_pair_min=%.4f cert_noemb_min=%.4f "
              "trip_noemb_max=%.4f trip_pairemb_max=%.4f pk01_cert=%.4f "
              "pk01_agree=%.4f ctl=%.4f"
              % (ef, a["min_pairwise_separation_traversal"],
                 min(a["certified_pairwise"].values()),
                 min(a["certified_no_embedding"].values()),
                 max(a["triple_no_embedding"].values()),
                 max(a["triple_pairwise_embedding_holds"].values()),
                 a["pk_pair01_certified"], a["pk_pair01_answer_agree"],
                 a["control_one_generation_rate"]))
    print("jaccard(all-levels) per pair:",
          json.dumps(meta["jaccard_all_levels"]))
    print("jaccard(level-0) per pair:", json.dumps(meta["jaccard_level0"]))


if __name__ == "__main__":
    main()
