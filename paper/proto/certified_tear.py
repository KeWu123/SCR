# -*- coding: utf-8 -*-
"""Certified structural tear: replay the beam over the published adjacency.

THE QUESTION. The paper's A5 is an existence statement over *adjacency*:
  certified tear  <=>  NOT EXISTS published generation s with
                        Adj(o) UNION Adj(o')  subset of  E_s.
The shipped detector (results_hnsw_tear.json) only compares returned node
sets (tear_any / tear_structural) and the paper says a certified count would
need "per-episode edge sets with generation identifiers, which the run does
not record".  This script closes that gap the way the paper itself proposes:
REPLAY the beam with edge-set recording.

DESIGN (paper config, identical to real_hnsw_tear.py):
  - faiss-cpu IndexHNSWFlat, N=20000, d=64, M=32, efConstruction=40,
    data uniform in [0,1)^d, L2, seeds [20261008, 20261009, 20261010].
  - Generation A: built over xb (natural order).
    Generation B: same vectors, insertion order permuted -> a different but
    equally legitimate graph over the SAME node set.  B's local ids are
    remapped to the shared id space (perm[u]).
  - Replay = standard HNSW layer-0 beam search executed directly over each
    generation's exported adjacency (engine entry point, ef-bounded
    candidate set), recording every expanded edge (u, v).
  - The two published generations are {A, B}; an episode's two retrievals are
    served by the currently published generation across the swap (exactly the
    structure-causal deployment of Thm. char's (<=) direction).

PREDICATE. edges_A walks only E_A and edges_B only E_B, so
  union <= E_A  <=>  edges_B <= E_A,      union <= E_B  <=>  edges_A <= E_B,
hence
  certified  <=>  (edges_B has an edge outside E_A) AND
                  (edges_A has an edge outside E_B).

CONTROL (distinguishes 'two different traversals' from 'two different
structures'): both retrievals served by generation A, the second from a
different (random) entry point.  The two traversals then differ, but both
walk E_A, so the certified rate must be 0.0000.

OUTPUT: paper/proto/results_certified_tear.json
"""
import heapq
import json
import os
import time

import numpy as np
import faiss

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, "results_certified_tear.json")

N, D, M, EFC = 20000, 64, 32, 40
SEEDS = [20261008, 20261009, 20261010]
NQ, K = 200, 10
EFS = [16, 32, 64, 128]


# ----------------------------------------------------------------- graphs -- #
def _np(x, dt):
    return faiss.rev_swig_ptr(x, x.size()).view(dt) if hasattr(faiss, "rev_swig_ptr") \
        else np.array(x, dtype=dt)


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


def level0_neighbors(offs, nbrs, u, slots):
    return nbrs[offs[u]:offs[u] + slots]


# ------------------------------------------------------------------ replay -- #
def beam_search(xb, q, entry, offs, nbrs, slots, ef, k):
    """HNSW layer-0 beam search. Returns (top-k list, walked-edge set)."""
    d2 = ((xb - q) ** 2).sum(1)
    visited = bytearray(xb.shape[0])
    cand = [(d2[entry], entry)]          # min-heap on distance
    visited[entry] = 1
    res = [(-d2[entry], entry)]          # max-heap via negation
    heapq.heapify(cand)
    heapq.heapify(res)
    walked = set()

    while cand:
        du, u = heapq.heappop(cand)
        if len(res) >= ef and du > -res[0][0]:
            break
        for v in level0_neighbors(offs, nbrs, u, slots):
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
    topk = sorted((-nd, v) for nd, v in res)[:k]
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
    E_A = set(eA)
    E_B = set((int(perm[u]), int(perm[v])) for (u, v) in eB)   # shared ids
    # B's graph structure in shared ids (for the replay over B)
    offB_map = offB            # offsets identical (same node count/slots)
    nbB_map = nbrs_map = None
    # remap B's neighbor array lazily per-node inside a wrapper
    perm_arr = perm

    slotsA = idxA.hnsw.nb_neighbors(0)
    slotsB = idxB.hnsw.nb_neighbors(0)

    entryA = int(getattr(idxA.hnsw, "entry_point", 0) or 0)
    entryB_local = int(getattr(idxB.hnsw, "entry_point", 0) or 0)
    entryB = int(perm_arr[entryB_local])

    class BGraph:
        """B's adjacency, remapped to shared ids on the fly."""

        def neighbors(self, u_shared):
            j = int(np.where(perm_arr == u_shared)[0][0]) if False else None  # unused
    # Simpler: precompute B's level-0 neighbor lists in shared ids.
    # (N x 64 ints ~ 5 MB, fine.)
    nbB_shared = np.full((N, slotsB), -1, dtype=np.int64)
    for j in range(N):
        row = nbB[offB[j]:offB[j] + slotsB]
        u_shared = int(perm_arr[j])
        for t, v in enumerate(row):
            if v >= 0:
                nbB_shared[u_shared, t] = int(perm_arr[v])
    nbA_l0 = np.full((N, slotsA), -1, dtype=np.int64)
    for u in range(N):
        row = nbA[offA[u]:offA[u] + slotsA]
        for t, v in enumerate(row):
            if v >= 0:
                nbA_l0[u, t] = int(v)

    def search_A(q, entry, ef):
        d2 = ((xb - q) ** 2).sum(1)
        visited = bytearray(N)
        cand = [(d2[entry], entry)]
        res = [(-d2[entry], entry)]
        visited[entry] = 1
        walked = set()
        heapq.heapify(cand); heapq.heapify(res)
        while cand:
            du, u = heapq.heappop(cand)
            if len(res) >= ef and du > -res[0][0]:
                break
            for v in nbA_l0[u]:
                if v < 0 or visited[v]:
                    continue
                visited[v] = 1
                walked.add((u, int(v)))
                dv = d2[v]
                if len(res) < ef:
                    heapq.heappush(cand, (dv, v)); heapq.heappush(res, (-dv, v))
                elif dv < -res[0][0]:
                    heapq.heapreplace(res, (-dv, v)); heapq.heappush(cand, (dv, v))
        topk = sorted((-nd, v) for nd, v in res)[:K]
        return [v for _, v in topk], walked

    def search_B(q, entry, ef):
        d2 = ((xb - q) ** 2).sum(1)
        visited = bytearray(N)
        cand = [(d2[entry], entry)]
        res = [(-d2[entry], entry)]
        visited[entry] = 1
        walked = set()
        heapq.heapify(cand); heapq.heapify(res)
        while cand:
            du, u = heapq.heappop(cand)
            if len(res) >= ef and du > -res[0][0]:
                break
            for v in nbB_shared[u]:
                if v < 0 or visited[v]:
                    continue
                visited[v] = 1
                walked.add((u, int(v)))
                dv = d2[v]
                if len(res) < ef:
                    heapq.heappush(cand, (dv, v)); heapq.heappush(res, (-dv, v))
                elif dv < -res[0][0]:
                    heapq.heapreplace(res, (-dv, v)); heapq.heappush(cand, (dv, v))
        topk = sorted((-nd, v) for nd, v in res)[:K]
        return [v for _, v in topk], walked

    # exact top-k for recall of the replay
    exact = np.argsort(((xb[None, :, :] - qb[:, None, :]) ** 2).sum(2), axis=1)[:, :K] \
        if False else None
    # (chunked exact to bound memory)
    exact = []
    for i in range(NQ):
        d2 = ((xb - qb[i]) ** 2).sum(1)
        exact.append(set(np.argpartition(d2, K)[:K].tolist()))

    per_ef = {}
    for ef in EFS:
        cert = agree = cert_agree = 0
        esc_A = esc_B = walked_cnt = 0
        ctl_cert = 0
        recA = recB = 0
        for i in range(NQ):
            q = qb[i]
            topA, wA = search_A(q, entryA, ef)
            topB, wB = search_B(q, entryB, ef)
            walked_cnt += len(wA) + len(wB)
            outA = [e for e in wB if e not in E_A]   # B-walked edges absent from E_A
            outB = [e for e in wA if e not in E_B]   # A-walked edges absent from E_B
            esc_A += len(outA); esc_B += len(outB)
            certified = len(outA) > 0 and len(outB) > 0
            cert += certified
            same = set(topA) == set(topB)
            agree += same
            cert_agree += (certified and same)
            recA += len(set(topA) & exact[i]) / K
            recB += len(set(topB) & exact[i]) / K
            # control: two traversals of ONE generation, different entry points
            e2 = int(rng.integers(N))
            _, wA2 = search_A(q, e2, ef)
            outA2 = [e for e in wA2 if e not in E_A]
            ctl_cert += (len(outA2) > 0)     # must stay 0: wA2 walks E_A only
        per_ef[ef] = {
            "certified_rate": cert / NQ,
            "answer_agree_rate": agree / NQ,
            "certified_given_agree": (cert_agree / agree) if agree else None,
            "certified_and_agree_rate": cert_agree / NQ,
            "mean_escape_edges_B_not_in_A": esc_A / NQ,
            "mean_escape_edges_A_not_in_B": esc_B / NQ,
            "mean_walked_edges_per_retrieval": walked_cnt / (2 * NQ),
            "replay_recall_A": recA / NQ,
            "replay_recall_B": recB / NQ,
            "control_one_generation_rate": ctl_cert / NQ,
        }
        print("  seed %d ef=%3d  certified=%.4f agree=%.4f cert|agree=%s "
              "esc(B\\A)=%.1f esc(A\\B)=%.1f recall=%.3f/%.3f ctl=%.4f"
              % (seed, ef, per_ef[ef]["certified_rate"], per_ef[ef]["answer_agree_rate"],
                 ("%.4f" % per_ef[ef]["certified_given_agree"]) if agree else "n/a",
                 per_ef[ef]["mean_escape_edges_B_not_in_A"],
                 per_ef[ef]["mean_escape_edges_A_not_in_B"],
                 per_ef[ef]["replay_recall_A"], per_ef[ef]["replay_recall_B"],
                 per_ef[ef]["control_one_generation_rate"]))
    meta = {
        "edges_A": len(E_A),
        "edges_B": len(E_B),
        "entry_A": entryA, "entry_B": entryB,
    }
    return per_ef, meta


def main():
    t0 = time.time()
    all_seeds = {}
    meta = None
    for seed in SEEDS:
        print("seed %d ..." % seed)
        per_ef, meta = run_seed(seed)
        all_seeds[seed] = per_ef
    # aggregate across seeds
    agg = {}
    for ef in EFS:
        agg[ef] = {}
        for key in all_seeds[SEEDS[0]][ef]:
            vals = [all_seeds[s][ef][key] for s in SEEDS
                    if all_seeds[s][ef][key] is not None]
            agg[ef][key] = (sum(vals) / len(vals)) if vals else None
    out = {
        "protocol": {
            "engine": "faiss-cpu %s" % faiss.__version__,
            "index": "IndexHNSWFlat",
            "N": N, "d": D, "M": M, "efConstruction": EFC,
            "k": K, "n_queries": NQ, "ef_values": EFS,
            "seeds": SEEDS,
            "data": "uniform [0,1)^d, L2",
            "generations": "A: natural insertion order; B: permuted insertion "
                           "order over the same vectors, ids remapped",
            "replay": "HNSW layer-0 beam search over the exported adjacency "
                      "(engine entry point), every expanded edge recorded",
            "predicate": "certified <=> some B-walked edge outside E_A AND "
                         "some A-walked edge outside E_B "
                         "(union of walked edges embeds in neither published "
                         "generation)",
            "control": "two traversals of generation A with different entry "
                       "points; must be 0",
        },
        "per_seed": {str(s): all_seeds[s] for s in SEEDS},
        "aggregate": agg,
        "graph_meta": meta,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote", OUT_JSON)
    for ef in EFS:
        a = agg[ef]
        print("ef=%3d certified=%.4f agree=%.4f cert|agree=%s ctl=%.4f"
              % (ef, a["certified_rate"], a["answer_agree_rate"],
                 ("%.4f" % a["certified_given_agree"])
                 if a["certified_given_agree"] is not None else "n/a",
                 a["control_one_generation_rate"]))


if __name__ == "__main__":
    main()
