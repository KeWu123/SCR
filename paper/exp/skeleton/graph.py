#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simplified hierarchical HNSW (two layers) + generation-aware maintenance.

Design notes
------------
* Two layers: a coarse upper layer (~N/l1_ratio nodes) for long-range
  navigation, and a full base layer. Insertion is greedy + ef-bounded beam
  with diversity pruning; search descends the coarse layer, then beams the
  base layer.
* Adjacency lives in plain dicts {node -> np.int32 array}; a *generation* is
  one such dict for the BASE layer (index maintenance -- merges/compactions --
  rewrites base-layer adjacency only, which is what production systems
  rebuild; the coarse layer is static and tiny).
* The SCR variant retains base-layer generations copy-on-write; the native
  variant keeps only the current one.
* Search modes implement the four predicate placements of E7:
    - 'all'      : no visibility filtering (sanity upper bound)
    - 'through'  : traverse everything, collect visible results only
    - 'post'     : standard beam, filter the result set afterwards
    - 'strict'   : navigate inside the visible subgraph only
"""
from __future__ import annotations

import heapq

import numpy as np

_EMPTY = ()


class NSG:
    """Two-layer HNSW-style graph (simplified: no per-node level sampling)."""

    def __init__(self, vecs, M=12, efC=64, seed=0, l1_ratio=16):
        self.V = np.ascontiguousarray(vecs, dtype=np.float32)
        self.N = self.V.shape[0]
        self.M = int(M)
        self.efC = int(efC)
        self.l1_ratio = int(l1_ratio)
        self.rng = np.random.default_rng(seed)
        self.entry = 0
        self.adj1 = {}

    # ---------------------------------------------------------------- dists
    def _dq(self, i, q):
        return 1.0 - float(self.V[i] @ q)

    def _d(self, i, j):
        return 1.0 - float(self.V[i] @ self.V[j])

    # ---------------------------------------------------------------- build
    def _select(self, node, cands, m):
        """Diversity pruning: keep c if closer to node than to any accepted."""
        chosen = []
        for d, c in cands:
            if len(chosen) >= m:
                break
            ok = True
            for a in chosen:
                if self._d(c, a) < d:
                    ok = False
                    break
            if ok:
                chosen.append(c)
        if len(chosen) < m:                    # fill with nearest remaining
            for d, c in cands:
                if c not in chosen:
                    chosen.append(c)
                    if len(chosen) >= m:
                        break
        return chosen

    def _insert(self, adj, entry, idx, mmax):
        cand = self._beam_std(adj, entry, self.V[idx], self.efC)
        neigh = self._select(idx, cand, mmax)
        adj[idx] = neigh
        for j in neigh:
            lst = adj[j]
            lst.append(idx)
            if len(lst) > mmax:
                adj[j] = self._select(j, sorted((self._d(j, c), c) for c in lst), mmax)
        return neigh

    def _finalize(self, adj):
        return {i: np.asarray(sorted(set(l)), dtype=np.int32) for i, l in adj.items()}

    def build(self):
        order = self.rng.permutation(self.N).tolist()
        # -- coarse layer membership: the global first node is always included,
        #    so the coarse entry is a valid base-layer node from the start.
        sel = [order[0]] + [i for i in order[1:]
                            if self.rng.random() < 1.0 / self.l1_ratio]
        entry0 = int(order[0])
        entry1 = int(order[0])
        adj0, adj1 = {entry0: []}, {entry1: []}
        sel_set = set(sel)
        for raw in order[1:]:
            idx = int(raw)
            e0 = self._beam_std(adj1, entry1, self.V[idx], 8)[0][1] if len(adj1) > 1 else entry1
            self._insert(adj0, e0, idx, self.M)
            if idx in sel_set:
                self._insert(adj1, entry1, idx, self.M)
        self.entry = entry0
        self.adj1 = adj1
        return self._finalize(adj0)

    # ---------------------------------------------------------------- search
    def _descent(self, q):
        """Descend the coarse layer; return a base-layer entry node."""
        if len(self.adj1) <= 1:
            return self.entry
        out = self._beam_std(self.adj1, self.entry1(), q, 8)
        return out[0][1] if out else self.entry

    def entry1(self):
        if not hasattr(self, "_e1"):
            self._e1 = next(iter(self.adj1))
        return self._e1

    def _beam_std(self, adj, entry, q, ef):
        d0 = self._dq(entry, q)
        visited = {entry}
        cand = [(d0, entry)]
        res = [(-d0, entry)]
        while cand:
            cd, cu = heapq.heappop(cand)
            if len(res) >= ef and cd > -res[0][0]:
                break
            for v in adj.get(cu, _EMPTY):
                v = int(v)
                if v in visited:
                    continue
                visited.add(v)
                dv = self._dq(v, q)
                if len(res) < ef or dv < -res[0][0]:
                    heapq.heappush(cand, (dv, v))
                    heapq.heappush(res, (-dv, v))
                    if len(res) > ef:
                        heapq.heappop(res)
        return sorted((-d, i) for d, i in res)

    def _beam_through(self, adj, entry, q, ef, vis):
        d0 = self._dq(entry, q)
        visited = {entry}
        cand = [(d0, entry)]
        res = [(-d0, entry)] if vis[entry] else []
        while cand:
            cd, cu = heapq.heappop(cand)
            if len(res) >= ef and cd > -res[0][0]:
                break
            for v in adj.get(cu, _EMPTY):
                v = int(v)
                if v in visited:
                    continue
                visited.add(v)
                dv = self._dq(v, q)
                if vis[v]:
                    if len(res) < ef or dv < -res[0][0]:
                        heapq.heappush(res, (-dv, v))
                        if len(res) > ef:
                            heapq.heappop(res)
                if len(res) < ef or dv < -res[0][0]:
                    heapq.heappush(cand, (dv, v))
        return sorted((-d, i) for d, i in res)

    def _beam_strict(self, adj, entry, q, ef, vis):
        d0 = self._dq(entry, q)
        visited = {entry}
        cand = [(d0, entry)]
        res = [(-d0, entry)]
        while cand:
            cd, cu = heapq.heappop(cand)
            if len(res) >= ef and cd > -res[0][0]:
                break
            for v in adj.get(cu, _EMPTY):
                v = int(v)
                if v in visited or not vis[v]:
                    continue
                visited.add(v)
                dv = self._dq(v, q)
                if len(res) < ef or dv < -res[0][0]:
                    heapq.heappush(cand, (dv, v))
                    heapq.heappush(res, (-dv, v))
                    if len(res) > ef:
                        heapq.heappop(res)
        return sorted((-d, i) for d, i in res)

    def _nearest_visible(self, q, vis):
        idx = np.nonzero(vis)[0]
        if idx.size == 0:
            return None
        sims = self.V[idx] @ q
        return int(idx[np.argmax(sims)])

    def search(self, adj, q, k, ef, vis=None, mode="through"):
        """Return [(dist, node)] of up to k results, nearest first.

        `adj` is a BASE-layer generation (dict node -> np.array).
        """
        if mode == "all" or vis is None:
            return self._beam_std(adj, self._descent(q), q, max(ef, k))[:k]
        if mode == "post":
            out = self._beam_std(adj, self._descent(q), q, ef)
            return [t for t in out if vis[t[1]]][:k]
        if mode == "through":
            return self._beam_through(adj, self._descent(q), q, ef, vis)[:k]
        if mode == "strict":
            e = self._nearest_visible(q, vis)
            if e is None:
                return []
            return self._beam_strict(adj, e, q, ef, vis)[:k]
        raise ValueError("unknown mode: %r" % mode)


# --------------------------------------------------------------------------- #
# maintenance: region rebuild = merge / compaction
# --------------------------------------------------------------------------- #
def rebuild_region(vecs, region_ids, adj, M, diverse=True):
    """Recompute adjacency for `region_ids` over region + existing neighbours.

    Models a repair-preserving destructive merge: the affected nodes' neighbour
    lists are recomputed on the merged union AUGMENTED with their current
    long-range links (so the graph stays healthy), and the previous adjacency
    is replaced. The candidate pool grows with each merge, so lists genuinely
    change -- which is the churn E6 measures.

    diverse=True applies the standard diversity prune; diverse=False keeps the
    plain top-M list (a coarser re-link pass).
    """
    region_ids = np.asarray(region_ids, dtype=np.int64)
    cand_set = set(region_ids.tolist())
    for i in region_ids.tolist():
        for v in adj.get(i, _EMPTY):
            cand_set.add(int(v))
    cand = np.asarray(sorted(cand_set), dtype=np.int64)
    C = vecs[cand]
    m_target = min(M, len(cand) - 1)
    out = {}
    for i in region_ids.tolist():
        sims = C @ vecs[i]
        order = np.argsort(-sims)
        if not diverse:
            chosen = []
            for oi in order:
                c = int(cand[oi])
                if c != i:
                    chosen.append(c)
                    if len(chosen) >= m_target:
                        break
            out[int(i)] = np.asarray(sorted(chosen), dtype=np.int32)
            continue
        chosen = []
        for oi in order:
            c = int(cand[oi])
            if c == i:
                continue
            d = 1.0 - float(sims[oi])
            ok = True
            for a in chosen:
                if 1.0 - float(vecs[c] @ vecs[a]) < d:
                    ok = False
                    break
            if ok:
                chosen.append(c)
                if len(chosen) >= m_target:
                    break
        if len(chosen) < m_target:           # fill with nearest remaining
            for oi in order:
                c = int(cand[oi])
                if c != i and c not in chosen:
                    chosen.append(c)
                    if len(chosen) >= m_target:
                        break
        out[int(i)] = np.asarray(sorted(chosen), dtype=np.int32)
    return out


def merged_adjacency(adj, region_new):
    """Copy-on-write: new generation shares unchanged arrays by reference."""
    out = dict(adj)
    out.update(region_new)
    return out


def reach_fraction(adj, vis, start):
    """BFS inside the visible subgraph; returns reachable / |visible|."""
    if start is None:
        return 0.0
    seen = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        for v in adj.get(u, _EMPTY):
            v = int(v)
            if v not in seen and vis[v]:
                seen.add(v)
                stack.append(v)
    n_vis = int(vis.sum())
    return len(seen) / max(1, n_vis)
