#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SCR v2 experiment skeleton -- shared utilities (numpy only).

Conventions mirror paper/proto/ so numbers stay comparable across v1/v2:
cosine similarity, commit-timestamped row versions, deterministic embeddings.
"""
from __future__ import annotations

import hashlib

import numpy as np

DIM = 64


# --------------------------------------------------------------------------- #
# embeddings / corpora
# --------------------------------------------------------------------------- #
def embed(text: str, dim: int = DIM) -> np.ndarray:
    """Deterministic hashed bag-of-words projection (same as the v1 prototype)."""
    vec = np.zeros(dim, dtype=np.float32)
    for t in text.lower().replace(",", " ").replace(".", " ").split():
        h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0 if (h >> 17) & 1 else -1.0
    n = np.linalg.norm(vec)
    if n > 1e-9:
        vec /= n
    return vec


def make_text(rng, vocab=24, length=12):
    return " ".join("w%d" % rng.integers(0, vocab) for _ in range(length))


def clustered_vectors(n, dim=DIM, n_clusters=40, spread=0.35, rng=None):
    """Clustered unit vectors: realistic neighbourhood structure for ANN graphs.

    Pure-random vectors have no community structure; navigability would be
    meaningless on them. Clustered data is the standard cheap stand-in.
    """
    rng = rng or np.random.default_rng(0)
    centers = rng.normal(size=(n_clusters, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    assign = rng.integers(0, n_clusters, size=n)
    pts = centers[assign] + spread * rng.normal(size=(n, dim)).astype(np.float32)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts.astype(np.float32)


# --------------------------------------------------------------------------- #
# MVCC base table (row versions)
# --------------------------------------------------------------------------- #
class RowVersion:
    __slots__ = ("row_id", "ver_id", "commit_ts", "text", "deleted", "vec")

    def __init__(self, row_id, ver_id, commit_ts, text, deleted):
        self.row_id = row_id
        self.ver_id = ver_id
        self.commit_ts = commit_ts
        self.text = text
        self.deleted = deleted
        self.vec = embed(text) if not deleted else np.zeros(DIM, dtype=np.float32)


class BaseTable:
    def __init__(self):
        self.versions = []
        self._next_ver = 0
        self.latest = {}

    def _new_ver(self, row_id, text, ts, deleted=False):
        v = RowVersion(row_id, self._next_ver, ts, text, deleted)
        self._next_ver += 1
        self.versions.append(v)
        if not deleted:
            self.latest[row_id] = v.ver_id
        return v

    def insert(self, row_id, text, ts):
        return self._new_ver(row_id, text, ts)

    def update(self, row_id, text, ts):
        return self._new_ver(row_id, text, ts)

    def delete(self, row_id, ts):
        return self._new_ver(row_id, "", ts, deleted=True)


# --------------------------------------------------------------------------- #
# small stats helpers
# --------------------------------------------------------------------------- #
def pct(arr, q):
    return float(np.percentile(arr, q)) if len(arr) else float("nan")


def jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, len(sa | sb))


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial rate (report as mean [lo, hi])."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - r) / d, (c + r) / d)


def mean_std(xs):
    a = np.asarray(xs, dtype=float)
    return float(a.mean()), float(a.std(ddof=0))


# --------------------------------------------------------------------------- #
# memory accounting for generations (copy-on-write aware)
# --------------------------------------------------------------------------- #
def unique_adj_bytes(frames):
    """Sum nbytes over *unique* adjacency arrays across retained generations.

    Copy-on-write: arrays shared between generations are counted once.
    """
    seen = set()
    total = 0
    for f in frames:
        for a in f.values():
            if id(a) not in seen:
                seen.add(id(a))
                total += int(a.nbytes)
    return total
