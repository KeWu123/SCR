#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
原型：混合关系-向量工作负载下的一致性检索（Snapshot-Consistent Retrieval, SCR）

用于验证论文核心论点：
  1) 现有 RAG 架构把向量索引当作与基表解耦的"最终一致"物化视图，
     LLM Agent 在同一个 episode 内同时读表与读索引，会产生跨存储的读异常
     （脏读未来版本 / 陈旧版本 / 读到已删除行 / 撕裂 episode）。
  2) "新鲜度"与"一致性"正交：可以做到"陈旧但一致"，
     在不牺牲一致性的前提下用有界陈旧换取低延迟。

三种对照（查询路径均已向量化以排除 Python 层开销对延迟测量的干扰）：
  A. AsyncIndex      —— 工业界常见的异步索引（滞后 L，允许跨存储异常）
  B. LockIndex       —— 全局锁串行化（正确，但读需等待在途嵌入）
  C. VersionedIndex  —— 本文方案：带提交时间戳的版本化向量索引 + 快照过滤
                        支持 exact-snapshot 与 bounded-staleness 两种模式

依赖：仅 numpy。
"""
import hashlib

import numpy as np

DIM = 64


# --------------------------------------------------------------------------- #
# 确定性嵌入函数（可复现；无需外部模型）
# --------------------------------------------------------------------------- #
def embed(text, dim=DIM):
    """Hashed bag-of-words -> signed random projection -> L2 normalize。"""
    vec = np.zeros(dim, dtype=np.float32)
    for t in text.lower().replace(",", " ").replace(".", " ").split():
        h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0 if (h >> 17) & 1 else -1.0
    n = np.linalg.norm(vec)
    if n > 1e-9:
        vec /= n
    return vec


class RowVersion:
    __slots__ = ("row_id", "ver_id", "commit_ts", "text", "deleted", "vec")

    def __init__(self, row_id, ver_id, commit_ts, text, deleted):
        self.row_id = row_id
        self.ver_id = ver_id
        self.commit_ts = commit_ts
        self.text = text
        self.deleted = deleted
        self.vec = embed(text)


# --------------------------------------------------------------------------- #
# 基表：MVCC
# --------------------------------------------------------------------------- #
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
# 公共：向量化"每行取 <=T 的最新版本"解析
# --------------------------------------------------------------------------- #
def _resolve_latest(cand_idx, rows, ts, vers=None):
    """Resolve each row by (timestamp, version id); storage ordinal is fallback."""
    if len(cand_idx) == 0:
        return cand_idx
    r = rows[cand_idx]
    t = ts[cand_idx]
    tie = cand_idx if vers is None else vers[cand_idx]
    order = np.lexsort((tie, t, r))
    ci = cand_idx[order]
    r = r[order]
    last = np.append(r[1:] != r[:-1], True)
    return ci[last]


def _topk(vecs, q, k):
    """返回 (选中下标, 对应相似度)，两者对齐。"""
    sims = vecs @ q
    k = min(k, len(sims))
    top = np.argpartition(-sims, k - 1)[:k]
    top = top[np.argsort(-sims[top])]
    return top, sims[top]


# --------------------------------------------------------------------------- #
# A. 异步索引（最终一致）
# --------------------------------------------------------------------------- #
class AsyncIndex:
    """追加式异步索引：写入 ts+L 之后才对读可见，且不回收旧版本。"""

    def __init__(self, lag):
        self.lag = lag
        self.eff = []      # 生效时间 = commit_ts + lag
        self.rows = []
        self.vers = []
        self.ts = []
        self.dele = []
        self.vecs = []
        self._dirty = False
        self._A = {}

    def on_write(self, rv: RowVersion):
        self.eff.append(rv.commit_ts + self.lag)
        self.rows.append(rv.row_id)
        self.vers.append(rv.ver_id)
        self.ts.append(rv.commit_ts)
        self.dele.append(rv.deleted)
        self.vecs.append(rv.vec)
        self._dirty = True

    def _arrays(self):
        if self._dirty:
            self._A = {
                "eff": np.asarray(self.eff, np.int64),
                "rows": np.asarray(self.rows, np.int64),
                "vers": np.asarray(self.vers, np.int64),
                "ts": np.asarray(self.ts, np.int64),
                "dele": np.asarray(self.dele, bool),
                "vecs": np.stack(self.vecs) if self.vecs else np.zeros((0, DIM), np.float32),
            }
            self._dirty = False
        return self._A

    def query(self, q, k, snapshot):
        A = self._arrays()
        if A["rows"].size == 0:
            return []
        cand = np.nonzero(A["eff"] <= snapshot)[0]
        cand = _resolve_latest(cand, A["rows"], A["eff"], A["vers"])
        cand = cand[~A["dele"][cand]]
        if cand.size == 0:
            return []
        top, scores = _topk(A["vecs"][cand], q, k)
        return [(int(A["rows"][cand[i]]), int(A["vers"][cand[i]]), float(scores[j]))
                for j, i in enumerate(top)]

    def index_bytes(self):
        A = self._arrays()
        return int(A["vecs"].nbytes + A["ts"].nbytes + A["vers"].nbytes)


# --------------------------------------------------------------------------- #
# B. 全局锁
# --------------------------------------------------------------------------- #
class LockIndex:
    """全局锁：写与读互斥，索引与基表始终同步（读需等待在途嵌入）。"""

    def __init__(self):
        self.vers = {}      # row_id -> ver_id
        self.vecs = {}      # row_id -> vec
        self._M = None
        self._ids = None
        self._dirty = True

    def on_write(self, rv: RowVersion):
        if rv.deleted:
            self.vers.pop(rv.row_id, None)
            self.vecs.pop(rv.row_id, None)
        else:
            self.vers[rv.row_id] = rv.ver_id
            self.vecs[rv.row_id] = rv.vec
        self._dirty = True

    def _arrays(self):
        if self._dirty:
            self._ids = np.asarray(list(self.vecs.keys()), np.int64)
            self._M = (np.stack([self.vecs[i] for i in self._ids])
                       if len(self._ids) else np.zeros((0, DIM), np.float32))
            self._dirty = False
        return self._M, self._ids

    def query(self, q, k, snapshot):
        M, ids = self._arrays()
        if ids.size == 0:
            return []
        top, scores = _topk(M, q, k)
        return [(int(ids[i]), int(self.vers[int(ids[i])]), float(scores[j]))
                for j, i in enumerate(top)]

    def index_bytes(self):
        M, ids = self._arrays()
        return int(M.nbytes + ids.nbytes)


# --------------------------------------------------------------------------- #
# C. 本文方案：版本化向量索引 + 快照过滤
# --------------------------------------------------------------------------- #
class VersionedIndex:
    """
    向量索引作为基表的物化视图，每个节点携带 (ver, commit_ts, deleted) 溯源。
    查询时按快照可见性过滤 + 每行解析到该快照下的最新可见版本。
    """

    def __init__(self, gc_enabled=True):
        self.rows, self.vers, self.ts, self.dele, self.vecs = [], [], [], [], []
        self._dirty = True
        self._A = {}
        self.gc_enabled = gc_enabled
        self.watermark = 0
        self.pruned = 0

    def on_write(self, rv: RowVersion):
        self.rows.append(rv.row_id)
        self.vers.append(rv.ver_id)
        self.ts.append(rv.commit_ts)
        self.dele.append(rv.deleted)
        self.vecs.append(rv.vec)
        self.watermark = max(self.watermark, rv.commit_ts)
        self._dirty = True

    def _arrays(self):
        if self._dirty:
            self._A = {
                "rows": np.asarray(self.rows, np.int64),
                "vers": np.asarray(self.vers, np.int64),
                "ts": np.asarray(self.ts, np.int64),
                "dele": np.asarray(self.dele, bool),
                "vecs": np.stack(self.vecs) if self.vecs else np.zeros((0, DIM), np.float32),
            }
            self._dirty = False
        return self._A

    def gc(self, oldest_active_snapshot):
        """版本回收：每行只保留 <= s_min 的最新版本 + 所有 > s_min 的版本。"""
        if not self.gc_enabled or len(self.ts) < 3:
            return
        A = self._arrays()
        keep = []
        by_row = {}
        for i, rid in enumerate(A["rows"].tolist()):
            by_row.setdefault(rid, []).append(i)
        for rid, idxs in by_row.items():
            older = [i for i in idxs if A["ts"][i] <= oldest_active_snapshot]
            newer = [i for i in idxs if A["ts"][i] > oldest_active_snapshot]
            if older:
                keep.append(max(older, key=lambda i: (A["ts"][i], A["vers"][i])))
                self.pruned += len(older) - 1
            keep.extend(newer)
        keep.sort()
        if len(keep) < len(self.ts):
            self.rows = [self.rows[i] for i in keep]
            self.vers = [self.vers[i] for i in keep]
            self.ts = [self.ts[i] for i in keep]
            self.dele = [self.dele[i] for i in keep]
            self.vecs = [self.vecs[i] for i in keep]
            self._dirty = True

    def query(self, q, k, snapshot):
        A = self._arrays()
        if A["rows"].size == 0:
            return []
        cand = np.nonzero(A["ts"] <= snapshot)[0]
        cand = _resolve_latest(cand, A["rows"], A["ts"], A["vers"])
        cand = cand[~A["dele"][cand]]
        if cand.size == 0:
            return []
        top, scores = _topk(A["vecs"][cand], q, k)
        return [(int(A["rows"][cand[i]]), int(A["vers"][cand[i]]), float(scores[j]))
                for j, i in enumerate(top)]

    def index_bytes(self):
        A = self._arrays()
        return int(A["vecs"].nbytes + A["ts"].nbytes + A["vers"].nbytes + A["dele"].nbytes)
