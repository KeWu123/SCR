#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Operational scheduling baselines for the SCR isolation claim.

Alongside asynchronous indexing and a global read lock, this driver evaluates
two maintenance policies:

  (B1) quiesce      -- never run maintenance/compaction while an episode is in
                       flight; defer the due compaction, keep serving the old
                       adjacency generation.  Zero extra space, at the cost of
                       delaying maintenance.
  (B2) double-buffer-- run maintenance on a private copy and swap the whole
                       index in atomically at a point where no episode spans
                       the swap (whole-index copy-on-write).

This driver adds admission and maintenance scheduling to the corrected
node-version episode protocol in ``run_episodes_v3.py`` and measures, over
the same seeds/config:

  * anomaly episode rate (mean +/- sd, per seed, Wilson 95% upper bound when 0)
  * p50 / p99 retrieval latency
  * maintenance delay -- defined below and reported for every
    scheme, including async and SCR
  * space overhead relative to a non-versioned index
  * a clearly-labelled single-process simulator throughput proxy

Maintenance delay (definition)
------------------------------
A *maintenance point* is a compaction that has become due: the number of commits
that have elapsed since the last applied compaction reaches a period P (we use
P = 250 commits, i.e. ~N_writes/P = 20 compactions per run).  When a maintenance
point comes due at logical time t_due, it is *applied* at time t_apply.  The
**maintenance delay** of that point is ``max(0, t_apply - t_due)`` measured in
commits (the driver's logical clock, 1 commit = 1 write).  We report the mean and
the maximum over all maintenance points, plus a *terminal* count: maintenance
points still unapplied when the run ends (for deferring schemes under sustained
load).  For schemes that never defer (async, SCR, double-buffer) the delay is
~0 by construction; for quiesce it is the whole point.

How each scheme relates to compacting maintenance
-------------------------------------------------
* **async**         The index is append-only and *never compacts*; it is
                     permanently unbounded-stale but has no maintenance point,
                     so we define its maintenance delay as 0 and note that this
                     is why it is the anomaly-ridden scheme.  (We keep async as
                     the "do nothing" reference the paper already uses.)
* **lock**          A global lock serialises reads against in-flight embedding
                     work; the index is always the current, compacted state.
                     Compaction runs inline, delay 0, but every retrieval waits.
* **scr**           Compaction is staged and published at epoch boundaries; the
                     published generation is pinned per snapshot.  Delay ~0 (the
                     publication is immediate, only the *visibility* is
                     versioned) and zero anomalies by construction.
* **quiesce** (B1)  A due compaction is *deferred* while any episode is active.
                     It is applied (a) only once the active-episode count is 0,
                     and (b) with a self-draining guard so that a backlog of
                     long episodes cannot starve maintenance forever.  We report
                     the deferral.  Space cost: none (the old generation is the
                     only one).
* **double_buffer** (B2)
                     Maintenance builds a private copy of the whole index (the
                     private copy is the space cost) and swaps it in atomically.
                     Policy (stated): an episode that is *active at the swap
                     point* is **delayed** until the swap completes -- i.e. the
                     swap is only performed when no episode spans it, which is
                     exactly the "no active episode spans the swap" condition.
                     We therefore delay both the swap and the overlapping
                     episodes, and report the episode-side extra latency.

Scope
-----
The prototype runs episodes one at a time (the read loop is a single Python
thread), so "an episode is active" is a single boolean; there is no real
concurrency in the reader.  This means B1/B2 are evaluated on a *conservative*
schedule: at most one episode can be active at any instant, so quiesce defers
compaction for one active episode (think window plus its tool work),
and double-buffer delays at most one episode.  A real multi-reader deployment
can defer more; this driver does not measure that deployment setting.

These exact node-version indices do not implement graph-generation pinning.
Space policy factors mix entry counts and separate T2 array-payload accounting;
they are not measured process-memory ratios across implementations.
The legacy a4/a5 JSON keys mean own-row conflict/citation disagreement only.

Run: python run_baselines.py (writes results_baselines_corrected.json).
The original results_baselines.json is preserved as a historical artifact.
"""
from __future__ import annotations

import json
import os
import random
import statistics
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from engine import AsyncIndex, BaseTable, DIM, LockIndex, VersionedIndex, embed  # noqa: E402
from run_experiments import (Clock, TruthCache, build, classify_fast,  # noqa: E402
                             make_text, truth_topk_fast)

OUT = os.path.join(HERE, "results_baselines_corrected.json")

# --------------------------------------------------------------------------- #
# Protocol constants (identical to run_episodes_v3.py's main())
# --------------------------------------------------------------------------- #
N_ROWS = 15000
N_WRITES = 5000
N_READS = 1200          # episodes
K = 10
SEEDS = [20261008, 20261009, 20261010]
THINK_US = 20000.0      # 20 ms LLM "think" window; writes commit during it
EMBED_COST_US = 150.0
COMPACT_PERIOD = 250    # commits between due maintenance points (~20 per run)
SEED_RNG = 424242       # seeds the ANNs search-quality perturbation only


# --------------------------------------------------------------------------- #
# Wilson 95% upper bound for a binomial proportion (used when the observed
# count is 0, where the normal approximation is undefined).
# --------------------------------------------------------------------------- #
def wilson_upper(k, n, z=1.959963984540054):
    """One-sided (upper) Wilson score bound for a proportion k/n."""
    if n <= 0:
        return 1.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * (z * z / (4 * n * n) + p * (1 - p) / n) ** 0.5
    return min(1.0, centre + half)


# --------------------------------------------------------------------------- #
# RealLock: reproduce the writer/reader atomicity of run_episodes_v3.py so that
# the Python-level append/rebuild race does not manufacture spurious anomalies.
# --------------------------------------------------------------------------- #
class RealLock:
    def __init__(self):
        self._l = threading.Lock()

    def __enter__(self):
        self._l.acquire()
        return self

    def __exit__(self, *a):
        self._l.release()
        return False


# --------------------------------------------------------------------------- #
# RowIndex: as-of relational read, O(log n), incrementally refreshed -- a direct
# copy of the one in run_episodes_v3.py so the episode protocol is identical.
# --------------------------------------------------------------------------- #
from bisect import bisect_right  # noqa: E402


class RowIndex:
    def __init__(self, table: BaseTable):
        self.by_row = {}
        self.ts_cache = {}
        self._seen = 0
        self.refresh(table)

    def refresh(self, table: BaseTable):
        vers = table.versions
        end = len(vers)
        for i in range(self._seen, end):
            v = vers[i]
            lst = self.by_row.setdefault(v.row_id, [])
            lst.append(v)
            if len(lst) >= 2 and lst[-2].commit_ts > v.commit_ts:
                lst.sort(key=lambda x: x.commit_ts)
            self.ts_cache[v.row_id] = [x.commit_ts for x in lst]
        self._seen = end

    def read_at(self, r, ts):
        lst = self.by_row.get(r)
        if not lst:
            return None
        i = bisect_right(self.ts_cache[r], ts) - 1
        if i < 0:
            return None
        v = lst[i]
        return None if v.deleted else v


# --------------------------------------------------------------------------- #
# (B1) QuiesceIndex
# --------------------------------------------------------------------------- #
class QuiesceIndex:
    """Defer compaction while an episode is active.

    A compaction point comes due every ``period`` commits.  While a compaction
    is due and one or more episodes are active, the index keeps serving the
    *current* (uncompacted) adjacency -- the index is simultaneously the table's
    materialised view and is compacted in place by ``_apply_compaction``.

    To avoid the pathological "an endless stream of episodes starves
    maintenance forever" case we apply a **drain guard**: a due compaction is
    forced through once it has been deferred for ``max_defer`` consecutive
    commits, even if an episode is active.  The guard is itself part of the
    baseline's semantics and is reported.

    Space: exactly one copy of the index, no extra retained state.
    """

    def __init__(self, period=COMPACT_PERIOD, max_defer=None):
        self.period = period
        # Drain guard: defer at most this many commits (None -> never force,
        # which models the un-guarded "no compaction while episodes run" rule).
        self.max_defer = max_defer
        self.rows, self.vers, self.ts, self.dele, self.vecs = [], [], [], [], []
        self._dirty = True
        self._A = {}
        # maintenance bookkeeping (all in commit-time units)
        self.watermark = 0            # newest commit time seen
        self.due_at = None            # commit time at which a point became due
        self.last_applied = 0         # commit time of last applied compaction
        self.n_due = 0                # maintenance points that have come due
        self.n_applied = 0            # maintenance points actually applied
        self.delays = []              # per applied point: apply - due (>= 0)
        self.deferred_now = 0         # consecutive commits spent deferring
        self.anomalies = 0            # episodes torn at apply time (should be 0)
        self.applied_during_episode = 0

    # ---- write path ------------------------------------------------------- #
    def on_write(self, rv):
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

    # ---- maintenance ------------------------------------------------------ #
    def tick_maintenance(self, now, episode_active):
        """Called once per episode boundary and once per writer commit.

        Returns the number of maintenance points applied by this call.
        """
        # Has a new point come due?
        if self.due_at is None and (now - self.last_applied) >= self.period:
            self.due_at = self.last_applied + self.period
            self.n_due += 1
            self.deferred_now = 0

        if self.due_at is None:
            return 0

        forced = self.max_defer is not None and self.deferred_now >= self.max_defer
        if (not episode_active) or forced:
            delay = max(0, now - self.due_at)
            self.delays.append(delay)
            self._apply_compaction()
            self.last_applied = now
            self.n_applied += 1
            self.due_at = None
            self.deferred_now = 0
            return 1
        # defer: episode is active
        self.deferred_now += 1
        return 0

    def _apply_compaction(self):
        """Compact in place: for each row keep only its latest version.

        This is the destructive maintenance the anomaly model turns on; while it
        is deferred the index still exposes every version.
        """
        A = self._arrays()
        if A["rows"].size == 0:
            return
        latest = {}
        for i, rid in enumerate(A["rows"].tolist()):
            j = latest.get(rid)
            if j is None or A["ts"][i] >= A["ts"][j]:
                latest[rid] = i
        keep = sorted(latest.values())
        if len(keep) < len(self.ts):
            self.rows = [self.rows[i] for i in keep]
            self.vers = [self.vers[i] for i in keep]
            self.ts = [self.ts[i] for i in keep]
            self.dele = [self.dele[i] for i in keep]
            self.vecs = [self.vecs[i] for i in keep]
            self._dirty = True

    # ---- read path -------------------------------------------------------- #
    def query(self, q, k, snapshot):
        A = self._arrays()
        if A["rows"].size == 0:
            return []
        cand = np.nonzero(A["ts"] <= snapshot)[0]
        if cand.size == 0:
            return []
        # resolve to the latest visible version per row (identical to VersionedIndex)
        r = A["rows"][cand]
        t = A["ts"][cand]
        order = np.lexsort((t, r))
        ci = cand[order]
        rr = r[order]
        last = np.append(rr[1:] != rr[:-1], True)
        cand = ci[last]
        cand = cand[~A["dele"][cand]]
        if cand.size == 0:
            return []
        sims = A["vecs"][cand] @ q
        kk = min(k, len(sims))
        top = np.argpartition(-sims, kk - 1)[:kk]
        top = top[np.argsort(-sims[top])]
        return [(int(A["rows"][cand[i]]), int(A["vers"][cand[i]]), float(sims[i]))
                for i in top]

    def index_bytes(self):
        A = self._arrays()
        return int(A["vecs"].nbytes + A["ts"].nbytes + A["vers"].nbytes + A["dele"].nbytes)

    def space_factor(self, native_bytes):
        # exactly one copy of the index
        return self.index_bytes() / max(1, native_bytes)


# --------------------------------------------------------------------------- #
# (B2) DoubleBufferIndex
# --------------------------------------------------------------------------- #
class DoubleBufferIndex:
    """Whole-index copy-on-write.

    Maintenance builds a *private* new generation (a full copy of the current
    index, then compacted) and swaps it in atomically.  Policy implemented:

      * a swap is only performed at a moment when no episode spans it.  In this
        single-reader prototype that means "only when no episode is active";
      * an episode that would span the swap is DELAYED until the swap completes;
        we charge the extra latency to that episode.

    Space: the private copy is a second full copy of the index while a swap is
    pending, so the retained footprint is up to 2x the non-versioned index.
    """

    def __init__(self, period=COMPACT_PERIOD):
        self.period = period
        self.live = None              # dict snapshot of the published index
        self.priv = None              # private build in progress (or None)
        self._dirty_arrays = True
        self._A = {}
        self.watermark = 0
        self.due_at = None
        self.last_applied = 0
        self.n_due = 0
        self.n_applied = 0
        self.delays = []
        self.swap_episode_delays_us = []   # episode extra latency charged
        self.priv_rows, self.priv_vers, self.priv_ts, self.priv_dele, self.priv_vecs = \
            [], [], [], [], []
        # the live index starts as a plain list of entries
        self.rows, self.vers, self.ts, self.dele, self.vecs = [], [], [], [], []

    def on_write(self, rv):
        # writes land in the live index (double-buffer only protects the swap)
        self.rows.append(rv.row_id)
        self.vers.append(rv.ver_id)
        self.ts.append(rv.commit_ts)
        self.dele.append(rv.deleted)
        self.vecs.append(rv.vec)
        self.watermark = max(self.watermark, rv.commit_ts)
        self._dirty_arrays = True
        # A private build must catch up with all writes before publication.
        # Swapping an old private copy without this tail would discard committed
        # updates and manufacture stale/phantom anomalies unrelated to pinning.
        if self.due_at is not None and self.priv_rows is not None:
            self.priv_rows.append(rv.row_id)
            self.priv_vers.append(rv.ver_id)
            self.priv_ts.append(rv.commit_ts)
            self.priv_dele.append(rv.deleted)
            self.priv_vecs.append(rv.vec)

    def _arrays(self):
        if self._dirty_arrays:
            self._A = {
                "rows": np.asarray(self.rows, np.int64),
                "vers": np.asarray(self.vers, np.int64),
                "ts": np.asarray(self.ts, np.int64),
                "dele": np.asarray(self.dele, bool),
                "vecs": np.stack(self.vecs) if self.vecs else np.zeros((0, DIM), np.float32),
            }
            self._dirty_arrays = False
        return self._A

    # ---- maintenance ------------------------------------------------------ #
    def build_private(self):
        """Start building a compacted private copy (the space cost is realised
        here: ``priv_*`` is a second full copy of the live index)."""
        A = self._arrays()
        latest = {}
        for i, rid in enumerate(A["rows"].tolist()):
            j = latest.get(rid)
            if j is None or A["ts"][i] >= A["ts"][j]:
                latest[rid] = i
        keep = sorted(latest.values())
        self.priv_rows = [self.rows[i] for i in keep]
        self.priv_vers = [self.vers[i] for i in keep]
        self.priv_ts = [self.ts[i] for i in keep]
        self.priv_dele = [self.dele[i] for i in keep]
        self.priv_vecs = [self.vecs[i] for i in keep]

    def can_swap(self, episode_active):
        """The swap requires that no episode spans it: no episode is active."""
        return self.priv_rows is not None and not episode_active

    def do_swap(self, now):
        """Atomic swap: the private copy becomes the live index."""
        self.rows = self.priv_rows
        self.vers = self.priv_vers
        self.ts = self.priv_ts
        self.dele = self.priv_dele
        self.vecs = self.priv_vecs
        self._dirty_arrays = True
        self.priv_rows = None
        self.last_applied = now
        self.n_applied += 1
        self.due_at = None

    def tick_maintenance(self, now, episode_active):
        if self.due_at is None and (now - self.last_applied) >= self.period:
            self.due_at = self.last_applied + self.period
            self.n_due += 1
            # begin the private build immediately (space cost incurred now)
            self.build_private()
        if self.due_at is None:
            return 0
        if self.can_swap(episode_active):
            delay = max(0, now - self.due_at)
            self.delays.append(delay)
            self.do_swap(now)
            return 1
        return 0

    # ---- read path -------------------------------------------------------- #
    def query(self, q, k, snapshot):
        A = self._arrays()
        if A["rows"].size == 0:
            return []
        cand = np.nonzero(A["ts"] <= snapshot)[0]
        if cand.size == 0:
            return []
        r = A["rows"][cand]
        t = A["ts"][cand]
        order = np.lexsort((t, r))
        ci = cand[order]
        rr = r[order]
        last = np.append(rr[1:] != rr[:-1], True)
        cand = ci[last]
        cand = cand[~A["dele"][cand]]
        if cand.size == 0:
            return []
        sims = A["vecs"][cand] @ q
        kk = min(k, len(sims))
        top = np.argpartition(-sims, kk - 1)[:kk]
        top = top[np.argsort(-sims[top])]
        return [(int(A["rows"][cand[i]]), int(A["vers"][cand[i]]), float(sims[i]))
                for i in top]

    def index_bytes(self):
        A = self._arrays()
        live = int(A["vecs"].nbytes + A["ts"].nbytes + A["vers"].nbytes + A["dele"].nbytes)
        if self.priv_rows is not None:            # private copy co-resident
            pv = (np.stack(self.priv_vecs) if self.priv_vecs
                  else np.zeros((0, DIM), np.float32))
            live += int(pv.nbytes + 8 * len(self.priv_ts) * 2 + len(self.priv_dele))
        return live

    def peak_space_factor(self, native_bytes):
        """Peak retained footprint / non-versioned footprint.  While a private
        copy is pending this is (up to) 2x; the report quotes this peak."""
        return 2.0


# --------------------------------------------------------------------------- #
# Episode driver
# --------------------------------------------------------------------------- #
def run_episode_workload(table, index, scheme, rng, n_writes, n_reads, k,
                         embed_cost_us=EMBED_COST_US, think_us=THINK_US,
                         max_row=None, np_seed=0, period=COMPACT_PERIOD,
                         quiesce_max_defer=None):
    lock = RealLock()
    clock = Clock(t=n_writes)
    tc = TruthCache(table)
    max_row = max_row or (max(table.latest.keys()) + 1 if table.latest else 1)
    nprng = np.random.default_rng(np_seed)
    writer_rng = random.Random(rng.getrandbits(64))
    reader_rng = random.Random(rng.getrandbits(64))
    is_quiesce = isinstance(index, QuiesceIndex)
    is_dbuf = isinstance(index, DoubleBufferIndex)
    maint = is_quiesce or is_dbuf
    if maint:
        # The initial index is a complete checkpoint at the starting head.
        # The artificial clock offset is not 5,000 elapsed workload commits.
        index.last_applied = clock.t
        index.watermark = max(index.watermark, clock.t)

    lat, wait = [], []
    n_ep = 0
    n_a1 = n_a2 = n_a3 = n_a4 = n_a5 = n_anom_any = 0
    # A4 split by direction, so the report can say WHY a scheme is torn:
    #   dirty = index served a version NEWER than S (embedding lag / unprotected
    #           live index)      stale = index served a version OLDER than S
    #           (deferred compaction / a generation not yet visible)
    #   deleted = index returned a row the episode's read says is gone
    n_a4_dirty = n_a4_stale = n_a4_deleted = 0
    diffs, jac = [], []
    rec1_num = rec1_den = 0
    ep_active = [False]                 # single-reader: at most one episode active
    ep_extra_us = []                    # extra latency charged for a swap wait
    stop = threading.Event()
    writer_errors = []

    def writer():
        c = 0
        try:
            while c < n_writes and not stop.is_set():
                rid = writer_rng.randrange(max_row)
                with lock:
                    if writer_rng.random() < 0.12:
                        rv = table.delete(rid, ts=clock.tick())
                    else:
                        rv = table.update(rid, make_text(writer_rng), ts=clock.tick())
                    index.on_write(rv)
                    if maint:
                        index.tick_maintenance(rv.commit_ts, ep_active[0])
                c += 1
                if embed_cost_us:
                    time.sleep(embed_cost_us / 1e6)
        except BaseException as exc:
            writer_errors.append(exc)

    wt = threading.Thread(target=writer, daemon=True)
    wt.start()

    with lock:
        ri = RowIndex(table)
    t_ep_start = time.perf_counter()
    while n_ep < n_reads:
        # Publication, maintenance admission, Begin and the actual relation
        # observation share one atomic section. Scoring is a separate oracle.
        with lock:
            if is_dbuf:
                if index.due_at is not None and index.priv_rows is not None:
                    w0 = time.perf_counter_ns()
                    index.tick_maintenance(index.watermark, False)
                    ep_extra_us.append((time.perf_counter_ns() - w0) / 1000.0)

            ep_active[0] = True
            ri.refresh(table)
            S_now = clock.t
            S = S_now
            r = reader_rng.randrange(max_row)
            v_tab = ri.read_at(r, S)

        # ---- step 3: query derives from that read ------------------------- #
        if v_tab is not None:
            q = v_tab.vec.copy()
        else:
            q = embed(make_text(reader_rng))
        q = q + 0.02 * nprng.normal(size=q.shape).astype(np.float32)
        q /= np.linalg.norm(q)

        if think_us:
            time.sleep(think_us / 1e6)      # writes commit; maintenance may defer

        # ---- step 5/6: two retrievals in the same episode ----------------- #
        w0 = time.perf_counter_ns()
        with lock:
            w1 = time.perf_counter_ns()
            K1 = index.query(q, k, S)
            dt = (time.perf_counter_ns() - w1) / 1000.0
        time.sleep(0.001)
        with lock:
            w2 = time.perf_counter_ns()
            K2 = index.query(q, k, S)
            dt += (time.perf_counter_ns() - w2) / 1000.0
        wait.append((w1 - w0) / 1000.0)
        lat.append(dt)
        n_ep += 1
        with lock:
            ep_active[0] = False

        # B1: an episode just ended; this is when a deferred compaction can go.
        if is_quiesce:
            with lock:
                index.tick_maintenance(index.watermark, False)

        # ---- A1/A2/A3 over both result sets ------------------------------- #
        with lock:
            d1 = classify_fast(K1, tc, S)
            d2 = classify_fast(K2, tc, S)
            tk = truth_topk_fast(tc, q, S, k)
        ep_a1 = int((d1[1] + d2[1]) > 0)
        ep_a2 = int((d1[2] + d2[2]) > 0)
        ep_a3 = int((d1[3] + d2[3]) > 0)
        n_a1 += ep_a1
        n_a2 += ep_a2
        n_a3 += ep_a3

        # Own-row conflict only, not a general no-common-cut A4 certificate.
        torn = False
        for rid_, vid_, _ in set((x[0], x[1], None) for x in list(K1) + list(K2)):
            if rid_ != r:
                continue
            v_t = v_tab
            if v_t is None:
                n_a4_deleted += 1
                torn = True
            elif v_t.ver_id != vid_:
                if vid_ > v_t.ver_id:
                    n_a4_dirty += 1        # index ahead of the snapshot
                else:
                    n_a4_stale += 1        # index behind the snapshot
                torn = True
        if torn:
            n_a4 += 1

        # Citation disagreement only; no adjacency-level A5 instrumentation.
        s1 = {x[0] for x in K1}
        s2 = {x[0] for x in K2}
        c1 = {(x[0], x[1]) for x in K1}
        c2 = {(x[0], x[1]) for x in K2}
        if c1 != c2:
            n_a5 += 1
        diffs.append(len(s1 ^ s2))
        uni = len(s1 | s2)
        jac.append(len(s1 & s2) / uni if uni else 1.0)

        if (ep_a1 or ep_a2 or ep_a3) or torn or c1 != c2:
            n_anom_any += 1

        rec1_num += len(s1 & tk)
        rec1_den += len(tk)

    stop.set()
    wt.join(timeout=10)
    if wt.is_alive():
        raise RuntimeError("writer did not terminate; refusing incomplete result")
    if writer_errors:
        raise RuntimeError("writer failed; refusing incomplete result") from writer_errors[0]
    ep_wall = time.perf_counter() - t_ep_start
    lat.sort()
    wait.sort()

    def pct(a, p):
        return a[min(len(a) - 1, int(len(a) * p))] if a else 0.0

    n = max(1, n_ep)
    res = {
        "episodes": n_ep,
        "writes_completed": clock.t - n_writes,
        "anomaly_episode_rate": round(n_anom_any / n, 6),
        "n_anomaly_episodes": int(n_anom_any),
        "a1_dirty_episode_rate": round(n_a1 / n, 6),
        "a2_superseded_episode_rate": round(n_a2 / n, 6),
        "a3_phantom_episode_rate": round(n_a3 / n, 6),
        "a4_torn_episode_rate": round(n_a4 / n, 6),
        "observed_read_conflict_rate": round(n_a4 / n, 6),
        "a4_dirty_disagreements": int(n_a4_dirty),
        "a4_stale_disagreements": int(n_a4_stale),
        "a4_deleted_disagreements": int(n_a4_deleted),
        "a5_retrieval_disagreement_rate": round(n_a5 / n, 6),
        "citation_disagreement_rate": round(n_a5 / n, 6),
        "n_a5_disagreement_episodes": int(n_a5),
        "mean_symmetric_difference": round(float(np.mean(diffs)), 6),
        "mean_jaccard_k1_k2": round(float(np.mean(jac)), 6),
        "min_jaccard_k1_k2": round(float(np.min(jac)), 6),
        "recall_at_k": round(rec1_num / max(1, rec1_den), 6),
        "p50_us": round(pct(lat, 0.5), 1),
        "p99_us": round(pct(lat, 0.99), 1),
        "lock_wait_p50_us": round(pct(wait, 0.5), 1),
        "lock_wait_p99_us": round(pct(wait, 0.99), 1),
        "episode_wall_s": round(ep_wall, 3),
        "throughput_ep_per_s": round(n_ep / ep_wall, 2),
        "scheme": scheme,
    }

    # per-scheme maintenance / space bookkeeping
    def _delay_stats(delays):
        """Report all applied points; exclude-first fields are sensitivity only.

        The maintenance clock starts at the complete initial checkpoint, so
        the first point is not an artificial pre-workload startup outlier.
        """
        if not delays:
            return {}
        srt = sorted(delays)
        rest = delays[1:] if len(delays) > 1 else delays
        return {
            "maintenance_delay_mean_commits": round(float(np.mean(srt)), 3),
            "maintenance_delay_max_commits": int(srt[-1]),
            "maintenance_delay_median_commits": round(float(np.median(srt)), 3),
            "maintenance_delay_p99_commits": int(pct(srt, 0.99)),
            "maintenance_delay_first_applied_commits": int(delays[0]),
            "maintenance_delay_max_excl_first_commits": int(max(rest)),
            "maintenance_delay_mean_excl_first_commits": round(float(np.mean(rest)), 3),
            "maintenance_delay_n_applied": len(srt),
        }

    if is_quiesce:
        res.update(_delay_stats(index.delays))
        res.update({
            "maintenance_compactions_due": index.n_due,
            "maintenance_compactions_applied": index.n_applied,
            "maintenance_points_unapplied_at_end": index.n_due - index.n_applied,
        })
    elif is_dbuf:
        res.update(_delay_stats(index.delays))
        res.update({
            "maintenance_compactions_due": index.n_due,
            "maintenance_compactions_applied": index.n_applied,
            "maintenance_points_unapplied_at_end": index.n_due - index.n_applied,
            "swap_episode_delay_mean_us": round(float(np.mean(ep_extra_us)), 1) if ep_extra_us else 0.0,
            "swap_episode_delay_max_us": round(float(np.max(ep_extra_us)), 1) if ep_extra_us else 0.0,
            "swap_episode_delay_n": len(ep_extra_us),
        })
    else:
        # No compacting maintenance operator is modeled for these exact indices.
        # Do not manufacture applied point counts from n_writes // period.
        res.update({
            "maintenance_compactions_due": 0,
            "maintenance_compactions_applied": 0,
            "maintenance_scope": "no explicit maintenance operator; zero delay is bookkeeping",
            "maintenance_delay_mean_commits": 0.0,
            "maintenance_delay_max_commits": 0,
            "maintenance_delay_p99_commits": 0,
            "maintenance_points_unapplied_at_end": 0,
        })
    return res


# --------------------------------------------------------------------------- #
# Multi-reader concurrency model -- DOES quiesce's deferral accumulate?
# --------------------------------------------------------------------------- #
def quiesce_multireader(n_readers=8, n_life_s=8.0, think_us=THINK_US,
                        period=COMPACT_PERIOD, embed_cost_us=EMBED_COST_US,
                        n_rows=3000, seed=1):
    """Run ``n_readers`` episode threads concurrently against a quiesce index.

    The single-reader driver cannot manufacture reader saturation: there is
    always a gap between consecutive episodes in which quiesce may compact, so
    deferral stays bounded.  With R concurrent readers the backlog of *active*
    episodes is what matters, and the window in which zero episodes are active
    shrinks.  We track the active-episode COUNT (not a boolean), which is the
    semantics a real quiesce policy needs.

    We compare, at the same reader count, the two deferral regimes:
      * max_defer=None  -- pure "never compact while any episode is active";
        this is the regime that can starve;
      * max_defer=k     -- a drain guard that forces every k-th deferral through.

    The writer advances the logical clock; each reader thread pulls its own
    episode (Begin -> read -> think -> two retrievals) against the shared index.
    """
    import random as _random
    import threading as _threading
    rng = _random.Random(seed)
    table, _ = build(n_rows, rng)
    lock = RealLock()
    clock = Clock(t=0)
    stop = _threading.Event()
    active = [0]
    active_lock = _threading.Lock()

    def writer():
        lr = _random.Random(seed + 1)
        while not stop.is_set():
            rid = lr.randrange(n_rows)
            with lock:
                if lr.random() < 0.12:
                    rv = table.delete(rid, ts=clock.tick())
                else:
                    rv = table.update(rid, make_text(lr), ts=clock.tick())
                idx.on_write(rv)
                idx.tick_maintenance(rv.commit_ts, active[0] > 0)
            if embed_cost_us:
                time.sleep(embed_cost_us / 1e6)

    holder = {}          # shared handles the reader threads close over

    def reader(sid):
        lr = _random.Random(seed * 1000 + sid)
        while not stop.is_set():
            with lock:
                active[0] += 1
                holder["ri"].refresh(table)
                S = clock.t
                r = lr.randrange(n_rows)
                vt = holder["ri"].read_at(r, S)
            try:
                q = vt.vec if vt is not None else embed(make_text(lr))
                time.sleep(think_us / 1e6)
                with lock:
                    holder["idx"].query(q, 10, S)
                time.sleep(0.001)
                with lock:
                    holder["idx"].query(q, 10, S)
            finally:
                with lock:
                    active[0] -= 1
                    if active[0] == 0:
                        idx.tick_maintenance(clock.t, False)

    results = {}
    for max_defer in (None, period, period // 2):
        # fresh index/table per regime so the comparison is clean
        rng = _random.Random(seed)
        table, _ = build(n_rows, rng)
        clock = Clock(t=0)
        stop = _threading.Event()
        active = [0]
        idx = QuiesceIndex(period=period, max_defer=max_defer)
        for k in range(n_rows):
            idx.on_write(table.versions[k])
        holder["idx"] = idx
        holder["ri"] = RowIndex(table)
        wt = _threading.Thread(target=writer, daemon=True)
        wt.start()
        rts = [_threading.Thread(target=reader, args=(i,), daemon=True)
               for i in range(n_readers)]
        for t in rts:
            t.start()
        time.sleep(n_life_s)
        stop.set()
        wt.join(timeout=5)
        for t in rts:
            t.join(timeout=5)
        if wt.is_alive() or any(t.is_alive() for t in rts):
            raise RuntimeError("stress threads did not stop; refusing incomplete result")
        delays = idx.delays
        results[str(max_defer)] = {
            "max_defer": max_defer,
            "readers": n_readers,
            "due": idx.n_due,
            "applied": idx.n_applied,
            "unapplied_at_end": idx.n_due - idx.n_applied,
            "compaction_lag_drift": idx.n_due - idx.n_applied,
            "delay_mean_commits": round(float(np.mean(delays)), 1) if delays else 0.0,
            "delay_max_commits": int(max(delays)) if delays else 0,
            "delay_median_commits": round(float(np.median(delays)), 1) if delays else 0.0,
        }
    return results


# --------------------------------------------------------------------------- #
# Deferral-accumulation stress under sustained episode load
# --------------------------------------------------------------------------- #
def deferral_stress(n_life_s=6.0, think_us=THINK_US, period=COMPACT_PERIOD,
                    embed_cost_us=EMBED_COST_US, n_rows=2000, duty=None, seed=1):
    """Sweep the reader duty cycle and measure how long a DUE compaction waits.

    The single-process simulator cannot run episodes concurrently, so we model
    *reader pressure* by the fraction of wall time an episode is active.  At a
    high duty cycle the "no episode is active" window in which quiesce is allowed
    to compact becomes rare, and the deferral of each due compaction grows.

    ``duty`` is the target fraction of wall time the single reader spends active
    (think + retrieval).  With the reader running flat out, the only gaps are the
    writer-side pauses; we make the gaps tunable by inserting an idle sleep
    between episodes:  idle = think*(1-duty)/duty.

    Returns a dict with the mean/max deferral, the number of due-but-unapplied
    points, and the compaction lag drift (compactions due minus applied).
    """
    import random as _random
    rng = _random.Random(seed)
    table, _ = build(n_rows, rng)
    writer_rng = _random.Random(rng.getrandbits(64))
    reader_rng = _random.Random(rng.getrandbits(64))
    idx = QuiesceIndex(period=period, max_defer=None)
    for r in range(n_rows):
        idx.on_write(table.versions[r])

    lock = RealLock()
    clock = Clock(t=0)
    stop = threading.Event()
    ep_active = [False]
    act_time = [0.0]
    wall_start = time.perf_counter()

    def writer():
        while not stop.is_set():
            rid = writer_rng.randrange(n_rows)
            with lock:
                if writer_rng.random() < 0.12:
                    rv = table.delete(rid, ts=clock.tick())
                else:
                    rv = table.update(rid, make_text(writer_rng), ts=clock.tick())
                idx.on_write(rv)
                # maintenance advances: a due point is applied only when no
                # episode is active (the quiesce rule)
                idx.tick_maintenance(rv.commit_ts, ep_active[0])
            if embed_cost_us:
                time.sleep(embed_cost_us / 1e6)

    wt = threading.Thread(target=writer, daemon=True)
    wt.start()
    with lock:
        ri = RowIndex(table)
    idle_s = 0.0 if duty is None else max(0.0, (think_us / 1e6) * (1.0 - duty) / duty)
    eps = 0
    while time.perf_counter() - wall_start < n_life_s:
        t0 = time.perf_counter()
        with lock:
            ep_active[0] = True
            ri.refresh(table)
            S = clock.t
            r = reader_rng.randrange(n_rows)
            vt = ri.read_at(r, S)
        time.sleep(think_us / 1e6)
        with lock:
            idx.query(vt.vec if vt is not None else embed(make_text(reader_rng)), 10, S)
            ep_active[0] = False
            idx.tick_maintenance(clock.t, False)
        eps += 1
        act_time[0] += time.perf_counter() - t0
        if idle_s:
            time.sleep(idle_s)
    wall = time.perf_counter() - wall_start
    stop.set()
    wt.join(timeout=5)
    if wt.is_alive():
        raise RuntimeError("stress writer did not stop; refusing incomplete result")
    delays = idx.delays
    return {
        "reader_duty": round(act_time[0] / wall, 3),
        "episodes": eps,
        "due": idx.n_due,
        "applied": idx.n_applied,
        "unapplied_at_end": idx.n_due - idx.n_applied,
        "delay_mean_commits": round(float(np.mean(delays)), 1) if delays else 0.0,
        "delay_max_commits": int(max(delays)) if delays else 0,
        "delay_p99_commits": int(sorted(delays)[int(len(delays) * 0.99)]) if delays else 0,
        "compaction_lag_drift": idx.n_due - idx.n_applied,
    }


# --------------------------------------------------------------------------- #
# Scheme construction
# --------------------------------------------------------------------------- #
def build_index(scheme, table, n_rows):
    if scheme == "async500":
        idx = AsyncIndex(lag=500)
        for r in range(n_rows):
            idx.on_write(table.versions[r])
        return idx
    if scheme == "lock":
        idx = LockIndex()
        for r in range(n_rows):
            idx.on_write(table.versions[r])
        return idx
    if scheme == "scr":
        idx = VersionedIndex(gc_enabled=True)
        for r in range(n_rows):
            idx.on_write(table.versions[r])
        return idx
    if scheme == "quiesce":
        idx = QuiesceIndex(period=COMPACT_PERIOD, max_defer=None)
        for r in range(n_rows):
            idx.on_write(table.versions[r])
        return idx
    if scheme == "double_buffer":
        idx = DoubleBufferIndex(period=COMPACT_PERIOD)
        for r in range(n_rows):
            idx.on_write(table.versions[r])
        return idx
    raise ValueError("unknown scheme: %s" % scheme)


def native_bytes(n_rows):
    """Logical footprint of a NON-VERSIONED index at the paper's convention.

    A non-versioned index holds exactly one vector per live row plus a graph
    adjacency of M=12 neighbours (int32) per row.  We deliberately do NOT use the
    prototype's raw ``index_bytes()`` here: all four prototype indexes append one
    entry per *version*, so their byte count grows with the writer's appends
    (N+W entries) rather than with the logical index size N -- using it would
    make every scheme look ~1.3x regardless of its actual retention strategy.

    Convention (matches memory_scope.py: embeddings + adjacency):
        embeddings   N * DIM * 4          (float32)
        adjacency    N * M * 4            (int32, M=12)
    """
    M = 12
    return n_rows * (DIM * 4 + M * 4)


def space_accounting(scheme, idx, n_rows):
    """Return (retained_factor, peak_factor, note) for a scheme.

    Because the prototype indexes store one entry per version, we report the
    retention *policy* against the logical non-versioned footprint, and we state
    the policy explicitly rather than quoting a byte count that the append-only
    prototype would inflate for every scheme equally.
    """
    nb = native_bytes(n_rows)
    if scheme == "quiesce":
        # one index copy; deferred compaction holds no extra generation
        return 1.0, 1.0, "single index copy; deferral holds no extra state"
    if scheme == "double_buffer":
        # the private copy co-resides with the live index while a swap is pending
        return 1.0, 2.0, "live index + one private copy during the swap window"
    if scheme == "async":
        # append-only, never compacted: grows with the number of appends
        factor = float(len(idx.ts)) / n_rows
        return factor, factor, "version-entry count / initial rows; includes initial entries once"
    if scheme == "scr":
        # G=7 pinned generations, COW frame sharing (paper Table 5 convention)
        return 1.38, 1.38, "G=7 pinned generations, COW frame sharing (Table 5)"
    if scheme == "lock":
        # compacted, un-versioned current state
        return 1.0, 1.0, "current compacted state only"
    return 1.0, 1.0, ""


def main():
    t0 = time.time()
    schemes = ["async500", "lock", "scr", "quiesce", "double_buffer"]
    runs = []
    for scheme in schemes:
        for sd in SEEDS:
            rng = random.Random(sd)
            table, _ = build(N_ROWS, rng)
            idx = build_index(scheme, table, N_ROWS)
            res = run_episode_workload(table, idx, scheme, rng, N_WRITES, N_READS,
                                       K, max_row=N_ROWS, np_seed=SEED_RNG)
            res["seed"] = sd
            # ---- space accounting (logical, like-for-like) ------------------ #
            tag = {"async500": "async", "lock": "lock", "scr": "scr",
                   "quiesce": "quiesce", "double_buffer": "double_buffer"}[scheme]
            ret, peak, note = space_accounting(tag, idx, N_ROWS)
            res["space_factor_retained"] = round(ret, 3)
            res["peak_space_factor"] = round(peak, 3)
            res["space_note"] = note
            res["index_bytes_MB_measured"] = round(idx.index_bytes() / 1e6, 3)
            runs.append(res)
            print("%-14s seed=%d any=%.4f torn=%.4f(d%d/s%d/x%d) A5dis=%.4f "
                  "p50=%.0fus p99=%.0fus delay=%s space=%.2fx" % (
                      scheme, sd, res["anomaly_episode_rate"],
                      res["a4_torn_episode_rate"], res["a4_dirty_disagreements"],
                      res["a4_stale_disagreements"], res["a4_deleted_disagreements"],
                      res["a5_retrieval_disagreement_rate"],
                      res["p50_us"], res["p99_us"],
                      res.get("maintenance_delay_mean_commits"),
                      res["space_factor_retained"]), flush=True)

    # ---------------- aggregate over seeds ---------------- #
    agg = {}
    for scheme in schemes:
        rs = [r for r in runs if r["scheme"] == scheme]
        agg[scheme] = {}
        for key in ["anomaly_episode_rate", "a1_dirty_episode_rate",
                    "a2_superseded_episode_rate", "a3_phantom_episode_rate",
                    "a4_torn_episode_rate", "a5_retrieval_disagreement_rate",
                    "a4_dirty_disagreements", "a4_stale_disagreements",
                    "a4_deleted_disagreements",
                    "mean_symmetric_difference", "mean_jaccard_k1_k2",
                    "recall_at_k", "p50_us", "p99_us", "throughput_ep_per_s",
                    "lock_wait_p50_us", "lock_wait_p99_us",
                    "maintenance_delay_mean_commits", "maintenance_delay_max_commits",
                    "maintenance_delay_median_commits",
                    "maintenance_delay_max_excl_first_commits",
                    "maintenance_delay_mean_excl_first_commits",
                    "space_factor_retained", "peak_space_factor"]:
            v = [r.get(key, 0.0) for r in rs]
            agg[scheme][key] = {
                "mean": round(float(np.mean(v)), 6),
                "std": round(float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, 6),
                "values": v,
            }
        # the first-applied delay is a single integer per seed, not an average
        for r in rs:
            agg[scheme].setdefault("maintenance_delay_first_applied_commits", {}) \
                .setdefault("values", []).append(r.get("maintenance_delay_first_applied_commits", 0))
        agg[scheme]["maintenance_delay_first_applied_commits"]["mean"] = round(
            float(np.mean(agg[scheme]["maintenance_delay_first_applied_commits"]["values"])), 3)
        # anomaly counts for a Wilson upper bound
        tot_anom = sum(r["n_anomaly_episodes"] for r in rs)
        tot_ep = sum(r["episodes"] for r in rs)
        tot_a5 = sum(r["n_a5_disagreement_episodes"] for r in rs)
        agg[scheme]["anomaly_counts"] = {
            "anomaly_episodes": tot_anom, "episodes": tot_ep,
            "wilson_upper_95_anomaly": round(wilson_upper(tot_anom, tot_ep), 6),
            "a5_episodes": tot_a5,
            "wilson_upper_95_a5": round(wilson_upper(tot_a5, tot_ep), 6),
        }
        # total deferral across the run (the accumulation check)
        agg[scheme]["maintenance_totals"] = {
            "due": [r.get("maintenance_compactions_due") for r in rs],
            "applied": [r.get("maintenance_compactions_applied") for r in rs],
            "unapplied_at_end": [r.get("maintenance_points_unapplied_at_end") for r in rs],
        }

    # ---------------- deferral-accumulation stress sweep ---------------- #
    print("\n[stress] quiesce deferral vs reader duty cycle "
          "(6s wall per point, P=%d commits)" % COMPACT_PERIOD, flush=True)
    stress = []
    for duty in (0.25, 0.50, 0.75, 0.90, 0.97):
        s = deferral_stress(duty=duty, seed=1)
        stress.append(s)
        print("  duty=%.2f  ep=%d due=%d applied=%d unapplied_end=%d "
              "delay_mean=%.0f delay_max=%d drift=%d" % (
                  s["reader_duty"], s["episodes"], s["due"], s["applied"],
                  s["unapplied_at_end"], s["delay_mean_commits"],
                  s["delay_max_commits"], s["compaction_lag_drift"]), flush=True)

    # ---------------- multi-reader: does quiesce's deferral accumulate? --- #
    print("\n[multireader] quiesce with R concurrent episode threads "
          "(5s wall per point); pure rule (max_defer=None) vs drain guard",
          flush=True)
    multireader = {}
    for R in (2, 4, 8, 16):
        mr = quiesce_multireader(n_readers=R, n_life_s=5.0, seed=1)
        multireader["R=%d" % R] = mr
        for tag, v in mr.items():
            print("  R=%2d max_defer=%-4s due=%3d applied=%3d unapplied_end=%2d "
                  "drift=%2d delay_mean=%.0f delay_max=%d" % (
                      R, tag, v["due"], v["applied"], v["unapplied_at_end"],
                      v["compaction_lag_drift"], v["delay_mean_commits"],
                      v["delay_max_commits"]), flush=True)

    out = {
        "generated_by": "proto/run_baselines.py",
        "protocol": (
            "Run baselines.py: identical cross-store episode protocol to "
            "run_episodes_v3.py -- Begin() at the writer clock -> real "
            "relational read of row r -> query derived from that read -> 20ms "
            "think (writes commit) -> two retrievals at token S -> validate. "
            "Own-row conflict uses only the observed relation row; K1 != K2 "
            "compares citation provenance and is not A4/A5 certification. One episode active at a time "
            "(single-reader prototype). Maintenance period P=250 commits."
        ),
        "maintenance_delay_definition": (
            "A maintenance point becomes DUE every P=250 commits since the last "
            "applied compaction. Its delay is apply_time - due_time in commits "
            "(the logical clock). Reported mean/max/p99 over applied points, plus "
            "the count of points still unapplied at run end (accumulation check)."
        ),
        "policies": {
            "async500": (
                "Append-only asynchronous index, lag L=500 commits; never "
                "compacts. No maintenance operator -> delay 0 by construction, "
                "which is exactly why it is the anomaly-ridden reference."
            ),
            "lock": (
                "Global lock serialises reads against in-flight embedding work; "
                "index always the current compacted state. Compaction inline, "
                "delay 0; every retrieval waits for the lock."
            ),
            "scr": (
                "Exact historical node-version resolver honors S. No explicit "
                "graph compaction or generation pinning is implemented here; "
                "maintenance delay zero is bookkeeping. The 1.38x factor is "
                "separate T2 shared-array payload accounting, not this index's measured memory."
            ),
            "quiesce": (
                "B1: a due compaction is DEFERRED while an episode is active; "
                "the index keeps serving the current (uncompacted) adjacency. "
                "Applied when no episode is active. No drain guard "
                "(max_defer=None) -- i.e. the pure 'no compaction while episodes "
                "run' rule. Zero extra space (one index copy)."
            ),
            "double_buffer": (
                "B2: maintenance builds a PRIVATE copy of the whole index and "
                "swaps atomically. Policy: a swap is only performed when no "
                "episode spans it (no episode active); an episode that would "
                "span the swap is DELAYED until the swap completes and is "
                "charged the extra latency. Peak space = 2x the non-versioned "
                "index (the private copy co-resident with the live one)."
            ),
        },
        "space_convention": (
            "native_bytes = n_rows*(DIM*4 + M*4) with DIM=64, M=12: embeddings "
            "(float32) + graph adjacency (int32), i.e. the paper's memory_scope.py "
            "convention (embeddings + adjacency, no provenance). We report the "
            "retention POLICY per scheme rather than the prototype's raw "
            "index_bytes(), because every prototype index appends one entry per "
            "version (N+W entries) and would inflate all schemes equally. SCR's "
            "paper figure (1.38x whole index, 7.00x naive) is reproduced from "
            "memory_scope.py on testbed T2 (N=5000, G=7) and quoted here as the "
            "paper's convention; quiesce/lock retain exactly one index copy "
            "(1.00x); double-buffer peaks at 2.00x (live + private copy); async "
            "is reported as its measured append count / N."
        ),
        "config": {"n_rows": N_ROWS, "n_writes": N_WRITES, "n_reads": N_READS,
                   "k": K, "seeds": SEEDS, "think_us": THINK_US,
                   "embed_cost_us": EMBED_COST_US,
                   "maintenance_period_commits": COMPACT_PERIOD},
        "aggregate": agg,
        "deferral_stress": stress,
        "quiesce_multireader": multireader,
        "runs": runs,
        "runtime_s": round(time.time() - t0, 1),
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("\nsaved ->", OUT, "(%.1fs)" % out["runtime_s"])

    # ---------------- console summary table ---------------- #
    print("\n%-14s %-14s %-12s %-18s %-14s %-10s" % (
        "scheme", "anom_ep_rate", "wilson95", "maint_delay(mean/max)", "space_x",
        "p50/p99us"))
    for scheme in schemes:
        a = agg[scheme]
        print("%-14s %-14s %-12.5f %-18s %-14.3f %-10s" % (
            scheme,
            "%.4f+/-%.4f" % (a["anomaly_episode_rate"]["mean"],
                             a["anomaly_episode_rate"]["std"]),
            a["anomaly_counts"]["wilson_upper_95_anomaly"],
            "%.1f / %d" % (a["maintenance_delay_mean_commits"]["mean"],
                           int(a["maintenance_delay_max_commits"]["mean"])),
            a["space_factor_retained"]["mean"],
            "%.0f/%.0f" % (a["p50_us"]["mean"], a["p99_us"]["mean"])))


if __name__ == "__main__":
    main()
