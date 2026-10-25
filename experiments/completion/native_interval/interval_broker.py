"""Integrated exact multi-interval native admission; local reference authority.

All inherited ownership and reference rules are unchanged. Source writers must
seal commit cuts and route pruning/native cleanup through this authority.
"""
from pathlib import Path
import sys, time, sqlite3
from dataclasses import dataclass
from types import MappingProxyType
from bisect import bisect_right
from threading import local

HERE = Path(__file__).resolve().parent
RUN = HERE.parent.parent
sys.path.insert(0, str(RUN / 'vendor'))
sys.path.insert(0, str(RUN))
sys.path.insert(0, str(HERE.parent))
import numpy as np
import pyarrow as pa
from live_broker import NativeBroker
from support_spectrum import Support, certify, lifetimes_from_events


@dataclass(frozen=True)
class IntervalManifest:
    name: str
    low: int
    high: int
    table_name: str
    native_version: int
    keys: frozenset
    catalogue: object
    initial_support: object
    settings: tuple = (4, 128, 10)


class IntervalBroker(NativeBroker):
    """Reference policy scans a committed catalogue and immutable native table.

    Point mode is a strong comparator: identical cached native inspection and
    catalogue, independently certified point cuts, then the same lifecycle.
    It is not a correctness-weakened wrapper. Neither arm's query is optimized.
    """
    def __init__(self, sql, native_db, mode='interval'):
        if mode not in ('interval', 'point'):
            raise ValueError('unknown certification mode')
        super().__init__(sql, native_db)
        self.mode = mode
        self.exact_support = {}
        self._views = {}
        self._populations = {}
        self.metrics = {'publication': [], 'admission': [], 'query': []}
        self._hook_context = local()

    def _reject_hook_reentry(self):
        if getattr(self._hook_context, 'active', False):
            raise ValueError('broker actions cannot reenter observation callbacks')

    def _invoke_hook(self, hook, *args):
        if hook is None:
            return
        self._reject_hook_reentry()
        self._hook_context.active = True
        try:
            return hook(*args)
        finally:
            self._hook_context.active = False

    def _configured_table(self, table):
        """A name never establishes physical identity for a supplied handle."""
        import os
        configured = self.native_db.open_table(table.name)
        supplied, actual = os.stat(table.uri), os.stat(configured.uri)
        if (supplied.st_dev, supplied.st_ino) != (actual.st_dev, actual.st_ino):
            raise ValueError('native table outside configured artifact domain')
        return configured

    def _own_table(self, table):
        # This also protects the inherited cleanup and maintain implementations.
        return super()._own_table(self._configured_table(table))

    @staticmethod
    def _native_key(record):
        row, version = record['row'], record['version']
        if (not isinstance(row, int) or isinstance(row, bool) or not -(1 << 63) <= row < (1 << 63)
            or not isinstance(version, int) or isinstance(version, bool) or not 1 <= version < (1 << 63)):
            raise ValueError('native keys require exact SQLite-domain integers')
        return row, version

    @staticmethod
    def _native_key_schema(arrow):
        for name in ('row', 'version'):
            if arrow.column_names.count(name) != 1 or not pa.types.is_integer(arrow.schema.field(name).type):
                raise ValueError('native key fields must have Arrow integer types')

    def _catalogue(self):
        self.reader.execute('begin')
        try:
            head, floor = self.reader.execute('select epoch,floor from meta').fetchone()
            if not (isinstance(head, int) and isinstance(floor, int) and 0 <= floor <= head):
                raise ValueError('invalid committed source frontier')
            records = self.reader.execute('select row,epoch,version,deleted,vector,payload from history order by row,epoch').fetchall()
            rows, keys, events = {}, set(), []
            for row, epoch, version, deleted, vector, payload in records:
                if not (isinstance(row, int) and isinstance(version, int) and version >= 1 and isinstance(epoch, int) and 0 <= epoch <= head and deleted in (0, 1)):
                    raise ValueError('invalid or uncommitted source record')
                key = (row, version)
                if key in keys:
                    raise ValueError('source version identity reused')
                keys.add(key)
                value = None if deleted else bytes(vector)
                if value is not None and (not value or len(value) % 4):
                    raise ValueError('invalid float32 embedding')
                rows.setdefault(row, []).append((epoch, version, value, payload))
                events.append((row, epoch, key, value))
            # Complete lineages are an input contract: SQL is the source's
            # authority, not an untrusted event subset. GC retains anchors.
            catalogue = MappingProxyType({r: tuple(v) for r, v in rows.items()})
            return head, floor, catalogue, lifetimes_from_events(events, head)
        finally:
            self.reader.rollback()

    @staticmethod
    def _view(catalogue, snapshot):
        result = {}
        for row, events in catalogue.items():
            i = bisect_right(events, (snapshot, float('inf'))) - 1
            if i >= 0:
                epoch, version, vector, payload = events[i]
                if vector is not None:
                    result[row] = (version, payload)
        return MappingProxyType(result)

    @staticmethod
    def _point_covered(catalogue, population, snapshot):
        checks = 0
        for row, events in catalogue.items():
            checks += 1
            i = bisect_right(events, (snapshot, float('inf'))) - 1
            if i >= 0:
                epoch, version, vector, payload = events[i]
                if vector is not None and population.get((row, version)) != vector:
                    return False, checks
        return True, checks

    def publish(self, name, table, requested_cuts=None, registered_hook=None):
        self._reject_hook_reentry()
        started = time.perf_counter_ns()
        with self.lock:
            acquired = time.perf_counter_ns()
            if name in self.manifests:
                raise ValueError('duplicate manifest')
            v = int(table.version() if callable(table.version) else table.version)
            private = self._configured_table(table)
            domain = self._own_table(private)
            private.checkout(v)
            cache_key = (domain, v)
            scan_start = time.perf_counter_ns()
            cached = cache_key in self._populations
            if not cached:
                population = {}
                native_arrow = private.to_arrow()
                self._native_key_schema(native_arrow)
                vector_type = native_arrow.schema.field('vector').type
                if not pa.types.is_fixed_size_list(vector_type) or not pa.types.is_float32(vector_type.value_type):
                    raise ValueError('native embeddings must be fixed-size Float32 vectors')
                for record in native_arrow.to_pylist():
                    key = self._native_key(record)
                    vector = np.asarray(record['vector'], dtype=np.float32)
                    if key in population or not np.isfinite(vector).all():
                        raise ValueError('duplicate candidate or nonfinite embedding')
                    population[key] = vector.tobytes()
                self._populations[cache_key] = MappingProxyType(population)
            population = self._populations[cache_key]
            scan_end = time.perf_counter_ns()
            head, floor, catalogue, lifetimes = self._catalogue()
            catalogue_end = time.perf_counter_ns()
            if self.mode == 'interval':
                support = certify(lifetimes, population, floor, head, head)
                work = len(lifetimes)
                comparisons = sum(v.identity is not None for v in lifetimes)
            else:
                cuts = tuple(range(floor, head + 1)) if requested_cuts is None else tuple(requested_cuts)
                if len(set(cuts)) != len(cuts) or any(not isinstance(s, int) or isinstance(s, bool) or not floor <= s <= head for s in cuts):
                    raise ValueError('invalid point certification requests')
                accepted, comparisons = [], 0
                for s in sorted(cuts):
                    covered, checks = self._point_covered(catalogue, population, s)
                    comparisons += checks
                    if covered:
                        accepted.append(s)
                # Coalescing accepted points is merely a compact representation
                # of certificates already independently established one by one.
                intervals = []
                for s in accepted:
                    if intervals and intervals[-1][1] == s - 1:
                        intervals[-1] = (intervals[-1][0], s)
                    else:
                        intervals.append((s, s))
                support = Support(tuple(intervals))
                work = len(cuts) * len(catalogue)  # upper bound, early exit allowed
            certified = time.perf_counter_ns()
            m = IntervalManifest(name, floor, head, table.name, v, frozenset(population), catalogue, support)
            # Observation/fault-injection callbacks run on staged state. Public
            # broker actions are forbidden from their thread, and the guard
            # excludes other threads until registration. A callback exception
            # therefore cannot create an externally admitted orphan token.
            self._invoke_hook(registered_hook, m)
            self.manifests[name] = m
            self.exact_support[name] = support
            self.support[name] = self._envelope(support)
            self._manifest_domains[name] = domain
            registered = time.perf_counter_ns()
            ended = time.perf_counter_ns()
            self.metrics['publication'].append({'name': name, 'head': head, 'floor': floor, 'support': support.intervals,
                'started_ns': started, 'acquired_ns': acquired, 'registered_ns': registered, 'ended_ns': ended,
                'wait_ns': acquired-started, 'hold_ns': ended-acquired, 'native_scan_ns': scan_end-scan_start,
                'catalogue_ns': catalogue_end-scan_end, 'certification_ns': certified-catalogue_end,
                'native_cache_hit': cached, 'native_population': len(population), 'source_versions': len(lifetimes),
                'certification_work_upper_bound': work, 'certification_records_visited': comparisons})
            return m

    @staticmethod
    def _envelope(support):
        return (support.intervals[0][0], support.intervals[-1][1]) if support.intervals else None

    def begin(self, s, name, hook=None):
        self._reject_hook_reentry()
        started = time.perf_counter_ns()
        with self.lock:
            acquired = time.perf_counter_ns()
            support = self.exact_support.get(name)
            if support is None or not support.contains(s):
                self.metrics['admission'].append({'snapshot': s, 'name': name, 'started_ns': started, 'ended_ns': time.perf_counter_ns(), 'wait_ns': acquired-started, 'accepted': False})
                raise ValueError('snapshot support closed')
            self.frontier(s)
            m = self.manifests[name]
            handle = self.checkout(m)
            if int(handle.version() if callable(handle.version) else handle.version) != m.native_version:
                raise ValueError('native generation identity changed')
            viewkey = (name, s)
            if viewkey not in self._views:
                self._views[viewkey] = self._view(m.catalogue, s)
            self.serial += 1
            token = self.serial
            self.pins[token] = (s, name)
            try:
                self._invoke_hook(hook)
            except BaseException:
                del self.pins[token]
                raise
            ended = time.perf_counter_ns()
            self.metrics['admission'].append({'snapshot': s, 'name': name, 'started_ns': started, 'ended_ns': ended, 'wait_ns': acquired-started, 'accepted': True})
            return token

    def query(self, token, vector, hook=None):
        self._reject_hook_reentry()
        started = time.perf_counter_ns()
        with self.lock:
            acquired = time.perf_counter_ns()
            if token in self._ending:
                raise ValueError('episode closing')
            s, name = self.pins[token]
            m = self.manifests[name]
            source = self._views[(name, s)]
            self.ops[token] = self.ops.get(token, 0) + 1
        try:
            handle = self.checkout(m)
            actual = int(handle.version() if callable(handle.version) else handle.version)
            if actual != m.native_version:
                raise ValueError('native generation identity changed')
            self._invoke_hook(hook)
            result_arrow = handle.search(vector).nprobes(m.settings[0]).ef(m.settings[1]).limit(120).to_arrow()
            self._native_key_schema(result_arrow)
            records = result_arrow.to_pylist()
            if int(handle.version() if callable(handle.version) else handle.version) != m.native_version:
                raise ValueError('native generation identity changed during query')
            answers = []
            for r in records:
                key = self._native_key(r)
                if key[0] in source and source[key[0]][0] == key[1]:
                    answers.append((key[0], key[1], source[key[0]][1]))
            return answers[:m.settings[2]]
        finally:
            with self.lock:
                self.ops[token] -= 1
                if not self.ops[token]:
                    del self.ops[token]
                if token in self._ending and token not in self.ops:
                    del self.pins[token]
                    self._ending.remove(token)
            self.metrics['query'].append({'name': name, 'snapshot': s, 'started_ns': started, 'ended_ns': time.perf_counter_ns(), 'wait_ns': acquired-started})

    def audit_hold(self, token):
        # A held audit can execute the same immutable source view.
        self._reject_hook_reentry()
        return super().audit_hold(token)

    def end(self, token):
        self._reject_hook_reentry()
        return super().end(token)

    def cleanup(self, table):
        self._reject_hook_reentry()
        return super().cleanup(table)

    def maintain(self, table):
        self._reject_hook_reentry()
        return super().maintain(table)

    def withdraw_component(self, name, low, high):
        self._reject_hook_reentry()
        if any(not isinstance(x, int) or isinstance(x, bool) for x in (low, high)) or low > high:
            raise ValueError('invalid closed withdrawal range')
        with self.lock:
            support = self.exact_support[name]
            result = []
            for a, b in support.intervals:
                if b < low or a > high:
                    result.append((a, b))
                else:
                    if a < low:
                        result.append((a, low-1))
                    if b > high:
                        result.append((high+1, b))
            self.exact_support[name] = Support(tuple(result))
            self.support[name] = self._envelope(self.exact_support[name])

    def withdraw(self, name):
        self._reject_hook_reentry()
        with self.lock:
            if name not in self.manifests:
                raise ValueError('unknown manifest')
            self.exact_support[name] = Support(())
            self.support[name] = None

    def drop_retired_table(self, table):
        """Delete an entirely retired table under the same ownership guard."""
        self._reject_hook_reentry()
        with self.lock:
            configured = self._configured_table(table)
            domain = self._own_table(configured)
            affected = {name for name, key in self._manifest_domains.items() if key == domain}
            if any(self.support[name] is not None for name in affected) or any(name in affected for s, name in self.pins.values()):
                raise ValueError('native table still protected')
            self.native_db.drop_table(configured.name)
            for key in [k for k in self._populations if k[0] == domain]:
                del self._populations[key]

    def gc(self):
        self._reject_hook_reentry()
        with self.lock:
            # Keep base retirement and derived-state removal in one critical
            # section: a retired registry name may be reused immediately.
            floor, removed = super().gc()
            for m in removed:
                self.exact_support.pop(m.name, None)
                for key in [k for k in self._views if k[0] == m.name]:
                    del self._views[key]
            return floor, removed

    def close(self):
        self._reject_hook_reentry()
        with self.lock:
            super().close()
            self.exact_support.clear()
            self._views.clear()
            self._populations.clear()
