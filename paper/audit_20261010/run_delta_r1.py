#!/usr/bin/env python
"""Fault checks and the narrowly scoped fresh bounded-delta flat control."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import threading
import time

# Fixed before importing NumPy: latency is a one-thread local control only.
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper" / "proto"))
import run_experiments as workload
from engine import VersionedIndex

AUDIT = Path(__file__).resolve().parent
RESULT = AUDIT / "results_delta_r1.json"
CHECKS = AUDIT / "checks_delta_r1.json"
SEED, N, W, R, K = 20261008, 15000, 5000, 1200, 10
DELTAS = [0, 10, 50, 200, 500, 1000]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fresh(n=40, seed=SEED, index_type=VersionedIndex):
    table, _ = workload.build(n, random.Random(seed))
    index = index_type()
    workload.attach(index, table)
    return table, index


class RecordingIndex(VersionedIndex):
    def __init__(self):
        super().__init__()
        self.query_digests = []
        self.query_cuts = []

    def query(self, q, k, snapshot):
        self.query_digests.append(hashlib.sha256(q.tobytes()).hexdigest())
        self.query_cuts.append(snapshot)
        return super().query(q, k, snapshot)


def checks():
    records = []
    # A short reader must not silently truncate the requested writer stream.
    table, index = fresh(index_type=RecordingIndex)
    row = workload.run_workload(table, index, random.Random(SEED), 25, 1, 5,
                                embed_cost_us=2000, think_us=0)
    assert row["writes_completed"] == 25
    assert row["writes_at_reader_end"] < 25
    assert len(table.versions) == 40 + 25
    records.append({"check": "reader_finishes_early_writer_still_completes",
                    "writes_at_reader_end": row["writes_at_reader_end"],
                    "writes_completed": row["writes_completed"]})

    # Different interleavings must not consume each other's random stream.
    streams = []
    for think in (0, 2000):
        table, index = fresh(index_type=RecordingIndex)
        row = workload.run_workload(table, index, random.Random(SEED), 25, 12, 5,
                                    embed_cost_us=1000, think_us=think)
        writes = [(v.row_id, v.text, v.deleted) for v in table.versions[40:]]
        streams.append((writes, index.query_digests, row))
    assert streams[0][0] == streams[1][0]
    assert streams[0][1] == streams[1][1]
    records.append({"check": "writer_and_query_streams_schedule_independent",
                    "write_trace_sha256": hashlib.sha256(json.dumps(streams[0][0]).encode()).hexdigest(),
                    "query_trace_sha256": hashlib.sha256(json.dumps(streams[0][1]).encode()).hexdigest(),
                    "overlap_counts": [s[2]["episodes_overlapped_write"] for s in streams]})

    class FailingIndex(VersionedIndex):
        def __init__(self):
            super().__init__()
            self.armed, self.calls = False, 0

        def on_write(self, version):
            if self.armed:
                self.calls += 1
                if self.calls == 3:
                    raise ValueError("injected third publication failure")
            super().on_write(version)

    table, index = fresh(index_type=FailingIndex)
    index.armed = True
    try:
        workload.run_workload(table, index, random.Random(SEED), 25, 12, 5,
                              embed_cost_us=100, think_us=5000)
        raise AssertionError("a failed publication returned benchmark scores")
    except RuntimeError as exc:
        assert isinstance(exc.__cause__, ValueError)
        assert "third publication" in str(exc.__cause__)
        records.append({"check": "writer_exception_propagated_no_score_result",
                        "exception": str(exc), "cause": str(exc.__cause__),
                        "table_appended_writes": len(table.versions) - 40,
                        "index_appended_writes": len(index.ts) - 40})

    # Pause precisely after table.append but before index publication. Begin
    # attempts to enter during that gap. Neither query nor oracle may observe it.
    publication_paused, begin_attempted, release = (threading.Event() for _ in range(3))

    class OwnedLock:
        def __init__(self):
            self.mutex = threading.RLock()
            self.owner, self.depth = None, 0

        def __enter__(self):
            if threading.current_thread().name == "fault-reader" and publication_paused.is_set():
                begin_attempted.set()
            self.mutex.acquire()
            self.owner = threading.get_ident()
            self.depth += 1
            return self

        def __exit__(self, *args):
            self.depth -= 1
            if not self.depth:
                self.owner = None
            self.mutex.release()
            return False

    mutex = OwnedLock()

    class GatedIndex(RecordingIndex):
        def __init__(self):
            super().__init__()
            self.armed = False
            self.table = None

        def on_write(self, version):
            if self.armed and not publication_paused.is_set():
                publication_paused.set()
                if not release.wait(3):
                    raise TimeoutError("fault gate release was not supplied")
            super().on_write(version)

        def query(self, q, k, snapshot):
            assert mutex.owner == threading.get_ident()
            assert len(self.table.versions) == len(self.ts)
            assert snapshot <= max(self.ts)
            return super().query(q, k, snapshot)

    table, index = fresh(index_type=GatedIndex)
    index.table, index.armed = table, True
    old_sync = workload.TruthCache.sync
    sync_calls = []

    def checked_sync(cache):
        assert mutex.owner == threading.get_ident()
        assert len(table.versions) == len(index.ts)
        sync_calls.append(len(table.versions))
        return old_sync(cache)

    output, errors = [], []

    def call_control():
        try:
            output.append(workload.run_workload(table, index, random.Random(SEED), 5, 3, 5,
                                                lock=mutex, embed_cost_us=1000, think_us=0))
        except BaseException as exc:
            errors.append(exc)

    workload.TruthCache.sync = checked_sync
    reader = threading.Thread(target=call_control, name="fault-reader")
    try:
        reader.start()
        assert publication_paused.wait(3), "writer did not reach publication fault gate"
        assert begin_attempted.wait(3), "Begin did not attempt to enter paused publication"
        assert len(table.versions) == len(index.ts) + 1
        assert not index.query_digests and not sync_calls
        release.set()
        reader.join(5)
        assert not reader.is_alive()
        if errors:
            raise errors[0]
        assert output[0]["writes_completed"] == 5
        assert output[0]["dirty_citations"] == 0
        records.append({"check": "atomic_publication_begin_query_and_oracle_refresh",
                        "begin_blocked_at_partial_publication": True,
                        "oracle_sync_calls": len(sync_calls),
                        "query_cuts": index.query_cuts,
                        "writes_completed": output[0]["writes_completed"]})
    finally:
        release.set()
        reader.join(5)
        workload.TruthCache.sync = old_sync

    for nw, nr in ((0, 3), (3, 0), (0, 0)):
        table, index = fresh()
        row = workload.run_workload(table, index, random.Random(SEED), nw, nr, 5,
                                    embed_cost_us=0, think_us=0)
        assert row["writes_completed"] == nw and row["episodes"] == nr
    table, index = fresh()
    try:
        workload.run_workload(table, index, random.Random(SEED), 1, 1, 5,
                              lock=workload.DummyLock())
        raise AssertionError("DummyLock was accepted")
    except ValueError:
        pass
    records.append({"check": "zero_counts_and_explicit_dummy_lock_rejection", "passed": True})
    result = {"passed": True, "checks": records,
              "scope": "finite fault injections and schedule checks; no production-engine proof",
              "run_experiments_sha256": digest(ROOT / "paper" / "proto" / "run_experiments.py"),
              "audit_driver_sha256": digest(__file__)}
    CHECKS.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("checks passed:", len(records), flush=True)
    return result


def rerun(check_result):
    result = {
        "config": {"seed": SEED, "n_rows": N, "n_writes": W, "n_reads": R, "k": K,
                   "deltas": DELTAS, "embed_cost_us": 150.0, "think_us": 20000.0,
                   "query_tokens": 12, "delete_probability": 0.12,
                   "blas_threads": {name: os.environ[name] for name in
                                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}},
        "scope": "one seed; one query per read; exact flat versioned index and exact same-cut oracle; random 12-token text queries; no ANN graph or production claim",
        "protocol": "independent writer/query RNGs; table/index publication and head advancement atomic with Begin; query and oracle refresh under publication mutex; every requested write completed before returning",
        "run_experiments_sha256": digest(ROOT / "paper" / "proto" / "run_experiments.py"),
        "audit_driver_sha256": digest(__file__),
        "checks_file": CHECKS.name,
        "checks_passed": check_result["passed"],
        "exp2": [],
    }
    started = time.perf_counter()
    for delta in DELTAS:
        cell_started = time.perf_counter()
        table, build_ms = workload.build(N, random.Random(SEED))
        index = VersionedIndex()
        attach_started = time.perf_counter()
        workload.attach(index, table)
        attach_seconds = time.perf_counter() - attach_started
        row = workload.run_workload(table, index, random.Random(SEED), W, R, K,
                                    mode="bounded", delta=delta,
                                    embed_cost_us=150.0, think_us=20000.0)
        row.update({"delta": delta, "scheme": "flat-bounded delta=%d" % delta,
                    "seed": SEED, "build_ms": round(build_ms, 6),
                    "attach_seconds": round(attach_seconds, 6),
                    "cell_seconds": round(time.perf_counter() - cell_started, 6),
                    "reads_requested": R, "reads_completed": row["episodes"]})
        assert row["writes_completed"] == W and row["episodes"] == R
        result["exp2"].append(row)
        result["run_seconds_so_far"] = round(time.perf_counter() - started, 6)
        result["complete"] = len(result["exp2"]) == len(DELTAS)
        RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: row[key] for key in
                          ("delta", "citation_validity", "recall_at_k", "anomaly_episode_rate",
                           "p50_us", "p99_us", "writes_completed", "workload_seconds",
                           "episodes_started_during_writer", "episodes_overlapped_write")}), flush=True)
    print("result sha256:", digest(RESULT), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checks-only", action="store_true")
    args = parser.parse_args()
    check_result = checks()
    if not args.checks_only:
        rerun(check_result)
