"""Independent Table 2 audit. Never imports/runs cross_store.py.

Reads the original raw/NPZ/native artifacts. Writes only to the selected output
directory and a new timestamped scratch directory. Uses source maps reconstructed from
dataset size and the published seeded edit schedule, native table bytes, and an
independent dense float64 distance oracle.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import copy
import datetime
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
EXPECTED_VERSIONS = {"numpy": "1.26.4", "pyarrow": "21.0.0", "lancedb": "0.40.0"}


class NativeEvidenceOutputError(ValueError):
    pass


def arguments():
    parser = argparse.ArgumentParser(description="Physical native Table 2 audit; no experiment regeneration and no silent database fallback.")
    parser.add_argument("--mode", choices=["physical-native"], default="physical-native")
    parser.add_argument("--run-dir", type=Path, default=HERE.parent, help="Evidence directory containing raw JSON, NPZ and source scripts.")
    parser.add_argument("--native-dbdir", type=Path, help="Explicit native-root override for a relocated artifact; otherwise the raw dbdir is used exactly.")
    parser.add_argument("--output-dir", type=Path, default=HERE, help="Directory for the audit JSON and a fresh diagnostic scratch directory.")
    parser.add_argument("--scratch-root", type=Path, help="Optional short writable parent for fresh diagnostic scratch, useful for deep Windows extraction paths.")
    return parser.parse_args()


def load_runtime(run):
    global np, pa, lancedb, load_digits
    vendor = (run / "vendor").resolve()
    if vendor.is_dir():
        sys.path.insert(0, str(vendor))
    import numpy as np
    import pyarrow as pa
    import lancedb
    from sklearn.datasets import load_digits
    runtime = {}
    for module in (np, pa, lancedb):
        origin = Path(module.__file__).resolve()
        if module.__version__ != EXPECTED_VERSIONS[module.__name__]:
            raise RuntimeError(f"Unexpected {module.__name__} version {module.__version__}; expected {EXPECTED_VERSIONS[module.__name__]}")
        runtime[module.__name__] = dict(version=module.__version__, origin=str(origin))
    return runtime


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def keypairs(values):
    return [(int(i), int(v)) for i, v in values]


def invalid_keys(answers, source):
    return [(i, v) for i, v in answers if i not in source or source[i] != v]


def exact_top10(vectors, versions, query):
    ids = np.array(sorted(versions), dtype=np.int64)
    vs = np.array([versions[int(i)] for i in ids], dtype=np.int64)
    mat = np.stack([vectors[int(i), int(v)] for i, v in zip(ids, vs)]).astype(np.float64)
    distance = np.einsum("ij,ij->i", mat - query.astype(np.float64), mat - query.astype(np.float64))
    order = np.lexsort((vs, ids, distance))[:10]
    return [(int(ids[k]), int(vs[k])) for k in order]


def native_population(table):
    records = table.to_arrow().to_pylist()
    keys = [(int(r["row"]), int(r["version"])) for r in records]
    vectors = {(int(r["row"]), int(r["version"])): np.array(r["vector"], dtype=np.float32) for r in records}
    assert len(keys) == len(set(keys)), "duplicate native keys"
    return keys, vectors


def search(table, queries, k):
    return [keypairs([(r["row"], r["version"]) for r in table.search(q).nprobes(4).ef(128).limit(k).to_arrow().to_pylist()]) for q in queries]


def file_inventory(root):
    return {str(p.relative_to(root)): sha(p) for p in root.rglob("*") if p.is_file() and p.suffix not in (".lock",) and not p.name.endswith(("-shm", "-wal"))}


def validate_native_root(native_root, cells):
    # LanceDB connect may create a missing directory. Reject bad pointers first.
    if not native_root.is_dir():
        raise FileNotFoundError(f"Native database pointer does not exist: {native_root}")
    lance = native_root / "lance"
    if not lance.is_dir():
        raise FileNotFoundError(f"Native database pointer has no lance/ directory: {native_root}")
    for c in cells:
        name = f"{c['dataset']}_{c['seed']}"
        for required in (lance / (name + ".lance"), native_root / (name + ".sqlite")):
            if not required.exists():
                raise FileNotFoundError(f"Native database pointer is incomplete or incorrect: {required}")
    return native_root


def main(args):
    global RUN
    if not __debug__:
        raise RuntimeError("Run without Python -O/-OO: this physical audit requires its assertions.")
    started = time.time()
    RUN = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    raw_path = RUN / "cross_store_results.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    assert {(c["dataset"], c["seed"]) for c in raw["cells"]} == {(d, s) for d in ("digits", "uniform") for s in (11, 29, 47)} and len(raw["cells"]) == 6
    native_root = (args.native_dbdir if args.native_dbdir is not None else Path(raw["dbdir"])).resolve()
    if output_dir.is_relative_to(native_root):
        raise NativeEvidenceOutputError("Audit output must be outside the native evidence directory.")
    validate_native_root(native_root, raw["cells"])
    runtime = load_runtime(RUN)
    assert raw["lancedb_version"] == lancedb.__version__
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_root = args.scratch_root.resolve() if args.scratch_root else output_dir
    if scratch_root.is_relative_to(native_root):
        raise NativeEvidenceOutputError("Audit scratch must be outside the native evidence directory.")
    scratch_root.mkdir(parents=True, exist_ok=True)
    scratch = scratch_root / ("za_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    scratch.mkdir()
    original_files = [raw_path, RUN / "cross_store.py", RUN / "verify_cross_store.py", RUN / "cross_store_protocol.md"]
    original_files += [RUN / f"{c['dataset']}_{c['seed']}_inputs.npz" for c in raw["cells"]]
    original_hashes = {str(p): sha(p) for p in original_files}
    assert original_hashes[str(RUN / "cross_store.py")] == raw["code_sha256"]
    assert original_hashes[str(RUN / "cross_store_protocol.md")] == raw["protocol_sha256"]
    native_before = file_inventory(native_root)
    db = lancedb.connect(str(native_root / "lance"))
    aggregated = defaultdict(list)
    seeds = []
    mutation_controls = []
    cut_swap = []
    native_plans = []
    checked_records = checked_returned = 0
    for c in raw["cells"]:
        name = f"{c['dataset']}_{c['seed']}"
        inp = RUN / (name + "_inputs.npz")
        assert sha(inp) == c["inputs_sha256"]
        z = np.load(inp)
        x, queries, replacements = z["x"], z["q"], z["replacements"]
        # Reconstruct all population/edit inputs without reading recorded maps.
        rng = np.random.default_rng(c["seed"])
        if c["dataset"] == "digits":
            digits = load_digits().data.astype(np.float32) / 16
            order = rng.permutation(len(digits))
            expected_queries, expected_x = digits[order[:100]], digits[order[100:]]
        else:
            expected_x = rng.random((2000, 64), dtype=np.float32)
            expected_queries = rng.random((100, 64), dtype=np.float32)
        expected_rep = expected_x[rng.permutation(len(expected_x))] + rng.normal(0, .02, expected_x.shape).astype(np.float32)
        assert np.array_equal(x, expected_x) and np.array_equal(queries, expected_queries) and np.array_equal(replacements, expected_rep)
        n = len(x)
        order = np.random.default_rng(c["seed"] + 777).permutation(n)
        updates = order[:n // 4]
        deletes = order[n // 4:n // 4 + n // 10]
        reinserts = deletes[:len(deletes) // 2]
        assert updates.tolist() == c["updates"] and deletes.tolist() == c["deletes"] and reinserts.tolist() == c["reinserts"]
        old_source = dict.fromkeys(range(n), 1)
        new_source = {i: 1 for i in range(n) if i not in set(map(int, deletes))}
        new_source.update({int(i): 2 for i in updates})
        new_source.update({int(i): 3 for i in reinserts})
        expected_vectors = {(i, v): x[i] if v == 1 else replacements[i] for i, v in set(old_source.items()) | set(new_source.items())}
        assert {int(k): int(v) for k, v in c["old_sqlite"].items()} == old_source
        assert c["old_sqlite"] == c["old_sqlite_after_writer_commit"]
        assert {int(k): int(v) for k, v in c["current_sqlite"].items()} == new_source
        assert c["old_epoch"] == 0 and c["current_epoch"] == 1
        old = db.open_table(name)
        old.checkout(c["old_lance_version"])
        current = db.open_table(name)
        current.checkout(c["current_lance_version"])
        for cut, handle in (("old", old), ("current", current)):
            plan = handle.search(queries[0]).nprobes(4).ef(128).limit(10).explain_plan()
            assert "ANNSubIndex" in plan and "ANNIvfPartition" in plan
            native_plans.append(dict(dataset=c["dataset"], seed=c["seed"], cut=cut, version=c[cut + "_lance_version"], plan=plan))
        old_keys, old_vectors = native_population(old)
        new_keys, new_vectors = native_population(current)
        assert set(old_keys) == set(old_source.items()) == set(keypairs(c["old_population"]))
        assert set(new_keys) == set(new_source.items()) == set(keypairs(c["current_population"]))
        for k, vector in old_vectors.items():
            assert vector.tobytes() == expected_vectors[k].tobytes(), (name, "old-vector", k)
        for k, vector in new_vectors.items():
            assert vector.tobytes() == expected_vectors[k].tobytes(), (name, "new-vector", k)
        # SQLite old reader was closed by the experiment; only the current DB
        # can be re-opened. Historical WAL assertions are recorded, not replayed.
        connection = sqlite3.connect((native_root / (name + ".sqlite")).as_uri() + "?mode=ro", uri=True)
        sql_current = list(connection.execute("select row,version,payload from docs order by row"))
        sql_epoch = connection.execute("select epoch from meta").fetchone()[0]
        connection.close()
        assert sql_epoch == 1 and {int(i): int(v) for i, v, p in sql_current} == new_source
        assert [list(a) for a in sql_current] == c["current_sqlite_payloads"]
        assert c["old_sqlite_payloads"] == [[i, 1, f"{i}:1"] for i in range(n)]
        assert all(p == f"{i}:{v}" for i, v, p in sql_current)
        rerun_old, rerun_new, rerun_40 = search(old, queries, 10), search(current, queries, 10), search(current, queries, 40)
        assert rerun_old == [keypairs(v) for v in c["answers_before"]] == [keypairs(v) for v in c["answers_after"]]
        assert rerun_40 == [keypairs(v) for v in c["native_candidates40"]]
        native_answers = {
            "admitted_old": rerun_old,
            "mismatched": rerun_new,
            "mismatched_filter": [[k for k in row if k in old_source.items()][:10] for row in rerun_40],
            "aligned_current": rerun_new,
        }
        old_truth = [exact_top10(expected_vectors, old_source, q) for q in queries]
        new_truth = [exact_top10(expected_vectors, new_source, q) for q in queries]
        cover_old = len(set(old_keys) & set(new_keys)) / len(old_keys)
        assert abs(cover_old - c["current_old_population_coverage"]) < 1e-14
        per_policy = {}
        for policy in native_answers:
            source = new_source if policy == "aligned_current" else old_source
            truth = new_truth if policy == "aligned_current" else old_truth
            records = sorted([r for r in c["records"] if r["policy"] == policy], key=lambda r: r["query"])
            assert len(records) == 100 and [r["query"] for r in records] == list(range(100))
            rows = []
            for j, (answer, target, saved) in enumerate(zip(native_answers[policy], truth, records)):
                invalid = len(invalid_keys(answer, source))
                recall = len(set(answer) & set(target)) / 10
                fill = len(answer) / 10
                assert answer == keypairs(saved["answers"]) and target == keypairs(saved["truth"])
                assert saved["invalid"] == invalid and abs(saved["recall"] - recall) < 1e-14 and saved["fill"] == fill
                rows.append(dict(invalid=invalid, recall=recall, fill=fill, returned=len(answer)))
                checked_records += 1
                checked_returned += len(answer)
            summary = dict(invalid=sum(r["invalid"] for r in rows), returned=sum(r["returned"] for r in rows), recall=sum(r["recall"] for r in rows) / 100, fill=sum(r["fill"] for r in rows) / 100,
                coverage=1.0 if policy in ("admitted_old", "aligned_current") else cover_old,
                query_invalid_range=[min(r["invalid"] for r in rows), max(r["invalid"] for r in rows)],
                query_recall_range=[min(r["recall"] for r in rows), max(r["recall"] for r in rows)])
            for metric in ("invalid", "returned", "recall", "fill"):
                assert abs(summary[metric] - c["summary"][policy][metric]) < 1e-12
            per_policy[policy] = summary
            aggregated[c["dataset"], policy].append(summary)
            swapped = old_source if policy == "aligned_current" else new_source
            cut_swap.append(dict(dataset=c["dataset"], seed=c["seed"], policy=policy,
                invalid_with_declared_cut=summary["invalid"], invalid_with_other_cut=sum(len(invalid_keys(a, swapped)) for a in native_answers[policy])))
            if summary["invalid"] == 0:
                baseline = native_answers[policy][0]
                bad_version = baseline.copy(); i, v = bad_version[0]; bad_version[0] = (i, v + 100)
                bad_id = baseline.copy(); bad_id[0] = (n + 10000, 1)
                changed_source = source.copy(); changed_source.pop(i)
                outcomes = dict(wrong_version=len(invalid_keys(bad_version, source)), absent_id=len(invalid_keys(bad_id, source)), changed_declared_source=len(invalid_keys(baseline, changed_source)))
                assert outcomes == dict(wrong_version=1, absent_id=1, changed_declared_source=1)
                mutation_controls.append(dict(dataset=c["dataset"], seed=c["seed"], policy=policy, original_invalid=0, outcomes=outcomes))
        seeds.append(dict(dataset=c["dataset"], seed=c["seed"], n=n, current_n=len(new_source), old_lance_version=c["old_lance_version"], current_lance_version=c["current_lance_version"], old_native_vectors_byte_equal=n, current_native_vectors_byte_equal=len(new_source), current_sqlite_rows_checked=len(sql_current), sql_epoch=sql_epoch, metrics=per_policy))
        print("native audit", name, "OK", per_policy["mismatched"]["invalid"], flush=True)
    table = []
    for (dataset, policy), values in aggregated.items():
        table.append(dict(dataset=dataset, policy=policy, invalid=sum(v["invalid"] for v in values), returned=sum(v["returned"] for v in values),
            coverage=sum(v["coverage"] for v in values) / len(values), recall=sum(v["recall"] for v in values) / len(values), fill=sum(v["fill"] for v in values) / len(values),
            per_seed_invalid=[v["invalid"] for v in values], per_seed_recall=[v["recall"] for v in values], seed_recall_range=[min(v["recall"] for v in values), max(v["recall"] for v in values)],
            zero_semantics=("exact output-version filter guarantees returned-key validity under the declared old-source map" if policy == "mismatched_filter" else "empirically observed against aligned source map" if policy in ("admitted_old", "aligned_current") else "deliberately mismatched source and candidate populations")))
    # Existing checker controls, performed only on a copied evidence bundle.
    checker_dir = scratch / "existing_checker_controls"
    checker_dir.mkdir()
    shutil.copy2(RUN / "verify_cross_store.py", checker_dir)
    for p in original_files:
        if p.suffix == ".npz": shutil.copy2(p, checker_dir)
    existing_checker_tests = []
    variants = {}
    altered = copy.deepcopy(raw); altered["dbdir"] = str(scratch / "nonexistent_native_db")
    variants["native_dbdir_missing"] = altered
    altered = copy.deepcopy(raw); altered["cells"][0]["records"][0]["answers"][0][1] += 100
    variants["returned_version_corrupted"] = altered
    altered = copy.deepcopy(raw); altered["cells"][0]["records"][0]["answers"][0][0] = 999999
    variants["returned_id_absent"] = altered
    altered = copy.deepcopy(raw); altered["cells"][0]["old_sqlite"]["0"] = 999
    variants["declared_source_map_corrupted"] = altered
    for label, evidence in variants.items():
        (checker_dir / "cross_store_results.json").write_text(json.dumps(evidence), encoding="utf-8")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(RUN / "vendor")
        result = subprocess.run([sys.executable, str(checker_dir / "verify_cross_store.py")], capture_output=True, text=True, env=environment)
        existing_checker_tests.append(dict(test=label, exit_code=result.returncode, passed=result.returncode == 0, output_tail=(result.stdout + result.stderr)[-800:]))
        assert (result.returncode == 0) == (label == "native_dbdir_missing")
        print("existing checker control", label, result.returncode, flush=True)
    # Fresh native same-key/wrong-vector control. This is deliberately outside
    # the original native root and original NPZ files.
    # Short diagnostic names avoid Windows native-writer path-length limits in
    # deeply nested extracted artifacts. These tables are fresh controls only.
    control_db = lancedb.connect(str(scratch / "vc"))
    control_x = np.zeros((4, 64), np.float32); control_x[:, 0] = np.arange(4, dtype=np.float32)
    bad_x = control_x.copy(); bad_x[0, 0] = 100
    def small_arrow(vectors):
        return pa.table({"row": pa.array(range(4), type=pa.int32()), "version": pa.array([1]*4, type=pa.int32()), "vector": pa.array(vectors.tolist(), type=pa.list_(pa.float32(), 64))})
    correct = control_db.create_table("a", small_arrow(control_x))
    corrupt = control_db.create_table("b", small_arrow(bad_x))
    good_keys, good_vec = native_population(correct); bad_keys, bad_vec = native_population(corrupt)
    control_source = dict.fromkeys(range(4), 1)
    top_good = keypairs([(r["row"], r["version"]) for r in correct.search(control_x[0]).limit(1).to_arrow().to_pylist()])
    top_bad = keypairs([(r["row"], r["version"]) for r in corrupt.search(control_x[0]).limit(1).to_arrow().to_pylist()])
    same_key_control = dict(native_key_population_equal=set(good_keys)==set(bad_keys), invalid_key_count_correct=len(invalid_keys(good_keys, control_source)), invalid_key_count_wrong_vector=len(invalid_keys(bad_keys, control_source)),
        vector_byte_mismatches=sum(good_vec[k].tobytes()!=bad_vec[k].tobytes() for k in good_vec), expected_top1=top_good, wrong_vector_top1=top_bad, recall_at_1_correct=1, recall_at_1_wrong_vector=len(set(top_good)&set(top_bad)))
    assert same_key_control["invalid_key_count_wrong_vector"] == 0 and same_key_control["vector_byte_mismatches"] == 1 and same_key_control["recall_at_1_wrong_vector"] == 0
    # The strengthened audit rejects invalid physical pointers before any
    # engine connection can create an empty database directory.
    path_controls = []
    missing_path = scratch / "missing_native_pointer"
    incorrect_path = scratch / "wrong_native_pointer"
    incorrect_path.mkdir()
    for label, path in (("missing_native_pointer", missing_path), ("incorrect_native_pointer", incorrect_path)):
        try:
            validate_native_root(path, raw["cells"])
        except FileNotFoundError as error:
            path_controls.append(dict(test=label, rejected=True, error=str(error)))
        else:
            raise AssertionError(f"Invalid physical-native pointer was accepted: {path}")
    assert not missing_path.exists(), "bad pointer must not cause engine-created database"
    assert {str(p): sha(p) for p in original_files} == original_hashes
    assert file_inventory(native_root) == native_before
    vector_byte_checks = sum(c["old_native_vectors_byte_equal"] + c["current_native_vectors_byte_equal"] for c in seeds)
    assert checked_records == 2400 and checked_returned == 24000 and vector_byte_checks == 21627
    assert len(mutation_controls) == 18 and len(native_plans) == 12
    out = dict(status="passed", mode=args.mode, checker_sha256=sha(Path(__file__)), evidence_run_dir=str(RUN), recorded_native_dbdir=raw["dbdir"], native_dbdir=str(native_root), native_dbdir_explicit_override=args.native_dbdir is not None, scratch=str(scratch), raw_sha256=original_hashes[str(raw_path)], original_file_hashes=original_hashes, pinned_runtime=runtime, lancedb_version=lancedb.__version__, sqlite_version=sqlite3.sqlite_version,
        original_files_unchanged=True, native_persisted_files_unchanged=True, records_checked=checked_records, returned_keys_checked=checked_returned, zero_row_returned_keys=sum(r["returned"] for r in table if r["invalid"]==0), mismatch_row_returned_keys=sum(r["returned"] for r in table if r["invalid"]), table=table, per_seed=seeds,
        native_vector_byte_checks=vector_byte_checks, physical_pointer_controls=path_controls, native_plans=native_plans, independent_mutation_controls=mutation_controls, source_cut_swap_controls=cut_swap, existing_checker_controls=existing_checker_tests, same_key_wrong_vector_native_control=same_key_control,
        limitations=["Historical SQLite epoch-0 read transaction was closed by the original experiment; this audit checks its recorded full map and payloads against an independently reconstructed old source, while the surviving SQLite DB is verified directly at epoch 1. This is not a replay of the old WAL transaction.", "Returned-version invalid counts check declared (ID, version) membership. They do not detect wrong embeddings retaining the same key; full native vector-byte identity is checked separately.", "Recorded native answers were replayed from the exact persisted Lance table versions 2 and 9, with the same search parameters; all 2,400 arm-query records matched. These seeded workload results are not a production anomaly-rate estimate.", "Zero is guaranteed only for the explicit exact postfilter under a correct source map. Pinned/aligned zeros are observed native results under aligned populations and immutable version checkout; the table does not establish universal implementation correctness."],
        elapsed_seconds=time.time()-started)
    (output_dir / "zero_table_audit.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(dict(status=out["status"], records=checked_records, keys=checked_returned, elapsed=out["elapsed_seconds"])))


if __name__ == "__main__":
    args = arguments()
    try:
        main(args)
    except Exception as error:
        if not isinstance(error, NativeEvidenceOutputError):
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "zero_table_audit.json").write_text(json.dumps(dict(status="failed", mode=args.mode, error_type=type(error).__name__, error=str(error)), indent=2), encoding="utf-8")
        raise
