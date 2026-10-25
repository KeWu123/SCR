# Reproducing the experiments

Run commands from the repository root. The latest evaluated manuscript is
identified by its title in README.md. This repository contains its research
code and recorded evidence; manuscript build files and editorial working
directories are excluded.

## Environment and checks

Install `requirements.txt` in a 64-bit Python 3.12 environment. The native
measurements used Python 3.12.3, SQLite 3.50.1, NumPy 1.26.4, PyArrow 21.0.0,
LanceDB 0.40.0 and FAISS 1.15.1. Python supplies SQLite; changing the Python
installation can change SQLite. The direct dependency pins do not lock every
transitive package. Earlier prototypes have their historical environment
record in `paper/audit_20261010/evidence_manifest_r1.json`.

```bash
python -m pip install -r requirements.txt
python run.py inventory
python run.py verify
python run.py verify --suite native
python run.py verify --suite evidence
```

The core suite checks the finite protocol, recorded async episodes, support
compiler and interval/scaling records. The native suite creates fresh small
SQLite/LanceDB fixtures for strict identities, retention, failure atomicity
and hook behavior. The evidence suite reconstructs scalar exact oracles for
the saved cross-store records and validates compressed full-hierarchy traces.
It takes longer than the other suites. `--suite all` runs all three.

The launcher uses a temporary copy for verification. Checks run with assertions
enabled; do not use Python's `-O` flag. A missing dependency or failed checker
fails the suite. Generated verification output describes the new check and is
discarded with the temporary copy. Stored verification JSON files describe
historical checks, rather than a new performance measurement.

## Main native experiments

Use a disposable checkout for full reruns: drivers can replace their result
JSON, input arrays and latest-run pointers. Their physical databases are
generated locally and excluded from Git. Once a driver finishes, run its
independent checker against those outputs.

| Experiment | Generate new measurements | Check new outputs |
|---|---|---|
| SQLite WAL/LanceDB snapshot alignment | `python run.py run experiments/cross_store.py` | `python run.py run experiments/verify_cross_store.py` |
| Point broker admission and reclamation | `python run.py run experiments/run_live_broker.py` | `python run.py run experiments/verify_live_broker.py` |
| Draining FIFO ingestion queue | `python run.py run experiments/async_queue.py` | `python run.py run experiments/verify_async.py` |
| Finite pin/admission model | — | `python run.py run experiments/verify_protocol.py` |
| Maximal support compiler | — | `python run.py run experiments/completion/verify_support_spectrum.py` |
| Correct native alternatives and costs | `python run.py run experiments/completion/compare_native_alternatives.py` | `python run.py run experiments/completion/verify_comparative_native.py` |
| Native disconnected interval support | `python run.py run experiments/completion/native_interval/run_native_intervals.py` | `python run.py run experiments/completion/native_interval/verify_native_intervals.py` |
| Source-history scaling | `python run.py run experiments/completion/native_interval/spectrum_scaling.py` | `python run.py run experiments/completion/native_interval/verify_spectrum_scaling.py` |

The comparative benchmark includes the 50,000-object cases and concurrency
controls; it is a full experiment rather than a smoke check. Its physical
checker requires the exact databases created by that new run. The same applies
to the live-broker and interval physical checkers. They must not silently
substitute fresh databases when checking an archived record. This source
repository does not include archived SQLite or LanceDB database directories.

For a physical audit of a newly generated alignment run, pass its actual
database directory explicitly:

```bash
python run.py run experiments/completion/verify_zero_table.py --mode physical-native --run-dir experiments --native-dbdir experiments/cross_store_db/NEW_RUN --output-dir validation/zero_table
```

Use the real directory name in place of `NEW_RUN`. Dependencies are checked
against the pinned versions. See [reference/README.md](reference/README.md) and
[the interval README](experiments/completion/native_interval/README.md) for
the lifecycle and boundary regression commands.

## Native table versions and memory accounting

```bash
python run.py run experiments/head_to_head/h2h_lance_vs_scr.py
python run.py run experiments/head_to_head/retention_instrumented.py
```

These drivers write separate timestamped outputs. Their READMEs define the
exact historical-cut schedule and distinguish allocated arrays from computed
full-copy byte totals. The native latest-version arm is a declared mismatched
source-cut control; it is not a claim that the backend violates its own API.

## Full native graph traces

[experiments/native_trace/README.md](experiments/native_trace/README.md)
documents the instrumented C++ harness, upstream source audit, native build
dependencies and rerun commands. The three gzip files retain every original
JSONL byte. The primary metrics use A, B, C and the repeated-A control; a later
D arm is retained and analyzed separately. A Python wheel alone is insufficient
to build the C++ harness.

## Earlier prototypes and controls

```bash
python run.py run paper/proto/run_experiments.py
python run.py run paper/proto/run_episodes_v3.py
python run.py run paper/proto/ann_experiments.py
python run.py run paper/exp/skeleton/run_all.py
python run.py run paper/exp/skeleton/e8_multiseed.py
python run.py run paper/exp/skeleton/run_e9.py
```

The corrected original exact diagnostic is retained in
`paper/audit_20261010/results_episodes_r1_corrected.json`. Its historical source
drivers are under `completion/measurement_sources/`; the active prototype
contains later guards. Historical and release file hashes are separate in
`PROVENANCE.json`, including documentation-only revisions to historical drivers.
Recorded ANN answer differences, native graph observations and modeled array
accounting measure different quantities.

PostgreSQL controls require a disposable PostgreSQL 17.4 database with pgvector
0.8.0 and `requirements-engine.txt`. They create and replace experiment tables
and indexes. Configure `SCR_PG_HOST`, `SCR_PG_PORT`, `SCR_PG_DB`, `SCR_PG_USER`
and standard PostgreSQL authentication. Drivers that start their own local
server also require `SCR_PGBIN`, `SCR_PGDATA`, `SCR_PGLOG`; set
`SCR_PGVECTOR_SOURCE` when the extension is outside the normal library path.
These inherited server-start helpers target Windows PostgreSQL installations.

```bash
python -m pip install -r requirements-engine.txt
python run.py run paper/proto/engine_multiseed.py
python run.py run paper/proto/engine_second.py
python run.py run paper/proto/engine_second_pin.py
python run.py run paper/proto/engine_matched2_lance.py
python run.py run paper/proto/engine_matched2_pg.py
python run.py run paper/proto/merge_matched2.py
python run.py run paper/proto/decompose_recall_conditioned.py
```

A rerun under a new compiler, SQLite version, numerical environment or scheduler
is a new measurement. Preserve its runtime versions and raw observations when
comparing it with the recorded results.
