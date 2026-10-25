# Native FAISS traversal evidence

The experiment instruments FAISS 1.15.1 `IndexHNSWFlat::search`. Three seeds
(11, 29, 47) supply 2,000 uniform 32-dimensional vectors and 200 queries each.
Search uses one OpenMP thread, `M=16`, construction beam 64, `k=10`, and
search beams 16, 64, and 128.

A uses natural insertion order; B inserts the same vectors in a fixed
permutation. C replaces the base-layer neighbor lists of 100 selected nodes
with B's lists, retaining A's upper layers. The primary structural comparison
tests every observation against the complete published universe `{A, B, C}`.
`AA` repeats A as a control. These four labels account for 7,200 native traces,
each paired with a trace-disabled call. The compressed records also retain a
later supplementary D arm (25% vector replacement and rebuild), adding 1,800
traces. D is excluded from the primary metrics and has a separate analysis.

## Verify the retained measurements

Install the repository's Python requirements, then run from this directory:

```sh
python verify_native.py
python analyze_maint_tear.py
```

The primary verifier accepts either freshly generated `raw_*.jsonl` or the
retained `raw_*.jsonl.gz`, preferring an uncompressed rerun when both exist.
It checks graph slots, trace on/off labels and distance bits, recorded neighbor
ranges, first-admission edges, exact distances, recall, and same-generation
controls. It writes `verification_summary.json`; `results_summary.json`
contains the retained primary statistics. Supplementary D records receive
trace on/off and graph-slot checks; their exact distance/recall oracle is
outside the primary verifier. The separate D analysis writes
`maintenance_verification.json`.

## Build and rerun

The included tagged FAISS archive is sufficient to prepare its source offline.
Building requires CMake 3.24+, a C++20 compiler, OpenMP, and system BLAS/LAPACK
development libraries. A pip FAISS wheel supplies the Python API, not these
C++ build dependencies. For example, Debian/Ubuntu packages are
`cmake g++ libopenblas-dev liblapack-dev`. On Windows use a configured C++
developer shell and supply a compatible BLAS library through CMake's standard
`BLAS_LIBRARIES` option if automatic discovery fails.

```sh
python prepare_source.py
python audit_source_patch.py
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target scr_native_trace --parallel
```

Run `build/scr_native_trace .` on single-configuration systems, or
`build/Release/scr_native_trace.exe .` with Visual Studio. Then run the Python
verifier. The harness reads the retained binary inputs and writes fresh graph
and raw JSONL files beside them. Use a disposable checkout to preserve the
archived measurements. Compiler and BLAS differences may change new ANN
builds; compare a rerun as a new measurement.

`prepare_inputs.py` regenerates the seeded input arrays and binaries. Existing
binary/NPZ bytes and their original hashes are retained. `HNSW.h.patch` and
`HNSW.cpp.patch` add observational hooks without replacing native search.
`audit_source_patch.py` verifies all 1,097 upstream files and permits exactly
those two changes. FAISS's MIT license is included as `LICENSE.faiss` and in
the upstream archive. No precompiled binaries or BLAS libraries are bundled.

## Source identities

`source_provenance.json` records the original supplied source hashes and each
compressed trace's compressed and uncompressed hashes. Compression uses an
empty stored filename and a zero timestamp. Packaging added archive extraction,
portable CMake configuration, explicit standard-library includes, and gzip I/O.
The verifier also selects the four primary labels explicitly because the
supplied raw files contain the later D arm. Primary scientific calculations
and stored graph/input/trace records are preserved. Historical summary file
inventories were removed; their scientific statistics remain. Original
instrumentation hashes describe the measurement host's line endings, while
the source audit reports hashes of the normalized extracted source.
