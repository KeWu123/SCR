# Actual native support spectrum scale

Nine controlled hot-row history cells use 2,000 initial vectors, 32 dimensions, seeds 11/29/47, and
32/128/512 subsequent sealed integer source commits. Each commit is a real
SQLite transaction that updates one of four rows, creates a fresh positive
version ID, and advances the committed source head. All vectors use one fixed
Float32 encoder. Candidate population P is the union of source populations at
cuts 0,8,16,...,T. Updating the same four rows repeatedly ensures that missing
intermediate versions create actual holes; updating different rows only once
would incorrectly make the initial/final union cover all intermediate cuts.
The exact maximal support consists of the T/8+1 selected singleton components.

Both arms receive identical copied native index files, complete source history,
queries, domain and candidate budget. Each performs exactly one complete
native population inspection and one complete committed source catalogue scan.
Interval mode compiles maximal support from missing lifetimes once. The strong
point comparator certifies every integer cut in [0,T] against the same cached
population/catalogue, with early rejection on an uncovered version. It protects
every accepted future point cut under the same lifecycle rules, so source
retention is equal. Coalescing established point certificates is just storage
compression. No weaker point-wrapper baseline or assumed speed advantage is
used. Native inspection, catalogue extraction, isolated certification and full
publication/guard time are recorded separately. Source commit/build/copy costs
are outside publication and separately recorded where applicable.
After these equal one-time scans, seven paired warm-cache certificate
repetitions use the already cached populations, catalogues and lifetimes.
Arm order is independently shuffled with a fixed cell seed for each pair.
The first cold publication and warm repeated certification are reported
separately. Point comparisons retain genuine early exits, often stopping
within the first four hot rows on invalid cuts; reported work is actual visited
records, never an artificial full N-times-T count.

Every integer cut receives an actual Begin attempt. Every supported cut is
followed by an actual native query; every hole must reject. IVF_HNSW_FLAT uses
L2/four partitions/m=16/ef_construction=64; queries use nprobes=4, ef=128,
120 candidates and up to ten source-version-valid outputs. Native arrays,
complete source histories, query arrays, copied native indexes and all raw
answers are retained. An independent checker reconstructs all cuts directly,
compares exact native/source bytes and both arms' answers, replays native
queries, and computes scalar Float64 top-ten oracles for selected cuts. This
is a focused reference-policy scaling study with three cells per T and seven
paired certificate repetitions per cell. Query IDs are fixed before execution
as (cut/8) mod 16, and fill/recall/failures are reported for that fixed set;
population coverage does not imply a general ANN-recall guarantee. This history
is a deliberate stress construction and carries no production-prevalence claim.
The study
it does not establish a general speedup, SLA, or a new interval-union theorem.
