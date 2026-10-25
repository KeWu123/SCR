# Native interval integration and continuing publication

This is a local reference implementation of the paper's maximal support and
retention protocol. It extends, and does not overwrite, the previously measured
point broker. One same-user cooperating authority owns each physical SQLite
source and native table. Committed integer cuts are sealed; external mutation
of old source rows or destructive native cleanup is outside the contract.
The embedding identity is the exact finite Float32 vector under one fixed
encoder. A complete committed SQLite history, including GC anchors, supplies
version lifetimes. No version identity may be reused.

Native row/version columns must have genuine Arrow integer types and exact
integer values in the signed SQLite int64 domain; negative row identifiers
are allowed and version identifiers are positive. Null, Boolean, string or
floating key fields/values cannot be coerced into valid identities. Native
embeddings have fixed-size Float32 schema. All supplied table handles,
including cleanup/maintenance/drop arguments, must match the configured
table's physical identity. A shared name alone does not authorize an action.
Optional fault-injection/observation callbacks are non-reentrant: they may
observe supplied immutable descriptors and coordinate test events, but cannot
invoke public broker actions from their own thread or wait for a broker action
on another thread. Publication callbacks execute before registry visibility;
callback failure leaves no admitted token or published entry. Base and derived
GC/name cleanup and close clearing share one continuous reentrant guard.

All cells use 2,000 initial rows, 32 dimensions, seeds 11/29/47, LanceDB 0.40.0,
and IVF_HNSW_FLAT (L2, four partitions, m=16, ef_construction=64). Queries use
nprobes=4, ef=128, 120 native candidates, ten outputs, followed by exact source
version filtering. Both certification arms execute identical queries and have
identical ownership, pin, operation, withdrawal and GC rules. Neither is
claimed to improve ANN recall or beat a correct equivalent wrapper.

Disconnected population: group A replaces at cut 2, deletes at 4 and reinserts
at 6; group B replaces at 3, 5 and 7. The immutable native population is the
union of source populations at cuts 0,1,4,7,8. The exact maximal supported set
is [0,1] union {4} union [7,8]; missing cuts are 2,3,5,6. Both arms cache one
full native population inspection and one complete catalogue scan. The
interval arm compiles missing lifetimes; the strong point comparator checks
all nine cuts separately using the same cached catalogue and native map.
Point certificate coalescing is just storage compression after all point
checks, not an interval inference. Costs include native scan, catalogue scan,
certification, actual admission and query. Sixteen queries are run at each
accepted cut. All future/negative/noninteger boundaries reject.

The oldest component is withdrawn with an episode, audit and an actual
in-flight native operation; releasing the episode and audit keeps the source
floor at zero until the operation completes. The floor then moves to four,
then seven after the middle component is withdrawn. A late episode retains
seven after all support is withdrawn; releasing it moves the floor to eight.
Cleanup rejects while other components remain open. A physical successor is
installed before actual cleanup and opening the old native generation must
fail. Raw SQL and native copies before GC permit independent full replay;
final physical sources/native directories permit GC and reclamation checks.

Continuing publication: eight source commits each update 10%, delete 2%, and
reinsert the preceding delete set. Two old-cut episodes query concurrently;
a third thread requests the observed source head against the latest registry
entry (three of four attempts) or its published cut (one of four). Source
commit precedes native building. Every cycle actually registers the successor,
then performs Begin and a native query at that new cut, withdraws the previous
successor and performs real GC/cleanup. The initial old episodes keep source
floor zero throughout, and releasing them advances it to cut eight. Both
arms use equivalent full source/native certification; interval mode compiles
all support through the observed head, point mode certifies the current cut.
Full native scans and source scans occur inside the retention guard. Native
building occurs before publication outside it. No incremental technique is
claimed. Raw monotonic timestamps record critical-section holds and waits,
queries, all admissions including rejected requests, snapshot age in commits,
and source-commit-to-publication/admission delay. Initial warm publication is
excluded from continuing publication summaries. Threads have no arrival-rate
or throughput SLA: this is a modest saturation/interference functional study.

Source and native databases, complete source lineage and query arrays are
retained. Every continuing native table is archived after building; copying is
separately timed and included in source-to-publication delay (not in building
or publication hold). A complete physical source copy is saved before GC.
Retired whole native tables are physically dropped through the authority;
the read-only archives are evidence copies, not serving registry entries.
The independent checker does not import broker or driver code; it
reconstructs cuts from the lineage, checks exact embedding bytes, replays
native results and computes scalar Python Float64 top-ten oracles. Rejections
and histories are verified against the algebra, not author assertions.
