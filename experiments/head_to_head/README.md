# Native versioning and retention diagnostics

These drivers reproduce the paper's nine-cut native-version comparison and
array-storage retention diagnostic. The JSON files beside the drivers are the
recorded measurements. Their `sha` fields identify the original measurement
scripts, before the path and documentation changes in this release.

Run from the repository root:

```sh
python experiments/head_to_head/retention_instrumented.py --output-dir outputs/retention
python experiments/head_to_head/h2h_lance_vs_scr.py --output-dir outputs/native_versions
```

The retention driver requires NumPy. The native-version driver additionally
requires LanceDB 0.40.0 and PyArrow. SQLite is included with Python. Without
`--output-dir`, each driver creates a new timestamped directory under `runs/`.
New runs preserve the recorded JSON files.

## Native version comparison

The workload uses 2,000 synthetic 32-dimensional vectors, two 500-row update
groups, source cuts 0 through 8, and seeds 11, 29 and 47. It evaluates:

- the latest native table version at every requested source cut;
- the native version recorded at that cut by the experiment's bookkeeping;
- SCR admission from source lifetimes and a retained version population.

The recorded native-version-at-cut arm has recall@10 of 1.00. The latest-version
arm can return a population inconsistent with the requested historical cut.
SCR's retained population supports cuts `{0, 1, 4, 7, 8}` and rejects the holes.
The SCR arm records admission decisions; it does not run a separate SCR search
engine or compare end-to-end search latency. The cleanup probe checks reopening
an old version after immediate native cleanup; it does not measure concurrent
pin protection.

## Retention diagnostic

The workload uses 5,000 rows, dimension 64, base degree 12 and seven generations.
Six publications each allocate adjacency frames for a sampled 40% of rows.
The recorded copy-on-write adjacency total is 816,000 bytes, or 3.40 times the
240,000-byte base. Shared embeddings occupy 1,280,000 bytes. Together these
arrays occupy 2,096,000 bytes, or 1.3789 times the current-generation total.

Base and copy-on-write adjacency arrays are allocated and measured with
`nbytes`. The shared-vector and full-copy alternatives are byte totals computed
from the allocated arrays. They are not native engine footprints or process
memory measurements. Python container-size fields depend on the runtime:
`per_node_list_overhead_bytes` includes owned array data, and the two object-array
fields count the same pointer storage from different views. Field names and
recorded values are retained for comparison with the original results.
