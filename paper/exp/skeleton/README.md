# Graph and exact-retrieval diagnostics

These NumPy experiments evaluate controlled graph rewrites, predicate placement
and exact node-version retrieval. Run the commands from the repository root:

```sh
python paper/exp/skeleton/run_all.py
python paper/exp/skeleton/e8_multiseed.py
python paper/exp/skeleton/run_e9.py
```

For the smaller E6/E7 smoke run, set `SCR_QUICK=1` before running `run_all.py`.
In PowerShell:

```powershell
$env:SCR_QUICK = '1'
python paper/exp/skeleton/run_all.py
Remove-Item Env:SCR_QUICK
```

The drivers write `paper/exp/results/results_v2.json`,
`results_v2_multiseed.json` and `results_v2_e9.json`, respectively. Run in a copy
of the repository when preserving the included historical results.

## Experiments

- E6 uses 5,000 nodes and initial retrieval cuts 45 through 125. The 500
  updates and 120 deletions commit before that window, at commits 1 through 40,
  so both retrievals see the same logical population. Six graph rebuilds run
  between commits 60 and 120. The arms use the current graph with visibility
  filtering, a snapshot-bound graph generation, and a full-copy reference.
- E7 uses 8,000 nodes and visible fractions from 0.9 to 0.05. It compares
  unfiltered search, traversal through invisible nodes, post-filtering with
  `ef = k * 4`, and navigation restricted to visible nodes. BFS measures
  reachability within the visible subgraph. Search budgets differ, so recall
  differences do not establish an equal-work latency comparison.
- E8 calls `paper/proto/run_experiments.py` for seeds 20261008, 20261009 and
  20261010. The arms are asynchronous lag 50, 200 and 500; a global lock; and
  exact version-bound retrieval. This is a node-version diagnostic, not native
  graph-generation pinning.
- E9 varies graph-rebuild region size, beam width and link repair, with seeds
  11 and 22.

## Accounting and interpretation

`fullcopy` memory is modeled as the number of generations times vector and
adjacency bytes. Current-graph and SCR memory totals count adjacency-array
payloads only and deduplicate shared copy-on-write arrays. Their accounting
scopes differ: the modeled full-copy total includes vectors. All totals exclude
Python container overhead, native engine files and process memory.

E6 and E9 use a Python graph model with controlled maintenance. Their rates do
not estimate production-engine anomaly rates. E7 measures recall and
reachability under predicate placement; it does not certify adjacency
incompatibility. Native FAISS and database-engine controls are separate
experiments in the repository.
