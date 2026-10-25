# Native interval experiments

`interval_broker.py` extends the measured point broker with exact, potentially
disconnected snapshot support. Both certification arms share native inspection,
source catalogue and lifecycle rules. Driver, broker and protocol bytes match
the identities stored in the measurement records.

The two retained runs include result JSON, source lineages, query arrays and
scaling input arrays. Saved machine paths are repository-relative; historical
and published hashes are recorded separately in `record_provenance.json`.
Physical SQLite/LanceDB snapshots are excluded. These saved records establish
the paper's observations; their presence does not establish a fresh replay.

Check support/answer records and source/input hashes with the standard library:

```bash
python experiments/completion/native_interval/verify_recorded_support.py
```

After installing the repository requirements, these regressions create fresh
databases and need no archived snapshot:

```bash
python experiments/completion/native_interval/domain_key_hook_regressions.py
```

Run the complete experiments and physical validators from the repository root:

```bash
python experiments/completion/native_interval/run_native_intervals.py
python experiments/completion/native_interval/verify_native_intervals.py
python experiments/completion/native_interval/adversarial_checks.py
python experiments/completion/native_interval/gc_name_reuse_regression.py
python experiments/completion/native_interval/spectrum_scaling.py
python experiments/completion/native_interval/verify_spectrum_scaling.py
```

The drivers create dated run directories and replace their respective
`latest.json` pointers. Use a disposable checkout to preserve the retained
measurement records. Physical validators fail when the selected databases are
missing; they do not substitute a fresh database for archived evidence.
Timing and scheduling can vary across runs.
