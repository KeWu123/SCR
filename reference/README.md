# Reference broker

`native_broker_strict.py` validates exact integer row/version identities and
committed source lineage before publishing a native generation. Publication
failure leaves the registry and newly acquired ownership locks unchanged.
This reference is separate from `experiments/live_broker.py`, whose original
source is retained for the reported measurements. The strict validation cost
is outside those timings.

Install the repository requirements, then run from the repository root:

```bash
python reference/verify_strict_broker.py
python reference/verify_retention_guards.py
```

Both commands create fresh SQLite/LanceDB fixtures under `reference/runs/`.
They check malformed keys and source histories, failure atomicity, lifecycle
retention and concurrent publication/admission guards. They require NumPy,
PyArrow and LanceDB; no database server or saved measurement database is needed.

`--historical-data` adds the historical native-schema check to the strict
checker. It requires the retained `experiments/live_db/` snapshot, which is
excluded from this source repository; missing data fails that check.
