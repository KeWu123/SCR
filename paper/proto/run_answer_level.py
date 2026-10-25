#!/usr/bin/env python
"""Citation validity against a declared snapshot, using the shared repaired driver.

An evidence bundle consists of the observed relational fact and the two returned
sets of (row, version) citations. A fixed-snapshot validator rejects a dirty,
superseded or deleted-at-s citation, or differing citations for identical exact
queries at the same token. This does not certify that no alternative snapshot
can explain the bundle, and it does not measure generated answer quality.

The original results_answer_level.json is historical. A default run writes
results_answer_level_corrected.json; --output can select another new artifact.
"""
import argparse
import json
import os
import random
import time
import numpy as np
from engine import AsyncIndex, VersionedIndex
from run_experiments import attach, build
from run_episodes_v3 import RealLock, run_episode_workload

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_answer_level_corrected.json")

def wilson_upper(k, n, z=1.959963984540054):
    if n <= 0:
        return None
    p, zz = k / n, z * z
    return (p + zz / (2*n) + z * ((p*(1-p) + zz/(4*n))/n)**0.5) / (1 + zz/n)

def run_answer_level_episodes(table, index, rng, n_writes, n_reads, k,
                              lock=None, embed_cost_us=150.0, think_us=20000.0,
                              max_row=None, np_seed=0, collect_witness=False,
                              between_calls_us=1000.0):
    r = run_episode_workload(table, index, rng, n_writes, n_reads, k,
                              lock=lock or RealLock(), embed_cost_us=embed_cost_us,
                              think_us=think_us, max_row=max_row, np_seed=np_seed,
                              between_calls_us=between_calls_us)
    return {**r, "dangling_evidence_rate": r["a2_superseded_episode_rate"],
            "phantom_citation_rate": r["a3_phantom_episode_rate"],
            "n_dangling": r["a2_count"], "n_phantom": r["a3_count"],
            "n_disagree": round(r["citation_disagreement_rate"] * r["episodes"]),
            "n_invalid": r["n_invalid_bundles"],
            "witnesses": r["witnesses"] if collect_witness else []}

def main(output_path=None):
    start = time.time()
    schemes = ("async50", "async200", "async500", "scr")
    seeds = (20261008, 20261009, 20261010)
    runs = []
    for scheme in schemes:
        for seed in seeds:
            rng = random.Random(seed)
            table, _ = build(15000, rng)
            index = AsyncIndex(int(scheme[5:])) if scheme.startswith("async") else VersionedIndex()
            attach(index, table)
            r = run_answer_level_episodes(table, index, rng, 5000, 1200, 10,
                                           max_row=15000, np_seed=seed,
                                           collect_witness=True)
            r.update(scheme=scheme, seed=seed)
            runs.append(r)
            print(scheme, seed, r["bundle_invalidation_rate"], flush=True)
    aggregate = {}
    for scheme in schemes:
        rs = [r for r in runs if r["scheme"] == scheme]
        a = {}
        for key in ("dangling_evidence_rate", "phantom_citation_rate",
                    "citation_disagreement_rate", "bundle_invalidation_rate"):
            vals = [r[key] for r in rs]
            a[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)), "values": vals}
        a["pooled_episodes"] = sum(r["episodes"] for r in rs)
        a["pooled_invalid"] = sum(r["n_invalid"] for r in rs)
        a["conditional_binomial_wilson_upper95"] = wilson_upper(a["pooled_invalid"], a["pooled_episodes"])
        aggregate[scheme] = a
    doc = {"generated_by": "proto/run_answer_level.py (shared run_episodes_v3 core)",
           "scoring": "fixed-snapshot citation validity; no no-state certificate; no generated-answer quality",
           "inference_scope": "pooled Wilson is a conditional binomial reference, not independent writer-trace replication",
           "config": {"n_rows": 15000, "n_writes": 5000, "n_reads": 1200, "k": 10,
                      "seeds": list(seeds), "think_us": 20000, "between_calls_us": 1000},
           "aggregate": aggregate, "runs": runs, "runtime_s": time.time() - start}
    target = output_path or OUT
    with open(target, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    print("saved", target)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=OUT)
    main(parser.parse_args().output)
