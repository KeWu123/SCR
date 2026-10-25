"""Check real retained index arrays and answers against unreclaimed history."""
from pathlib import Path
import json
import random
import sys
import numpy as np

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER / 'proto'))
from engine import BaseTable, VersionedIndex, embed, _resolve_latest
from run_experiments import build, make_text

def resolved(arrays, cut):
    out = {}
    for row, ver, ts, deleted in zip(arrays['rows'], arrays['vers'], arrays['ts'], arrays['dele']):
        if int(ts) <= cut and (int(row) not in out or (int(ts), int(ver)) > out[int(row)][:2]):
            out[int(row)] = (int(ts), int(ver), bool(deleted))
    return out

def main():
    seed, n, writes, floor = 20261008, 15000, 5000, 4500
    table, _ = build(n, random.Random(seed))
    idx = VersionedIndex()
    for v in table.versions:
        idx.on_write(v)
    rng = random.Random(seed)
    for i in range(writes):
        row = rng.randrange(n)
        idx.on_write(table.update(row, make_text(rng), ts=i + 1))
    cuts = [floor, 4750, 5000]
    full = {key: value.copy() for key, value in idx._arrays().items()}
    queries = [embed(make_text(random.Random(seed + j + 1))) for j in range(20)]
    before = {(s, j): idx.query(q, 10, s) for s in cuts for j, q in enumerate(queries)}
    raw_bytes, raw_versions = idx.index_bytes(), len(idx.ts)
    idx.gc(floor)
    retained = idx._arrays()
    for cut in cuts:
        assert resolved(retained, cut) == resolved(full, cut), ('lost version', cut)
    for cut in cuts:
        for j, query in enumerate(queries):
            assert idx.query(query, 10, cut) == before[cut, j], ('changed answer', cut, j)

    # A deleted row is reinserted above the floor, and an older routing entry
    # must never reappear. Check the actual arrays, including the tombstone.
    tiny, _ = build(2, random.Random(19))
    control = VersionedIndex()
    for v in tiny.versions:
        control.on_write(v)
    control.on_write(tiny.delete(0, ts=80))
    control.on_write(tiny.update(0, 'alpha beta', ts=100))
    original = {k: v.copy() for k, v in control._arrays().items()}
    control.gc(90)
    for cut in [90, 100, 110]:
        assert resolved(control._arrays(), cut) == resolved(original, cut)

    # Advancing the retention floor past a still-admitted cut loses a row's
    # required anchor. This fault must be observable to the independent oracle.
    faulty = VersionedIndex()
    for v in table.versions:
        faulty.on_write(v)
    faulty.gc(floor + 1)
    assert resolved(faulty._arrays(), floor) != resolved(full, floor)

    # Timestamp ties share the same declared version-id tie-break in querying
    # and reclamation, including a tombstone superseded at the same timestamp.
    tie_checks = 0
    for middle_deleted, last_deleted in [(False, False), (True, False), (False, True)]:
        tied_table, tied = BaseTable(), VersionedIndex()
        tied_table.insert(0, 'alpha', 0)
        if middle_deleted:
            tied_table.delete(0, 1)
        else:
            tied_table.update(0, 'beta', 1)
        if last_deleted:
            last = tied_table.delete(0, 1)
        else:
            last = tied_table.update(0, 'gamma', 1)
        for version in tied_table.versions:
            tied.on_write(version)
        answer = tied.query(embed('gamma'), 1, 1)
        assert answer == ([] if last_deleted else [(0, last.ver_id, 1.0)])
        tied.gc(1)
        assert tied.query(embed('gamma'), 1, 1) == answer
        tie_checks += 1
    for permutation in [[0, 1, 2], [2, 0, 1], [1, 2, 0]]:
        chosen = _resolve_latest(np.array(permutation), np.array([0, 0, 0]),
                                 np.array([1, 1, 1]), np.array([9, 3, 7]))
        assert chosen.tolist() == [0]
    result = {'seed': seed, 'N': n, 'writes': writes, 'floor': floor,
              'cuts': cuts, 'row_resolution_checks': n * len(cuts),
              'answer_checks': len(before), 'versions_before': raw_versions,
              'versions_after': len(idx.ts), 'pruned': idx.pruned,
              'represented_bytes_before': raw_bytes,
              'represented_bytes_after': idx.index_bytes(),
              'byte_scope': 'vectors, timestamps, versions, delete flags; excludes row ids, base table, Python/RSS',
              'delete_reinsert_checks': 3, 'wrong_floor_fault_detected': True,
              'timestamp_tie_gc_checks': tie_checks, 'tied_candidate_permutations': 3,
              'scope': 'sequential real-index lifetime check; atomic pin race checked separately'}
    (Path(__file__).parent / 'results_gc_r1.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
