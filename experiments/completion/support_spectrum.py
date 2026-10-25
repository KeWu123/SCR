"""Exact maximal snapshot support from immutable version lifetimes.

No ANN recall assertion is made. Candidate identities include the encoder and
embedding bytes. Source lineage and a contiguous ingestion prefix are inputs.
"""
from dataclasses import dataclass
from bisect import bisect_right

@dataclass(frozen=True)
class Lifetime:
    key: str
    begin: int
    end: int  # exclusive
    identity: bytes | None  # None denotes tombstone

@dataclass(frozen=True)
class Support:
    intervals: tuple  # inclusive integer endpoints

    def contains(self, snapshot):
        if not isinstance(snapshot,int) or isinstance(snapshot,bool):
            raise ValueError('integer commit cuts required')
        i = bisect_right(self.intervals, (snapshot, float('inf'))) - 1
        return i >= 0 and self.intervals[i][0] <= snapshot <= self.intervals[i][1]

def certify(lifetimes, population, floor, head, prefix):
    """Return all covered integer cuts in [floor,min(head,prefix)].

    Lifetimes must be complete for this domain, obtained from the committed
    source lineage (including each row's anchor at floor). A population is an
    immutable key -> exact encoder/embedding identity mapping. A source GC
    guard must cover certificate creation and registration of these intervals.
    """
    if any(not isinstance(x, int) or isinstance(x, bool) for x in (floor, head, prefix)):
        raise ValueError('integer commit cuts required')
    if not 0 <= floor <= head or prefix < 0:
        raise ValueError('invalid readable domain')
    upper = min(head, prefix)
    if upper < floor:
        return Support(())
    debt = []
    seen = set()
    for v in lifetimes:
        if any(not isinstance(x,int) or isinstance(x,bool) for x in (v.begin,v.end)) or v.key in seen or v.begin < 0 or v.begin >= v.end:
            raise ValueError('invalid or duplicate version lifetime')
        seen.add(v.key)
        if v.identity is not None and population.get(v.key) != v.identity:
            lo, hi = max(floor, v.begin), min(upper, v.end - 1)
            if lo <= hi:
                debt.append((lo, hi))
    debt.sort()
    result, cursor = [], floor
    for lo, hi in debt:
        if lo > cursor:
            result.append((cursor, lo - 1))
        cursor = max(cursor, hi + 1)
    if cursor <= upper:
        result.append((cursor, upper))
    return Support(tuple(result))

def lifetimes_from_events(events, head):
    """Group complete (row,commit,key,identity) source events in linear space."""
    if not isinstance(head,int) or isinstance(head,bool) or head<0:
        raise ValueError('integer source head required')
    rows, keys = {}, set()
    for row, t, key, identity in events:
        if not isinstance(t,int) or isinstance(t,bool) or key in keys or not 0 <= t <= head:
            raise ValueError('source lineage identity/commit error')
        keys.add(key)
        rows.setdefault(row, []).append((t, key, identity))
    result = []
    for values in rows.values():
        values.sort()
        if len({v[0] for v in values}) != len(values):
            raise ValueError('ambiguous row commit')
        for i, (t, key, identity) in enumerate(values):
            end = values[i + 1][0] if i + 1 < len(values) else head + 1
            result.append(Lifetime(key, t, end, identity))
    return tuple(result)
