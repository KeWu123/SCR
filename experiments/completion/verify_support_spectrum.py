"""Independent per-cut oracle, exhaustive small histories and scaled runs."""
from pathlib import Path
import itertools, json, random, time, hashlib
from support_spectrum import certify, lifetimes_from_events
HERE = Path(__file__).resolve().parent

def oracle(events, population, floor, head, prefix):
    valid = set()
    for s in range(floor, min(head, prefix) + 1):
        state = {}
        for row, t, key, identity in sorted(events, key=lambda e: e[1]):
            if t <= s:
                state[row] = (key, identity)
        if all(identity is None or population.get(key) == identity
               for key, identity in state.values()):
            valid.add(s)
    return valid

cases = 0
for rows in itertools.product(range(2), repeat=4):
    for deleted in itertools.product((False, True), repeat=4):
        events = [(row,t+1,str(t+1),None if deleted[t] else bytes([t+1])) for t,row in enumerate(rows)]
        life = lifetimes_from_events(events,4)
        for keep in itertools.product((False, True),repeat=4):
            p = {str(t+1):bytes([t+1]) for t in range(4) if keep[t] and not deleted[t]}
            for floor in (0,2,4):
                for prefix in (0,2,4):
                    support = certify(life,p,floor,4,prefix)
                    expected = oracle(events,p,floor,4,prefix)
                    assert {s for s in range(5) if support.contains(s)} == expected
                    cases += 1
# A missing middle version gives disconnected support; tombstones impose no
# candidate obligation, and later reinsertion does not resurrect its past.
events = [(0,0,'a',b'a'),(0,2,'b',b'b'),(0,4,'del',None),(0,6,'c',b'c')]
support = certify(lifetimes_from_events(events,8),{'a':b'a','c':b'c'},0,8,8)
assert support.intervals == ((0,1),(4,8))
wrong = certify(lifetimes_from_events(events,8),{'a':b'wrong','c':b'c'},0,8,8)
assert wrong.intervals == ((4,8),)

scaled=[]
for n in (1000,10000,100000):
    for seed in (11,29,47):
        rng=random.Random(seed);events=[]
        for i in range(n):events.append((i,0,str(i)+':0',i.to_bytes(8,'little')))
        # Every row is updated once at a different commit; a deterministic
        # mix retains old, new, or both identities.
        for i in range(n):events.append((i,i+1,str(i)+':1',(i+n).to_bytes(8,'little')))
        population={key:identity for row,t,key,identity in events if rng.random()<.85}
        lt=lifetimes_from_events(events,n)
        times=[]
        for rep in range(5):
            start=time.perf_counter_ns();s=certify(lt,population,0,n,n);times.append(time.perf_counter_ns()-start)
        probes=[0,n]+[rng.randrange(n+1) for _ in range(20)]
        for cut in probes:
            state={}
            for row,t,key,identity in events:
                if t<=cut:state[row]=(key,identity)
            assert s.contains(cut)==all(population.get(k)==v for k,v in state.values())
        scaled.append(dict(n=n,seed=seed,version_count=len(events),population_count=len(population),times_ns=times,intervals=s.intervals,checked_cuts=probes))
result=dict(exhaustive_cases=cases,disconnected_support=list(support.intervals),wrong_embedding_rejected=True,scaled=scaled,code_sha256=hashlib.sha256((HERE/'support_spectrum.py').read_bytes()).hexdigest())
(HERE/'support_spectrum_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('support spectrum:',cases,'exhaustive cases;',len(scaled),'scaled cells')
