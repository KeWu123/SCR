"""Independent boundary/random oracle for support_spectrum; no paper inputs."""
from pathlib import Path
import sys, random, json
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from support_spectrum import certify,lifetimes_from_events,Lifetime

rng=random.Random(773190);cases=0
for trial in range(10000):
    head=rng.randrange(0,18);floor=rng.randrange(head+1);prefix=rng.randrange(head+5)
    events=[];population={};serial=0
    for row in range(rng.randrange(8)):
        times=sorted(rng.sample(range(head+1),rng.randrange(min(7,head+1)+1)))
        for t in times:
            key=f'{row}/{serial}';serial+=1
            identity=None if rng.random()<.3 else bytes([serial%256])
            events.append((row,t,key,identity))
            if rng.random()<.7:population[key]=identity if rng.random()<.8 else b'wrong'
    rng.shuffle(events)
    # Compute the live row state separately at each integer cut.
    expected=[]
    for cut in range(floor,min(head,prefix)+1):
        valid=True
        for row in set(e[0] for e in events):
            past=[e for e in events if e[0]==row and e[1]<=cut]
            if past:
                latest=max(past,key=lambda e:e[1])
                if latest[3] is not None and population.get(latest[2])!=latest[3]:valid=False
        if valid:expected.append(cut)
    got=certify(lifetimes_from_events(events,head),population,floor,head,prefix)
    actual=[s for s in range(head+1) if got.contains(s)]
    assert actual==expected,(events,floor,head,prefix,population,got,expected)
    cases+=1

def attempt(fn):
    try:return {'accepted':str(fn())}
    except Exception as exc:return {'rejected':type(exc).__name__+': '+str(exc)}
integer_support=certify((),{},0,2,2)
out={'random_integer_lineage_cases':cases,'typed_domain_counterexample_found':False,
     'noninteger_contains':attempt(lambda:integer_support.contains(.5)),
     'noninteger_lifetime_intervals':attempt(lambda:certify((Lifetime('x',.5,1.5,b'x'),),{},0,2,2).intervals),
     'noninteger_event_accepted':attempt(lambda:lifetimes_from_events([('r',.5,'x',b'x')],2)),
     'interpretation':'Boundary validation gap only: integral Lifetime/event/contains inputs are intended typed-domain preconditions.'}
(HERE/'support_results.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out))
