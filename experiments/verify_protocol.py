"""Independent snapshot enumeration, two numerical rounds and a real GC race."""
from pathlib import Path
import itertools,json,threading,hashlib
from protocol import Version,Manifest,Registry
RUN=Path(__file__).resolve().parent
def oracle(source,s):
    result={}
    for row in set(v.row for v in source):
        candidates=[v for v in source if v.row==row and v.seq<=s]
        if candidates:
            selected=max(candidates,key=lambda v:v.seq)
            if selected.embedding is not None:result[row]=selected
    return result
def rejects(f):
    try:f()
    except (ValueError,KeyError):return True
    return False
counts=[]
for offset in [0,100]:
    source=[Version('a',1,'a1',(1+offset,),'a1'),Version('b',2,'b1',(2+offset,),'b1'),Version('a',3,'a2',(3+offset,),'a2'),Version('a',4,'aD',None,'delete'),Version('a',5,'a3',(5+offset,),'a3')]
    live=[v for v in source if v.embedding is not None];count=0
    for bits in itertools.product([False,True],repeat=len(live)):
      population=tuple((v.key,v.embedding) for yes,v in zip(bits,live) if yes)
      for low in range(6):
       for high in range(low,6):
        registry=Registry(source);m=Manifest('g',low,high,5,tuple(range(1,6)),population)
        required={v.key for s in range(low,high+1) for v in oracle(source,s).values()}
        expect=required<=dict(population).keys();assert (not rejects(lambda:registry.publish(m)))==expect;count+=1
        if expect:
            tokens=[registry.begin(s,'g') for s in range(low,high+1)]
            registry.gc()
            for s,token in zip(range(low,high+1),tokens):assert registry.resolve(token)==oracle(source,s)
            registry.withdraw('g');assert rejects(lambda:registry.begin(low,'g'));registry.gc();assert 'g' in registry.manifests
            for token in tokens:registry.end(token)
            assert registry.gc()[1]==['g']
    r=Registry(source);m=Manifest('g',0,5,5,(1,2,3,4,5),tuple((v.key,v.embedding) for v in live));r.publish(m)
    bad=Manifest('bad',0,5,5,(1,2,4,5),m.population);assert rejects(lambda:r.publish(bad))
    selected=threading.Event();release=threading.Event();attempted=threading.Event();done=threading.Event();token=[]
    def hook():selected.set();assert release.wait(5)
    def admit():token.append(r.begin(1,'g',hook))
    def collect():attempted.set();r.withdraw('g');r.gc();done.set()
    a=threading.Thread(target=admit);a.start();assert selected.wait(5);b=threading.Thread(target=collect);b.start();assert attempted.wait(5);assert not done.is_set();release.set();a.join(5);b.join(5);assert done.is_set() and r.resolve(token[0])==oracle(source,1);r.end(token[0]);assert r.gc()[1]==['g']
    counts.append(count)
out=dict(status='passed',rounds=2,exhaustive_configurations=sum(counts),threaded_interleavings=2,source_sha256=hashlib.sha256((RUN/'protocol.py').read_bytes()).hexdigest(),checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());(RUN/'protocol_verification.json').write_text(json.dumps(out,indent=2));print(out)
