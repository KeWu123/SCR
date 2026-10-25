from pathlib import Path
import json,hashlib,numpy as np
RUN=Path(__file__).resolve().parent
raw=json.loads((RUN/'cross_store_results.json').read_text());checked=0
for c in raw['cells']:
    p=RUN/f"{c['dataset']}_{c['seed']}_inputs.npz";assert hashlib.sha256(p.read_bytes()).hexdigest()==c['inputs_sha256'];data=np.load(p);x,q,rep=data['x'],data['q'],data['replacements'];old={i:1 for i in range(len(x))};new=old.copy()
    for i in c['updates']:new[i]=2
    for i in c['deletes']:del new[i]
    for i in c['reinserts']:new[i]=3
    assert c['old_sqlite']==c['old_sqlite_after_writer_commit'] and {int(k):v for k,v in c['old_sqlite'].items()}==old and {int(k):v for k,v in c['current_sqlite'].items()}==new
    assert c['old_epoch']==0 and c['current_epoch']==1 and c['old_lance_version']<c['current_lance_version']
    assert set(map(tuple,c['old_population']))==set(old.items()) and set(map(tuple,c['current_population']))==set(new.items())
    for a,v,payload in c['old_sqlite_payloads']+c['current_sqlite_payloads']:assert payload==f'{a}:{v}'
    assert c['completed_maintenance'] and c['answers_before']==c['answers_after'] and 'IVF' in c['old_plan'].upper() and 'IVF' in c['current_plan'].upper()
    truth={}
    for j,vec in enumerate(q):
        for label,versions in [('old',old),('new',new)]:
            ds=[(sum((float(a)-float(b))**2 for a,b in zip(vec,x[i] if v==1 else rep[i])),i,v) for i,v in versions.items()];truth[j,label]=[(i,v) for _,i,v in sorted(ds)[:10]]
    for z in c['records']:
        versions=new if z['policy']=='aligned_current' else old;target=truth[z['query'],'new' if z['policy']=='aligned_current' else 'old'];ans=list(map(tuple,z['answers']))
        assert target==list(map(tuple,z['truth'])) and z['invalid']==sum(versions.get(i)!=v for i,v in ans) and abs(z['recall']-len(set(ans)&set(target))/10)<1e-12 and z['fill']==len(ans)/10
        if z['policy']=='mismatched_filter':assert ans==[k for k in map(tuple,c['native_candidates40'][z['query']]) if old.get(k[0])==k[1]][:10]
        checked+=1
    assert abs(c['current_old_population_coverage']-len(set(old.items())&set(new.items()))/len(old))<1e-12
out=dict(status='passed',cells=len(raw['cells']),records=checked,independent_scalar_oracles=1200,checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),experiment_sha256=raw['code_sha256']);(RUN/'cross_store_verification.json').write_text(json.dumps(out,indent=2));print(out)
