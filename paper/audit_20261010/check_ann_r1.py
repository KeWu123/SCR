"""Adversarial snapshot-membership and dispatch/publication checks."""
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'paper'/'proto'))
import ann_experiments as A
from engine import BaseTable,VersionedIndex,embed
from run_experiments import build,attach,TruthCache,classify_fast
out={'ivf_checks':[],'publication_smoke':[]}
def fixture(tombstone=False,reinsert=False,split=False):
    tab=BaseTable();old=tab.insert(0,'alpha',0)
    for row in range(1,7):tab.insert(row,'alpha beta',0)
    if tombstone or reinsert:
        latest=tab.delete(0,1)
    else:latest=tab.update(0,'gamma',1)
    if reinsert:latest=tab.insert(0,'gamma',2)
    idx=A.IVF(nlist=2 if split else 1)
    idx.centroids=np.stack([embed('alpha'),embed('gamma')]) if split else np.zeros((1,64),np.float32)
    for v in tab.versions:idx.add(v)
    return tab,idx,old,latest
for name,tomb,reinsert,split in (('hidden_latest_below_beam',False,False,False),
         ('hidden_latest_unprobed',False,False,True),('tombstone_below_beam',True,False,False),
         ('deletion_then_reinsertion',True,True,True)):
    tab,idx,old,latest=fixture(tomb,reinsert,split)
    for mode in ('in','post'):
        for S in ((0,1,2) if reinsert else (0,1)):
            got=idx.query(embed('alpha'),1,S,nprobe=1,mode=mode)
            classified=classify_fast(got,TruthCache(tab),S)
            assert classified[1:]==(0,0,0),(name,mode,S,got,classified)
            if S==0:assert got[0][1]==old.ver_id
            if S==1 and tomb:assert all(r[0]!=0 for r in got)
            out['ivf_checks'].append({'case':name,'mode':mode,'S':S,'returned':got,
                                      'classification':list(classified)})
    if reinsert:
        got=idx.query(embed('gamma'),1,2,nprobe=2)
        assert got[0][1]==latest.ver_id
        out['reinsertion_latest_found']=True
out['tied_timestamp_checks']=[]
for name,tomb,reinsert in (('tied_update',False,False),('tied_tombstone',True,False),
                           ('tied_delete_reinsert',True,True)):
    tab=BaseTable();old=tab.insert(0,'alpha',1)
    if tomb:
        latest=tab.delete(0,1)
    else:
        latest=tab.update(0,'gamma',1)
    if reinsert:latest=tab.insert(0,'gamma',1)
    tab.insert(1,'alpha beta',0)
    for reverse in (False,True):
        idx=A.IVF(nlist=1);idx.centroids=np.zeros((1,64),np.float32)
        for v in reversed(tab.versions) if reverse else tab.versions:idx.add(v)
        for mode in ('in','post'):
            got=idx.query(embed('alpha' if tomb and not reinsert else 'gamma'),2,1,nprobe=1,mode=mode)
            c=classify_fast(got,TruthCache(tab),1)
            assert c[1:]==(0,0,0),(name,reverse,mode,got,c)
            own=[r for r in got if r[0]==0]
            if tomb and not reinsert:assert not own
            else:assert own and own[0][1]==latest.ver_id
            out['tied_timestamp_checks'].append({'case':name,'reverse_storage':reverse,'mode':mode,
                                                'returned':got,'classification':list(c)})
    # The ignored-token control must use the same version-ID tie break.
    ignored=A.NoSnapshotIndex()
    for v in reversed(tab.versions):ignored.on_write(v)
    got=ignored.query(embed('alpha' if tomb and not reinsert else 'gamma'),2,1)
    assert classify_fast(got,TruthCache(tab),1)[1:]==(0,0,0)
out['ignored_token_tie_controls']=3
before=json.loads(Path(__file__).with_name('results_ivf_before_r1.json').read_text())
out['static_reference_scope']=before['static_knobs']
out['static_after']=A.selectivity_sweep(**before['static_knobs'])
out['static_exact_numeric_parity']=out['static_after']==before['static_reference']
assert out['static_exact_numeric_parity']

class PauseIndex(VersionedIndex):
    stable_checks=0
    def query(self,q,k,snapshot):
        n=len(self.rows)
        time.sleep(.002)
        assert len(self.rows)==n,'writer appended during protected retrieval'
        self.stable_checks+=1
        return super().query(q,k,snapshot)
for seed in (1,2,3):
    tab,_=build(120,random.Random(seed));idx=PauseIndex();attach(idx,tab)
    r=A.run_ann(tab,idx,random.Random(seed),80,50,10,think_us=1000,embed_cost_us=200)
    assert r['n_anomaly_episodes']==0 and r['writes_completed']==80 and r['recall_at_k']==1
    out['publication_smoke'].append({'seed':seed,'episodes':50,'writes':80,'stable_checks':idx.stable_checks})
class InternalTypeError(VersionedIndex):
    calls=0
    def query(self,q,k,snapshot,nprobe=8,mode='in'):
        self.calls+=1
        raise TypeError('audit internal query failure; not dispatch')
tab,_=build(40,random.Random(1));idx=InternalTypeError();attach(idx,tab)
try:A.run_ann(tab,idx,random.Random(1),5,3,5,think_us=0)
except TypeError as exc:
    assert 'internal query failure' in str(exc) and idx.calls==1
    out['internal_typeerror_propagates_without_retry']=True
assert out.get('internal_typeerror_propagates_without_retry')
class BrokenWriter(VersionedIndex):
    fail=False
    def on_write(self,rv):
        if self.fail:raise ValueError('audit publication failure')
        return super().on_write(rv)
tab,_=build(40,random.Random(1));idx=BrokenWriter();attach(idx,tab);idx.fail=True
try:A.run_ann(tab,idx,random.Random(1),5,3,5,think_us=0)
except RuntimeError as exc:
    assert isinstance(exc.__cause__,ValueError)
    out['writer_error_rejects_result']=True
assert out.get('writer_error_rejects_result')
Path(__file__).with_name('results_ann_checks_r1.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print('IVF cases',len(out['ivf_checks']),'static parity',out['static_exact_numeric_parity'],
      'TypeError and writer failure guards pass')
