"""Directed final R3 interface checks, without any full-scope rerun."""
import json
import random
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'paper'/'proto'))
import ann_experiments as A
from engine import VersionedIndex,embed
from run_experiments import build,attach
out={'slow_writer_completion':[]}
for seed in (41,42,43):
    tab,_=build(80,random.Random(seed));idx=VersionedIndex();attach(idx,tab)
    r=A.run_ann(tab,idx,random.Random(seed),80,1,5,think_us=0,embed_cost_us=2000)
    assert r['writes_at_reader_end']<80
    assert r['writes_completed']==80 and len(tab.versions)==160
    assert 'temporal_anomaly_episode_rate' not in r and 'n_temporal_anomaly_episodes' not in r
    assert r['future_version_episodes']==0 and r['future_version_rate']==0
    out['slow_writer_completion'].append({k:r[k] for k in ('episodes','writes_at_reader_end','writes_completed',
                                                         'future_version_episodes','future_version_rate')})
tab,_=build(80,random.Random(44));idx=VersionedIndex();attach(idx,tab)
try:
    A.run_ann(tab,idx,random.Random(44),80,1,5,think_us=0,embed_cost_us=10000,
              writer_join_timeout_s=.001)
except RuntimeError as exc:
    assert 'did not terminate' in str(exc)
    out['unfinished_writer_rejected']=True
assert out.get('unfinished_writer_rejected')
time.sleep(.03)  # Allow the deliberately cancelled paced worker to exit.

class FixedResultIndex(VersionedIndex):
    def query(self,q,k,snapshot):return self.fixed
# Superseded and deleted citations are overall anomalies but not future versions.
tab,_=build(10,random.Random(45));old=tab.versions[0]
tab.update(0,'gamma delta',0)
idx=FixedResultIndex();attach(idx,tab);idx.fixed=[(old.row_id,old.ver_id,1.)]
stale=A.run_ann(tab,idx,random.Random(45),0,1,1,think_us=0)
assert stale['anomaly_episode_rate']==1 and stale['stale_rate']==1
assert stale['future_version_episodes']==0 and stale['future_version_rate']==0
tab.delete(0,0)
idx=FixedResultIndex();attach(idx,tab);idx.fixed=[(old.row_id,old.ver_id,1.)]
deleted=A.run_ann(tab,idx,random.Random(45),0,1,1,think_us=0)
assert deleted['anomaly_episode_rate']==1 and deleted['phantom_rate']==1
assert deleted['future_version_episodes']==0 and deleted['future_version_rate']==0
# A deliberately known future commit must enter the future field.
tab,_=build(10,random.Random(46));future=tab.update(0,'gamma delta',100)
idx=FixedResultIndex();attach(idx,tab);idx.fixed=[(future.row_id,future.ver_id,1.)]
ahead=A.run_ann(tab,idx,random.Random(46),0,1,1,think_us=0)
assert ahead['anomaly_episode_rate']==1 and ahead['future_version_episodes']==1 and ahead['future_version_rate']==1
out['scope_cases']={name:{k:r[k] for k in ('anomaly_episode_rate','future_version_episodes','future_version_rate',
     'dirty_rate','stale_rate','phantom_rate')} for name,r in (('superseded',stale),('deleted',deleted),('future',ahead))}
Path(__file__).with_name('results_ann_interfaces_r1.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
