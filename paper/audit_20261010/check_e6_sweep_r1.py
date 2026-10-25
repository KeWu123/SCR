"""Fresh N=5000 E6 sweep; reuse identical construction, execute every search."""
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'paper'/'exp'/'skeleton'))
import experiments as E
original_build=E.NSG.build
original_rebuild=E.rebuild_region
build_cache={}
region_cache={}
stats={'actual_initial_builds':0,'actual_region_rebuilds':0,'cached_initial_builds':0,'cached_region_rebuilds':0}
def digest_adj(adj):
    h=hashlib.sha256()
    for i,a in sorted(adj.items()):
        h.update(int(i).to_bytes(8,'little'))
        h.update(len(a).to_bytes(8,'little'))
        h.update(np.asarray(a,dtype=np.int32).tobytes())
    return h.hexdigest()
def cached_build(self):
    key=(hashlib.sha256(self.V.tobytes()).hexdigest(),self.M,self.efC,self.l1_ratio,
         json.dumps(self.rng.bit_generator.state,sort_keys=True))
    if key not in build_cache:
        adj=original_build(self)
        build_cache[key]=(adj,self.entry,self.adj1)
        stats['actual_initial_builds']+=1
    else:
        stats['cached_initial_builds']+=1
    adj,self.entry,self.adj1=build_cache[key]
    return adj
def cached_rebuild(V,region,cur,M,diverse=True):
    key=(hashlib.sha256(V.tobytes()).hexdigest(),hashlib.sha256(region.tobytes()).hexdigest(),
         digest_adj(cur),M,diverse)
    if key not in region_cache:
        region_cache[key]=original_rebuild(V,region,cur,M,diverse=diverse)
        stats['actual_region_rebuilds']+=1
    else:
        stats['cached_region_rebuilds']+=1
    return region_cache[key]
E.NSG.build=cached_build
E.rebuild_region=cached_rebuild
old=json.loads((ROOT/'paper'/'exp'/'results'/'results_v2.json').read_text(encoding='utf-8'))
out={'generated_by':'audit_20261010/check_e6_sweep_r1.py',
     'scope':'fresh simulated graph searches; content-keyed construction reused across beams; no engine A5 certificate',
     'config':{'N':5000,'dim':64,'M':12,'efConstruction':64,'episodes_per_run':300,
               'seeds':[11,22],'efs':[16,24,48],'rebuild_diverse':[True,False]},
     'runs':[],'aggregate':[],'archived_comparisons':[]}
start=time.time()
path=Path(__file__).with_name('results_t2_e6_sweep_r1.json')
for div in (True,False):
    for ef in (16,24,48):
        group=[]
        for seed in (11,22):
            r=E.e6_run(seed,ef=ef,rebuild_diverse=div)
            assert r['scr_answer_change_count']==0
            assert r['filter_violations']==0
            group.append(r); out['runs'].append(r)
            prev=next((a for a in old['E6']['sweep_runs'] if a['seed']==seed and a['ef']==ef
                       and a['rebuild_diverse']==div),None)
            if prev:
                out['archived_comparisons'].append({'seed':seed,'ef':ef,'diverse':div,
                    'all_old_numeric_fields_match':all(r[k]==v for k,v in prev.items())})
            out['runtime_s']=round(time.time()-start,1)
            out['construction_reuse']=stats.copy()
            path.write_text(json.dumps(out,indent=2),encoding='utf-8')
            print('E6',seed,ef,div,'native',r['native_answer_change_count'],
                  'pinned',r['scr_answer_change_count'],flush=True)
        rates=[r['native_answer_change_count']/r['episodes'] for r in group]
        out['aggregate'].append({'ef':ef,'rebuild_diverse':div,'mean':float(np.mean(rates)),
            'population_sd':float(np.std(rates,ddof=0)),'sample_sd':float(np.std(rates,ddof=1)),
            'native_count':sum(r['native_answer_change_count'] for r in group),
            'pinned_count':sum(r['scr_answer_change_count'] for r in group),
            'pinned_rate':sum(r['scr_answer_change_count'] for r in group)/600,'episodes':600})
out['runtime_s']=round(time.time()-start,1)
out['construction_reuse']=stats.copy()
path.write_text(json.dumps(out,indent=2),encoding='utf-8')
print('all archived matched',all(c['all_old_numeric_fields_match'] for c in out['archived_comparisons']),flush=True)
