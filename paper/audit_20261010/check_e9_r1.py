"""Fresh full diverse-rebuild E9 matrix with actually measured pinned calls."""
import json
import sys
import time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper' / 'exp' / 'skeleton'))
import experiments as E
old = json.loads((ROOT / 'paper' / 'exp' / 'results' / 'results_v2_e9.json').read_text(encoding='utf-8'))
out = {'generated_by': 'audit_20261010/check_e9_r1.py',
       'scope': 'fresh simulated diverse region rebuild; actual two same-generation calls; not engine A5',
       'config': {'N': 5000, 'dim': 64, 'M': 12, 'efConstruction': 64,
                  'seeds': [11,22], 'episodes_per_cell_seed': 200,
                  'fracs': [.05,.1,.2,.4], 'efs': [16,24,48], 'rebuild_diverse': True},
       'runs': [], 'aggregate': [], 'archived_comparisons': []}
start = time.time()
path = Path(__file__).with_name('results_t2_e9_r1.json')
for ef in (16,24,48):
    for frac in (.05,.1,.2,.4):
        group = []
        for seed in (11,22):
            row = E.e9_run(seed, frac, n_episodes=200, ef=ef, rebuild_diverse=True)
            assert row['filter_violations'] == 0
            assert row['pinned_answer_change_count'] == 0
            prev = next(a for a in old['runs'] if a['seed']==seed and a['merge_frac']==frac
                        and a['ef']==ef and a['rebuild_diverse'])
            out['archived_comparisons'].append({'seed':seed, 'frac':frac, 'ef':ef,
                'all_old_numeric_fields_match': all(row[k]==v for k,v in prev.items())})
            out['runs'].append(row)
            group.append(row)
            print('E9', seed, ef, frac, row['tear_rate'], 'pinned', row['pinned_answer_change_count'], flush=True)
            out['runtime_s'] = round(time.time()-start,1)
            path.write_text(json.dumps(out,indent=2),encoding='utf-8')
        rates = [r['tear_rate'] for r in group]
        out['aggregate'].append({'ef':ef,'merge_frac':frac, 'mean':float(np.mean(rates)),
                                  'sample_sd':float(np.std(rates,ddof=1)),
                                  'pinned_count':sum(r['pinned_answer_change_count'] for r in group),
                                  'episodes':400})
out['runtime_s'] = round(time.time()-start,1)
path.write_text(json.dumps(out,indent=2),encoding='utf-8')
print('all archived numbers match',all(c['all_old_numeric_fields_match'] for c in out['archived_comparisons']),flush=True)
