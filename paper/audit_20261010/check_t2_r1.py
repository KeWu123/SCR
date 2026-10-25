"""Fresh targeted T2 checks; preserve all original simulator artifacts."""
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper' / 'exp' / 'skeleton'))
import experiments as E

class FakeGraph:
    """Inject only a pinned second-call fault, so constant zero must fail."""
    def __init__(self, V, **kwargs):
        self.N = len(V)
        self.calls = 0
    def build(self):
        return {i: np.array([(i + 1) % self.N], dtype=np.int32) for i in range(self.N)}
    def search(self, adj, q, k, ef, **kwargs):
        self.calls += 1
        return [(1., 1 if self.calls % 4 == 0 else 0)]

original_graph, original_rebuild = E.NSG, E.rebuild_region
E.NSG = FakeGraph
E.rebuild_region = lambda V, region, cur, M, diverse: {}
try:
    injected = E.e6_run(11, N=600, dim=4, M=2, efC=4, n_episodes=7, k=1, ef=2)
    assert injected['native_answer_change_count'] == 0
    assert injected['scr_answer_change_count'] == 7
    assert injected['scr_tear_rate'] == 1.
finally:
    E.NSG, E.rebuild_region = original_graph, original_rebuild

out = {'generated_by': 'audit_20261010/check_t2_r1.py',
       'classification': 'fresh personally executed targeted simulator validation; not engine instrumentation',
       'injected_pinned_fault': {'episodes': 7, 'observed_changes': 7, 'rate': 1.,
                               'native_changes': 0, 'hardcoded_zero_would_fail': True},
       'E6_headline_runs': [], 'E7_runs': [], 'E9_main_corner_runs': []}
path = Path(__file__).with_name('results_t2_r1.json')
start = time.time()
def save():
    out['runtime_s'] = round(time.time() - start, 1)
    path.write_text(json.dumps(out, indent=2), encoding='utf-8')

save()
for seed in (11, 22, 33):
    row = E.e6_run(seed, ef=48, rebuild_diverse=True)
    assert row['scr_answer_change_count'] == 0
    assert row['filter_violations'] == 0
    out['E6_headline_runs'].append(row)
    save()
    print('E6', seed, row['native_answer_change_count'], row['scr_answer_change_count'], flush=True)
for seed in (7, 8):
    row = E.e7_run(seed)
    out['E7_runs'].append(row)
    save()
    print('E7', seed, 'completed', flush=True)
for seed in (11, 22):
    row = E.e9_run(seed, merge_frac=.4, ef=16, n_episodes=200)
    assert row['filter_violations'] == 0
    out['E9_main_corner_runs'].append(row)
    save()
    print('E9', seed, row['tear_rate'], flush=True)

old = json.loads((ROOT / 'paper' / 'exp' / 'results' / 'results_v2.json').read_text(encoding='utf-8'))
comparisons = []
for row, prev in zip(out['E6_headline_runs'], old['E6']['headline_runs']):
    assert row['seed'] == prev['seed']
    comparisons.append({'seed': row['seed'], 'matching_keys': all(row[k] == prev[k] for k in (
        'native_tear_rate_set', 'native_recall_r1', 'scr_recall', 'retained_MB_native', 'retained_MB_scr')),
        'fresh_native_change_count': row['native_answer_change_count'],
        'fresh_pinned_change_count': row['scr_answer_change_count'],
        'old_pinned_rate_was_constant': prev['scr_tear_rate']})
out['E6_archived_key_comparisons'] = comparisons
out['E7_archived_comparison'] = out['E7_runs'] == old['E7']['runs']
out['E7_archived_numeric_comparison'] = all(a['rows'] == b['rows'] for a,b in zip(out['E7_runs'], old['E7']['runs']))
out['memory_scope'] = {'unique_adj_arrays': 'measured NumPy array payload only; dictionary/object allocator omitted',
                       'whole_index': 'array payload plus shared embedding array; excludes Python metadata',
                       'naive_fullcopy': 'modeled G*(N*d*4+N*M*4), not an allocated implementation'}
save()
print('E6 archive comparison', comparisons, flush=True)
print('E7 numeric parity', out['E7_archived_numeric_comparison'], flush=True)
