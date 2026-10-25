"""Maintenance-tear analysis: replace 25% of vectors + rebuild, node set unchanged.

Measures cross-maintenance (A vs D) answer agreement and positive-edge
non-embeddability using the same instrumentation traces as the native-trace
experiment. Independent of the manuscript; reads saved graphs + raw traces only.
"""
from pathlib import Path
import json, collections, statistics, gzip
import numpy as np

ROOT = Path(__file__).resolve().parent
summary = {'scope': 'maintenance tear: replace last 25% of vectors then rebuild; node set unchanged',
           'seeds': []}

for seed in (11, 29, 47):
    inp = np.load(ROOT / f'input_{seed}.npz')
    xb = inp['vectors']; queries = inp['queries']
    edges = {}; neighbor_sets = {}
    for name in ('A', 'D'):
        g = json.loads((ROOT / f'graph_{seed}_{name}.json').read_text())
        assert g['generation'] == name and len(g['nodes']) == 2000
        edges[name] = set(); neighbor_sets[name] = {}
        for node in g['nodes']:
            for layer in node['layers']:
                lev, u, slots = layer['level'], node['id'], layer['slots']
                positive = [v for v in slots if v >= 0]
                neighbor_sets[name][lev, u] = frozenset(positive)
                edges[name].update((lev, u, v) for v in positive)
    A_only = sum(1 for e in edges['A'] if e not in edges['D'])
    D_only = sum(1 for e in edges['D'] if e not in edges['A'])
    common = sum(1 for e in edges['A'] if e in edges['D'])
    jaccard = common / (common + A_only + D_only) if (common + A_only + D_only) else 1.0

    records = {}
    raw=ROOT/f'raw_{seed}.jsonl'
    if not raw.exists():
        raw=raw.with_suffix('.jsonl.gz')
    opener=gzip.open if raw.suffix=='.gz' else open
    for line in opener(raw,mode='rt',encoding='utf-8'):
        r = json.loads(line)
        if r['generation'] not in ('A', 'D'):
            continue
        observed = set()
        for lev, u, slots in r['expansions']:
            observed.update((lev, u, v) for v in slots if v >= 0)
        records[r['ef'], r['query'], r['generation']] = {'labels': r['labels_on'], 'edges': observed}

    out = {'seed': seed, 'adjacency_jaccard': round(jaccard, 4),
           'A_only_edges': A_only, 'D_only_edges': D_only, 'common_edges': common, 'cells': []}
    for ef in (16, 64, 128):
        data = collections.Counter()
        for qi in range(200):
            a = records[ef, qi, 'A']; d = records[ef, qi, 'D']
            union = a['edges'] | d['edges']
            noemb = not (union <= edges['A'] or union <= edges['D'])
            agree = set(a['labels']) == set(d['labels'])
            data['episodes'] += 1
            data['positive_edge_no_embedding'] += noemb
            data['answer_agree'] += agree
            data['invisible_tear'] += (noemb and agree)
        data = dict(data)
        data.update({'ef': ef,
                     'no_embedding_rate': data['positive_edge_no_embedding'] / 200,
                     'answer_agree_rate': data['answer_agree'] / 200,
                     'invisible_tear_rate': data['invisible_tear'] / 200})
        out['cells'].append(data)
    summary['seeds'].append(out)

summary['aggregate'] = []
for ef in (16, 64, 128):
    rs = [c for s in summary['seeds'] for c in s['cells'] if c['ef'] == ef]
    row = {'ef': ef, 'episodes': sum(r['episodes'] for r in rs)}
    for key in ('no_embedding_rate', 'answer_agree_rate', 'invisible_tear_rate'):
        vals = [r[key] for r in rs]
        row[key] = {'mean': round(statistics.mean(vals), 4),
                    'min': min(vals), 'max': max(vals), 'per_seed': vals}
    summary['aggregate'].append(row)

(ROOT / 'maintenance_verification.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps({'verified': True, 'episodes_per_beam': 600, 'answer_agreement': [row['answer_agree_rate']['mean'] for row in summary['aggregate']]}))
