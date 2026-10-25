"""Independent full raw-trace verifier; no original manuscript/evidence inputs."""
from pathlib import Path
import hashlib,json,collections,statistics,gzip
import numpy as np
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
summary={'scope':'Native FAISS 1.15.1 IndexHNSWFlat::search with observational hooks; controlled builds and local rewrite, not production maintenance prevalence',
         'source':'https://github.com/facebookresearch/faiss/tree/v1.15.1', 'seeds':[], 'total_native_traces':0,
         'total_trace_on_off_bit_identical':0,'supplementary_native_traces':0,'same_generation_controls':0,'same_generation_control_failures':0,
         'verification_passes':['C++ native harness invariants and trace-on/off memcmp','independent saved-input/graph/trace Python verification'],
         'files_sha256':{}}
for seed in (11,29,47):
    inp=np.load(ROOT/f'input_{seed}.npz');xb=inp['vectors'];queries=inp['queries'];rewritten=set(inp['rewritten'].tolist())
    graph={};edges={};neighbor_sets={};graph_slots={};entries={}
    for name in ('A','B','C','D'):
        p=ROOT/f'graph_{seed}_{name}.json';g=json.loads(p.read_text());assert g['generation']==name
        assert {node['id'] for node in g['nodes']}==set(range(2000))
        graph[name]=g;edges[name]=set();neighbor_sets[name]={};graph_slots[name]={};entries[name]=g['entry_point']
        for node in g['nodes']:
            assert [layer['level'] for layer in node['layers']]==list(range(len(node['layers'])))
            for layer in node['layers']:
                lev,u,slots=layer['level'],node['id'],layer['slots']
                assert len(slots)==(32 if lev==0 else 16)
                positive=[v for v in slots if v>=0]
                assert len(set(positive))==len(positive)
                assert all(0<=v<2000 and v!=u for v in positive)
                assert slots==positive+[-1]*(len(slots)-len(positive))
                neighbor_sets[name][lev,u]=frozenset(positive);graph_slots[name][lev,u]=slots
                edges[name].update((lev,u,v) for v in positive)
        summary['files_sha256'][p.name]=sha(p)
    changed={u for lev,u in graph_slots['A'] if graph_slots['A'][lev,u]!=graph_slots['C'].get((lev,u))}
    assert changed<=rewritten and changed
    assert all(graph_slots['A'][key]==graph_slots['C'][key] for key in graph_slots['A'] if key[0]>0)
    exact=[]
    for q in queries:
        distance=np.array([sum((float(a)-float(b))**2 for a,b in zip(v,q)) for v in xb])
        exact.append(set(np.lexsort((np.arange(2000),distance))[:10].tolist()))
    records={};counts=collections.defaultdict(lambda:collections.Counter());recall=collections.defaultdict(list)
    raw=ROOT/f'raw_{seed}.jsonl'
    if not raw.exists():
        raw=raw.with_suffix('.jsonl.gz')
    if raw.suffix=='.gz':
        provenance=json.loads((ROOT/'source_provenance.json').read_text(encoding='utf-8'))
        assert sha(raw)==provenance['raw_traces'][raw.name]['compressed_sha256'], 'compressed trace hash mismatch'
    opener=gzip.open if raw.suffix=='.gz' else open
    with opener(raw,mode='rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);assert r['seed']==seed and r['ef'] in (16,64,128) and 0<=r['query']<200
            if r['generation']=='D':
                assert r['labels_off']==r['labels_on'] and r['distance_bits_off']==r['distance_bits_on']
                assert len(r['labels_on'])==10 and len(set(r['labels_on']))==10
                assert all(0<=label<2000 for label in r['labels_on'])
                distance=np.array(r['distance_bits_on'],dtype=np.uint32).view(np.float32)
                assert np.isfinite(distance).all() and np.all(distance[:-1]<=distance[1:])
                observed=set()
                for lev,u,slots in r['expansions']:
                    assert slots==graph_slots['D'][lev,u], 'supplementary graph-slot mismatch'
                    observed.update((lev,u,v) for v in slots if v>=0)
                first=[tuple(x) for x in r['first_visited_edges']]
                assert first and set(first)<=observed
                assert all(lev==0 for lev,u,v in first)
                assert len({v for lev,u,v in first})==len(first)
                summary['supplementary_native_traces']+=1
                continue
            assert r['generation'] in ('A','B','C','AA')
            gen='A' if r['generation']=='AA' else r['generation'];ef,qi=r['ef'],r['query']
            assert r['labels_off']==r['labels_on'] and r['distance_bits_off']==r['distance_bits_on']
            assert len(r['labels_on'])==10 and len(set(r['labels_on']))==10
            got=np.array(r['distance_bits_on'],dtype=np.uint32).view(np.float32)
            assert np.isfinite(got).all() and np.all(got[:-1]<=got[1:])
            expected=np.array([sum((float(a)-float(b))**2 for a,b in zip(xb[i],queries[qi])) for i in r['labels_on']])
            assert np.allclose(got,expected,atol=2e-5,rtol=2e-6),(seed,ef,qi,gen,'distance correspondence')
            observed=set();lists={};ordered=[]
            for lev,u,slots in r['expansions']:
                assert slots==graph_slots[gen][lev,u],(seed,ef,qi,gen,lev,u,'full expanded adjacency mismatch')
                positive=[v for v in slots if v>=0]
                observed.update((lev,u,v) for v in positive)
                if (lev,u) in lists:assert lists[lev,u]==frozenset(positive)
                lists[lev,u]=frozenset(positive)
                ordered.append((lev,u,tuple(slots)))
            first=[tuple(x) for x in r['first_visited_edges']]
            assert first and set(first)<=observed
            assert all(lev==0 for lev,u,v in first)
            assert len({v for lev,u,v in first})==len(first), 'target visited more than once'
            assert observed<=edges[gen]
            key=(ef,qi,r['generation']);assert key not in records
            records[key]={'raw':r,'edges':observed,'lists':lists,'ordered':ordered}
            recall[ef,gen].append(len(set(r['labels_on'])&exact[qi])/10)
            counts[ef,gen]['expansions']+=len(ordered);counts[ef,gen]['positive_edges']+=len(observed);counts[ef,gen]['first_visited']+=len(first)
            summary['total_native_traces']+=1;summary['total_trace_on_off_bit_identical']+=1
    assert len(records)==3*200*4
    out={'seed':seed,'rewritten_node_count':len(rewritten),'changed_base_neighbor_lists':len(changed),'cells':[]}
    for ef in (16,64,128):
        for pair in (('A','B'),('A','C'),('A','AA')):
            data=collections.Counter();witnesses=[]
            for qi in range(200):
                a,b=records[ef,qi,pair[0]],records[ef,qi,pair[1]]
                union=a['edges']|b['edges']
                embedding=[g for g in ('A','B','C') if union<=edges[g]]
                compatible=[]
                for g in ('A','B','C'):
                    if all(neighbor_sets[g].get(key)==ns for record in (a,b) for key,ns in record['lists'].items()):compatible.append(g)
                cert=not embedding;full=not compatible;same=set(a['raw']['labels_on'])==set(b['raw']['labels_on'])
                data['episodes']+=1;data['positive_edge_no_embedding']+=cert;data['full_neighbor_set_no_embedding']+=full
                data['answer_agree']+=same;data['certified_and_answer_agree']+=cert and same
                if cert and len(witnesses)<3:
                    witnesses.append({'query':qi,'answer_agrees':same,'escape_edges_by_generation':{
                        g:list(sorted(union-edges[g]))[:4] for g in ('A','B','C')},'observed_positive_edge_union_count':len(union)})
                if pair==('A','AA'):
                    identical=a['raw']['labels_on']==b['raw']['labels_on'] and a['raw']['distance_bits_on']==b['raw']['distance_bits_on'] and a['ordered']==b['ordered'] and a['raw']['first_visited_edges']==b['raw']['first_visited_edges']
                    summary['same_generation_controls']+=1
                    summary['same_generation_control_failures']+=not identical or cert or full
                    assert identical and not cert and not full
            data=dict(data);data.update({'ef':ef,'pair':'/'.join(pair),'positive_edge_no_embedding_rate':data['positive_edge_no_embedding']/200,
                 'full_neighbor_set_no_embedding_rate':data['full_neighbor_set_no_embedding']/200,'answer_agree_rate':data['answer_agree']/200,
                 'positive_edge_no_embedding_given_answer_agreement':data['certified_and_answer_agree']/data['answer_agree'] if data['answer_agree'] else None,
                 'witnesses':witnesses})
            out['cells'].append(data)
        triple=collections.Counter()
        for qi in range(200):
            rs=[records[ef,qi,g] for g in ('A','B','C')]
            noemb=not any(set.union(*(r['edges'] for r in rs))<=e for e in (edges[g] for g in ('A','B','C')))
            pairwise=all(any((rs[a]['edges']|rs[b]['edges'])<=e for e in (edges[g] for g in ('A','B','C'))) for a,b in ((0,1),(0,2),(1,2)))
            triple['episodes']+=1;triple['triple_no_embedding']+=noemb;triple['all_pairs_embed_somewhere']+=pairwise;triple['triple_only_residue']+=noemb and pairwise
        out.setdefault('triples',[]).append({'ef':ef,**triple})
    out['native_search_means']=[{'ef':ef,'generation':g,'recall_at_10':statistics.mean(recall[ef,g]),
          **{k:v/len(recall[ef,g]) for k,v in counts[ef,g].items()}} for ef,g in sorted(recall)]
    summary['seeds'].append(out);summary['files_sha256'][raw.name]=sha(raw)
summary['aggregate']=[]
for ef in (16,64,128):
    for pair in ('A/B','A/C','A/AA'):
        rs=[c for s in summary['seeds'] for c in s['cells'] if c['ef']==ef and c['pair']==pair]
        row={'ef':ef,'pair':pair,'episodes':sum(r['episodes'] for r in rs)}
        for key in ('positive_edge_no_embedding_rate','full_neighbor_set_no_embedding_rate','answer_agree_rate'):
            values=[r[key] for r in rs];row[key]={'mean':statistics.mean(values),'min':min(values),'max':max(values),'per_seed':values}
        row['certified_among_answer_agree_count']=sum(r['certified_and_answer_agree'] for r in rs)
        row['answer_agree_count']=sum(r['answer_agree'] for r in rs)
        summary['aggregate'].append(row)
for p in ROOT.iterdir():
    if p.is_file() and p.suffix in ('.py','.cpp','.md','.ps1','.bin','.exe') and p.name!='results_summary.json':summary['files_sha256'][p.name]=sha(p)
assert summary['total_native_traces']==7200 and summary['same_generation_controls']==1800 and summary['same_generation_control_failures']==0
assert summary['supplementary_native_traces'] in (0,1800)
if all(not (ROOT/f'raw_{seed}.jsonl').exists() for seed in (11,29,47)):
    retained=json.loads((ROOT/'results_summary.json').read_text(encoding='utf-8'))
    assert summary['aggregate']==retained['aggregate'], 'retained statistics differ'
summary['passed']=True
(ROOT/'verification_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({'passed': True, 'native_traces': summary['total_native_traces'],
    'trace_on_off_bit_identical': summary['total_trace_on_off_bit_identical'],
    'same_generation_controls': summary['same_generation_controls'],
    'supplementary_native_traces': summary['supplementary_native_traces']}))
