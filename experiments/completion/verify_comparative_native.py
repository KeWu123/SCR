"""Independent native/scalar checker. Does not import broker or benchmark code."""
from pathlib import Path
import sys, json, math, hashlib, sqlite3, time
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'vendor'))
sys.path.insert(0,str(ROOT.parents[1]))
from release_identity import matches as identity_matches
import numpy as np, lancedb

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def version(table):return int(table.version() if callable(table.version) else table.version)
def inventory(value):
    assert sum(r['bytes'] for r in value['files'])==value['bytes']
    assert sum(value['categories'].values())==value['bytes']
    for k,v in value['categories'].items():assert sum(r['bytes'] for r in value['files'] if r['category']==k)==v
def check_source_sizes(value):assert sum(v for k,v in value.items() if k!='total_bytes')==value['total_bytes']
def initial_native(cell,x):
    c=sqlite3.connect(cell['initial_sqlite'])
    assert c.execute('select epoch,floor from meta').fetchone()==(0,0)
    source=c.execute('select row,epoch,version,deleted,vector,payload from history order by row').fetchall();c.close()
    assert len(source)==cell['n']
    for i,(row,epoch,v,deleted,vector,payload) in enumerate(source):
        assert (row,epoch,v,deleted,payload)==(i,0,1,0,f'{i}:1')
        assert bytes(vector)==x[i].tobytes()
    db=lancedb.connect(cell['initial_lance']);table=db.open_table('objects');table.checkout(cell['initial_native_version'])
    pop=table.to_arrow().to_pylist();assert len(pop)==cell['n']
    keys=set()
    for row in pop:
        i=int(row['row']);v=int(row['version']);assert (i,v) not in keys;keys.add((i,v))
        assert 0<=i<cell['n'] and v==1 and np.asarray(row['vector'],dtype=np.float32).tobytes()==x[i].tobytes()
    assert keys=={(i,1) for i in range(cell['n'])}
    assert 'ANNIVFPARTITION' in cell['indexed_plan'].upper() and 'ANNSUBINDEX' in cell['indexed_plan'].upper(),cell['indexed_plan']
    stats=table.index_stats('vector_idx')
    assert stats.index_type=='IVF_HNSW_FLAT' and stats.num_indexed_rows==cell['n'] and stats.num_unindexed_rows==0
    return keys
def scalar_truth(x,q,k=10):
    # Scalar Python Float64 arithmetic; no vectorized distance/search implementation.
    result=[]
    for query in q:
        query=[float(a) for a in query]
        scored=[]
        for row,vec in enumerate(x):
            distance=math.fsum((float(a)-b)*(float(a)-b) for a,b in zip(vec,query))
            scored.append((distance,row,1))
        result.append([(row,v) for distance,row,v in sorted(scored)[:k]])
    return result
def main():
    result=json.loads((ROOT/'comparative_results.json').read_text(encoding='utf-8'))
    assert identity_matches(result['protocol_sha256'],ROOT/'comparative_protocol.md')
    assert identity_matches(result['driver_sha256'],ROOT/'compare_native_alternatives.py')
    assert identity_matches(result['broker_sha256'],ROOT.parent/'live_broker.py')
    assert result['versions']['numpy']=='1.26.4' and result['versions']['pyarrow']=='21.0.0'
    assert {(c['n'],c['d'],c['seed']) for c in result['cells']}=={(n,d,s) for n,d in [(2000,32),(10000,64),(50000,64)] for s in [11,29,47]}
    checked=[];writer_count=0;query_count=0;oracle_count=0
    for cell in result['cells']:
        assert sha(cell['inputs_path'])==cell['inputs_sha256']
        inputs=np.load(cell['inputs_path']);x,q,rep=inputs['x'],inputs['q'],inputs['replacements']
        assert x.shape==(cell['n'],cell['d']) and q.shape==(64,cell['d']) and rep.shape==x.shape
        initial_native(cell,x);truth=scalar_truth(x,q[:16]);oracle_count+=16
        canonical=cell['arms']['episode_lock']['answers'];recalls=[]
        expected_source_hash=hashlib.sha256()
        for row in range(cell['n']):expected_source_hash.update(str((row,1,f'{row}:1')).encode());expected_source_hash.update(x[row].tobytes())
        assert len(canonical)==64
        for j,answers in enumerate(canonical):
            assert len(answers)==10 and len({r for r,v,p in answers})==10
            assert all(0<=row<cell['n'] and v==1 and p==f'{row}:1' for row,v,p in answers)
            if j<16:recalls.append(len({(r,v) for r,v,p in answers}&set(truth[j]))/10)
        metadata=set()
        for kind,arm in cell['arms'].items():
            assert arm['answers']==canonical
            metadata.add(arm['metadata']['sha256'])
            for key,n in [('publication_ns',3),('query_ns',240),('admission_ns',240),('end_ns',240),('episode16_ns',3)]:
                samples=arm[key];assert len(samples)==n and all(isinstance(v,int) and v>0 for v in samples)
                saved=arm['summary'][key];assert saved['n']==n
                for name,p in [('p50_ns',50),('p95_ns',95),('p99_ns',99)]:assert math.isclose(saved[name],float(np.percentile(samples,p)),rel_tol=1e-12)
                assert math.isclose(saved['mean_ns'],sum(samples)/n,rel_tol=1e-12)
            query_count+=len(arm['query_ns'])
        assert len(metadata)==1,'same population/source metadata must match'
        inventory(cell['initial_storage']['native']);check_source_sizes(cell['initial_storage']['source'])
        if cell['n']<=10000:assert len(cell['concurrency'])==12
        else:assert cell['concurrency']==[]
        for cc in cell['concurrency']:
            w=cc['writer'];writer_count+=1
            assert cc['actual_hold_ns']==cc['release_ns']-cc['hold_start_ns']
            assert cc['actual_hold_ns']>=cc['requested_hold_ms']*1_000_000
            assert cc['hold_start_ns']<=w['attempt_ns']<=w['acquired_ns']<=w['source_commit_ns']<=w['published_ns']<=w['end_ns']
            assert w['gate_wait_ns']==w['acquired_ns']-w['attempt_ns'] and w['total_ns']==w['end_ns']-w['attempt_ns']
            assert w['source_commit_before_release']==(w['source_commit_ns']<cc['release_ns'])
            assert w['publication_before_release']==(w['published_ns']<cc['release_ns'])
            if cc['arm']=='episode_lock':assert not w['source_commit_before_release'] and w['acquired_ns']>cc['hold_start_ns']+cc['requested_hold_ms']*1_000_000
            updates=set(cc['updates']);deletes=set(cc['deletes']);reinserts=set(cc['reinserts'])
            assert len(updates)==cell['n']//4 and len(deletes)==cell['n']//10 and len(reinserts)==len(deletes)//2
            assert not updates&deletes and reinserts<=deletes
            expected={row:(2 if row in updates else 3 if row in reinserts else 1) for row in range(cell['n']) if row not in deletes or row in reinserts}
            assert w['current_population']==[[r,v] for r,v in sorted(expected.items())]
            assert len(cc['reader_threads'])==cc['readers']
            all_samples=[]
            for z in cc['reader_threads']:
                assert len(z['query_ns'])==len(z['query_ids'])>0 and z['mismatches']==0
                assert all(t>0 for t in z['query_ns']) and all(0<=j<64 for j in z['query_ids'])
                all_samples+=z['query_ns']
            assert cc['reader_query_summary']['n']==len(all_samples)
            assert math.isclose(cc['reader_query_summary']['p99_ns'],float(np.percentile(all_samples,99)),rel_tol=1e-12)
            assert cc['old_answers_after_concurrency']==canonical[:cc['readers']]
            assert len(cc['source_snapshot_witnesses'])==cc['readers']
            for witness in cc['source_snapshot_witnesses']:
                assert witness['logical_source_cut']==0 and witness['population']==cell['n'] and witness['source_rows_sha256']==expected_source_hash.hexdigest()
                assert witness['sqlite_snapshot_epoch']==(None if cc['arm']=='scr' else 0)
            assert 0<cc['hold_audit_ns']<cc['actual_hold_ns']
            for thread in cc['reader_threads']:assert cc['hold_start_ns']<=thread['start_ns']<thread['end_ns']<cc['release_ns']
            assert cc['old_native_reopen_rejected'] and cc['gc_floor']==1
            for stage in ['storage_before','storage_during_hold','storage_after_writer','storage_after_gc']:
                inventory(cc[stage]['native']);check_source_sizes(cc[stage]['source'])
            assert cc['storage_before']['history_rows']==cell['n']
            assert cc['storage_after_writer']['history_rows']==cell['n']+len(updates)+len(deletes)
            assert cc['storage_after_gc']['history_rows']==cell['n']
            if cc['database_retained']:
                c=sqlite3.connect(str(Path(cc['path'])/'source.sqlite'))
                assert c.execute('select epoch,floor from meta').fetchone()==(1,1)
                actual={r:(v,bytes(vec),p) for r,v,deleted,vec,p in c.execute('select row,version,deleted,vector,payload from history') if not deleted};c.close()
                assert set(actual)==set(expected)
                for row,v in expected.items():assert actual[row]==(v,(x[row] if v==1 else rep[row]).tobytes(),f'{row}:{v}')
                db=lancedb.connect(str(Path(cc['path'])/'lance'));table=db.open_table('objects');native=table.to_arrow().to_pylist()
                assert {(int(z['row']),int(z['version'])) for z in native}==set(expected.items())
                for z in native:
                    row=int(z['row']);v=int(z['version']);assert np.asarray(z['vector'],dtype=np.float32).tobytes()==(x[row] if v==1 else rep[row]).tobytes()
        checked.append({'n':cell['n'],'d':cell['d'],'seed':cell['seed'],'scalar_oracle_queries':16,'mean_recall10':sum(recalls)/16,'min_recall10':min(recalls),'all_three_arm_answers_equal':True,'invalid_returned_versions':0,'population_checked':cell['n']})
        print('VERIFIED',cell['n'],cell['d'],cell['seed'],'scalar recall',round(sum(recalls)/16,4),flush=True)
    assert {r['arm'] for r in result['races']}=={'episode_lock','retained_wrapper','scr'}
    for race in result['races']:
        for key in ['inflight_reference_retained','new_query_after_end_rejected','inflight_answer_preserved','admission_withdraw_atomic','withdraw_rejects_future']:assert race[key] is True
        assert race['held_floor']==0
    output={'passed':True,'result_sha256':sha(ROOT/'comparative_results.json'),'checker_sha256':sha(__file__),'broker_sha256':result['broker_sha256'],'cells':checked,'timed_query_samples_checked':query_count,'real_writer_cases_checked':writer_count,'scalar_oracle_queries':oracle_count,'deterministic_race_arms':3,'notes':['Scalar Float64 distance oracle independently recomputed; native initial vectors and retained final source/native vectors inspected.','p99 is exploratory: 240 timing samples per arm/cell; concurrency tails use their explicitly recorded per-case sample counts.','N and d increase together; these cells are joint scale settings, not an isolated dimensionality causal experiment.','Retained wrapper and SCR are equivalent at exact point support; runtime differences reflect implementation choices.']}
    (ROOT/'comparative_verification.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print('PASS',query_count,'timed native queries;',writer_count,'real writer cases;',oracle_count,'scalar oracles',flush=True)
if __name__=='__main__':main()
