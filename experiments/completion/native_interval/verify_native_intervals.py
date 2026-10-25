"""Independent reconstruction/native replay. Imports no implementation code."""
from pathlib import Path
from _evidence import load_record, require_database
import sys, json, hashlib, sqlite3, math
HERE=Path(__file__).resolve().parent
RUN=HERE.parent.parent
sys.path.insert(0,str(RUN/'vendor'))
import numpy as np, pyarrow as pa, lancedb

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def version(t):return int(t.version() if callable(t.version) else t.version)
def lineage(path):
    records=json.loads(Path(path).read_text(encoding='utf-8'))
    out=[]
    for r in records:out.append((r['row'],r['epoch'],r['version'],r['deleted'],bytes.fromhex(r['vector_hex']),r['payload']))
    return out
def visible(events,cut):
    latest={}
    for e in events:
        if e[1]<=cut and (e[0] not in latest or latest[e[0]][1]<e[1]):latest[e[0]]=e
    return {r:e for r,e in latest.items() if not e[3]}
def sqlrows(path):
    c=sqlite3.connect(str(path));frontier=c.execute('select epoch,floor from meta').fetchone();rows=c.execute('select row,epoch,version,deleted,vector,payload from history order by row,epoch').fetchall();c.close()
    return frontier,[(r,e,v,d,bytes(x),p) for r,e,v,d,x,p in rows]
def canonical(events):return sorted(events,key=lambda x:(x[0],x[1]))
def exact_native_rows(arrow):
    # Independent raw-schema/value checks; no coercion from float/string/bool.
    for name in ['row','version']:
        assert arrow.column_names.count(name)==1 and pa.types.is_integer(arrow.schema.field(name).type),'noninteger native key schema'
    vector_type=arrow.schema.field('vector').type
    assert pa.types.is_fixed_size_list(vector_type) and pa.types.is_float32(vector_type.value_type)
    rows=arrow.to_pylist()
    for record in rows:
        row,version=record['row'],record['version']
        assert type(row) is int and -(2**63)<=row<2**63
        assert type(version) is int and 1<=version<2**63
        assert all(math.isfinite(x) for x in record['vector'])
    return rows
def population(db,table_name,v,expected):
    t=db.open_table(table_name);t.checkout(v);actual={}
    for r in exact_native_rows(t.to_arrow()):
        key=(r['row'],r['version']);assert key not in actual;actual[key]=np.asarray(r['vector'],dtype=np.float32).tobytes()
    assert actual==expected,'physical candidate/embedding mismatch'
    stats=t.index_stats('vector_idx');assert stats.index_type=='IVF_HNSW_FLAT' and stats.num_unindexed_rows==0 and stats.num_indexed_rows==len(actual)
    return t
def native_answers(t,src,q):
    records=exact_native_rows(t.search(q).nprobes(4).ef(128).limit(120).to_arrow())
    return [[r['row'],r['version'],src[r['row']][5]] for r in records if r['row'] in src and r['version']==src[r['row']][2]][:10]
def validate_answers(record,src):
    ans=record['answers'];assert len(ans)==10 and len({r for r,v,p in ans})==10
    for r,v,p in ans:assert r in src and (v,p)==(src[r][2],src[r][5])
def scalar(src,q):
    query=[float(a) for a in q];values=[]
    for r,e in src.items():
        vec=np.frombuffer(e[4],dtype=np.float32)
        distance=math.fsum((float(a)-b)**2 for a,b in zip(vec,query));values.append((distance,r,e[2]))
    return {(r,v) for distance,r,v in sorted(values)[:10]}
def summaries(actual,saved):
    assert saved['n']==len(actual)
    if actual:
        for name,p in [('p50_ns',50),('p95_ns',95),('p99_ns',99)]:assert math.isclose(saved[name],float(np.percentile(actual,p)),rel_tol=1e-12)
        assert saved['max_ns']==max(actual)
def main():
    latest=load_record(HERE/'latest.json');path=Path(latest['results']);assert sha(path)==latest['sha256']
    result=load_record(path)
    for field,p in [('protocol_sha256',HERE/'protocol.md'),('driver_sha256',HERE/'run_native_intervals.py'),('extension_sha256',HERE/'interval_broker.py'),('base_broker_sha256',RUN/'live_broker.py'),('support_compiler_sha256',HERE.parent/'support_spectrum.py')]:assert result[field]==sha(p),field
    require_database(result['disconnected'][0]['base_source'])
    assert result['settings']['candidate_budget']==120 and result['settings']['ef']==128
    counts={'disconnected_admitted_cuts':0,'disconnected_rejected_holes':0,'native_replayed_queries':0,'scalar_oracles':0,'checked_answer_records':0,'successor_publications':0,'physical_native_tables':0,'physical_gc_sources':0}
    recalls=[]
    for cell in result['disconnected']:
        events=lineage(cell['lineage_file']);q=np.load(cell['query_file'])['q'];domain=list(range(9))
        assert len(events)==3200 and len(q)==16
        expected={}
        for cut in [0,1,4,7,8]:
            for r,e in visible(events,cut).items():expected[(r,e[2])]=e[4]
        frontier,rows=sqlrows(cell['base_source']);assert frontier==(8,0) and rows==canonical(events)
        db=lancedb.connect(cell['base_native']);t=population(db,'objects',cell['native_version'],expected);counts['physical_native_tables']+=1
        supported=[s for s in domain if all(expected.get((r,e[2]))==e[4] for r,e in visible(events,s).items())]
        assert supported==[0,1,4,7,8]
        allanswers={}
        for mode,arm in cell['arms'].items():
            assert arm['support']==[[0,1],[4,4],[7,8]]
            assert [r['cut'] for r in arm['tests'] if r['accepted']]==supported
            assert [r['cut'] for r in arm['tests'] if not r['accepted']]==[2,3,5,6]
            counts['disconnected_admitted_cuts']+=5;counts['disconnected_rejected_holes']+=4
            for test in arm['tests']:
                if not test['accepted']:continue
                src=visible(events,test['cut'])
                for record in test['queries']:
                    validate_answers(record,src);counts['checked_answer_records']+=1
                    key=(record['cut'],record['query_id'])
                    actual=native_answers(t,src,q[record['query_id']]);counts['native_replayed_queries']+=1
                    assert record['answers']==actual
                    if key not in allanswers:
                        allanswers[key]=actual
                        if record['query_id']<4:
                            truth=scalar(src,q[record['query_id']]);counts['scalar_oracles']+=1;recalls.append(len({(r,v) for r,v,p in actual}&truth)/10)
                    else:assert allanswers[key]==actual
            assert all(arm['boundaries'].values())
            for answers,cut,qid in [(arm['inflight_answer'],0,0),(arm['late_answer'],7,1)]:
                src=visible(events,cut);validate_answers({'answers':answers},src);counts['checked_answer_records']+=1
                assert answers==native_answers(t,src,q[qid]);counts['native_replayed_queries']+=1
            assert [r['floor'] for r in arm['lifecycle']]==[0,0,4,7,7]
            assert [r['state']['history_rows'] for r in arm['lifecycle']]==[3200,3200,2600,2000,2000]
            assert arm['lifecycle'][0]['ops'] and len(arm['lifecycle'][0]['pins'])==3
            expected_final=[]
            last={}
            for e in events:
                if e[0] not in last or e[1]>last[e[0]][1]:last[e[0]]=e
            expected_final=canonical(last.values());frontier,rows=sqlrows(arm['final_source'])
            assert frontier==(8,8) and rows==expected_final;counts['physical_gc_sources']+=1
            finaldb=lancedb.connect(arm['final_native']);finaltable=finaldb.open_table('objects');assert version(finaltable)>=arm['successor_native_version']
            finalpop={(r['row'],r['version']):np.asarray(r['vector'],dtype=np.float32).tobytes() for r in exact_native_rows(finaltable.to_arrow())}
            assert set(finalpop)==set(expected)|{(cell['n']+1,1)} and all(finalpop[k]==v for k,v in expected.items())
            try:finaltable.checkout(cell['native_version'])
            except (ValueError,RuntimeError):pass
            else:raise AssertionError('old generation still physically available')
            assert arm['old_native_unavailable']
            assert len(arm['metrics']['query'])==82
        assert [r['queries'] for r in cell['arms']['interval']['tests'] if r['accepted']][0][0]['answers']==[r['queries'] for r in cell['arms']['point']['tests'] if r['accepted']][0][0]['answers']
    for cell in result['continuing']:
        events=lineage(cell['lineage_file']);q=np.load(cell['query_file'])['q'];scalar_cache={}
        assert cell['cycles']==8
        for mode,arm in cell['arms'].items():
            frontier,rows=sqlrows(arm['complete_source_file']);assert frontier==(8,0) and rows==canonical(events)
            current={}
            for e in events:
                if e[0] not in current or e[1]>current[e[0]][1]:current[e[0]]=e
            frontier,rows=sqlrows(arm['source_file']);assert frontier==(8,8) and rows==canonical(current.values());counts['physical_gc_sources']+=1
            assert not list(Path(arm['native_directory']).glob('*.lance')),'retired serving table still physical'
            archive=lancedb.connect(arm['native_archive']);tables={}
            for name,v in arm['table_versions']:
                cut=int(name[1:]);src=visible(events,cut);pop={(r,e[2]):e[4] for r,e in src.items()}
                tables[name]=population(archive,name,v,pop);counts['physical_native_tables']+=1
                plan=tables[name].search(q[0]).nprobes(4).ef(128).limit(120).explain_plan().upper();assert 'ANNIVFPARTITION' in plan and 'ANNSUBINDEX' in plan
            records=arm['old_queries']+arm['new_queries']+[p['newcut_query'] for p in arm['publications']]+[arm['final_query']]
            replay_cache={}
            for record in records:
                src=visible(events,record['cut']);validate_answers(record,src);counts['checked_answer_records']+=1
                key=(record['name'],record['query_id'])
                if key not in replay_cache:
                    replay_cache[key]=native_answers(tables[record['name']],src,q[record['query_id']]);counts['native_replayed_queries']+=1
                assert record['answers']==replay_cache[key]
                scalar_key=(record['cut'],record['query_id'])
                if record['query_id']<4 and scalar_key not in scalar_cache:
                    scalar_cache[scalar_key]=scalar(src,q[record['query_id']]);counts['scalar_oracles']+=1
                    recalls.append(len({(r,v) for r,v,p in record['answers']}&scalar_cache[scalar_key])/10)
            assert arm['old_after_all_publications']==native_answers(tables['g0'],visible(events,0),q[0])
            counts['native_replayed_queries']+=1;counts['checked_answer_records']+=1
            assert len(arm['publications'])==8 and len(arm['metrics']['publication'])==9
            counts['successor_publications']+=8
            for pub,metric in zip(arm['publications'],arm['metrics']['publication'][1:]):
                assert pub['cut']==metric['head'] and pub['name']==metric['name'] and pub['newcut_query']['cut']==pub['cut']
                assert pub['source_committed_ns']<=pub['native_build_start_ns']<=pub['native_built_ns']<=pub['publication_started_ns']<=metric['started_ns']<=metric['acquired_ns']<=metric['registered_ns']<=metric['ended_ns']<=pub['published_ns']<=pub['newcut_begin_started_ns']<=pub['newcut_admitted_ns']
                assert pub['floor']==0 and pub['history_rows']>=2000
                assert metric['hold_ns']==metric['ended_ns']-metric['acquired_ns'] and metric['wait_ns']==metric['acquired_ns']-metric['started_ns']
                assert pub['source_to_publication_ns']==pub['published_ns']-pub['source_committed_ns']
                assert pub['source_to_admission_ns']==pub['newcut_admitted_ns']-pub['source_committed_ns']
                assert metric['support']==[[pub['cut'],pub['cut']]],metric['support']
            assert [r['removed'] for r in arm['publications']]==[[]]+[[f'g{j}'] for j in range(1,8)]
            accepted=[r for r in arm['requests'] if r['accepted']];rejected=[r for r in arm['requests'] if not r['accepted']]
            assert accepted and rejected and arm['old_queries'] and arm['new_queries']
            for r in accepted:assert r['requested_cut']==int(r['name'][1:]) and r['snapshot_age_commits']==r['source_head_observed']-r['requested_cut']>=0
            sums=arm['summary'];assert sums['accepted_requests']==len(accepted) and sums['rejected_requests']==len(rejected)
            summaries([r['hold_ns'] for r in arm['metrics']['publication'][1:]],sums['publication_hold']);summaries([r['wait_ns'] for r in arm['metrics']['publication'][1:]],sums['publication_wait']);summaries([r['latency_ns'] for r in arm['old_queries']],sums['old_query']);summaries([r['latency_ns'] for r in arm['new_queries']],sums['new_query']);summaries([r['ended_ns']-r['started_ns'] for r in arm['requests']],sums['admission']);summaries([r['wait_ns'] for r in arm['metrics']['admission']],sums['admission_wait']);summaries([r['source_to_publication_ns'] for r in arm['publications']],sums['source_to_publication'])
    report={'status':'passed','results':str(path),'results_sha256':sha(path),'checker_sha256':sha(__file__),'counts':counts,'scalar_recall_mean':sum(recalls)/len(recalls),'scalar_recall_min':min(recalls),'limits':['Single cooperating local authority; fixed encoder and sealed cuts.','2,000 rows/32 dimensions, three seeds, eight successor cycles per arm.','Scalar oracles cover up to four query IDs per cut; all stored answers receive source/version checks and each unique native query is physically replayed.','No throughput SLA, distributed crash recovery or incremental certification.','Evidence archive copies are retained separately from the reclaimed serving registries.']}
    (HERE/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
