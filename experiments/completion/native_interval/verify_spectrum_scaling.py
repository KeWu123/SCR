"""Independent direct-cut, physical-native and scalar spectrum checker."""
from pathlib import Path
from _evidence import load_record, require_database
import sys,json,sqlite3,math,hashlib
HERE=Path(__file__).resolve().parent;RUN=HERE.parent.parent
sys.path.insert(0,str(RUN/'vendor'));sys.path.insert(0,str(HERE))
import numpy as np,pyarrow as pa,lancedb
from verify_native_intervals import exact_native_rows
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def quantile(values):return {'n':len(values),'median_ms':float(np.percentile(values,50))/1e6,'p95_ms':float(np.percentile(values,95))/1e6,'min_ms':min(values)/1e6,'max_ms':max(values)/1e6}
def oracle(src,q):
    values=[];qq=[float(x) for x in q]
    for r,(v,x) in src.items():values.append((math.fsum((float(a)-b)**2 for a,b in zip(np.frombuffer(x,dtype=np.float32),qq)),r,v))
    return {(r,v) for distance,r,v in sorted(values)[:10]}
def main():
    pointer=load_record(HERE/'spectrum_latest.json');path=Path(pointer['results']);assert sha(path)==pointer['sha256'];raw=load_record(path)
    for field,p in [('driver_sha256',HERE/'spectrum_scaling.py'),('broker_sha256',HERE/'interval_broker.py'),('compiler_sha256',HERE.parent/'support_spectrum.py'),('protocol_sha256',HERE/'spectrum_protocol.md')]:assert raw[field]==sha(p),field
    require_database(raw['cells'][0]['base_source'])
    assert {(c['commits'],c['seed']) for c in raw['cells']}=={(t,s) for t in [32,128,512] for s in [11,29,47]}
    counts={'cells':0,'direct_cut_checks':0,'admitted_cut_queries':0,'rejected_holes':0,'physical_native_replays':0,'scalar_oracles':0,'native_tables_checked':0,'point_certification_records_visited':0};recalls=[];fills=[];cellrows=[]
    for cell in raw['cells']:
        counts['cells']+=1;T=cell['commits'];assert sha(cell['input_file'])==cell['input_sha256'];data=np.load(cell['input_file']);q=data['q'];assert q.shape==(16,32)
        events=[(int(r),int(s),int(v),x.tobytes()) for r,s,v,x in zip(data['event_rows'],data['event_cuts'],data['event_versions'],data['event_vectors'])]
        assert len(events)==2000+T and len({(r,s) for r,s,v,x in events})==len(events)
        assert [s for r,s,v,x in events[2000:]]==list(range(1,T+1)) and [r for r,s,v,x in events[2000:]]==[s%4 for s in range(1,T+1)]
        assert len(cell['source_commit_ns'])==T and all(n>0 for n in cell['source_commit_ns'])
        expected_population={}
        sources={};current={};bycut={}
        for r,s,v,x in events:bycut.setdefault(s,[]).append((r,v,x))
        for cut in range(T+1):
            for r,v,x in bycut.get(cut,[]):current[r]=(v,x)
            sources[cut]=dict(current)
            if cut%8==0:expected_population.update({(r,v):x for r,(v,x) in current.items()})
        array_population={(int(r),int(v)):x.tobytes() for r,v,x in zip(data['population_rows'],data['population_versions'],data['population_vectors'])}
        assert array_population==expected_population and len(array_population)==2000+T//2
        c=sqlite3.connect(cell['base_source']);assert c.execute('select epoch,floor from meta').fetchone()==(T,0);sqlrows=c.execute('select row,epoch,version,deleted,vector,payload from history order by epoch,row').fetchall();c.close()
        assert [(r,s,v,bytes(x)) for r,s,v,d,x,p in sqlrows]==sorted(events,key=lambda e:(e[1],e[0])) and all(d==0 and p==f'{r}:{v}' for r,s,v,d,x,p in sqlrows)
        supported=[];point_visits=0
        for cut,src in sources.items():
            covered=True
            for r in sorted(src):
                point_visits+=1;v,x=src[r]
                if expected_population.get((r,v))!=x:covered=False;break
            if covered:supported.append(cut)
            counts['direct_cut_checks']+=1
        assert supported==list(range(0,T+1,8));counts['point_certification_records_visited']+=point_visits
        expected_intervals=[[s,s] for s in supported];answers_by_cut={};oracle_indices=sorted(set([0,len(supported)//4,len(supported)//2,3*len(supported)//4,len(supported)-1]))
        for mode,arm in cell['arms'].items():
            assert arm['initial_support']==expected_intervals and arm['publication']['support']==expected_intervals
            pub=arm['publication'];assert not pub['native_cache_hit'] and pub['source_versions']==2000+T and pub['native_population']==len(expected_population)
            assert pub['certification_records_visited']==(2000+T if mode=='interval' else point_visits)
            assert pub['started_ns']<=pub['acquired_ns']<=pub['registered_ns']<=pub['ended_ns'] and pub['hold_ns']==pub['ended_ns']-pub['acquired_ns'] and pub['hold_ns']<=arm['publication_elapsed_ns']
            assert len(arm['cached_certificate_ns'])==7 and all(n>0 for n in arm['cached_certificate_ns'])
            assert len(arm['tests'])==T+1 and [r['cut'] for r in arm['tests'] if r['accepted']]==supported
            assert all(not r['accepted'] for r in arm['tests'] if r['cut'] not in supported)
            assert arm['protected_gc_floor']==0
            nativepath=Path(cell['base_native']).parent.parent/mode/'lance';db=lancedb.connect(str(nativepath));table=db.open_table('objects');table.checkout(cell['native_version']);pop={}
            for r in exact_native_rows(table.to_arrow()):
                key=(r['row'],r['version']);assert key not in pop;pop[key]=np.asarray(r['vector'],dtype=np.float32).tobytes()
            assert pop==expected_population;counts['native_tables_checked']+=1
            stats=table.index_stats('vector_idx');assert stats.index_type=='IVF_HNSW_FLAT' and stats.num_indexed_rows==len(pop) and stats.num_unindexed_rows==0
            for test in arm['tests']:
                if not test['accepted']:counts['rejected_holes']+=1;continue
                cut=test['cut'];src=sources[cut];qid=test['query_id'];assert qid==(cut//8)%16
                ans=test['answers'];counts['admitted_cut_queries']+=1;fills.append(len(ans));assert len(ans)<=10 and len({r for r,v,p in ans})==len(ans)
                for r,v,p in ans:assert r in src and src[r][0]==v and p==f'{r}:{v}'
                native_records=exact_native_rows(table.search(q[qid]).nprobes(4).ef(128).limit(120).to_arrow())
                replay=[[r['row'],r['version'],f"{r['row']}:{r['version']}"] for r in native_records if r['row'] in src and src[r['row']][0]==r['version']][:10]
                counts['physical_native_replays']+=1;assert replay==ans
                if cut in answers_by_cut:assert ans==answers_by_cut[cut]
                else:
                    answers_by_cut[cut]=ans
                    if cut//8 in oracle_indices:
                        truth=oracle(src,q[qid]);counts['scalar_oracles']+=1;recalls.append(len({(r,v) for r,v,p in ans}&truth)/10)
            finalsql=sqlite3.connect(nativepath.parent/'source.sqlite');assert finalsql.execute('select epoch,floor from meta').fetchone()==(T,T) and finalsql.execute('select count(*) from history').fetchone()[0]==2000;finalsql.close()
        assert len(cell['cached_certificate_order'])==7 and all(set(x)=={'interval','point'} for x in cell['cached_certificate_order'])
        cellrows.append({'commits':T,'seed':cell['seed'],'supported_cuts':len(supported),'holes':T+1-len(supported),'components':len(supported),'native_vectors':len(expected_population),'source_versions':len(events),'point_records_visited':point_visits,'interval_records_visited':len(events)})
    summary={'status':'passed','results':str(path),'results_sha256':sha(path),'checker_sha256':sha(__file__),'key_checker_sha256':sha(HERE/'verify_native_intervals.py'),'counts':counts,'query_fill_min':min(fills),'query_fill_max':max(fills),'scalar_recall_mean':sum(recalls)/len(recalls),'scalar_recall_min':min(recalls),'cells':cellrows,'scale':[]}
    for T in [32,128,512]:
        cells=[c for c in raw['cells'] if c['commits']==T];row={'commits':T,'components':T//8+1,'supported_fraction':(T//8+1)/(T+1)}
        for mode in ['interval','point']:
            arms=[c['arms'][mode] for c in cells];row[mode]={'warm_certificate':quantile([x for a in arms for x in a['cached_certificate_ns']]),'cold_publication':quantile([a['publication_elapsed_ns'] for a in arms]),'cold_native_scan':quantile([a['publication']['native_scan_ns'] for a in arms]),'cold_catalogue_scan':quantile([a['publication']['catalogue_ns'] for a in arms]),'first_certificate':quantile([a['publication']['certification_ns'] for a in arms]),'records_visited':arms[0]['publication']['certification_records_visited']}
        summary['scale'].append(row)
    (HERE/'spectrum_verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
