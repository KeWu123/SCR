"""Native support spectrum versus correctly enumerated point certificates."""
from pathlib import Path
import sys,json,time,datetime,hashlib,sqlite3,shutil
HERE=Path(__file__).resolve().parent;RUN=HERE.parent.parent
sys.path.insert(0,str(RUN/'vendor'));sys.path.insert(0,str(HERE))
import numpy as np,pyarrow as pa,lancedb
from interval_broker import IntervalBroker
from support_spectrum import certify,lifetimes_from_events
def ns():return time.perf_counter_ns()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def version(t):return int(t.version() if callable(t.version) else t.version)
def main():
    output=HERE/('spectrum_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));output.mkdir()
    result={'driver_sha256':sha(__file__),'broker_sha256':sha(HERE/'interval_broker.py'),'compiler_sha256':sha(HERE.parent/'support_spectrum.py'),'protocol_sha256':sha(HERE/'spectrum_protocol.md'),'n':2000,'d':32,'settings':{'index':'IVF_HNSW_FLAT','metric':'l2','num_partitions':4,'m':16,'ef_construction':64,'nprobes':4,'ef':128,'native_candidates':120,'outputs':10},'versions':{'numpy':np.__version__,'pyarrow':pa.__version__,'lancedb':lancedb.__version__},'cells':[]}
    for commits in [32,128,512]:
        for seed in [11,29,47]:
            root=output/f't{commits}_s{seed}';root.mkdir();base=root/'base';base.mkdir()
            rng=np.random.default_rng(seed+commits*1000);x=rng.random((2000,32),dtype=np.float32);q=rng.random((16,32),dtype=np.float32)
            sqlpath=base/'source.sqlite';c=sqlite3.connect(sqlpath,isolation_level=None);c.execute('pragma journal_mode=WAL');c.execute('create table history(row integer,epoch integer,version integer,deleted integer,vector blob,payload text,primary key(row,epoch))');c.execute('create table meta(epoch integer,floor integer)');c.execute('insert into meta values(0,0)')
            events=[(r,0,1,0,x[r].tobytes(),f'{r}:1') for r in range(2000)];current={r:(1,x[r].tobytes()) for r in range(2000)};population={(r,1):x[r].tobytes() for r in range(2000)}
            c.execute('begin');c.executemany('insert into history values(?,?,?,?,?,?)',events);c.commit();commit_ns=[]
            for cut in range(1,commits+1):
                row=cut%4;v=current[row][0]+1;vector=rng.random(32,dtype=np.float32).tobytes();event=(row,cut,v,0,vector,f'{row}:{v}');start=ns();c.execute('begin immediate');c.execute('insert into history values(?,?,?,?,?,?)',event);c.execute('update meta set epoch=?',(cut,));c.commit();commit_ns.append(ns()-start);events.append(event);current[row]=(v,vector)
                if cut%8==0:
                    for r,(vv,xx) in current.items():population[(r,vv)]=xx
            c.execute('pragma wal_checkpoint(truncate)');c.close()
            inputpath=root/'inputs.npz';np.savez_compressed(inputpath,q=q,event_rows=np.array([e[0] for e in events],dtype=np.int64),event_cuts=np.array([e[1] for e in events],dtype=np.int64),event_versions=np.array([e[2] for e in events],dtype=np.int64),event_vectors=np.array([np.frombuffer(e[4],dtype=np.float32) for e in events]),population_rows=np.array([r for r,v in population],dtype=np.int64),population_versions=np.array([v for r,v in population],dtype=np.int64),population_vectors=np.array([np.frombuffer(xx,dtype=np.float32) for xx in population.values()]))
            db=lancedb.connect(str(base/'lance'));arrow=pa.table({'row':pa.array([r for r,v in population],type=pa.int64()),'version':pa.array([v for r,v in population],type=pa.int64()),'vector':pa.array([np.frombuffer(xx,dtype=np.float32).tolist() for xx in population.values()],type=pa.list_(pa.float32(),32))});native_start=ns();table=db.create_table('objects',arrow);table.create_index(index_type='IVF_HNSW_FLAT',metric='l2',num_partitions=4,m=16,ef_construction=64);build_ns=ns()-native_start;native_version=version(table)
            cell={'commits':commits,'seed':seed,'n':2000,'d':32,'input_file':str(inputpath),'input_sha256':sha(inputpath),'base_source':str(sqlpath),'base_native':str(base/'lance'),'native_version':native_version,'source_commit_ns':commit_ns,'native_build_ns':build_ns,'arms':{}}
            # Rotate order by seed; artifacts and source contents are identical.
            order=['interval','point'] if seed!=29 else ['point','interval']
            cell['arm_order']=order
            instances={}
            for mode in order:
                work=root/mode;work.mkdir();shutil.copy2(sqlpath,work/'source.sqlite');shutil.copytree(base/'lance',work/'lance');sql=sqlite3.connect(work/'source.sqlite',check_same_thread=False,isolation_level=None);native=lancedb.connect(str(work/'lance'));t=native.open_table('objects');t.checkout(native_version);a=IntervalBroker(sql,native,mode)
                before=ns();manifest=a.publish('spectrum',t,requested_cuts=range(commits+1));full_elapsed=ns()-before;tests=[]
                for cut in range(commits+1):
                    start=ns()
                    try:token=a.begin(cut,'spectrum')
                    except ValueError:tests.append({'cut':cut,'accepted':False,'admission_ns':ns()-start});continue
                    admitted=ns();qid=(cut//8)%16;answer=a.query(token,q[qid]);ended=ns();a.end(token);tests.append({'cut':cut,'accepted':True,'admission_ns':admitted-start,'query_ns':ended-admitted,'query_id':qid,'answers':answer})
                assert [r['cut'] for r in tests if r['accepted']]==list(range(0,commits+1,8))
                cell['arms'][mode]={'publication_elapsed_ns':full_elapsed,'publication':a.metrics['publication'][0],'initial_support':manifest.initial_support.intervals,'tests':tests,'metrics':a.metrics,'protected_gc_floor':a.gc()[0]}
                assert cell['arms'][mode]['protected_gc_floor']==0
                catalogue=manifest.catalogue
                lifetimes=lifetimes_from_events([(r,epoch,(r,v),xx) for r,ev in catalogue.items() for epoch,v,xx,payload in ev],commits)
                instances[mode]=(a,sql,manifest,lifetimes,next(iter(a._populations.values())))
                cell['arms'][mode]['cached_certificate_ns']=[]
                print('spectrum',commits,seed,mode,'components',len(manifest.initial_support.intervals),'cert_ms',round(cell['arms'][mode]['publication']['certification_ns']/1e6,3),flush=True)
            cell['cached_certificate_order']=[]
            # Seven paired warm-cache repetitions; random arm order fixed by
            # the cell seed. Source/catalogue/native inspection is not repeated.
            timing_rng=np.random.default_rng(seed+commits+99991)
            for repeat in range(7):
                paired=list(timing_rng.permutation(['interval','point']));cell['cached_certificate_order'].append(paired)
                for mode in paired:
                    a,sql,manifest,lifetimes,pop=instances[mode];start=ns()
                    if mode=='interval':support=certify(lifetimes,pop,0,commits,commits).intervals
                    else:
                        accepted=[]
                        for cut in range(commits+1):
                            covered,visited=a._point_covered(manifest.catalogue,pop,cut)
                            if covered:accepted.append(cut)
                        support=tuple((s,s) for s in accepted)
                    elapsed=ns()-start;assert support==tuple((s,s) for s in range(0,commits+1,8));cell['arms'][mode]['cached_certificate_ns'].append(elapsed)
            for a,sql,manifest,lifetimes,pop in instances.values():a.withdraw('spectrum');a.gc();a.close();sql.close()
            result['cells'].append(cell);(output/'partial.json').write_text(json.dumps(result),encoding='utf-8')
    path=output/'results.json';path.write_text(json.dumps(result,indent=2),encoding='utf-8');(HERE/'spectrum_latest.json').write_text(json.dumps({'results':str(path),'sha256':sha(path)},indent=2),encoding='utf-8');print('DONE',path,flush=True)
if __name__=='__main__':main()
