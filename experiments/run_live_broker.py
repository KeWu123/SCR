from pathlib import Path
import sys,sqlite3,json,threading,datetime,time,hashlib,gc
RUN=Path(__file__).resolve().parent;sys.path.insert(0,str(RUN/'vendor'))
from live_broker import NativeBroker
from cross_store import arrow,version
import numpy as np,lancedb
root=RUN/'live_db'/datetime.datetime.now().strftime('%Y%m%d%H%M%S%f');root.mkdir(parents=True);db=lancedb.connect(str(root/'lance'));cells=[]
def rejected(f):
    try:f()
    except (ValueError,KeyError):return True
    return False
for dataset in ['digits','uniform']:
 for seed in [11,29,47]:
    inputs=np.load(RUN/f'{dataset}_{seed}_inputs.npz');x,q,rep=inputs['x'],inputs['q'],inputs['replacements'];n=len(x);name=f'{dataset}_{seed}'
    sql=sqlite3.connect(root/(name+'.sqlite'),check_same_thread=False);sql.execute('pragma journal_mode=WAL');sql.execute('create table history(row integer,epoch integer,version integer,deleted integer,vector blob,payload text,primary key(row,epoch))');sql.execute('create table meta(epoch integer,floor integer)');sql.execute('insert into meta values(0,0)');sql.executemany('insert into history values(?,?,?,?,?,?)',[(i,0,1,0,x[i].tobytes(),f'{i}:1') for i in range(n)]);sql.commit()
    table=db.create_table(name,arrow(np.arange(n),np.ones(n,dtype=int),x));table.create_index(index_type='IVF_HNSW_FLAT',metric='l2',num_partitions=4,m=16,ef_construction=64);oldv=version(table);old=db.open_table(name);old.checkout(oldv)
    broker=NativeBroker(sql,db);started=time.perf_counter_ns();broker.publish('old',0,old);old_publish_ms=(time.perf_counter_ns()-started)/1e6;token=broker.begin(0,'old');pre=[broker.query(token,vec) for vec in q]
    perm=np.random.default_rng(seed+777).permutation(n);u=perm[:n//4];d=perm[n//4:n//4+n//10];rein=d[:len(d)//2];changes=[]
    for i in u:changes.append((int(i),1,2,0,rep[i].tobytes(),f'{int(i)}:2'))
    for i in d:
        v=3 if i in rein else 2;deleted=0 if i in rein else 1;changes.append((int(i),1,v,deleted,rep[i].tobytes(),f'{int(i)}:{v}' if not deleted else 'delete'))
    sql.executemany('insert into history values(?,?,?,?,?,?)',changes);sql.execute('update meta set epoch=1');sql.commit()
    table.delete('row IN ('+','.join(map(str,np.concatenate([u,d])))+')');table.add(arrow(u,np.full(len(u),2),rep[u]));table.add(arrow(rein,np.full(len(rein),3),rep[rein]));table.create_index(index_type='IVF_HNSW_FLAT',metric='l2',num_partitions=4,m=16,ef_construction=64,replace=True);broker.maintain(table);newv=version(table);current=db.open_table(name);current.checkout(newv)
    rejected_mismatch=rejected(lambda:broker.publish('bad',0,current));assert rejected_mismatch;broker.publish('current',1,current)
    post=[broker.query(token,vec) for vec in q];assert pre==post
    old.checkout(newv);assert broker.query(token,q[0])==pre[0];old.checkout(oldv)
    clone=broker.checkout(broker.manifests['old']);clone.checkout(newv);assert broker.query(token,q[0])==pre[0]
    assert not hasattr(broker.manifests['old'],'handle')
    sql.execute('update meta set epoch=2');assert rejected(lambda:broker.publish('uncommitted',2,current)) and rejected(broker.gc);sql.rollback()
    before_gc=sql.execute('select count(*) from history').fetchone()[0]
    def bytes_on_disk():return sum(p.stat().st_size for p in (root/'lance'/(name+'.lance')).rglob('*') if p.is_file())
    bytes_before_gc=bytes_on_disk();floor,_=broker.gc();future=broker.begin(0,'old');assert floor==0 and len(broker.source(0))==n;broker.end(future)
    audit=broker.audit_hold(token);broker.end(token);assert rejected(lambda:broker.query(token,q[0]))
    selected=threading.Event();release=threading.Event();attempted=threading.Event();done=threading.Event();race=[]
    def hook():selected.set();assert release.wait(10)
    def begin():race.append(broker.begin(0,'old',hook))
    def withdraw():attempted.set();broker.withdraw('old');broker.gc();done.set()
    a=threading.Thread(target=begin);a.start();assert selected.wait(10);b=threading.Thread(target=withdraw);b.start();assert attempted.wait(10);blocked=not done.is_set();release.set();a.join(10);b.join(10);assert done.is_set() and blocked
    after_withdraw=[broker.query(audit,vec) for vec in q];assert after_withdraw==pre;assert rejected(lambda:broker.begin(0,'old'));broker.end(race[0]);broker.gc();assert 'old' in broker.manifests
    entered=threading.Event();finish=threading.Event();inflight=[]
    def pause():entered.set();assert finish.wait(10)
    worker=threading.Thread(target=lambda:inflight.append(broker.query(audit,q[0],pause)));worker.start();assert entered.wait(10);broker.end(audit);broker.gc();assert 'old' in broker.manifests and rejected(lambda:broker.query(audit,q[0]));finish.set();worker.join(10);assert inflight==[pre[0]]
    floor,removed=broker.gc();assert floor==1 and [m.name for m in removed]==['old'];del removed;del old;gc.collect()
    broker.withdraw('current');_,retired=broker.gc();del retired;del current;gc.collect()
    cleanup=broker.cleanup(table)
    try:
        dead=db.open_table(name);dead.checkout(oldv);dead.to_arrow();unavailable=False
    except Exception:unavailable=True
    assert unavailable
    current=db.open_table(name);current.checkout(version(table));broker.publish('current',1,current)
    assert rejected(lambda:broker.publish('old_resurrected',0,current)) and rejected(lambda:broker.source(0))
    assert rejected(lambda:NativeBroker(sql,db))
    currenttoken=broker.begin(1,'current');currentanswers=[broker.query(currenttoken,vec) for vec in q]
    timings=[];begin_ns=[]
    handle=broker.checkout(broker.manifests['current'])
    for j,vec in enumerate(q):
        def direct():return [(int(a['row']),int(a['version'])) for a in handle.search(vec).nprobes(4).ef(128).limit(40).to_arrow().to_pylist()][:10]
        def wrapped():return [(a,v) for a,v,payload in broker.query(currenttoken,vec)]
        times={};answers={}
        for name_fn,fn in ([('direct',direct),('broker',wrapped)] if j%2==0 else [('broker',wrapped),('direct',direct)]):
            start=time.perf_counter_ns();answers[name_fn]=fn();times[name_fn]=time.perf_counter_ns()-start
        assert answers['direct']==answers['broker'];timings.append(times)
        start=time.perf_counter_ns();temp=broker.begin(1,'current');begin_ns.append(time.perf_counter_ns()-start);broker.end(temp)
    broker.end(currenttoken)
    after_gc=sql.execute('select count(*) from history').fetchone()[0];sourcecurrent={i:value[0] for i,value in broker.source(1).items()};assert all(sourcecurrent[a]==v for answers in currentanswers for a,v,p in answers)
    cells.append(dict(dataset=dataset,seed=seed,n=n,updates=u.tolist(),deletes=d.tolist(),reinserts=rein.tolist(),old_version=oldv,current_version=newv,old_publish_ms=old_publish_ms,paired_query_ns=timings,begin_ns=begin_ns,metadata_serialized_bytes=len(json.dumps(dict(broker.manifests['current'].source_view)).encode()),external_alias_isolated=True,returned_handle_alias_isolated=True,manifest_has_no_handle=True,uncommitted_source_rejected=True,republish_below_floor_rejected=True,restart_preserves_read_frontier=True,rejected_mismatched_publish=rejected_mismatch,atomic_begin_blocks_gc=blocked,inflight_native_query_protected=True,old_native_reopen_rejected=unavailable,native_bytes_before_gc=bytes_before_gc,native_bytes_after_gc=bytes_on_disk(),source_rows_before_gc=before_gc,source_rows_after_gc=after_gc,old_before=pre,old_after_maintenance=post,old_after_withdraw_audit=after_withdraw,current_after_gc=currentanswers,native_old_plan=handle.search(q[0]).nprobes(4).ef(128).limit(10).explain_plan(),cleanup=str(cleanup)))
    broker.close();restarted=NativeBroker(sql,db);assert rejected(lambda:restarted.publish('old_restart',0,current));restarted.close()
    cells[-1]['second_authority_rejected']=True
    print(name,'lifecycle passed',before_gc,after_gc,flush=True);sql.close()
out=dict(cells=cells,protocol_sha256=hashlib.sha256((RUN/'live_broker_protocol.md').read_bytes()).hexdigest(),code_sha256=hashlib.sha256((RUN/'live_broker.py').read_bytes()).hexdigest(),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),dbdir=str(root));(RUN/'live_broker_results.json').write_text(json.dumps(out,indent=2))
