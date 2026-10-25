"""Native correct-baseline benchmark. All outputs are confined to completion/."""
from pathlib import Path
import sys, os, time, json, hashlib, sqlite3, threading, shutil, datetime, traceback
from dataclasses import dataclass
from types import MappingProxyType
ROOT=Path(__file__).resolve().parent
RUN=ROOT.parent
sys.path.insert(0,str(RUN/'vendor'))
sys.path.insert(0,str(RUN))
import numpy as np, pyarrow as pa, lancedb
from live_broker import NativeBroker

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ns():return time.perf_counter_ns()
def ver(t):return int(t.version() if callable(t.version) else t.version)
def arrow(ids,vs,x):
    return pa.table({'row':pa.array(ids,type=pa.int32()),'version':pa.array(vs,type=pa.int32()),'vector':pa.array(x.tolist(),type=pa.list_(pa.float32(),x.shape[1]))})
def source(c,s):
    rows=c.execute('select row,version,deleted,vector,payload from history h where epoch=(select max(epoch) from history z where z.row=h.row and z.epoch<=?) and h.epoch<=? order by row',(s,s)).fetchall()
    return {r:(v,bytes(x),p) for r,v,d,x,p in rows if not d}
def connection(path):
    c=sqlite3.connect(str(path),check_same_thread=False,timeout=60,isolation_level=None)
    c.execute('pragma busy_timeout=60000')
    return c
def create_source(path,x):
    c=connection(path);c.execute('pragma journal_mode=WAL')
    c.execute('create table history(row integer,epoch integer,version integer,deleted integer,vector blob,payload text,primary key(row,epoch))')
    c.execute('create table meta(epoch integer,floor integer)');c.execute('insert into meta values(0,0)')
    c.execute('begin');c.executemany('insert into history values(?,?,?,?,?,?)',[(i,0,1,0,x[i].tobytes(),f'{i}:1') for i in range(len(x))]);c.commit()
    c.execute('pragma wal_checkpoint(truncate)');return c
def files(path):
    records=[]
    for p in Path(path).rglob('*'):
        if p.is_file():
            rel=p.relative_to(path).as_posix()
            category='index' if '_indices/' in rel else 'manifest' if '_versions/' in rel else 'transaction' if '_transactions/' in rel else 'data' if '/data/' in '/'+rel else 'other'
            records.append({'path':rel,'bytes':p.stat().st_size,'category':category})
    totals={k:sum(r['bytes'] for r in records if r['category']==k) for k in ['data','index','manifest','transaction','other']}
    return {'bytes':sum(totals.values()),'categories':totals,'files':records}
def source_files(path):
    result={}
    for suffix in ['', '-wal','-shm','.scr-retention.lock']:
        p=Path(str(path)+suffix);result[suffix or 'database']=p.stat().st_size if p.exists() else 0
    result['total_bytes']=sum(result.values());return result
def deep_size(obj,seen=None):
    if seen is None:seen=set()
    if id(obj) in seen:return 0
    seen.add(id(obj));total=sys.getsizeof(obj)
    if hasattr(obj,'items'):total+=sum(deep_size(k,seen)+deep_size(v,seen) for k,v in obj.items())
    elif isinstance(obj,(list,tuple,set,frozenset)):total+=sum(deep_size(v,seen) for v in obj)
    return total
@dataclass(frozen=True)
class Manifest:
    table_name:str
    native_version:int
    source_view:object
    keys:frozenset
    low:int=0
    high:int=0
def validate(c,db,table,s):
    c.execute('begin')
    try:
        head,floor=c.execute('select epoch,floor from meta').fetchone()
        if not floor<=s<=head:raise ValueError('unsupported source cut')
        src=source(c,s)
    finally:c.rollback()
    handle=db.open_table(table.name);v=ver(table);handle.checkout(v);pop={}
    for r in handle.to_arrow().to_pylist():
        key=(int(r['row']),int(r['version']))
        if key in pop:raise ValueError('duplicate key')
        pop[key]=np.asarray(r['vector'],dtype=np.float32).tobytes()
    for r,(version,vector,payload) in src.items():
        if pop.get((r,version))!=vector:raise ValueError('coverage or vector mismatch')
    return Manifest(table.name,v,MappingProxyType({r:(v,p) for r,(v,x,p) in src.items()}),frozenset(pop),s,s)
def metadata(m):
    obj={'table_name':m.table_name,'native_version':m.native_version,'low':m.low,'high':m.high,'keys':sorted(m.keys),'source_view':sorted((r,v,p) for r,(v,p) in m.source_view.items())}
    encoded=json.dumps(obj,separators=(',',':')).encode()
    return {'json_bytes':len(encoded),'python_deep_bytes':deep_size((m.table_name,m.native_version,m.low,m.high,m.keys,m.source_view)),'sha256':hashlib.sha256(encoded).hexdigest()}
def search(handle,view,q):
    rows=handle.search(q).nprobes(4).ef(128).limit(40).to_arrow().to_pylist()
    return [(int(r['row']),int(r['version']),view[int(r['row'])][1]) for r in rows if int(r['row']) in view and view[int(r['row'])][0]==int(r['version'])][:10]
class Gate:
    def __init__(self):self.cv=threading.Condition();self.readers=0;self.writer=False
    def read(self):
        with self.cv:
            while self.writer:self.cv.wait()
            self.readers+=1
    def unread(self):
        with self.cv:self.readers-=1;self.cv.notify_all()
    def write(self):
        with self.cv:
            while self.writer or self.readers:self.cv.wait()
            self.writer=True
    def unwrite(self):
        with self.cv:self.writer=False;self.cv.notify_all()
class Retained:
    """Correct point-support wrapper, retaining an actual SQL txn and native handle."""
    def __init__(self,path,c,db,kind):
        self.path=path;self.c=c;self.db=db;self.kind=kind;self.lock=threading.RLock();self.gate=Gate();self.support=True;self.tokens={};self.ops={};self.ending=set();self.serial=0;self.m=None
    def publish(self,name,s,t):
        with self.lock:self.m=validate(self.c,self.db,t,s);return self.m
    def begin(self,s=0,name='old',hook=None):
        if self.kind=='episode_lock':self.gate.read()
        try:
            with self.lock:
                if not self.support or self.m.low!=s:raise ValueError('support closed')
                reader=connection(self.path);reader.execute('pragma query_only=ON');reader.execute('begin')
                try:
                    head,floor=reader.execute('select epoch,floor from meta').fetchone()
                    if not floor<=s<=head or (self.kind=='episode_lock' and head!=s):raise ValueError('source unavailable/current changed')
                    h=self.db.open_table(self.m.table_name);h.checkout(self.m.native_version)
                    if hook:hook()
                except BaseException:reader.rollback();reader.close();raise
                self.serial+=1;self.tokens[self.serial]=(reader,h,self.m);return self.serial
        except BaseException:
            if self.kind=='episode_lock':self.gate.unread()
            raise
    def query(self,token,q,hook=None):
        with self.lock:
            if token in self.ending:raise ValueError('episode closing')
            reader,h,m=self.tokens[token];self.ops[token]=self.ops.get(token,0)+1
        try:
            if hook:hook()
            if ver(h)!=m.native_version:raise ValueError('native generation changed')
            return search(h,m.source_view,q)
        finally:
            with self.lock:
                self.ops[token]-=1
                if not self.ops[token]:del self.ops[token]
                if token in self.ending and token not in self.ops:self._finish(token)
    def _finish(self,token):
        reader,h,m=self.tokens.pop(token);reader.rollback();reader.close();self.ending.discard(token)
        if self.kind=='episode_lock':self.gate.unread()
    def end(self,token):
        with self.lock:
            if token in self.ops:self.ending.add(token)
            else:self._finish(token)
    def withdraw(self,name='old'):
        with self.lock:self.support=False
    def gc(self):
        with self.lock:
            head=self.c.execute('select epoch from meta').fetchone()[0]
            floor=self.m.low if self.support or self.tokens else head
            self.c.execute('begin immediate')
            try:
                self.c.execute('delete from history where epoch<? and epoch<(select max(epoch) from history z where z.row=history.row and z.epoch<=?)',(floor,floor));self.c.execute('update meta set floor=?',(floor,));self.c.commit()
            except BaseException:self.c.rollback();raise
            return floor,[]
    def close(self):
        if self.tokens:raise ValueError('active tokens')
def arm(kind,path,c,db):return NativeBroker(c,db) if kind=='scr' else Retained(path,c,db,kind)
def get_m(a,kind,name):return a.manifests[name] if kind=='scr' else a.m
def begin(a):return a.begin(0,'old')
ARMS=['episode_lock','retained_wrapper','scr']
def percentile(samples,p):return float(np.percentile(samples,p)) if samples else None
def summary(samples):return {'n':len(samples),'p50_ns':percentile(samples,50),'p95_ns':percentile(samples,95),'p99_ns':percentile(samples,99),'mean_ns':float(np.mean(samples)) if samples else None}
def safe_remove(path):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT.resolve()) or path==ROOT.resolve():raise ValueError('outside completion')
    shutil.rmtree(path)

def benchmark_cell(output,n,d,seed,idx,concurrency=True):
    cellroot=output/f'n{n}_d{d}_s{seed}';cellroot.mkdir()
    rng=np.random.default_rng(seed);x=rng.random((n,d),dtype=np.float32);q=rng.random((64,d),dtype=np.float32)
    rep=x[rng.permutation(n)]+rng.normal(0,.02,x.shape).astype(np.float32)
    inputpath=cellroot/'inputs.npz';np.savez_compressed(inputpath,x=x,q=q,replacements=rep)
    start=ns();sqlpath=cellroot/'initial.sqlite';c=create_source(sqlpath,x);sourcebuild=ns()-start
    start=ns();db=lancedb.connect(str(cellroot/'initial_lance'));t=db.create_table('objects',arrow(np.arange(n),np.ones(n,dtype=np.int32),x));nativewrite=ns()-start
    start=ns();t.create_index(index_type='IVF_HNSW_FLAT',metric='l2',num_partitions=4,m=16,ef_construction=64);indexbuild=ns()-start
    oldv=ver(t);initial_native=files(cellroot/'initial_lance');initial_source=source_files(sqlpath)
    canonical_handle=db.open_table('objects');canonical_handle.checkout(oldv)
    canonical_view={i:(1,f'{i}:1') for i in range(n)}
    canonical=[search(canonical_handle,canonical_view,vec) for vec in q]
    results={};instances={};order=ARMS[idx%3:]+ARMS[:idx%3]
    for kind in order:
        a=arm(kind,sqlpath,c,db);instances[kind]=a
        warm=ns();a.publish('warm',0,t);warm=ns()-warm
        publications=[]
        for j in range(3):
            start=ns();a.publish('old' if j==2 else f'publication{j}',0,t);publications.append(ns()-start)
        if kind=='scr':
            for name in ['warm','publication0','publication1']:a.withdraw(name)
            a.gc()
        m=get_m(a,kind,'old');token=begin(a)
        answers=[a.query(token,vec) for vec in q]
        assert answers==canonical,(n,seed,kind,'answer difference')
        for vec in q[:48]:a.query(token,vec)
        results[kind]={'publication_warmup_ns':warm,'publication_ns':publications,'metadata':metadata(m),'answers':answers,'query_ns':[],'admission_ns':[],'end_ns':[],'episode16_ns':[]}
        results[kind]['token']=token
    # Interleave each independent timed operation to reduce arm/cache ordering bias.
    for j in range(240):
        rotating=order[j%3:]+order[:j%3]
        for kind in rotating:
            a=instances[kind];r=results[kind]
            start=ns();ans=a.query(r['token'],q[j%48]);r['query_ns'].append(ns()-start)
            assert ans==canonical[j%48]
            start=ns();temporary=begin(a);r['admission_ns'].append(ns()-start)
            start=ns();a.end(temporary);r['end_ns'].append(ns()-start)
    for kind in order:
        a=instances[kind];r=results[kind];a.end(r.pop('token'))
        for j in range(3):
            start=ns();token=begin(a)
            for vec in q[:16]:a.query(token,vec)
            a.end(token);r['episode16_ns'].append(ns()-start)
        a.close()
        r['summary']={key:summary(r[key]) for key in ['publication_ns','query_ns','admission_ns','end_ns','episode16_ns']}
    c.execute('pragma wal_checkpoint(truncate)');c.close()
    cell={'n':n,'d':d,'seed':seed,'inputs_path':str(inputpath),'inputs_sha256':sha(inputpath),'initial_sqlite':str(sqlpath),'initial_lance':str(cellroot/'initial_lance'),'initial_native_version':oldv,'arm_order':order,'source_materialization_ns':sourcebuild,'native_table_write_ns':nativewrite,'native_index_build_ns':indexbuild,'initial_storage':{'source':initial_source,'native':initial_native},'indexed_plan':canonical_handle.search(q[0]).nprobes(4).ef(128).limit(40).explain_plan(),'arms':results,'concurrency':[]}
    if concurrency:
        for kind in order:
            for hold_ms in [100,300]:
                for readers in [1,4]:
                    cc=concurrency_case(cellroot,kind,hold_ms,readers,x,q,rep,oldv,canonical,keep=(hold_ms==300 and readers==4))
                    cell['concurrency'].append(cc)
                    print(n,d,seed,kind,hold_ms,readers,'wait_ms',round(cc['writer']['gate_wait_ns']/1e6,2),'commit_before_release',cc['writer']['source_commit_before_release'],flush=True)
    (cellroot/'cell.json').write_text(json.dumps(cell,indent=2),encoding='utf-8')
    return cell

def concurrency_case(cellroot,kind,hold_ms,readers,x,q,rep,oldv,canonical,keep):
    case=cellroot/f'{kind}_h{hold_ms}_r{readers}';case.mkdir()
    sqlpath=case/'source.sqlite';shutil.copy2(cellroot/'initial.sqlite',sqlpath);shutil.copytree(cellroot/'initial_lance',case/'lance')
    c=connection(sqlpath);c.execute('pragma journal_mode=WAL');db=lancedb.connect(str(case/'lance'));t=db.open_table('objects');t.checkout_latest()
    a=arm(kind,sqlpath,c,db);start=ns();a.publish('old',0,t);publication=ns()-start
    tokens=[];admit=[]
    for i in range(readers):
        start=ns();tokens.append(begin(a));admit.append(ns()-start)
    storage_before={'source':source_files(sqlpath),'native':files(case/'lance'),'history_rows':c.execute('select count(*) from history').fetchone()[0]}
    n=len(x);perm=np.random.default_rng(int(cellroot.name.split('_s')[-1])+777).permutation(n)
    updates=perm[:n//4];deletes=perm[n//4:n//4+n//10];reinserts=deletes[:len(deletes)//2]
    reinset=set(map(int,reinserts));changed=[]
    for row in updates:changed.append((int(row),1,2,0,rep[row].tobytes(),f'{int(row)}:2'))
    for row in deletes:
        v=3 if int(row) in reinset else 2;deleted=0 if int(row) in reinset else 1
        changed.append((int(row),1,v,deleted,rep[row].tobytes(),f'{int(row)}:{v}' if not deleted else 'delete'))
    attempted=threading.Event();stop=threading.Event();barrier=threading.Barrier(readers+1);writer={};errors=[];thread_results=[None]*readers
    def reader_fn(i):
        try:
            barrier.wait(30);reader_start=ns();latencies=[];queries=[];mismatches=0;j=i
            while not stop.is_set():
                started=ns();ans=a.query(tokens[i],q[j%len(q)]);latencies.append(ns()-started);queries.append(j%len(q));mismatches+=int(ans!=canonical[j%len(q)]);j+=1
            thread_results[i]={'query_ns':latencies,'query_ids':queries,'mismatches':mismatches,'start_ns':reader_start,'end_ns':ns()}
        except BaseException:errors.append(traceback.format_exc())
    def writer_fn():
        wc=None
        try:
            writer['attempt_ns']=ns();attempted.set()
            if kind=='episode_lock':a.gate.write()
            writer['acquired_ns']=ns();writer['gate_wait_ns']=writer['acquired_ns']-writer['attempt_ns']
            wc=connection(sqlpath);start=ns();wc.execute('begin immediate');wc.executemany('insert into history values(?,?,?,?,?,?)',changed);wc.execute('update meta set epoch=1');wc.commit();writer['source_write_ns']=ns()-start;writer['source_commit_ns']=ns()
            start=ns();t.delete('row IN ('+','.join(map(str,np.concatenate([updates,deletes])))+')');t.add(arrow(updates,np.full(len(updates),2),rep[updates]));t.add(arrow(reinserts,np.full(len(reinserts),3),rep[reinserts]));writer['native_write_ns']=ns()-start
            start=ns();newm=validate(wc,db,t,1);writer['publication_ns']=ns()-start;writer['new_version']=newm.native_version;writer['new_metadata']=metadata(newm);writer['published_ns']=ns()
            writer['current_population']=sorted(newm.keys);writer['end_ns']=ns()
        except BaseException:errors.append(traceback.format_exc())
        finally:
            if wc:wc.close()
            if kind=='episode_lock' and 'acquired_ns' in writer:a.gate.unwrite()
    reader_threads=[threading.Thread(target=reader_fn,args=(i,)) for i in range(readers)]
    for thread in reader_threads:thread.start()
    hold_start=ns();barrier.wait(30);wt=threading.Thread(target=writer_fn);wt.start();assert attempted.wait(30)
    deadline=hold_start+int(hold_ms*1e6)
    remainder=(deadline-ns())/1e9
    if remainder>0:time.sleep(remainder)
    stop.set()
    for thread in reader_threads:thread.join(120)
    assert all(not thread.is_alive() for thread in reader_threads)
    audit_start=ns();old_answers=[a.query(tokens[i],q[i]) for i in range(readers)]
    assert old_answers==[canonical[i] for i in range(readers)]
    source_witness=[]
    for token in tokens:
        if kind=='scr':
            held=a.source(0);observed_epoch=None
        else:
            reader=a.tokens[token][0];observed_epoch=reader.execute('select epoch from meta').fetchone()[0];held=source(reader,0)
            assert observed_epoch==0
        digest=hashlib.sha256()
        for row,(v,vec,payload) in sorted(held.items()):digest.update(str((row,v,payload)).encode());digest.update(vec)
        assert len(held)==n and all(held[i]==(1,x[i].tobytes(),f'{i}:1') for i in range(n))
        source_witness.append({'sqlite_snapshot_epoch':observed_epoch,'logical_source_cut':0,'population':len(held),'source_rows_sha256':digest.hexdigest()})
    retained={'source':source_files(sqlpath),'native':files(case/'lance'),'history_rows':c.execute('select count(*) from history').fetchone()[0]}
    hold_audit_ns=ns()-audit_start
    for token in tokens:a.end(token)
    release=ns();wt.join(120);assert not wt.is_alive();assert not errors,errors
    writer['source_commit_before_release']=writer['source_commit_ns']<release;writer['publication_before_release']=writer['published_ns']<release;writer['total_ns']=writer['end_ns']-writer['attempt_ns']
    assert all(z['mismatches']==0 and len(z['query_ns'])>0 for z in thread_results)
    if kind=='episode_lock':assert not writer['source_commit_before_release']
    after_writer={'source':source_files(sqlpath),'native':files(case/'lance'),'history_rows':c.execute('select count(*) from history').fetchone()[0]}
    a.withdraw('old');start=ns();floor,_=a.gc();source_gc=ns()-start;assert floor==1
    start=ns();cleanup=a.cleanup(t) if kind=='scr' else t.optimize(cleanup_older_than=datetime.timedelta(seconds=0),delete_unverified=True);native_gc=ns()-start
    a.close();del a
    unavailable=False
    try:h=db.open_table('objects');h.checkout(oldv);h.to_arrow()
    except Exception:unavailable=True
    assert unavailable
    c.execute('pragma wal_checkpoint(truncate)')
    after_gc={'source':source_files(sqlpath),'native':files(case/'lance'),'history_rows':c.execute('select count(*) from history').fetchone()[0]}
    result={'arm':kind,'requested_hold_ms':hold_ms,'readers':readers,'publication0_ns':publication,'admission_ns':admit,'hold_start_ns':hold_start,'release_ns':release,'actual_hold_ns':release-hold_start,'hold_audit_ns':hold_audit_ns,'source_snapshot_witnesses':source_witness,'writer':writer,'reader_threads':thread_results,'reader_query_summary':summary([v for z in thread_results for v in z['query_ns']]),'old_answers_after_concurrency':old_answers,'updates':updates.tolist(),'deletes':deletes.tolist(),'reinserts':reinserts.tolist(),'storage_before':storage_before,'storage_during_hold':retained,'storage_after_writer':after_writer,'storage_after_gc':after_gc,'source_gc_ns':source_gc,'native_gc_ns':native_gc,'old_native_reopen_rejected':unavailable,'gc_floor':floor,'cleanup':str(cleanup),'database_retained':keep,'path':str(case)}
    c.close();del t,db
    if not keep:safe_remove(case)
    return result

def race_cases(output):
    # Reuse one real indexed initial database, with a private physical copy per arm.
    base=output/'n2000_d32_s11';inputs=np.load(base/'inputs.npz');q=inputs['q'];races=[]
    for kind in ARMS:
        case=output/f'races_{kind}';case.mkdir();path=case/'source.sqlite';shutil.copy2(base/'initial.sqlite',path);shutil.copytree(base/'initial_lance',case/'lance');c=connection(path);db=lancedb.connect(str(case/'lance'));t=db.open_table('objects');a=arm(kind,path,c,db);a.publish('old',0,t);token=begin(a);expected=a.query(token,q[0])
        entered=threading.Event();release=threading.Event();answers=[];errors=[]
        def query_hook():entered.set();assert release.wait(30)
        def query_worker():
            try:answers.append(a.query(token,q[0],query_hook))
            except BaseException:errors.append(traceback.format_exc())
        thread=threading.Thread(target=query_worker);thread.start();assert entered.wait(30);a.withdraw('old');a.end(token);floor,_=a.gc();held=(token in a.pins if kind=='scr' else token in a.tokens)
        rejected=False
        try:a.query(token,q[0])
        except (ValueError,KeyError):rejected=True
        release.set();thread.join(30);assert not thread.is_alive() and not errors;assert held and rejected and answers==[expected]
        # Fresh wrapper/broker permits admission/withdraw race on real native objects.
        a.close();a=arm(kind,path,c,db);a.publish('old',0,t)
        selected=threading.Event();finish=threading.Event();attempted=threading.Event();done=threading.Event();tokens=[]
        def admission_hook():selected.set();assert finish.wait(30)
        def admit_worker():tokens.append(a.begin(0,'old',admission_hook))
        def withdraw_worker():attempted.set();a.withdraw('old');a.gc();done.set()
        th1=threading.Thread(target=admit_worker);th1.start();assert selected.wait(30);th2=threading.Thread(target=withdraw_worker);th2.start();assert attempted.wait(30);blocked=not done.is_set();finish.set();th1.join(30);th2.join(30)
        assert not th1.is_alive() and not th2.is_alive() and blocked and done.is_set() and a.query(tokens[0],q[0])==expected
        future_rejected=False
        try:begin(a)
        except (ValueError,KeyError):future_rejected=True
        assert future_rejected;a.end(tokens[0]);a.gc();a.close();c.close()
        races.append({'arm':kind,'inflight_reference_retained':held,'new_query_after_end_rejected':rejected,'inflight_answer_preserved':answers==[expected],'admission_withdraw_atomic':blocked,'withdraw_rejects_future':future_rejected,'held_floor':floor})
    return races

def main():
    output=ROOT/('comparative_native_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));output.mkdir()
    result={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol_sha256':sha(ROOT/'comparative_protocol.md'),'driver_sha256':sha(__file__),'broker_sha256':sha(RUN/'live_broker.py'),'versions':{'numpy':np.__version__,'pyarrow':pa.__version__,'lancedb':lancedb.__version__,'sqlite':sqlite3.sqlite_version,'python':sys.version},'output':str(output),'cells':[],'scope':{'warmup_queries':48,'timed_queries_per_arm':240,'publication_warmups':1,'publication_repetitions':3,'scalar_oracle_queries':16,'query_candidates':40,'output_k':10,'index_partitions':4,'nprobes':4,'ef':128,'m':16,'ef_construction':64}}
    selected=[(2000,32),(10000,64),(50000,64)]
    for n,d in selected:
        for i,seed in enumerate([11,29,47]):
            start=ns();cell=benchmark_cell(output,n,d,seed,i,concurrency=(n<=10000));result['cells'].append(cell);result['last_cell_wall_ns']=ns()-start
            (ROOT/'comparative_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print('CELL COMPLETE',n,d,seed,'seconds',round(result['last_cell_wall_ns']/1e9,2),flush=True)
    result['races']=race_cases(output)
    (ROOT/'comparative_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('DONE',str(ROOT/'comparative_results.json'),flush=True)
if __name__=='__main__':main()
