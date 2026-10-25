"""Actual native disconnected-support and continuing-publication workload."""
from pathlib import Path
import sys, json, sqlite3, time, hashlib, threading, shutil, datetime, traceback
HERE = Path(__file__).resolve().parent
RUN = HERE.parent.parent
sys.path.insert(0, str(RUN/'vendor'))
sys.path.insert(0, str(HERE))
import numpy as np, pyarrow as pa, lancedb
from interval_broker import IntervalBroker

def ns(): return time.perf_counter_ns()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ver(t): return int(t.version() if callable(t.version) else t.version)
def conn(p):
    c=sqlite3.connect(str(p),check_same_thread=False,isolation_level=None,timeout=30)
    c.execute('pragma busy_timeout=30000');return c
def create_sql(path, events, head):
    c=conn(path);c.execute('pragma journal_mode=WAL')
    c.execute('create table history(row integer,epoch integer,version integer,deleted integer,vector blob,payload text,primary key(row,epoch))')
    c.execute('create table meta(epoch integer,floor integer)');c.execute('insert into meta values(?,0)',(head,))
    c.execute('begin');c.executemany('insert into history values(?,?,?,?,?,?)',events);c.commit();c.execute('pragma wal_checkpoint(truncate)');return c
def table_records(events, cut):
    latest={}
    for event in events:
        if event[1]<=cut and (event[0] not in latest or latest[event[0]][1]<event[1]):latest[event[0]]=event
    return {r:e for r,e in latest.items() if not e[3]}
def arrow(pop,d):
    return pa.table({'row':pa.array([r for r,v in pop],type=pa.int32()),'version':pa.array([v for r,v in pop],type=pa.int32()),'vector':pa.array([np.frombuffer(x,dtype=np.float32).tolist() for x in pop.values()],type=pa.list_(pa.float32(),d))})
def make_table(db,name,pop,d):
    t=db.create_table(name,arrow(pop,d));t.create_index(index_type='IVF_HNSW_FLAT',metric='l2',num_partitions=4,m=16,ef_construction=64);return t
def stat_history(c):
    head,floor=c.execute('select epoch,floor from meta').fetchone()
    return {'head':head,'floor':floor,'history_rows':c.execute('select count(*) from history').fetchone()[0]}
def snapshot(c,path):
    target=sqlite3.connect(str(path));c.backup(target);target.close()
def query_record(a,token,q,qid):
    start=ns();ans=a.query(token,q)
    s,name=a.pins[token]
    return {'name':name,'cut':s,'query_id':qid,'answers':ans,'latency_ns':ns()-start}
def reject(f):
    try:f()
    except (ValueError,KeyError):return True
    return False
def events_json(events):
    return [{'row':r,'epoch':e,'version':v,'deleted':d,'vector_hex':bytes(x).hex(),'payload':p} for r,e,v,d,x,p in events]
def summary(samples):
    return {'n':len(samples),'p50_ns':float(np.percentile(samples,50)) if samples else None,'p95_ns':float(np.percentile(samples,95)) if samples else None,'p99_ns':float(np.percentile(samples,99)) if samples else None,'max_ns':max(samples) if samples else None}

def disconnected(output, seed, n=2000,d=32):
    root=output/f'disconnected_s{seed}';root.mkdir()
    rng=np.random.default_rng(seed);x=rng.random((n,d),dtype=np.float32);q=rng.random((16,d),dtype=np.float32)
    e=[(r,0,1,0,x[r].tobytes(),f'{r}:1') for r in range(n)]
    width=n//10
    for r in range(width):
        e.extend([(r,2,2,0,rng.random(d,dtype=np.float32).tobytes(),f'{r}:2'),(r,4,3,1,b'',f'{r}:deleted'),(r,6,4,0,rng.random(d,dtype=np.float32).tobytes(),f'{r}:4')])
    for r in range(width,2*width):
        for epoch,v in [(3,2),(5,3),(7,4)]:e.append((r,epoch,v,0,rng.random(d,dtype=np.float32).tobytes(),f'{r}:{v}'))
    pop={}
    for cut in [0,1,4,7,8]:
        for r,record in table_records(e,cut).items():pop[(r,record[2])]=record[4]
    base=root/'base';base.mkdir();c=create_sql(base/'source.sqlite',e,8)
    db=lancedb.connect(str(base/'lance'));t=make_table(db,'objects',pop,d);v=ver(t)
    np.savez_compressed(root/'queries.npz',q=q)
    (root/'lineage.json').write_text(json.dumps(events_json(e)),encoding='utf-8')
    c.close()
    cell={'seed':seed,'n':n,'d':d,'query_file':str(root/'queries.npz'),'lineage_file':str(root/'lineage.json'),'base_source':str(base/'source.sqlite'),'base_native':str(base/'lance'),'native_version':v,'population_keys':list(pop),'arms':{}}
    for mode in ['interval','point']:
        work=root/mode;work.mkdir();shutil.copy2(base/'source.sqlite',work/'source.sqlite');shutil.copytree(base/'lance',work/'lance')
        c=conn(work/'source.sqlite');db=lancedb.connect(str(work/'lance'));t=db.open_table('objects');t.checkout(v)
        a=IntervalBroker(c,db,mode);m=a.publish('g',t,requested_cuts=list(range(9)))
        assert m.initial_support.intervals==((0,1),(4,4),(7,8)),m.initial_support
        tests=[]
        for s in range(9):
            start=ns()
            try:token=a.begin(s,'g')
            except ValueError:tests.append({'cut':s,'accepted':False,'latency_ns':ns()-start});continue
            rec={'cut':s,'accepted':True,'latency_ns':ns()-start,'queries':[]}
            for i,query in enumerate(q):rec['queries'].append(query_record(a,token,query,i))
            a.end(token);tests.append(rec)
        boundaries={str(s):reject(lambda s=s:a.begin(s,'g')) for s in [-1,9,True,0.5]}
        assert all(boundaries.values())
        lifecycle=[]
        old=a.begin(0,'g');audit=a.audit_hold(old);late=a.begin(7,'g')
        entered=threading.Event();release=threading.Event();operation=[];errors=[]
        def run_query():
            try:operation.append(a.query(old,q[0],hook=lambda:(entered.set(),release.wait(30))))
            except BaseException:errors.append(traceback.format_exc())
        thread=threading.Thread(target=run_query);thread.start();assert entered.wait(30)
        a.withdraw_component('g',0,1)
        assert reject(lambda:a.begin(0,'g')) and reject(lambda:a.cleanup(t)) and reject(lambda:a.drop_retired_table(t))
        a.end(old);floor,_=a.gc();assert floor==0
        lifecycle.append({'action':'withdraw_first_with_inflight_and_audit','floor':floor,'state':stat_history(c),'pins':dict(a.pins),'ops':dict(a.ops)})
        a.end(audit);floor,_=a.gc();assert floor==0
        lifecycle.append({'action':'release_audit_keep_inflight','floor':floor,'state':stat_history(c)})
        release.set();thread.join(30);assert not thread.is_alive() and not errors,errors
        floor,_=a.gc();assert floor==4
        lifecycle.append({'action':'complete_inflight','floor':floor,'state':stat_history(c)})
        a.withdraw_component('g',4,4);floor,_=a.gc();assert floor==7
        assert reject(lambda:a.begin(4,'g')) and reject(lambda:a.cleanup(t))
        probe=a.begin(8,'g');a.end(probe)
        lifecycle.append({'action':'withdraw_middle_later_component_open','floor':floor,'state':stat_history(c)})
        late_answer=a.query(late,q[1]);a.withdraw('g');floor,_=a.gc();assert floor==7
        lifecycle.append({'action':'withdraw_all_late_episode_pinned','floor':floor,'state':stat_history(c)})
        a.end(late);floor,removed=a.gc();assert floor==8 and len(removed)==1
        # Install another physical generation then perform actual cleanup.
        t.checkout_latest();t.add(arrow({(n+1,1):rng.random(d,dtype=np.float32).tobytes()},d));successor=ver(t)
        a.cleanup(t)
        h=db.open_table('objects')
        unavailable=reject(lambda:h.checkout(v));assert unavailable
        snapshot(c,work/'final_source.sqlite')
        result={'support':m.initial_support.intervals,'tests':tests,'boundaries':boundaries,'lifecycle':lifecycle,'inflight_answer':operation[0],'late_answer':late_answer,'final_state':stat_history(c),'old_native_unavailable':unavailable,'successor_native_version':successor,'final_source':str(work/'final_source.sqlite'),'final_native':str(work/'lance'),'metrics':a.metrics}
        a.close();c.close();cell['arms'][mode]=result
        print('disconnected',seed,mode,'verified 5 accepted cuts / 4 holes; GC 0 -> 4 -> 7 -> 8',flush=True)
    return cell

def continuing(output,seed,n=2000,d=32,cycles=8):
    root=output/f'continuing_s{seed}';root.mkdir()
    rng=np.random.default_rng(seed+1000);x=rng.random((n,d),dtype=np.float32);q=rng.random((16,d),dtype=np.float32)
    events=[(r,0,1,0,x[r].tobytes(),f'{r}:1') for r in range(n)]
    changes=[];versions=np.ones(n,dtype=np.int32);deleted=set()
    for cut in range(1,cycles+1):
        records=[]
        ids=np.random.default_rng(seed+cut*91).permutation(n)
        update=set(map(int,ids[:n//10]));newdelete=set(map(int,ids[n//10:n//10+n//50]));reinserts=set(deleted)
        for row in sorted(update|newdelete|reinserts):
            versions[row]+=1;isdelete=row in newdelete
            vector=b'' if isdelete else rng.random(d,dtype=np.float32).tobytes()
            records.append((row,cut,int(versions[row]),int(isdelete),vector,f'{row}:{int(versions[row])}'))
        deleted=newdelete;changes.append(records)
    np.savez_compressed(root/'queries.npz',q=q)
    (root/'lineage.json').write_text(json.dumps(events_json(events+[r for records in changes for r in records])),encoding='utf-8')
    cell={'seed':seed,'n':n,'d':d,'cycles':cycles,'query_file':str(root/'queries.npz'),'lineage_file':str(root/'lineage.json'),'arms':{}}
    for mode in ['interval','point']:
        work=root/mode;work.mkdir();c=create_sql(work/'source.sqlite',events,0);db=lancedb.connect(str(work/'lance'))
        initialpop={(r,v):vec for r,epoch,v,deleted,vec,payload in events};t=make_table(db,'g0',initialpop,d)
        archive=work/'native_archive';archive.mkdir();shutil.copytree(Path(t.uri),archive/'g0.lance')
        a=IntervalBroker(c,db,mode);m0=a.publish('g0',t,requested_cuts=[0]);oldtokens=[a.begin(0,'g0') for j in range(2)]
        state={'head':0,'latest_name':'g0','latest_cut':0,'committed_at':ns()};state_lock=threading.Lock();stop=threading.Event();started=threading.Barrier(4)
        oldqueries=[];newqueries=[];requests=[];errors=[];pubrecords=[];alltables=[('g0',ver(t))];history=list(events)
        def old_worker(index):
            try:
                started.wait();j=index
                while not stop.is_set():
                    oldqueries.append(query_record(a,oldtokens[index],q[j%len(q)],j%len(q)));j+=2
            except BaseException:errors.append(traceback.format_exc());stop.set()
        def admission_worker():
            try:
                started.wait();j=0
                while not stop.is_set():
                    with state_lock:current=dict(state)
                    requested=current['head'] if j%4 else current['latest_cut'];name=current['latest_name'];start=ns()
                    rec={'requested_cut':requested,'source_head_observed':current['head'],'published_cut_observed':current['latest_cut'],'name':name,'started_ns':start}
                    try:token=a.begin(requested,name)
                    except ValueError:rec.update(accepted=False,ended_ns=ns());requests.append(rec);time.sleep(.001);j+=1;continue
                    rec.update(accepted=True,ended_ns=ns(),snapshot_age_commits=current['head']-requested);requests.append(rec)
                    newqueries.append(query_record(a,token,q[j%len(q)],j%len(q)));a.end(token);j+=1
            except BaseException:errors.append(traceback.format_exc());stop.set()
        threads=[threading.Thread(target=old_worker,args=(i,)) for i in range(2)]+[threading.Thread(target=admission_worker)]
        for th in threads:th.start()
        started.wait();time.sleep(.02)
        for cut,records in enumerate(changes,1):
            c.execute('begin immediate');c.executemany('insert into history values(?,?,?,?,?,?)',records);c.execute('update meta set epoch=?',(cut,));c.commit();committed=ns();history.extend(records)
            with state_lock:state.update(head=cut,committed_at=committed)
            population={(r,e[2]):e[4] for r,e in table_records(history,cut).items()}
            build_start=ns();table=make_table(db,f'g{cut}',population,d);built=ns();alltables.append((f'g{cut}',ver(table)))
            archive_start=ns();shutil.copytree(Path(table.uri),archive/f'g{cut}.lance');archived=ns()
            publication_started=ns()
            a.publish(f'g{cut}',table,requested_cuts=[cut]);published=ns()
            with state_lock:previous=state['latest_name'];state.update(latest_name=f'g{cut}',latest_cut=cut)
            # This is a real registry Begin and native query at the new cut.
            begin_start=ns();token=a.begin(cut,f'g{cut}');admitted=ns();answer=query_record(a,token,q[cut%len(q)],cut%len(q));a.end(token)
            if previous!='g0':a.withdraw(previous)
            gc_start=ns();floor,removed=a.gc();gc_end=ns()
            reclaimed=[]
            for m in removed:
                oldtable=db.open_table(m.table_name);a.drop_retired_table(oldtable);reclaimed.append(m.table_name)
            pubrecords.append({'cut':cut,'name':f'g{cut}','source_committed_ns':committed,'native_build_start_ns':build_start,'native_built_ns':built,'archive_copy_ns':archived-archive_start,'publication_started_ns':publication_started,'published_ns':published,'newcut_begin_started_ns':begin_start,'newcut_admitted_ns':admitted,'newcut_query':answer,'source_to_publication_ns':published-committed,'source_to_admission_ns':admitted-committed,'native_build_ns':built-build_start,'gc_ns':gc_end-gc_start,'floor':floor,'removed':reclaimed,'history_rows':stat_history(c)['history_rows']})
            print('continuing',seed,mode,'cut',cut,'installed + queried; hold_ms',round(a.metrics['publication'][-1]['hold_ns']/1e6,2),flush=True)
            time.sleep(.015)
        stop.set()
        for th in threads:th.join(60)
        assert not any(th.is_alive() for th in threads) and not errors,errors
        assert all(r['floor']==0 for r in pubrecords),'old source episode must stay protected'
        snapshot(c,work/'complete_source.sqlite')
        oldafter=a.query(oldtokens[0],q[0]);a.end(oldtokens[0]);a.end(oldtokens[1]);a.withdraw('g0');floor,removed=a.gc();assert floor==cycles
        for m in removed:a.drop_retired_table(db.open_table(m.table_name))
        finalname=f'g{cycles}';last=a.begin(cycles,finalname);finalquery=query_record(a,last,q[0],0);a.end(last);a.withdraw(finalname);floor,removed=a.gc()
        for m in removed:a.drop_retired_table(db.open_table(m.table_name))
        snapshot(c,work/'final_source.sqlite')
        result={'source_file':str(work/'final_source.sqlite'),'complete_source_file':str(work/'complete_source.sqlite'),'native_directory':str(work/'lance'),'native_archive':str(archive),'table_versions':alltables,'publications':pubrecords,'old_queries':oldqueries,'new_queries':newqueries,'requests':requests,'old_after_all_publications':oldafter,'final_query':finalquery,'final_state':stat_history(c),'metrics':a.metrics,'summary':{'publication_hold':summary([r['hold_ns'] for r in a.metrics['publication'][1:]]),'publication_wait':summary([r['wait_ns'] for r in a.metrics['publication'][1:]]),'old_query':summary([r['latency_ns'] for r in oldqueries]),'new_query':summary([r['latency_ns'] for r in newqueries]),'admission':summary([r['ended_ns']-r['started_ns'] for r in requests]),'admission_wait':summary([r['wait_ns'] for r in a.metrics['admission']]),'source_to_publication':summary([r['source_to_publication_ns'] for r in pubrecords]),'rejected_requests':sum(not r['accepted'] for r in requests),'accepted_requests':sum(r['accepted'] for r in requests),'admitted_snapshot_age_commits':[r['snapshot_age_commits'] for r in requests if r['accepted']]}}
        a.close();c.close();cell['arms'][mode]=result
    return cell

def main():
    output=HERE/('run_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));output.mkdir()
    result={'protocol_sha256':sha(HERE/'protocol.md'),'driver_sha256':sha(__file__),'extension_sha256':sha(HERE/'interval_broker.py'),'base_broker_sha256':sha(RUN/'live_broker.py'),'support_compiler_sha256':sha(HERE.parent/'support_spectrum.py'),'versions':{'numpy':np.__version__,'pyarrow':pa.__version__,'lancedb':lancedb.__version__},'settings':{'index':'IVF_HNSW_FLAT','metric':'l2','num_partitions':4,'m':16,'ef_construction':64,'nprobes':4,'ef':128,'candidate_budget':120,'outputs':10},'output_directory':str(output),'disconnected':[],'continuing':[]}
    for seed in [11,29,47]:
        result['disconnected'].append(disconnected(output,seed));(output/'partial.json').write_text(json.dumps(result),encoding='utf-8')
    for seed in [11,29,47]:
        result['continuing'].append(continuing(output,seed));(output/'partial.json').write_text(json.dumps(result),encoding='utf-8')
    path=output/'results.json';path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    (HERE/'latest.json').write_text(json.dumps({'results':str(path),'sha256':sha(path)},indent=2),encoding='utf-8')
    print('DONE',path,flush=True)
if __name__=='__main__':main()
