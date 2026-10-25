"""Isolated regressions for exact key identities and lifecycle safety."""
from pathlib import Path
import sys,json,sqlite3,threading,hashlib,datetime,traceback,os
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent
RUN=PROJECT/'experiments'
if not (RUN/'live_broker.py').is_file():raise RuntimeError('could not locate experiments/live_broker.py')
sys.path[:0]=[str(HERE),str(RUN)]
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--historical-data', action='store_true', help='Also reopen the retained historical live databases; these are not included in the source repository.')
arguments=parser.parse_args()
import numpy as np,pyarrow as pa,lancedb
from native_broker_strict import NativeBroker,_retention_owner
ROOT=HERE/'runs'/('strict_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));ROOT.mkdir(parents=True)
results={'root':str(ROOT),'checks':[],'runtime':sys.version,'lancedb':lancedb.__version__}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def error(f):
    try:f()
    except Exception as e:return type(e).__name__+': '+str(e)
    return None
def record(name,fn):
    try:results['checks'].append({'name':name,'status':'passed','detail':fn()})
    except BaseException as e:results['checks'].append({'name':name,'status':'FAILED','exception':str(e),'traceback':traceback.format_exc()})
    print(name,results['checks'][-1]['status'],flush=True)
    (ROOT/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
def vector(a=1,b=0):return np.array([a,b],dtype=np.float32).tobytes()
def fixture(tag,events=None,head=0):
    p=ROOT/tag;p.mkdir(parents=True)
    c=sqlite3.connect(p/'source.sqlite',check_same_thread=False);c.execute('pragma journal_mode=WAL')
    c.execute('create table history(row integer,epoch integer,version integer,deleted integer,vector blob,payload text,primary key(row,epoch))')
    c.execute('create table meta(epoch integer,floor integer)');c.execute('insert into meta values(?,0)',(head,))
    c.executemany('insert into history values(?,?,?,?,?,?)',events or [(0,0,1,0,vector(),'source')]);c.commit()
    return c,lancedb.connect(str(p/'lance'))
def table(db,name,rows,versions,rt=pa.int64(),vt=pa.int64(),vectors=None):
    return db.create_table(name,pa.table({'row':pa.array(rows,type=rt),'version':pa.array(versions,type=vt),'vector':pa.array(vectors or [[1,0] for _ in rows],type=pa.list_(pa.float32(),2))}))
def state(b):return (dict(b.manifests),dict(b.support),dict(b._manifest_domains),set(b._native_owners),dict(b.pins),dict(b.ops))

def malformed_candidates():
    c,db=fixture('malformed_candidates');b=NativeBroker(c,db)
    good=table(db,'good',[0],[1]);b.publish('good',0,good);initial=state(b);out=[]
    cases=[('fractional',[0.9],[1.9],pa.float64(),pa.float64()),('strings',['0'],['1'],pa.string(),pa.string()),('bools',[False],[True],pa.bool_(),pa.bool_()),('nullrow',[None],[1],pa.int64(),pa.int64()),('nullversion',[0],[None],pa.int64(),pa.int64()),('rowoverflow',[1<<63],[1],pa.uint64(),pa.uint64()),('versionoverflow',[0],[1<<63],pa.uint64(),pa.uint64()),('negativeversion',[0],[-1],pa.int64(),pa.int64()),('zeroversion',[0],[0],pa.int64(),pa.int64()),('negative_row_not_alias',[ -1],[1],pa.int64(),pa.int64()),('duplicates',[0,0],[1,1],pa.int64(),pa.int64())]
    for name,rs,vs,rt,vt in cases:
        t=table(db,name,rs,vs,rt,vt);err=error(lambda:b.publish(name,0,t));assert err and state(b)==initial
        # Failed publication must release the newly acquired physical lock.
        identity=os.stat(t.uri);probe=_retention_owner(identity,'test release','native-');probe.close()
        out.append({'case':name,'error':err,'registry_unchanged':True,'new_native_owner_released':True})
    # Failure on an already owned table must preserve that table's old lock.
    good.add(pa.table({'row':pa.array([0],type=pa.int64()),'version':pa.array([1],type=pa.int64()),'vector':pa.array([[1,0]],type=pa.list_(pa.float32(),2))}))
    err=error(lambda:b.publish('same_domain_bad',0,good));assert err and state(b)==initial
    blocked=error(lambda:_retention_owner(os.stat(good.uri),'test preserved','native-'));assert blocked
    out.append({'case':'same_domain_duplicate','error':err,'registry_unchanged':True,'previous_native_owner_preserved':True})
    b.withdraw('good');b.gc();b.close();c.close();return out
record('malformed/overflow/noncanonical candidates and publication atomicity',malformed_candidates)

def canonical_candidates():
    keys=[(-(1<<63),1),(-1,7),(0,1),((1<<63)-1,(1<<63)-1)]
    c,db=fixture('canonical_keys',[(r,0,v,0,vector(),'payload '+str(r)) for r,v in keys]);b=NativeBroker(c,db)
    t=table(db,'canonical',[r for r,v in keys],[v for r,v in keys]);m=b.publish('g',0,t);tok=b.begin(0,'g');ans=b.query(tok,[1,0])
    assert m.keys==frozenset(keys) and {(r,v) for r,v,p in ans}==set(keys)
    unsigned=table(db,'safe_unsigned',[0],[1],pa.uint64(),pa.uint64());err=error(lambda:b.publish('unsigned_missing_other_rows',0,unsigned));assert err
    b.end(tok);b.withdraw('g');b.gc();b.close();c.close()
    return {'canonical_signed_int64_keys_accepted':keys,'answers':ans,'negative_row_policy':'Signed row IDs are valid; exact equality required, no coercion.','safe_unsigned_but_incomplete_population_error':err}
record('canonical keys including negative row and int64 boundaries',canonical_candidates)

def malformed_sources():
    bads=[('fractional_row',0.9,1),('text_row','not-an-id',1),('fractional_version',0,1.9),('text_version',0,'1x'),('zero_version',0,0),('negative_version',0,-1),('row_overflow_real',float(1<<63),1),('version_overflow_real',0,float(1<<63))]
    out=[]
    for tag,r,v in bads:
        c,db=fixture('source_'+tag,[(r,0,v,0,vector(),'source')]);t=table(db,'native',[0],[1]);b=NativeBroker(c,db);before=state(b)
        err=error(lambda:b.publish('bad',0,t));assert err and state(b)==before and not c.in_transaction
        stored=c.execute('select typeof(row),typeof(version) from history').fetchone()
        out.append({'case':tag,'stored_sqlite_types':stored,'error':err,'failure_atomic':True});b.close();c.close()
    c,db=fixture('source_reused_version',[(0,0,1,0,vector(),'old'),(0,1,1,0,vector(),'reused')],head=1)
    b=NativeBroker(c,db);t=table(db,'native',[0],[1]);err=error(lambda:b.publish('bad',1,t));assert err and not b.manifests and not b._native_owners;out.append({'case':'source_version_reused','error':err,'failure_atomic':True});b.close();c.close()
    c,db=fixture('public_snapshot_domain');b=NativeBroker(c,db);t=table(db,'native',[0],[1])
    for i,snapshot in enumerate([None,True,False,1.5,'0',-1]):
        before=state(b);source_err=error(lambda:b.source(snapshot));publish_err=error(lambda:b.publish('bad'+str(i),snapshot,t))
        assert source_err and publish_err and state(b)==before and not b._native_owners and not b.reader.in_transaction
        out.append({'case':'invalid_public_snapshot_'+repr(snapshot),'source_error':source_err,'publish_error':publish_err,'failure_atomic':True})
    assert b.source(0)=={0:(1,vector(),'source')};b.close();c.close()
    return out
record('canonical source keys, reused versions and no source key casts',malformed_sources)

def lifecycle():
    ev=[(0,0,1,0,vector(0,0),'old'),(0,2,2,0,vector(2,0),'revised'),(0,4,3,1,vector(),'deleted'),(0,6,4,0,vector(6,0),'reborn'),(1,0,1,0,vector(0,1),'stable')]
    c,db=fixture('lifecycle',ev,8);b=NativeBroker(c,db);t=table(db,'objects',[0,0,1],[1,4,1],vectors=[[0,0],[6,0],[0,1]])
    b.publish('g',1,t);b.publish('gp',0,t);b.publish('new',4,t)
    tok=b.begin(1,'g');answer=b.query(tok,[0,0]);assert set(map(tuple,answer))=={(0,1,'old'),(1,1,'stable')}
    audit=b.audit_hold(tok);b.end(tok);assert error(lambda:b.query(tok,[0,0]))
    b.withdraw('g');assert error(lambda:b.begin(1,'g'));assert b.query(audit,[0,0])==answer
    assert b.gc()[0]==0 # other manifest's old support is global protection
    b.withdraw('gp');assert b.gc()[0]==1
    entered=[threading.Event(),threading.Event()];finish=[threading.Event(),threading.Event()];answers=[];errors=[]
    def work(i):
        def hook():entered[i].set();assert finish[i].wait(10)
        try:answers.append(b.query(audit,[0,0],hook))
        except BaseException as e:errors.append(str(e))
    workers=[threading.Thread(target=work,args=(i,)) for i in range(2)]
    for th in workers:th.start()
    assert all(e.wait(10) for e in entered);b.end(audit)
    assert audit in b.pins and b.ops[audit]==2 and error(lambda:b.query(audit,[0,0]))
    assert b.gc()[0]==1 and error(lambda:b.cleanup(t))
    finish[0].set();workers[0].join(10);assert b.ops[audit]==1 and audit in b.pins and b.gc()[0]==1
    finish[1].set();workers[1].join(10);assert not errors and answers==[answer,answer] and audit not in b.pins
    floor,removed=b.gc();assert floor==4 and [m.name for m in removed]==['g']
    retained=c.execute('select row,epoch,version,deleted from history order by row,epoch').fetchall();assert retained==[(0,4,3,1),(0,6,4,0),(1,0,1,0)]
    shared=error(lambda:b.cleanup(t));assert shared # new support still uses shared native table
    b.withdraw('new');b.gc();b.cleanup(t);b.close();c.close()
    return {'old_source_payloads':answer,'source_floors':[0,1,1,1,4],'audit_pin_holds_source_and_native':True,'two_inflight_ops_survive_End_until_both_complete':True,'source_anchor_after_floor4':retained,'shared_open_support_cleanup_error':shared,'all_support_closed_zero_refs_cleanup_succeeded':True}
record('point End, audit, two in-flight operations, global source and shared native GC',lifecycle)

def guards():
    c,db=fixture('guards',[(0,0,1,0,vector(),'old'),(0,4,2,0,vector(4,0),'new')],8)
    b=NativeBroker(c,db);t=table(db,'objects',[0,0],[1,2],vectors=[[1,0],[4,0]])
    entered,release,attempted,done=[threading.Event() for _ in range(4)];errors=[];out={}
    original=b.source
    def paused(s):x=original(s);entered.set();assert release.wait(10);return x
    b.source=paused
    def worker(fn):
        try:fn()
        except BaseException as e:errors.append(traceback.format_exc())
    def gc():attempted.set();out['certificate_floor']=b.gc()[0];done.set()
    a=threading.Thread(target=lambda:worker(lambda:b.publish('g',1,t)));a.start();assert entered.wait(10)
    z=threading.Thread(target=lambda:worker(gc));z.start();assert attempted.wait(10);out['certification_guard_blocks_gc']=not done.wait(.2);assert out['certification_guard_blocks_gc']
    release.set();a.join(10);z.join(10);assert not errors and out['certificate_floor']==1;b.source=original
    for e in [entered,release,attempted,done]:e.clear()
    tokens=[]
    def hook():entered.set();assert release.wait(10)
    def withdraw():attempted.set();b.withdraw('g');out['begin_floor']=b.gc()[0];done.set()
    a=threading.Thread(target=lambda:worker(lambda:tokens.append(b.begin(1,'g',hook))));a.start();assert entered.wait(10)
    z=threading.Thread(target=lambda:worker(withdraw));z.start();assert attempted.wait(10);out['atomic_begin_blocks_withdraw_and_gc']=not done.wait(.2);assert out['atomic_begin_blocks_withdraw_and_gc']
    release.set();a.join(10);z.join(10);assert not errors and out['begin_floor']==1 and b.query(tokens[0],[1,0])==[(0,1,'old')]
    b.end(tokens[0]);b.gc();b.close();c.close();out['errors']=errors;return out
record('strict certification/publication and Begin linearization guards',guards)

def gc_fail_closed():
    normal=[(0,0,1,0,vector(),'old'),(0,1,2,0,vector(2,0),'new')]
    cases=[('fractional_head',normal,1.5,0),('fractional_floor',normal,8,.5),('negative_head',normal,-1,0),('negative_floor',normal,8,-1),('floor_above_head',normal,1,2),('text_head',normal,'bad',0),
      ('key_reuse',[(0,0,1,0,vector(),'old'),(0,1,1,0,vector(),'reused')],8,0),
      ('ambiguous_commit',[(0,0,1,0,vector(),'old'),(0,0,2,0,vector(),'ambiguous')],8,0),
      ('float_epoch',[(0,.5,1,0,vector(),'bad')],8,0),('negative_epoch',[(0,-1,1,0,vector(),'bad')],8,0),('future_epoch',[(0,9,1,0,vector(),'bad')],8,0),
      ('bad_tombstone',[(0,0,1,2,vector(),'bad')],8,0),('zero_version',[(0,0,0,0,vector(),'bad')],8,0),('float_row',[(.9,0,1,0,vector(),'bad')],8,0),('text_row',[('bad',0,1,0,vector(),'bad')],8,0),('bad_float32_width',[(0,0,1,0,b'bad','bad')],8,0)]
    out=[]
    def stored(c):return (c.execute('select rowid,epoch,floor from meta order by rowid').fetchall(),c.execute('select rowid,* from history order by rowid').fetchall())
    for tag,rows,head,floor in cases:
        c,db=fixture('gc_'+tag)
        c.execute('drop table history');c.execute('create table history(row integer,epoch integer,version integer,deleted integer,vector blob,payload text)')
        c.executemany('insert into history values(?,?,?,?,?,?)',rows);c.execute('update meta set epoch=?,floor=?',(head,floor));c.commit()
        b=NativeBroker(c,db);before=stored(c);registry=state(b);err=error(b.gc)
        assert err and stored(c)==before and state(b)==registry and not b.collector.in_transaction
        out.append({'case':tag,'error':err,'source_and_registry_unchanged':True});b.close();c.close()
    c,db=fixture('gc_duplicate_meta',normal,8);c.execute('insert into meta values(8,0)');c.commit();b=NativeBroker(c,db);before=stored(c);err=error(b.gc);assert err and stored(c)==before and not b.collector.in_transaction;out.append({'case':'multiple_meta_records','error':err,'source_and_registry_unchanged':True});b.close();c.close()
    c,db=fixture('gc_regressed_head',[(0,0,1,0,vector(),'old')],8);b=NativeBroker(c,db);t=table(db,'objects',[0],[1]);b.publish('g',1,t);tok=b.begin(1,'g');c.execute('update meta set epoch=0');c.commit()
    before=stored(c);registry=state(b);err=error(b.gc);assert err and stored(c)==before and state(b)==registry and not b.collector.in_transaction
    out.append({'case':'protected_cut_above_regressed_head','error':err,'source_and_registry_unchanged':True});b.end(tok);b.close();c.close()
    c,db=fixture('gc_closed');b=NativeBroker(c,db);b.close();before=stored(c);err=error(b.gc);assert err=='ValueError: retention authority closed' and stored(c)==before
    out.append({'case':'closed_authority','error':err,'source_and_registry_unchanged':True});c.close()
    return out
record('GC rejects malformed frontier/lineage before deletion without mutation',gc_fail_closed)

def measured_data_impact():
    p=RUN/'live_broker_results.json';r=json.loads(p.read_text());data=RUN/'live_db'/r['dbdir'].replace('\\','/').rstrip('/').rsplit('/',1)[-1]
    if not (data/'lance').is_dir():raise FileNotFoundError('Retained historical live databases are excluded from this source repository: '+str(data))
    db=lancedb.connect(str(data/'lance'));out=[];answers=0
    for cell in r['cells']:
        name=f"{cell['dataset']}_{cell['seed']}";t=db.open_table(name);actual=t.to_arrow();NativeBroker._native_key_schema(actual)
        keys=[NativeBroker._exact_key(row['row'],row['version']) for row in actual.to_pylist()]
        c=sqlite3.connect(data/(name+'.sqlite'));types=c.execute('select distinct typeof(row),typeof(version),typeof(epoch),typeof(deleted) from history').fetchall();records=c.execute('select row,version from history').fetchall()
        assert all(x==('integer','integer','integer','integer') for x in types)
        for row,version in records:NativeBroker._exact_key(row,version)
        for field in ['old_before','old_after_maintenance','old_after_withdraw_audit','current_after_gc']:
            for query in cell[field]:
                for row,version,payload in query:NativeBroker._exact_key(row,version);answers+=1
        out.append({'cell':name,'latest_native_version':t.version,'native_row_type':str(actual.schema.field('row').type),'native_version_type':str(actual.schema.field('version').type),'native_rows_checked':len(keys),'source_rows_checked':len(records),'source_storage_types':types,'negative_native_rows':sum(row<0 for row,version in keys)})
        c.close()
    assert r['code_sha256']==sha(RUN/'live_broker.py') and r['driver_sha256']==sha(RUN/'run_live_broker.py')
    return {'cells':out,'stored_answer_rows_checked':answers,'historical_code_binding_unchanged':True,'historical_result_sha256':sha(p),'arrow_helper_sha256':sha(RUN/'cross_store.py'),'historical_driver_uses':'cross_store.arrow line 8 sets both key fields to Arrow int32, fixed-size Float32 vectors.','observed_affected_reported_rows':0,'limits':'Latest surviving tables, retained SQLite source and every stored answer checked. Old deleted native versions cannot be reopened; their integer schema is established by the hash-bound driver helper, not a new physical inspection. No historical timings or results rerun/rebound.'}
if arguments.historical_data:
    record('actual historical native schemas/source keys and reported-row impact',measured_data_impact)
results['historical_data_check_requested']=arguments.historical_data

results['strict_module_sha256']=sha(HERE/'native_broker_strict.py');results['checker_sha256']=sha(__file__);results['historical_broker_sha256']=sha(RUN/'live_broker.py')
results['status']='passed' if all(x['status']=='passed' for x in results['checks']) else 'FAILED'
(ROOT/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
(HERE/'runs'/'strict_latest.json').write_text(json.dumps({'result':str(ROOT/'results.json'),'result_sha256':sha(ROOT/'results.json'),'strict_module_sha256':results['strict_module_sha256'],'status':results['status']},indent=2),encoding='utf8')
print(json.dumps({'status':results['status'],'result':str(ROOT/'results.json'),'module_sha256':results['strict_module_sha256']},indent=2))
if results['status']!='passed':sys.exit(1)
