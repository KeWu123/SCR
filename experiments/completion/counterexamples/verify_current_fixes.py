"""Actual current-code checks for ownership and SQLite transaction repairs."""
from pathlib import Path
import sys,sqlite3,threading,uuid,json,subprocess,time,hashlib
HERE=Path(__file__).resolve().parent;RUN=HERE.parents[1]
sys.path.insert(0,str(RUN));sys.path.insert(0,str(RUN/'vendor'));sys.path.insert(0,str(HERE.parent))
import numpy as np,pyarrow as pa,lancedb
from live_broker import NativeBroker
out=[]
def record(name,**kwargs):out.append(dict(case=name,**kwargs));print(json.dumps(out[-1]),flush=True)
def error(fn):
    try:fn();return None
    except Exception as e:return type(e).__name__+': '+str(e)
def setup(name):
    root=HERE/(name+'_'+uuid.uuid4().hex[:8]);root.mkdir();path=root/'source.sqlite'
    sql=sqlite3.connect(path,check_same_thread=False);sql.execute('pragma journal_mode=WAL')
    sql.execute('create table history(row integer,epoch integer,version integer,deleted integer,vector blob,payload text,primary key(row,epoch))');sql.execute('create table meta(epoch integer,floor integer)');sql.execute('insert into meta values(0,0)')
    sql.execute('insert into history values(?,?,?,?,?,?)',(0,0,1,0,np.array([1.,0.],dtype=np.float32).tobytes(),'old'));sql.commit()
    db=lancedb.connect(str(root/'lance'));t=db.create_table('docs',pa.table({'row':pa.array([0],type=pa.int32()),'version':pa.array([1],type=pa.int32()),'vector':pa.array([[1.,0.]],type=pa.list_(pa.float32(),2))}))
    return path,sql,db,t,NativeBroker(sql,db)

path,sql,db,t,a=setup('fixed_owner');sql_b=sqlite3.connect(path,check_same_thread=False)
failure=error(lambda:NativeBroker(sql_b,db));assert failure and failure.startswith('ValueError:')
child_code='import sys,sqlite3;sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2]);from live_broker import NativeBroker;\ntry: NativeBroker(sqlite3.connect(sys.argv[3]),None)\nexcept Exception as e: print(type(e).__name__+": "+str(e))\nelse: print("UNEXPECTED_ACCEPT")'
child=subprocess.run([sys.executable,'-c',child_code,str(RUN),str(RUN/'vendor'),str(path)],capture_output=True,text=True,timeout=20)
assert child.returncode==0 and 'ValueError:' in child.stdout
a.publish('old',0,t);token=a.begin(0,'old');close_active=error(a.close);assert close_active
a.end(token);a.close();b=NativeBroker(sql_b,db);restart=b.source(0);b.close()
record('exclusive_owner_same_and_other_process',same_process_error=failure,child_stdout=child.stdout.strip(),close_with_pin_error=close_active,closed_owner_restart_rows=len(restart))

path,sql,db,t,a=setup('fixed_writer')
staged=threading.Event();done=threading.Event();result=[];errors=[]
def stage_before_collector_begin(stmt):
    if stmt.lower().startswith('begin immediate') and not staged.is_set():
        sql.execute('begin immediate');sql.execute('insert into history values(?,?,?,?,?,?)',(0,1,2,0,np.array([2.,0.],dtype=np.float32).tobytes(),'uncommitted'))
        sql.execute('update meta set epoch=1');staged.set()
def collect():
    try:result.append(a.gc()[0])
    except Exception as exc:errors.append(repr(exc))
    finally:done.set()
a.collector.set_trace_callback(stage_before_collector_begin)
worker=threading.Thread(target=collect);worker.start();assert staged.wait(10)
other=sqlite3.connect(path);uncommitted_visible=list(other.execute('select epoch,floor from meta'))
time.sleep(.05);blocked=not done.is_set();assert blocked and sql.in_transaction and uncommitted_visible==[(0,0)]
sql.rollback();worker.join(10);assert done.is_set() and not errors and result==[0]
assert list(other.execute('select epoch,floor from meta'))==[(0,0)]
assert other.execute('select count(*) from history').fetchone()[0]==1
record('gc_does_not_commit_transaction_started_after_check',gc_blocked=blocked,external_before=uncommitted_visible,gc_result=result,external_after=list(other.execute('select epoch,floor from meta')),history_count=1)
a.collector.set_trace_callback(None);a.close()

path,sql,db,t,a=setup('fixed_read')
sql.execute('insert into history values(?,?,?,?,?,?)',(0,1,2,0,np.array([2.,0.],dtype=np.float32).tobytes(),'new'));sql.execute('update meta set epoch=1');sql.commit()
external=sqlite3.connect(path);ran=[]
def destroy_before_history(stmt):
    if stmt.lower().startswith('select row,version') and not ran:
        ran.append(True);external.execute('delete from history where epoch=0');external.execute('update meta set floor=1');external.commit()
a.reader.set_trace_callback(destroy_before_history);source=a.source(0);a.reader.set_trace_callback(None)
assert source[0][0]==1 and source[0][2]=='old';next_error=error(lambda:a.source(0));assert next_error
record('frontier_and_history_share_sqlite_snapshot',coherent_source={k:[v[0],v[2]] for k,v in source.items()},next_read_error=next_error,note='External destructive source SQL is outside sole authority; used only to test read transaction atomicity.')
a.close()

path,sql,db,t,a=setup('fixed_tombstones')
sql.execute('delete from history')
events=[(0,0,1,0,1.,'a1'),(0,2,2,1,2.,'delete-a'),(0,4,3,0,3.,'a3'),(1,1,1,0,4.,'b1'),(1,3,2,0,5.,'b2'),(1,5,3,1,6.,'delete-b')]
sql.executemany('insert into history values(?,?,?,?,?,?)',[(r,e,v,d,np.array([x,0.],dtype=np.float32).tobytes(),p) for r,e,v,d,x,p in events]);sql.execute('update meta set epoch=5');sql.commit()
at3=db.create_table('at3',pa.table({'row':pa.array([1],type=pa.int32()),'version':pa.array([2],type=pa.int32()),'vector':pa.array([[5.,0.]],type=pa.list_(pa.float32(),2))}))
at5=db.create_table('at5',pa.table({'row':pa.array([0],type=pa.int32()),'version':pa.array([3],type=pa.int32()),'vector':pa.array([[3.,0.]],type=pa.list_(pa.float32(),2))}))
a.publish('at3',3,at3);tok=a.begin(3,'at3');before=[a.source(s) for s in (3,4,5)];floor=a.gc()[0];after=[a.source(s) for s in (3,4,5)]
assert floor==3 and before==after
anchored=list(sql.execute('select row,epoch,deleted from history order by row,epoch'))
assert anchored==[(0,2,1),(0,4,0),(1,3,0),(1,5,1)]
a.publish('at5',5,at5);a.withdraw('at3');a.gc();assert a.query(tok,[5.,0.])==[(1,2,'b2')]
a.end(tok);floor=a.gc()[0];assert floor==5
final_rows=list(sql.execute('select row,epoch,deleted from history order by row,epoch'));assert final_rows==[(0,4,0),(1,5,1)]
record('tombstone_and_reinsert_anchors',floor_with_open_support_and_pin=3,all_cuts_3_to_5_preserved=True,anchors_and_later=anchored,old_pin_query_after_withdraw=[[1,2,'b2']],floor_after_pin_release=floor,final_anchors=final_rows)
a.close()

from support_spectrum import certify,Lifetime,lifetimes_from_events
bad_lifetime=error(lambda:certify((Lifetime('x',.5,1.5,b'x'),),{},0,2,2))
bad_event=error(lambda:lifetimes_from_events([('r',.5,'x',b'x')],2))
bad_contains=error(lambda:certify((),{},0,2,2).contains(.5))
assert bad_lifetime and bad_event and bad_contains
record('integer_boundary_validation_repaired',fractional_lifetime_error=bad_lifetime,fractional_event_error=bad_event,fractional_query_error=bad_contains)
(HERE/'fixed_results.json').write_text(json.dumps({'source_sha256':hashlib.sha256((RUN/'live_broker.py').read_bytes()).hexdigest(),'cases':out},indent=2),encoding='utf-8')
