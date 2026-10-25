"""Two distinct persisted source authorities sharing one physical native table."""
from pathlib import Path
import sys,sqlite3,uuid,datetime,json,hashlib
HERE=Path(__file__).resolve().parent;RUN=HERE.parents[1]
sys.path.insert(0,str(RUN));sys.path.insert(0,str(RUN/'vendor'))
import numpy as np,pyarrow as pa,lancedb
from live_broker import NativeBroker
root=HERE/('shared_native_'+uuid.uuid4().hex[:8]);root.mkdir()
def source(path):
    sql=sqlite3.connect(path,check_same_thread=False);sql.execute('pragma journal_mode=WAL')
    sql.execute('create table history(row integer,epoch integer,version integer,deleted integer,vector blob,payload text,primary key(row,epoch))');sql.execute('create table meta(epoch integer,floor integer)');sql.execute('insert into meta values(0,0)')
    sql.execute('insert into history values(?,?,?,?,?,?)',(0,0,1,0,np.array([1.,0.],dtype=np.float32).tobytes(),'old'));sql.commit();return sql
def arrow(v,x):return pa.table({'row':pa.array([0],type=pa.int32()),'version':pa.array([v],type=pa.int32()),'vector':pa.array([[x,0.]],type=pa.list_(pa.float32(),2))})
sql_a=source(root/'source_a.sqlite');sql_b=source(root/'source_b.sqlite')
db=lancedb.connect(str(root/'lance'));t=db.create_table('shared',arrow(1,1.));a=NativeBroker(sql_a,db);b=NativeBroker(sql_b,db)
a.publish('old_a',0,t)
try:b.publish('old_b',0,t);publish_error=None
except Exception as e:publish_error=type(e).__name__+': '+str(e)
if publish_error is None:
    tok=b.begin(0,'old_b');before=b.query(tok,[1.,0.])
    sql_a.execute('insert into history values(?,?,?,?,?,?)',(0,1,2,0,np.array([2.,0.],dtype=np.float32).tobytes(),'new'));sql_a.execute('update meta set epoch=1');sql_a.commit()
    t.delete('row>=0');t.add(arrow(2,2.));a.publish('new_a',1,t);a.withdraw('old_a');removed=[m.name for m in a.gc()[1]]
    # The reference driver's physical cleanup pattern after its own logical GC.
    cleanup=t.optimize(cleanup_older_than=datetime.timedelta(seconds=0),delete_unverified=True)
    try:after=b.query(tok,[1.,0.]);query_error=None
    except Exception as e:after=None;query_error=type(e).__name__+': '+str(e)
    b.end(tok)
else:tok=None;before=None;after=None;removed=[];cleanup=None;query_error=None
out={'source_sha256':hashlib.sha256((RUN/'live_broker.py').read_bytes()).hexdigest(),'distinct_source_files':True,'shared_native_uri':t.uri,'second_publish_error':publish_error,'a_removed':removed,'b_before_cleanup':before,'b_after_cleanup':after,'b_query_error':query_error,'cleanup':str(cleanup),'interpretation':'Physical optimize call is outside NativeBroker.maintain, which conservatively retains all versions; tests whether driver GC-to-cleanup scope is enforceable across source domains.'}
(HERE/'shared_native_results.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out));a.close();b.close()
