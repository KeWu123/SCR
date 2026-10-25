"""Check native table ownership scope and controlled destructive cleanup."""
from pathlib import Path
import sys,sqlite3,uuid,json,hashlib
HERE=Path(__file__).resolve().parent;RUN=HERE.parents[1]
sys.path.insert(0,str(RUN));sys.path.insert(0,str(RUN/'vendor'))
import numpy as np,pyarrow as pa,lancedb
from live_broker import NativeBroker
root=HERE/('native_owner_'+uuid.uuid4().hex[:8]);root.mkdir()
def source(path):
    c=sqlite3.connect(path,check_same_thread=False);c.execute('create table meta(epoch integer,floor integer)');c.execute('insert into meta values(0,0)');c.execute('create table history(row integer,epoch integer,version integer,deleted integer,vector blob,payload text,primary key(row,epoch))');c.execute('insert into history values(?,?,?,?,?,?)',(0,0,1,0,np.array([1.,0.],dtype=np.float32).tobytes(),'old'));c.commit();return c
def arrow():return pa.table({'row':pa.array([0],type=pa.int32()),'version':pa.array([1],type=pa.int32()),'vector':pa.array([[1.,0.]],type=pa.list_(pa.float32(),2))})
def error(fn):
    try:fn();return None
    except Exception as e:return type(e).__name__+': '+str(e)
db=lancedb.connect(str(root/'lance'));t=db.create_table('shared',arrow());distinct=db.create_table('distinct',arrow())
a=NativeBroker(source(root/'a.sqlite'),db);b=NativeBroker(source(root/'b.sqlite'),db)
a.publish('a0',0,t);a.publish('a0_second',0,t);b.publish('b_distinct',0,distinct)
competing=error(lambda:b.publish('b_shared',0,t));assert competing
open_cleanup=error(lambda:a.cleanup(t));assert open_cleanup
tok=a.begin(0,'a0');a.withdraw('a0');a.withdraw('a0_second')
pinned_cleanup=error(lambda:a.cleanup(t));assert pinned_cleanup
a.end(tok);a.gc();closed_still_owned=error(lambda:b.publish('b_shared_after_retire',0,t));assert closed_still_owned
a.cleanup(t);a.close();b.publish('b_shared_after_close',0,t)
otherdb=lancedb.connect(str(root/'other_lance'));foreign=otherdb.create_table('shared',arrow())
wrong_database=error(lambda:b.publish('wrong_database',0,foreign));assert wrong_database and 'configured artifact domain' in wrong_database
b.close()
out={'source_sha256':hashlib.sha256((RUN/'live_broker.py').read_bytes()).hexdigest(),'multiple_manifests_one_owner':True,'distinct_tables_shared_database':True,'competing_owner_error':competing,'open_support_cleanup_error':open_cleanup,'active_pin_cleanup_error':pinned_cleanup,'retired_table_owner_held_until_close_error':closed_still_owned,'cleanup_after_closed_zero_refs_succeeded':True,'owner_transfer_after_close_succeeded':True,'same_name_foreign_database_error':wrong_database}
(HERE/'native_owner_results.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out))
