"""Same persisted SQLite file through two hard links, real native source/GC."""
from pathlib import Path
import sys,sqlite3,uuid,os,json,hashlib
HERE=Path(__file__).resolve().parent;RUN=HERE.parents[1]
sys.path.insert(0,str(RUN));sys.path.insert(0,str(RUN/'vendor'))
import numpy as np,pyarrow as pa,lancedb
from live_broker import NativeBroker
root=HERE/('hardlink_'+uuid.uuid4().hex[:8]);root.mkdir();path=root/'source.sqlite';alias=root/'same_file.sqlite'
sql=sqlite3.connect(path,check_same_thread=False)
sql.execute('pragma journal_mode=DELETE')
sql.execute('create table history(row integer,epoch integer,version integer,deleted integer,vector blob,payload text,primary key(row,epoch))')
sql.execute('create table meta(epoch integer,floor integer)');sql.execute('insert into meta values(0,0)')
sql.execute('insert into history values(?,?,?,?,?,?)',(0,0,1,0,np.array([1.,0.],dtype=np.float32).tobytes(),'old'));sql.commit()
db=lancedb.connect(str(root/'lance'));t=db.create_table('docs',pa.table({'row':pa.array([0],type=pa.int32()),'version':pa.array([1],type=pa.int32()),'vector':pa.array([[1.,0.]],type=pa.list_(pa.float32(),2))}))
a=NativeBroker(sql,db);a.publish('old',0,t);tok=a.begin(0,'old');os.link(path,alias)
sql_b=sqlite3.connect(alias,check_same_thread=False)
try:
    b=NativeBroker(sql_b,db);accepted=True;failure=None
except Exception as e:
    accepted=False;failure=type(e).__name__+': '+str(e)
if accepted:
    sql.execute('insert into history values(?,?,?,?,?,?)',(0,1,2,0,np.array([2.,0.],dtype=np.float32).tobytes(),'new'));sql.execute('update meta set epoch=1');sql.commit()
    floor=b.gc()[0]
    try:a.begin(0,'old');future_error=None
    except Exception as e:future_error=type(e).__name__+': '+str(e)
    history=list(sql.execute('select row,epoch,version from history'));b.close()
else:floor=None;future_error=None;history=[]
out={'source_sha256':hashlib.sha256((RUN/'live_broker.py').read_bytes()).hexdigest(),'os_samefile':os.path.samefile(path,alias),'first_path':str(path),'second_path':str(alias),'second_owner_accepted':accepted,'second_owner_error':failure,'second_gc_floor':floor,'history':history,'first_future_begin_error':future_error}
(HERE/'hardlink_results.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out))
a.end(tok);a.close()
