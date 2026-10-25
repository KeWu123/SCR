from pathlib import Path
import sys,json,sqlite3,threading,traceback,datetime
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
OUT=HERE/'runs'/('guards_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
OUT.mkdir(parents=True)
RUN=HERE.parent/'experiments'
sys.path[:0]=[str(RUN)]
import numpy as np,pyarrow as pa,lancedb
from live_broker import NativeBroker
p=OUT/'isolated_native'/'guards';p.mkdir(parents=True,exist_ok=False)
c=sqlite3.connect(p/'source.sqlite',check_same_thread=False);c.execute('pragma journal_mode=WAL')
c.execute('create table history(row integer,epoch integer,version integer,deleted integer,vector blob,payload text,primary key(row,epoch))')
c.execute('create table meta(epoch integer,floor integer)');c.execute('insert into meta values(8,0)')
c.executemany('insert into history values(?,?,?,?,?,?)',[(0,0,1,0,np.array([0,0],dtype=np.float32).tobytes(),'old'),(0,4,2,0,np.array([4,0],dtype=np.float32).tobytes(),'new')]);c.commit()
db=lancedb.connect(str(p/'lance'));t=db.create_table('g',pa.table({'row':pa.array([0,0],type=pa.int64()),'version':pa.array([1,2],type=pa.int64()),'vector':pa.array([[0,0],[4,0]],type=pa.list_(pa.float32(),2))}))
b=NativeBroker(c,db);out={};errors=[]
def worker(fn):
    try:fn()
    except BaseException as e:errors.append(traceback.format_exc())
selected,release,attempted,done=threading.Event(),threading.Event(),threading.Event(),threading.Event()
original=b.source
def paused(s):
    x=original(s);selected.set();assert release.wait(10);return x
b.source=paused
def gc_call():attempted.set();out['cert_gc_result']=b.gc()[0];done.set()
pub=threading.Thread(target=lambda:worker(lambda:b.publish('g',1,t)));pub.start();assert selected.wait(10)
gc=threading.Thread(target=lambda:worker(gc_call));gc.start();assert attempted.wait(10)
out['gc_blocked_between_certification_and_registration']=not done.wait(.2);assert out['gc_blocked_between_certification_and_registration']
release.set();pub.join(10);gc.join(10);assert not pub.is_alive() and not gc.is_alive() and not errors and out['cert_gc_result']==1
b.source=original
selected.clear();release.clear();attempted.clear();done.clear();tokens=[]
def hook():selected.set();assert release.wait(10)
def withdraw_gc():attempted.set();b.withdraw('g');out['begin_gc_result']=b.gc()[0];done.set()
admit=threading.Thread(target=lambda:worker(lambda:tokens.append(b.begin(1,'g',hook))));admit.start();assert selected.wait(10)
w=threading.Thread(target=lambda:worker(withdraw_gc));w.start();assert attempted.wait(10)
out['withdrawal_gc_blocked_during_atomic_begin']=not done.wait(.2);assert out['withdrawal_gc_blocked_during_atomic_begin']
release.set();admit.join(10);w.join(10);assert not admit.is_alive() and not w.is_alive() and not errors and out['begin_gc_result']==1
out['pin_query_survives_withdrawal']=b.query(tokens[0],[0,0])==[(0,1,'old')];assert out['pin_query_survives_withdrawal']
b.end(tokens[0]);out['floor_after_last_hold']=b.gc()[0];b.close();c.close();out['errors']=errors
(OUT/'guard_checks.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
