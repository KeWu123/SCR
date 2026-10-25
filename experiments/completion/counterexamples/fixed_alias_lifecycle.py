"""Current immutable-descriptor alias isolation and two-operation End schedule."""
from pathlib import Path
import sys,sqlite3,threading,uuid,json,hashlib
HERE=Path(__file__).resolve().parent;RUN=HERE.parents[1]
sys.path.insert(0,str(RUN));sys.path.insert(0,str(RUN/'vendor'))
import numpy as np,pyarrow as pa,lancedb
from live_broker import NativeBroker
root=HERE/('fixed_alias_'+uuid.uuid4().hex[:8]);root.mkdir()
sql=sqlite3.connect(root/'source.sqlite',check_same_thread=False)
sql.execute('pragma journal_mode=WAL');sql.execute('create table history(row integer,epoch integer,version integer,deleted integer,vector blob,payload text,primary key(row,epoch))')
sql.execute('create table meta(epoch integer,floor integer)');sql.execute('insert into meta values(0,0)')
sql.executemany('insert into history values(?,?,?,?,?,?)',[(i,0,1,0,np.array([100.+i,0.],dtype=np.float32).tobytes(),f'{i}:old') for i in range(2)]);sql.commit()
def arrow(rec):return pa.table({'row':pa.array([r[0] for r in rec],type=pa.int32()),'version':pa.array([1]*len(rec),type=pa.int32()),'vector':pa.array([r[1] for r in rec],type=pa.list_(pa.float32(),2))})
def error(fn):
    try:fn();return None
    except Exception as e:return type(e).__name__+': '+str(e)
db=lancedb.connect(str(root/'lance'));t=db.create_table('docs',arrow([(0,[100.,0.]),(1,[101.,0.])]))
a=NativeBroker(sql,db);m=a.publish('old',0,t);tok=a.begin(0,'old');baseline=a.query(tok,[100.,0.])
t.delete('row>=0');t.add(arrow([(0,[101.,0.]),(1,[100.,0.])]))
new=t.version() if callable(t.version) else t.version
external=a.checkout(m);external.checkout(new)
isolated=a.query(tok,[100.,0.]);assert baseline==isolated and not hasattr(m,'handle')
payload_mutation_error=error(lambda:m.source_view.__setitem__(0,(1,'changed')));assert payload_mutation_error
entered=[threading.Event(),threading.Event()];release=[threading.Event(),threading.Event()];answers=[None,None];fail=[]
def query(i):
    def hook():entered[i].set();assert release[i].wait(10)
    try:answers[i]=a.query(tok,[100.,0.],hook)
    except Exception as e:fail.append(repr(e))
threads=[threading.Thread(target=query,args=(i,)) for i in range(2)]
for th in threads:th.start()
assert all(e.wait(10) for e in entered)
a.end(tok);a.withdraw('old');a.gc();assert 'old' in a.manifests
audit_error=error(lambda:a.audit_hold(tok));assert audit_error
release[0].set();threads[0].join(10);assert tok in a.pins and 'old' in a.manifests
release[1].set();threads[1].join(10);assert tok not in a.pins and not fail and answers==[baseline,baseline]
removed=[m.name for m in a.gc()[1]];assert removed==['old']
out={'source_sha256':hashlib.sha256((RUN/'live_broker.py').read_bytes()).hexdigest(),'descriptor_exposes_handle':hasattr(m,'handle'),'baseline':baseline,'external_alias_changed_generation_query':isolated,'source_view_mutation_error':payload_mutation_error,'closing_audit_error':audit_error,'two_queries_retained_until_last_finish':True,'answers':answers,'removed_after_last_query':removed}
(HERE/'fixed_alias_results.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out));a.close()
