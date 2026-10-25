"""Actual-native regressions for physical domains, exact keys and callbacks."""
from pathlib import Path
import sys,json,sqlite3,datetime,hashlib
HERE=Path(__file__).resolve().parent;RUN=HERE.parent.parent
sys.path.insert(0,str(RUN/'vendor'));sys.path.insert(0,str(HERE))
import numpy as np,pyarrow as pa,lancedb
from interval_broker import IntervalBroker
from verify_native_intervals import exact_native_rows

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rejected(f):
    try:f()
    except (ValueError,AssertionError):return True
    return False
def main():
    root=HERE/('domain_key_hook_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));root.mkdir();checks=[]
    def expect(label,condition):assert condition,label;checks.append(label)
    for mode in ['interval','point']:
        work=root/mode;work.mkdir();sql=sqlite3.connect(work/'source.sqlite',check_same_thread=False,isolation_level=None)
        sql.execute('pragma journal_mode=WAL');sql.execute('create table history(row integer,epoch integer,version integer,deleted integer,vector blob,payload text,primary key(row,epoch))');sql.execute('create table meta(epoch integer,floor integer)');sql.execute('insert into meta values(0,0)')
        x=np.random.default_rng(9).random((16,4),dtype=np.float32);ids=[-(1<<63),(1<<63)-1,-1]+list(range(13));versions=[(1<<63)-1]*16
        sql.executemany('insert into history values(?,0,?,0,?,?)',[(r,v,x[i].tobytes(),str(r)) for i,(r,v) in enumerate(zip(ids,versions))])
        schema=pa.schema([('row',pa.int64()),('version',pa.int64()),('vector',pa.list_(pa.float32(),4))])
        records=[{'row':r,'version':v,'vector':x[i].tolist()} for i,(r,v) in enumerate(zip(ids,versions))]
        db=lancedb.connect(str(work/'lance'));t=db.create_table('objects',pa.Table.from_pylist(records,schema=schema));a=IntervalBroker(sql,db,mode);a.publish('base',t,requested_cuts=[0]);token=a.begin(0,'base')
        expect(mode+': signed row boundaries and max positive version valid',len(a.query(token,x[0]))==10)
        foreign=lancedb.connect(str(work/'foreign')).create_table('objects',t.to_arrow())
        for name,operation in [('drop',lambda:a.drop_retired_table(foreign)),('cleanup',lambda:a.cleanup(foreign)),('maintain',lambda:a.maintain(foreign))]:
            expect(mode+': foreign same-name '+name+' rejected',rejected(operation));expect(mode+': configured table survives '+name,Path(t.uri).exists() and len(a.query(token,x[0]))==10);expect(mode+': foreign table survives '+name,Path(foreign.uri).exists())
        operations=[('publish',lambda:a.publish('nested',t)),('begin',lambda:a.begin(0,'base')),('query',lambda:a.query(token,x[0])),('audit_hold',lambda:a.audit_hold(token)),('end',lambda:a.end(token)),('withdraw',lambda:a.withdraw('base')),('withdraw_component',lambda:a.withdraw_component('base',0,0)),('gc',lambda:a.gc()),('cleanup',lambda:a.cleanup(t)),('maintain',lambda:a.maintain(t)),('drop',lambda:a.drop_retired_table(t)),('close',lambda:a.close())]
        for label,operation in operations:
            def callback(*unused):operation()
            before=dict(a.pins)
            expect(mode+': publish hook rejects '+label,rejected(lambda:a.publish('stage_'+label,t,requested_cuts=[0],registered_hook=callback)))
            expect(mode+': failed staged publication leaves registry/pins coherent '+label,'stage_'+label not in a.manifests and 'stage_'+label not in a.exact_support and a.pins==before)
            expect(mode+': Begin hook rejects '+label,rejected(lambda:a.begin(0,'base',hook=callback)))
            expect(mode+': failed Begin leaves pins coherent '+label,a.pins==before)
            expect(mode+': query hook rejects '+label,rejected(lambda:a.query(token,x[0],hook=callback)))
            expect(mode+': failed query leaves refs coherent '+label,a.pins==before and not a.ops and not a._ending)
        expect(mode+': callbacks leave no persistent reentry flag',len(a.query(token,x[0]))==10)
        keycases=[('fractional',pa.float64(),pa.float64(),[r/16+.5 for r in range(16)],[1.5]*16),('integral_float',pa.float64(),pa.float64(),list(map(float,range(16))),[1.0]*16),('bool_row',pa.bool_(),pa.int64(),[True]*16,[1]*16),('string_row',pa.string(),pa.int64(),[str(r) for r in range(16)],[1]*16),('null_row',pa.int64(),pa.int64(),[None]+list(range(15)),[1]*16),('unsigned_row_overflow',pa.uint64(),pa.int64(),[(1<<63)+r for r in range(16)],[1]*16),('zero_version',pa.int64(),pa.int64(),list(range(16)),[0]*16),('negative_version',pa.int64(),pa.int64(),list(range(16)),[-1]*16),('bool_version',pa.int64(),pa.bool_(),list(range(16)),[True]*16),('null_version',pa.int64(),pa.int64(),list(range(16)),[None]+[1]*15),('unsigned_version_overflow',pa.int64(),pa.uint64(),list(range(16)),[(1<<63)]*16)]
        for label,rtype,vtype,rows,vs in keycases:
            arr=pa.table({'row':pa.array(rows,type=rtype),'version':pa.array(vs,type=vtype),'vector':pa.array(x.tolist(),type=pa.list_(pa.float32(),4))});bad=db.create_table('bad_'+label,arr)
            expect(mode+': production rejects '+label,rejected(lambda:a.publish(label,bad,requested_cuts=[0])))
            expect(mode+': independent checker rejects '+label,rejected(lambda:exact_native_rows(bad.to_arrow())))
            expect(mode+': rejected key schema creates no registry '+label,label not in a.manifests)
        a.end(token);a.withdraw('base');a.gc();a.close();sql.close()
    report={'status':'passed','count':len(checks),'checks':checks,'extension_sha256':sha(HERE/'interval_broker.py'),'checker_sha256':sha(__file__),'independent_checker_sha256':sha(HERE/'verify_native_intervals.py'),'scratch':str(root)}
    (HERE/'domain_key_hook_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))
if __name__=='__main__':main()
