"""Native boundary and ownership regressions."""
from pathlib import Path
from _evidence import load_record, require_database
import sys,json,sqlite3,shutil,datetime,threading,hashlib
HERE=Path(__file__).resolve().parent
RUN=HERE.parent.parent
sys.path.insert(0,str(RUN/'vendor'));sys.path.insert(0,str(HERE))
import numpy as np, pyarrow as pa,lancedb
from interval_broker import IntervalBroker

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rejection(f):
    try:f()
    except (ValueError,KeyError,RuntimeError):return True
    return False
def main():
    latest=load_record(HERE/'latest.json');raw=load_record(latest['results'])
    root=HERE/('adversarial_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));root.mkdir()
    checks=[]
    def expect(name,value):assert value,name;checks.append(name)
    base=raw['disconnected'][0];require_database(base['base_source'])
    for mode in ['interval','point']:
        work=root/mode;work.mkdir();shutil.copy2(base['base_source'],work/'source.sqlite');shutil.copytree(base['base_native'],work/'lance')
        c=sqlite3.connect(work/'source.sqlite',check_same_thread=False,isolation_level=None);db=lancedb.connect(str(work/'lance'));t=db.open_table('objects');t.checkout(base['native_version'])
        a=IntervalBroker(c,db,mode);m=a.publish('g',t,requested_cuts=range(9))
        expect(mode+': second source authority rejected',rejection(lambda:IntervalBroker(c,db,mode)))
        expect(mode+': duplicate registry name rejected',rejection(lambda:a.publish('g',t)))
        for value in [True,False,1.0,-1,9]:expect(mode+': bad boundary '+repr(value),rejection(lambda value=value:a.begin(value,'g')))
        expect(mode+': exposed catalogue immutable',rejection(lambda:m.catalogue.__setitem__(0,()))) if hasattr(m.catalogue,'__setitem__') else expect(mode+': exposed catalogue immutable',True)
        expect(mode+': cleanup while any support open rejected',rejection(lambda:a.cleanup(t)))
        expect(mode+': whole-table drop while support open rejected',rejection(lambda:a.drop_retired_table(t)))
        a.withdraw_component('g',0,0);expect(mode+': partial component split exact',a.exact_support['g'].intervals==((1,1),(4,4),(7,8)))
        expect(mode+': withdrawn boundary rejected',rejection(lambda:a.begin(0,'g')))
        token=a.begin(1,'g');a.end(token);floor,_=a.gc();expect(mode+': GC uses remaining component minimum',floor==1)
        token=a.begin(7,'g')
        # Hook executes after registered pin; failed Begin rolls it back.
        prior=dict(a.pins)
        def fail(*args):raise ValueError('hook failed')
        expect(mode+': Begin hook fails',rejection(lambda:a.begin(8,'g',hook=fail)))
        expect(mode+': failed Begin leaves no ghost pin',a.pins==prior)
        a.withdraw('g');floor,_=a.gc();expect(mode+': withdrawn support still held by episode',floor==7 and 'g' in a.manifests)
        expect(mode+': retired but pinned table drop rejected',rejection(lambda:a.drop_retired_table(t)))
        a.end(token);floor,removed=a.gc();expect(mode+': final release reclaims manifest',floor==8 and len(removed)==1)
        expect(mode+': old cut cannot resurrect',rejection(lambda:a.publish('old',t,requested_cuts=[0]))) if mode=='point' else expect(mode+': old cut cannot resurrect',a.publish('later',t).initial_support.intervals==((8,8),))
        if mode=='interval':a.withdraw('later');a.gc()
        # Failed publication callback must not leave support registered.
        expect(mode+': publication hook fails',rejection(lambda:a.publish('failure',t,requested_cuts=[8],registered_hook=fail)))
        expect(mode+': failed publication rolls registry back','failure' not in a.manifests and 'failure' not in a.exact_support)
        a.drop_retired_table(t);expect(mode+': actual retired table physically gone',not Path(t.uri).exists())
        a.close();c.close()
    # Independent candidate corruption: key identity alone is insufficient.
    work=root/'corruption';work.mkdir();shutil.copy2(base['base_source'],work/'source.sqlite')
    c=sqlite3.connect(work/'source.sqlite',check_same_thread=False,isolation_level=None);db=lancedb.connect(str(work/'lance'));original=lancedb.connect(base['base_native']).open_table('objects');original.checkout(base['native_version']);records=original.to_arrow().to_pylist()
    wrong=[dict(r) for r in records];victim=next(r for r in wrong if r['row']==0 and r['version']==1);victim['vector']=np.zeros(32,dtype=np.float32).tolist()
    t=db.create_table('wrong',pa.Table.from_pylist(wrong,schema=original.to_arrow().schema));a=IntervalBroker(c,db)
    m=a.publish('wrong',t);expect('embedding corruption removes exactly old component',m.initial_support.intervals==((4,4),(7,8)))
    expect('wrong embedding with valid key rejected at cut zero',rejection(lambda:a.begin(0,'wrong')))
    a.withdraw('wrong');a.gc()
    duplicate=db.create_table('duplicates',pa.Table.from_pylist(records+[records[0]],schema=original.to_arrow().schema))
    expect('duplicate native version identity rejected',rejection(lambda:a.publish('dup',duplicate)))
    c.execute('insert into history values(0,9,99,0,?,?)',(np.zeros(32,dtype=np.float32).tobytes(),'future'))
    expect('uncommitted future source event rejected',rejection(lambda:a.publish('future',t)))
    c.execute('delete from history where epoch=9');a.close();c.close()
    report={'status':'passed','checks':checks,'count':len(checks),'extension_sha256':sha(HERE/'interval_broker.py'),'checker_sha256':sha(__file__),'results_sha256':sha(latest['results']),'scratch':str(root),'scope':'Actual boundary, partial withdrawal, physical source/table ownership, callback rollback, native identity/embedding and future-event checks; this is not a complete proof.'}
    (HERE/'adversarial_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
