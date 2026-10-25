from pathlib import Path
import sys,json,sqlite3,hashlib,datetime,warnings
RUN=Path(__file__).resolve().parent;sys.path.insert(0,str(RUN/'vendor'))
warnings.filterwarnings('ignore')
import numpy as np,pyarrow as pa,lancedb
from sklearn.datasets import load_digits
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def arrow(ids,vs,x):return pa.table({'row':pa.array(ids,type=pa.int32()),'version':pa.array(vs,type=pa.int32()),'vector':pa.array(x.tolist(),type=pa.list_(pa.float32(),64))})
def version(t):return t.version() if callable(t.version) else t.version
def rows(c):return dict(c.execute('select row,version from docs order by row'))
def run():
    root=RUN/'cross_store_db'/datetime.datetime.now().strftime('%Y%m%d%H%M%S%f');root.mkdir(parents=True);db=lancedb.connect(str(root/'lance'));cells=[]
    for name in ['digits','uniform']:
      for seed in [11,29,47]:
        rng=np.random.default_rng(seed)
        if name=='digits':
            data=load_digits().data.astype(np.float32)/16;p=rng.permutation(len(data));q=data[p[:100]];x=data[p[100:]]
        else:x=rng.random((2000,64),dtype=np.float32);q=rng.random((100,64),dtype=np.float32)
        rep=x[rng.permutation(len(x))]+rng.normal(0,.02,x.shape).astype(np.float32);n=len(x);nameid=f'{name}_{seed}'
        inp=RUN/(nameid+'_inputs.npz');np.savez_compressed(inp,x=x,q=q,replacements=rep)
        w=sqlite3.connect(root/(nameid+'.sqlite'));w.execute('pragma journal_mode=WAL')
        w.execute('create table docs(row integer primary key,version integer,payload text)');w.execute('create table meta(epoch integer)');w.execute('insert into meta values(0)')
        w.executemany('insert into docs values(?,?,?)',[(i,1,f'{i}:1') for i in range(n)]);w.commit()
        t=db.create_table(nameid,arrow(np.arange(n),np.ones(n,dtype=int),x));t.create_index(index_type='IVF_HNSW_FLAT',metric='l2',num_partitions=4,m=16,ef_construction=64)
        ov=version(t);old=db.open_table(nameid);old.checkout(ov);r=sqlite3.connect(root/(nameid+'.sqlite'));r.execute('begin');initial=rows(r)
        def searches(h,k):return [[(int(a['row']),int(a['version'])) for a in h.search(vec).nprobes(4).ef(128).limit(k).to_arrow().to_pylist()] for vec in q]
        before=searches(old,10);order=np.random.default_rng(seed+777).permutation(n);u=order[:n//4];d=order[n//4:n//4+n//10];rein=d[:len(d)//2]
        w.executemany('update docs set version=2,payload=? where row=?',[(f'{int(i)}:2',int(i)) for i in u]);w.executemany('delete from docs where row=?',[(int(i),) for i in d]);w.executemany('insert into docs values(?,?,?)',[(int(i),3,f'{int(i)}:3') for i in rein]);w.execute('update meta set epoch=1');w.commit()
        current=rows(w);oldafter=rows(r);assert oldafter==initial
        t.delete('row IN ('+','.join(map(str,np.concatenate([u,d])))+')');t.add(arrow(u,np.full(len(u),2),rep[u]));t.add(arrow(rein,np.full(len(rein),3),rep[rein]));t.create_index(index_type='IVF_HNSW_FLAT',metric='l2',num_partitions=4,m=16,ef_construction=64,replace=True);t.optimize()
        after=searches(old,10);now=searches(t,10);over=searches(t,40)
        op=sorted((a['row'],a['version']) for a in old.to_arrow().select(['row','version']).to_pylist());cp=sorted((a['row'],a['version']) for a in t.to_arrow().select(['row','version']).to_pylist())
        assert op==sorted(initial.items()) and cp==sorted(current.items()) and before==after
        def truth(versions,vec):
            scored=[(float(np.sum((vec.astype(float)-(x[a] if v==1 else rep[a]).astype(float))**2)),a,v) for a,v in versions.items()]
            return [(a,v) for _,a,v in sorted(scored)[:10]]
        records=[]
        for j,vec in enumerate(q):
            ot=truth(initial,vec);nt=truth(current,vec);filtered=[k for k in over[j] if initial.get(k[0])==k[1]][:10]
            for policy,ans,target,source in [('admitted_old',after[j],ot,initial),('mismatched',now[j],ot,initial),('mismatched_filter',filtered,ot,initial),('aligned_current',now[j],nt,current)]:records.append(dict(query=j,policy=policy,answers=ans,truth=target,invalid=sum(source.get(a)!=v for a,v in ans),recall=len(set(ans)&set(target))/10,fill=len(ans)/10))
        summaries={}
        for policy in ['admitted_old','mismatched','mismatched_filter','aligned_current']:
            rr=[z for z in records if z['policy']==policy];summaries[policy]=dict(invalid=sum(z['invalid'] for z in rr),returned=sum(len(z['answers']) for z in rr),recall=float(np.mean([z['recall'] for z in rr])),fill=float(np.mean([z['fill'] for z in rr])))
        cells.append(dict(dataset=name,seed=seed,n=n,inputs_sha256=sha(inp),old_sqlite=initial,old_sqlite_after_writer_commit=oldafter,current_sqlite=current,old_sqlite_payloads=list(r.execute('select row,version,payload from docs order by row')),current_sqlite_payloads=list(w.execute('select row,version,payload from docs order by row')),old_epoch=r.execute('select epoch from meta').fetchone()[0],current_epoch=w.execute('select epoch from meta').fetchone()[0],old_lance_version=ov,current_lance_version=version(t),updates=u.tolist(),deletes=d.tolist(),reinserts=rein.tolist(),old_population=op,current_population=cp,current_old_population_coverage=len(set(op)&set(cp))/n,answers_before=before,answers_after=after,native_candidates40=over,completed_maintenance=True,old_plan=old.search(q[0]).nprobes(4).ef(128).limit(10).explain_plan(),current_plan=t.search(q[0]).nprobes(4).ef(128).limit(10).explain_plan(),records=records,summary=summaries))
        r.rollback();r.close();w.close();print(name,seed,summaries,flush=True)
    result=dict(code_sha256=sha(Path(__file__)),protocol_sha256=sha(RUN/'cross_store_protocol.md'),lancedb_version=lancedb.__version__,sqlite_version=sqlite3.sqlite_version,dbdir=str(root),cells=cells)
    (RUN/'cross_store_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
if __name__=='__main__':run()
