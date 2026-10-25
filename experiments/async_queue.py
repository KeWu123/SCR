from pathlib import Path
import numpy as np,threading,time,json,hashlib,collections
RUN=Path(__file__).resolve().parent
def run(seed,delay,focus_mode):
    rng=np.random.default_rng(seed);n=1024;d=16;x=rng.normal(size=(n,d)).astype(np.float32);initial=x.copy();ver=np.zeros(n,dtype=int);ix=x.copy();iv=ver.copy();lock=threading.Lock();queue=collections.deque();log=[];head=0;applied=0;recent=0;writer_done=threading.Event();ingester_done=threading.Event();history=[[(0,0,x[i].copy())] for i in range(n)]
    def writer():
        nonlocal head,recent
        rr=np.random.default_rng(seed+1000)
        for seq in range(1,251):
            row=int(rr.integers(n));vec=(x[row]+rr.normal(0,.01,d)).astype(np.float32)
            with lock:
                head=seq;recent=row;ver[row]+=1;x[row]=vec;history[row].append((seq,int(ver[row]),vec.copy()));ready=time.monotonic()+delay/1000;queue.append((seq,row,int(ver[row]),vec.copy(),ready));log.append(dict(seq=seq,row=row,version=int(ver[row]),ready=ready))
            time.sleep(.002)
        writer_done.set()
    def ingest():
        nonlocal applied
        while True:
            with lock:
                if queue and queue[0][4]<=time.monotonic():
                    seq,row,version,vec,ready=queue.popleft();ix[row]=vec;iv[row]=version;applied=seq
                finished=writer_done.is_set() and not queue
            if finished:ingester_done.set();return
            time.sleep(.0002)
    a=threading.Thread(target=writer);b=threading.Thread(target=ingest);a.start();b.start();episodes=[];post=0
    while post<20:
        with lock:
            s=head;c=applied;phase='drained' if ingester_done.is_set() else 'draining' if writer_done.is_set() else 'writing';focus=recent if focus_mode=='fresh' else int(rng.integers(n));query=x[focus].copy();truth=ver.copy();base_read=int(ver[focus]);pinnedx=ix.copy();pinnedv=iv.copy();historical={row:next(v for seq,v,vec in reversed(h) if seq<=c) for row,h in enumerate(history)};pinned_source_read=historical[focus];pinned_query=next(vec.copy() for seq,v,vec in reversed(history[focus]) if seq<=c)
        time.sleep(.001)
        with lock:currentx=ix.copy();currentv=iv.copy();c2=applied
        distances=((currentx-query)**2).sum(axis=1);ids=np.lexsort((np.arange(n),distances))[:10];answers=[(int(i),int(currentv[i])) for i in ids]
        pd=((pinnedx-pinned_query)**2).sum(axis=1);pids=np.lexsort((np.arange(n),pd))[:10];panswers=[(int(i),int(pinnedv[i])) for i in pids]
        episodes.append(dict(snapshot=s,pinned_snapshot=c,applied_at_call=c2,phase=phase,focus=focus,source_read=base_read,pinned_source_read=pinned_source_read,answers=answers,pinned_answers=panswers,invalid=int(sum(truth[i]!=v for i,v in answers)),pinned_invalid=int(sum(historical[i]!=v for i,v in panswers)),observed_conflict=any(i==focus and v!=base_read for i,v in answers),backlog=s-c,snapshot_age=s-c))
        if phase=='drained':post+=1
        time.sleep(.001)
    a.join();b.join();assert applied==head==250 and not queue
    return dict(seed=seed,delay_ms=delay,focus_mode=focus_mode,source_log=log,episodes=episodes,final_applied=applied,final_head=head)
if __name__=='__main__':
    cells=[]
    for focus in ['uniform','fresh']:
      for delay in [0,10,40]:
       for seed in [11,29,47]:
        c=run(seed,delay,focus);cells.append(c);print(focus,delay,seed,len(c['episodes']),flush=True)
    out=dict(code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((RUN/'async_protocol.md').read_bytes()).hexdigest(),cells=cells);(RUN/'async_results.json').write_text(json.dumps(out,indent=2))
