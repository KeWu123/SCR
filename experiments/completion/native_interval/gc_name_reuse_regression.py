"""Deterministic actual-native ABA attack at base/derived GC boundary."""
from pathlib import Path
from _evidence import load_record, require_database
import sys,json,sqlite3,shutil,datetime,threading,time,hashlib,traceback
HERE=Path(__file__).resolve().parent
RUN=HERE.parent.parent
sys.path.insert(0,str(RUN/'vendor'));sys.path.insert(0,str(HERE));sys.path.insert(0,str(RUN))
import numpy as np,lancedb
from interval_broker import IntervalBroker
from live_broker import NativeBroker
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    pointer=load_record(HERE/'latest.json');raw=load_record(pointer['results']);base=raw['disconnected'][0];require_database(base['base_source'])
    scratch=HERE/('gc_aba_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));scratch.mkdir();checks=[]
    for mode in ['interval','point']:
        work=scratch/mode;work.mkdir();shutil.copy2(base['base_source'],work/'source.sqlite');shutil.copytree(base['base_native'],work/'lance')
        c=sqlite3.connect(work/'source.sqlite',check_same_thread=False,isolation_level=None);db=lancedb.connect(str(work/'lance'));table=db.open_table('objects');table.checkout(base['native_version']);q=np.load(base['query_file'])['q'][0]
        a=IntervalBroker(c,db,mode);a.publish('reuse',table,requested_cuts=range(9));old=a.begin(0,'reuse');a.end(old);a.withdraw('reuse')
        base_retired=threading.Event();allow_cleanup=threading.Event();publisher_started=threading.Event();published=threading.Event();errors=[];new_tokens=[]
        original=NativeBroker.gc
        def paused_base_gc(self):
            value=original(self)
            if self is a:
                base_retired.set()
                if not allow_cleanup.wait(30):raise RuntimeError('test timeout')
            return value
        NativeBroker.gc=paused_base_gc
        def collect():
            try:a.gc()
            except BaseException:errors.append(traceback.format_exc())
        def republish():
            try:
                publisher_started.set();a.publish('reuse',table,requested_cuts=[8]);token=a.begin(8,'reuse');new_tokens.append(token);a.query(token,q);published.set()
            except BaseException:errors.append(traceback.format_exc())
        collector=threading.Thread(target=collect);publisher=threading.Thread(target=republish)
        try:
            collector.start();assert base_retired.wait(30)
            # Nonblocking acquisition makes the mutual-exclusion check
            # deterministic even if the publisher thread is not scheduled.
            acquired_in_gap=a.lock.acquire(blocking=False)
            if acquired_in_gap:a.lock.release()
            assert not acquired_in_gap,'retention lock released before derived cleanup'
            publisher.start();assert publisher_started.wait(30)
            # The attempted publisher must remain excluded while GC is paused
            # exactly after base retirement but before derived cleanup.
            leaked=published.wait(.2);assert not leaked,'new state published into stale-cleanup gap'
            allow_cleanup.set();collector.join(30);publisher.join(30)
            assert not collector.is_alive() and not publisher.is_alive() and not errors,errors
            assert published.is_set() and a.exact_support['reuse'].intervals==((8,8),)
            assert ('reuse',8) in a._views and len(a.query(new_tokens[0],q))==10
            a.end(new_tokens[0]);a.withdraw('reuse')
            checks.append({'mode':mode,'publisher_excluded_during_paused_gc':not leaked,'reused_name_survived_cleanup':True,'new_view_survived_cleanup':True,'new_native_query_returned':10})
        finally:
            allow_cleanup.set();NativeBroker.gc=original
            collector.join(30)
            if publisher.ident is not None:publisher.join(30)
            if not a.pins:a.close()
            c.close()
    report={'status':'passed','extension_sha256':sha(HERE/'interval_broker.py'),'checker_sha256':sha(__file__),'input_results_sha256':sha(pointer['results']),'checks':checks,'scratch':str(scratch)}
    (HERE/'gc_name_reuse_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
