"""Create compact, manuscript-ready tables from verified raw measurements."""
from pathlib import Path
import json, sys, hashlib
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'vendor'))
import numpy as np
raw=json.loads((ROOT/'comparative_results.json').read_text(encoding='utf-8'))
verified=json.loads((ROOT/'comparative_verification.json').read_text(encoding='utf-8'))
assert verified['passed'] and verified['result_sha256']==hashlib.sha256((ROOT/'comparative_results.json').read_bytes()).hexdigest()
arms=['episode_lock','retained_wrapper','scr']
labels={'episode_lock':'Episode lock','retained_wrapper':'Retained wrapper','scr':'SCR broker'}
def mdtable(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows])
def median(xs):return float(np.median(xs))
def f(value):return f'{value:.3f}'
rows=[];summary=[]
for n,d in [(2000,32),(10000,64),(50000,64)]:
    cells=[c for c in raw['cells'] if c['n']==n]
    oracle=[c['mean_recall10'] for c in verified['cells'] if c['n']==n]
    for arm in arms:
        pooled={key:[v for c in cells for v in c['arms'][arm][key]] for key in ['publication_ns','admission_ns','end_ns','query_ns','episode16_ns']}
        record={'n':n,'d':d,'arm':arm,'publication_median_ms':median(pooled['publication_ns'])/1e6,'admission_median_ms':median(pooled['admission_ns'])/1e6,'end_median_ms':median(pooled['end_ns'])/1e6,'query_median_ms':median(pooled['query_ns'])/1e6,'query_p99_ms':float(np.percentile(pooled['query_ns'],99))/1e6,'query_samples':len(pooled['query_ns']),'episode16_median_ms':median(pooled['episode16_ns'])/1e6,'recall10_mean':sum(oracle)/len(oracle),'json_metadata_bytes':median([c['arms'][arm]['metadata']['json_bytes'] for c in cells]),'python_metadata_bytes':median([c['arms'][arm]['metadata']['python_deep_bytes'] for c in cells])}
        summary.append(record)
        rows.append([f'{n:,}×{d}',labels[arm],f(record['publication_median_ms']),f(record['admission_median_ms']),f(record['query_median_ms']),f(record['query_p99_ms']),record['query_samples'],f(record['episode16_median_ms']),f(record['recall10_mean'])])
costtable=mdtable(['N×d','Arm','Publish ms','Begin ms','Query p50 ms','Query p99 ms','Query n','16-query episode ms','Recall@10'],rows)
concurrency=[];rows=[]
for n in [2000,10000]:
    for hold in [100,300]:
        for readers in [1,4]:
            for arm in arms:
                cases=[z for c in raw['cells'] if c['n']==n for z in c['concurrency'] if z['arm']==arm and z['requested_hold_ms']==hold and z['readers']==readers]
                samples=[v for z in cases for thread in z['reader_threads'] for v in thread['query_ns']]
                reader_wall=sum(max(t['end_ns'] for t in z['reader_threads'])-min(t['start_ns'] for t in z['reader_threads']) for z in cases)
                record={'n':n,'arm':arm,'requested_hold_ms':hold,'readers':readers,'seeds':len(cases),'gate_wait_median_ms':median([z['writer']['gate_wait_ns'] for z in cases])/1e6,'writer_median_ms':median([z['writer']['total_ns'] for z in cases])/1e6,'source_commit_during':sum(z['writer']['source_commit_before_release'] for z in cases),'publication_during':sum(z['writer']['publication_before_release'] for z in cases),'query_count':len(samples),'query_p99_ms':float(np.percentile(samples,99))/1e6,'reader_qps':len(samples)/(reader_wall/1e9),'actual_hold_median_ms':median([z['actual_hold_ns'] for z in cases])/1e6}
                record['source_commit_before_requested_deadline']=sum(z['writer']['source_commit_ns']<z['hold_start_ns']+z['requested_hold_ms']*1_000_000 for z in cases)
                record['validated_generation_before_requested_deadline']=sum(z['writer']['published_ns']<z['hold_start_ns']+z['requested_hold_ms']*1_000_000 for z in cases)
                concurrency.append(record)
                rows.append([f'{n:,}',f'{hold}/{readers}',labels[arm],f(record['actual_hold_median_ms']),f(record['gate_wait_median_ms']),f(record['writer_median_ms']),f"{record['source_commit_during']}/3",f"{record['publication_during']}/3",f(record['reader_qps']),f(record['query_p99_ms']),record['query_count']])
concurrencytable=mdtable(['N','Requested ms/readers','Arm','Actual hold ms','Writer wait ms','Writer full ms','Source commits during','Validated generations ready during','Reader queries/s','Reader p99 ms','Reader n'],rows)
rows=[];storage=[]
for n,d in [(2000,32),(10000,64),(50000,64)]:
    cells=[c for c in raw['cells'] if c['n']==n]
    record={'n':n,'d':d,'source_initial_bytes':median([c['initial_storage']['source']['total_bytes'] for c in cells]),'native_initial_bytes':median([c['initial_storage']['native']['bytes'] for c in cells]),'native_data_bytes':median([c['initial_storage']['native']['categories']['data'] for c in cells]),'native_index_bytes':median([c['initial_storage']['native']['categories']['index'] for c in cells]),'metadata_json_bytes':median([c['arms']['scr']['metadata']['json_bytes'] for c in cells]),'metadata_python_bytes':median([c['arms']['scr']['metadata']['python_deep_bytes'] for c in cells]),'source_build_ms':median([c['source_materialization_ns'] for c in cells])/1e6,'native_write_ms':median([c['native_table_write_ns'] for c in cells])/1e6,'index_build_ms':median([c['native_index_build_ns'] for c in cells])/1e6}
    storage.append(record)
    rows.append([f'{n:,}×{d}',f(record['source_initial_bytes']/1048576),f(record['native_initial_bytes']/1048576),f(record['native_data_bytes']/1048576),f(record['native_index_bytes']/1048576),f(record['metadata_json_bytes']/1048576),f(record['metadata_python_bytes']/1048576),f(record['index_build_ms'])])
storagetable=mdtable(['N×d','Source MiB','Native total MiB','Native data MiB','Native index MiB','Metadata JSON MiB','Metadata Python MiB','Index build ms'],rows)
retention=[];rows=[]
for n in [2000,10000]:
    for arm in arms:
        cases=[z for c in raw['cells'] if c['n']==n for z in c['concurrency'] if z['arm']==arm and z['requested_hold_ms']==300 and z['readers']==4]
        rec={'n':n,'arm':arm}
        row=[f'{n:,}',labels[arm]]
        for stage in ['storage_before','storage_during_hold','storage_after_writer','storage_after_gc']:
            src=median([z[stage]['source']['total_bytes'] for z in cases]);native=median([z[stage]['native']['bytes'] for z in cases]);rec[stage]={'source_bytes':src,'native_bytes':native};row.append(f'{src/1048576:.3f}/{native/1048576:.3f}')
        rec['source_gc_median_ms']=median([z['source_gc_ns'] for z in cases])/1e6;rec['native_gc_median_ms']=median([z['native_gc_ns'] for z in cases])/1e6;retention.append(rec);rows.append(row)
retentiontable=mdtable(['N','Arm','Initial source/native MiB','During hold source/native MiB','After writer source/native MiB','After GC source/native MiB'],rows)
all_cases=[z for c in raw['cells'] for z in c['concurrency']]
old_mismatches=sum(thread['mismatches'] for z in all_cases for thread in z['reader_threads'])
nominal={arm:{'source':sum(z['writer']['source_commit_ns']<z['hold_start_ns']+z['requested_hold_ms']*1_000_000 for z in all_cases if z['arm']==arm),'ready':sum(z['writer']['published_ns']<z['hold_start_ns']+z['requested_hold_ms']*1_000_000 for z in all_cases if z['arm']==arm)} for arm in arms}
report=f'''# Correct alternatives: native experiment report

All nine native scale/seed cells and all 72 real writer cases passed the independent checker. The three correct arms returned identical answers for each of 64 queries per cell, with zero version/payload errors and {old_mismatches} old-answer mismatches under concurrent updates. The scalar reference recomputed 16 exact queries per cell (144 total), and the checker inspected all initial native vectors and final retained source/native databases.

The correct retained wrapper is algorithmically equivalent to SCR for this exact-cut workload. Both permit new source/index publication while retaining old episodes. SCR's contribution here is an explicit support/lifecycle contract; these results do not establish a concurrency or correctness advantage over a correct retained wrapper. The updated reference SCR implementation opens a private native checkout on each query, while the wrapper keeps one per episode, so query overhead includes that implementation choice.

## Publication, admission and complete episodes

{costtable}

Values are medians pooled over three seeds. Publication has one warmup and three measured scans per arm/seed (n=9 per setting); full episodes have three repetitions per arm/seed (n=9). Each episode includes begin, 16 native searches, source-version/payload filtering, and end. Queries have 48 warmups then 240 measurements per arm/seed (720 pooled samples per setting). Begin/end each have 240 measurements interleaved after query warmup, with no separate begin/end warmup schedule. Query p99 uses those 720 samples; the raw per-seed p99 uses 240 and is exploratory. Query time excludes publication/admission, while the complete-episode column includes admission/close. Publication includes source snapshot scan, native open/checkout and population/vector scan, validation, Python metadata construction and insertion into the arm's registry; it excludes common table/index creation. Initial native builds are timed below. All arms use the same 40-candidate budget, 10-result limit, source cut, native generation and queries.

## Real writer progress and readers

{concurrencytable}

Each row pools three seeds. Requested holds are real elapsed 100/300 ms; actual holds can be longer because the readers finish in-flight searches and the driver audits actual source snapshots and retained files before releasing episodes. Writer wait is actual gate-acquisition delay; full writer time includes gate wait, atomic SQLite source update, native delete/add, complete new source/native validation and metadata construction. Source commit and validated-generation readiness are counted separately. The writer creates a complete checked new descriptor but does not insert it into the old-episode registry or measure admission of cut 1; the raw timestamp named `published_ns` denotes this readiness endpoint. Actual publication/registration cost is measured separately in the first table. The episode-lock writer cannot commit while an episode remains open. Retained-wrapper/SCR source writers committed during all 24 cases for each arm. Validated-generation readiness may occur after a short hold, especially at N=10,000; the table exposes this rather than equating source progress with completed native work. Reader throughput uses the measured reader-thread execution window (excluding the later source/file audit); it is a short-run CPU observation under simultaneous writer work, not an engine benchmark. Native search stays on the initial indexed generation; writer changes append unindexed fragments and no hidden rebuild is omitted.

Against the originally requested deadline, source commit occurred in {nominal['retained_wrapper']['source']}/24 retained-wrapper and {nominal['scr']['source']}/24 SCR cases; complete validated generations were ready in only {nominal['retained_wrapper']['ready']}/24 and {nominal['scr']['ready']}/24, respectively. Both counts are 0/24 for the episode lock. The longer actual holds at N=10,000 expose interference from the Python publication scan/GIL and audit, so the experiment does not establish behavior under a precise fixed hold duration.

## Physical storage and metadata

{storagetable}

All units are MiB (2^20 bytes). Source includes the SQLite database, WAL and SHM; native total includes data, vector index, manifests and transactions. Full per-file inventories are in raw results. The final broker uses zero-byte OS ownership-lock files in the system temporary directory; these do not add file-content bytes to the table. JSON metadata includes exact population keys plus immutable source-version/payload mapping; Python metadata is a recursive `getsizeof` estimate excluding engine internals, source vectors, external handles, SQLite transaction memory and per-episode engine memory. These are reference-representation costs, not minimal metadata bounds or total process-memory estimates. Common source creation/native table creation/index creation are separately measured in raw results; the last column is native index construction only.

{retentiontable}

Retention rows use the 300 ms/four-reader cases and medians over three seeds. The lock cannot have new source/index data during its hold, whereas the other arms may; these during-hold sizes therefore describe genuinely different writer progress. After-writer sizes all include the same logical update. Reclamation withdraws old support only after episode references are released, prunes source anchors to cut 1, then performs physical native cleanup. The old native version is no longer reopenable in all 72 cases. SQLite history returns from 1.35N records to N retained anchors; SQLite is checkpointed but not vacuumed, so its file need not shrink. Native optimize creates/compacts current representations and measured total native bytes increase despite obsolete-version unavailability. These observations do not demonstrate a native disk-byte reduction. Storage includes retained engine versions until explicit cleanup and should not be read as an optimized lock-vs-wrapper space bound.

## Correctness and scope

Paused native query/end/GC tests retained in-flight references for all three arms; new queries after end were rejected, and the paused answer completed unchanged. A paused admission raced with withdrawal/GC; registration serialized with withdrawal, the admitted token remained usable, and later admission was rejected. Initial plans show native ANN IVF scans, native index statistics confirm IVF_HNSW_FLAT with all N rows indexed and zero unindexed rows, and every initial row/version/vector was inspected. The verifier imports neither driver nor broker implementation and uses scalar Float64 arithmetic for exact distances.

This is a Windows CPU reference run on Intel Core i9-14900HX (24 physical/32 logical cores), without fixed CPU affinity. Native-library worker defaults apply. N and d increase together, so the three settings demonstrate joint population/dimension scaling and do not isolate dimensionality effects. Concurrency covers N=2,000 and N=10,000 only. There are three seeds, uniform synthetic vectors and modest 100/300 ms holds; no high-concurrency, distributed, crash-recovery, production throughput or production p99 claim is justified. Full-population publication is O(Nd) and serialized Python source metadata is O(N); the measurements make those costs visible.

Protocol: comparative_protocol.md. Driver: compare_native_alternatives.py. Raw: comparative_results.json. Independent checker: verify_comparative_native.py. Verification: comparative_verification.json. All hashes and library versions are recorded in raw/verification output.
'''
(ROOT/'comparative_report.md').write_text(report,encoding='utf-8')
(ROOT/'comparative_summary.json').write_text(json.dumps({'costs':summary,'concurrency':concurrency,'storage':storage,'retention':retention},indent=2),encoding='utf-8')
tex=[]
for row in summary:
    tex.append(f"{row['n']:,}/{row['d']} & {labels[row['arm']]} & {row['publication_median_ms']:.1f} & {row['admission_median_ms']:.2f} & {row['query_median_ms']:.2f} & {row['query_p99_ms']:.2f} & {row['episode16_median_ms']:.1f} \\\\")
(ROOT/'comparative_cost_rows.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8')
print('REPORT',str(ROOT/'comparative_report.md'))
