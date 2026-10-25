from pathlib import Path
import json,hashlib
RUN=Path(__file__).resolve().parent;data=json.loads((RUN/'async_results.json').read_text());count=0
for c in data['cells']:
    log=c['source_log'];assert [v['seq'] for v in log]==list(range(1,251)) and c['final_applied']==c['final_head']==250
    for e in c['episodes']:
        def resolved(row,s):return next((v['version'] for v in reversed(log) if v['row']==row and v['seq']<=s),0)
        assert e['source_read']==resolved(e['focus'],e['snapshot']);assert e['pinned_source_read']==resolved(e['focus'],e['pinned_snapshot'])
        assert e['invalid']==sum(resolved(row,e['snapshot'])!=v for row,v in e['answers'])
        assert e['pinned_invalid']==sum(resolved(row,e['pinned_snapshot'])!=v for row,v in e['pinned_answers'])==0
        assert e['observed_conflict']==any(row==e['focus'] and v!=e['source_read'] for row,v in e['answers'])
        if e['phase']=='drained':assert e['backlog']==0 and e['invalid']==0
        count+=1
out=dict(status='passed',cells=len(data['cells']),episodes=count,source_sha256=data['code_sha256'],checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());(RUN/'async_verification.json').write_text(json.dumps(out,indent=2));print(out)
