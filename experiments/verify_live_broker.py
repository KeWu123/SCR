from pathlib import Path
import json,hashlib,sqlite3
RUN=Path(__file__).resolve().parent;r=json.loads((RUN/'live_broker_results.json').read_text());count=0
for c in r['cells']:
    old={i:1 for i in range(c['n'])};new=old.copy()
    for i in c['updates']:new[i]=2
    for i in c['deletes']:new.pop(i)
    for i in c['reinserts']:new[i]=3
    assert c['rejected_mismatched_publish'] and c['atomic_begin_blocks_gc'] and c['old_native_reopen_rejected'] and c['inflight_native_query_protected']
    assert all(c[field] for field in ['external_alias_isolated','returned_handle_alias_isolated','manifest_has_no_handle','second_authority_rejected','uncommitted_source_rejected','republish_below_floor_rejected','restart_preserves_read_frontier'])
    assert c['old_before']==c['old_after_maintenance']==c['old_after_withdraw_audit']
    assert c['source_rows_before_gc']==c['n']+len(c['updates'])+len(c['deletes']) and c['source_rows_after_gc']==c['n']
    for field,truth in [('old_before',old),('old_after_maintenance',old),('old_after_withdraw_audit',old),('current_after_gc',new)]:
        for answers in c[field]:
            assert len(answers)==10 and len({a for a,v,p in answers})==10
            for row,version,payload in answers:assert truth.get(row)==version and payload==f'{row}:{version}';count+=1
    db=sqlite3.connect(Path(r['dbdir'])/(f"{c['dataset']}_{c['seed']}.sqlite"));actual=list(db.execute('select row,version,deleted from history order by row'));assert db.execute('select epoch,floor from meta').fetchone()==(1,1);db.close()
    assert len(actual)==c['n']
    for row,version,deleted in actual:
        assert (row not in new)==bool(deleted)
        if not deleted:assert new[row]==version
out=dict(status='passed',cells=len(r['cells']),returned_versions_checked=count,source_sha256=r['code_sha256'],driver_sha256=r['driver_sha256'],checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());(RUN/'live_broker_verification.json').write_text(json.dumps(out,indent=2));print(out)
