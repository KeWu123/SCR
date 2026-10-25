"""Read-only recomputation of engine artifacts and a live PostgreSQL environment probe."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "paper" / "proto"
out = {}
for filename in ("results_engine_matched2.json", "results_engine_floor_disjoint.json",
                 "results_engine_matched2_pg_decomp.json"):
    doc = json.loads((P / filename).read_text(encoding="utf-8"))
    checks, failures = [], []
    for name, cell in doc.get("cells", {}).items():
        pe = cell.get("per_episode", [])
        if not pe or not all("set_equal" in e for e in pe):
            continue
        diff = sum(not e["set_equal"] for e in pe) / len(pe)
        stated = cell.get("tear_rate_any_difference")
        if stated is not None and abs(diff - stated) > 1e-6:
            failures.append({"cell": name, "computed": diff, "stated": stated})
        eq = [e for e in pe if abs(e.get("recall_A_vs_exact", -1) - e.get("recall_B_vs_exact", -2)) < 1e-9]
        checks.append({"cell": name, "n": len(pe), "any_difference": diff,
                       "n_equal_recall": len(eq),
                       "conditional_difference": sum(not e["set_equal"] for e in eq) / len(eq) if eq else None})
    out[filename] = {"cells_checked": len(checks), "failures": failures, "checks": checks}
    assert not failures, failures

pg = json.loads((P / "results_engine_deletion.json").read_text(encoding="utf-8"))
out["pg_artifact_top_keys"] = list(pg)
def find_pin(v, path=""):
    if isinstance(v, dict):
        if "pinned_pass1_vs_pass2" in v:
            out.setdefault("pg_pinning_records", []).append({
                "path": path, "reindex_thread_alive_during_pinned_search2": v.get("reindex_thread_alive_during_pinned_search2"),
                "reindex_error": v.get("reindex_error"),
                "reindex_wall_s": v.get("reindex_concurrently_wall_s"),
                "pinned": v["pinned_pass1_vs_pass2"]})
        for k, w in v.items():
            find_pin(w, path + "/" + k)
    elif isinstance(v, list):
        for i, w in enumerate(v):
            find_pin(w, path + "/" + str(i))
find_pin(pg)
try:
    import psycopg2
    c = psycopg2.connect(host="127.0.0.1", port=5439, dbname="scr_engine", user="postgres", connect_timeout=5)
    cur = c.cursor()
    cur.execute("select version()")
    out["pg_live_version"] = cur.fetchone()[0]
    cur.execute("select typname from pg_type where typname = 'vector'")
    out["pg_live_vector_type"] = cur.fetchone()
    c.close()
except Exception as exc:
    out["pg_live_probe_error"] = repr(exc)
Path(__file__).with_name("artifact_checks_r1.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps({k: {a:b for a,b in v.items() if a != "checks"} if isinstance(v,dict) else v
                  for k,v in out.items() if k != "pg_pinning_records"}, indent=2))
print("pg pinning records:", len(out.get("pg_pinning_records", [])))
