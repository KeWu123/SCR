"""Compare native table-version reads with source-cut coverage admission.

SQLite WAL + LanceDB 0.40.0, using the paper's disconnected-support schedule
(two 500-row groups over cuts 0..8).

Schedule (matching the paper's 9-cut example):
  cut 0: all N rows v1@0.
  cut 2: group-1 (G1 rows) replaced -> v2@2. LanceDB: delete G1 v1, add G1 v2.
  cut 3: group-2 (G2 rows) get v2@3.     LanceDB: delete G2 v1, add G2 v2.
  cut 4: G1 tombstone.                    LanceDB: delete G1 v2.
  cut 5: G2 get v3@5.                     LanceDB: delete G2 v2, add G2 v3.
  cut 6: G1 reinserted -> v4@6.           LanceDB: add G1 v4.
  cut 7: G2 get v4@7.                     LanceDB: delete G2 v3, add G2 v4.

SCR retains the population observed at checkpoints {0,1,4,7,8} (each row's
latest version there): pop = all (row,1) + (G2,2) + (G1,4) + (G2,4). The
intermediate versions (G1,2) and (G2,3) are missing -> holes {2,3} and {5,6},
S_g* = {0,1,4,7,8}.

Arms:
  A1 (latest version): a client answers every requested source cut with the
      latest LanceDB version, which can differ from the historical population.
  A2 (LanceDB-native, version-at-time): the client that knows the exact
      version at cut time gets the right population -- but LanceDB provides no
      source-cut-to-version mapping in this experiment. Zero-retention cleanup
      can remove the historical version.
  B (SCR): certifies S_g* from source history + retained population, rejects
      admission at holes.
"""
import argparse, sqlite3, json, datetime, hashlib
from pathlib import Path
import numpy as np, lancedb
import pyarrow as pa

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '--output-dir', type=Path,
    default=HERE / 'runs' / ('native_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S%f')),
    help='Directory for fresh database files and results; defaults to a new runs/ directory.',
)
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)

N, D = 2000, 32
G1N, G2N = 500, 500
SEEDS = [11, 29, 47]

def arrow(ids, vs, x):
    return pa.table({'row': pa.array(ids, type=pa.int32()),
                     'version': pa.array(vs, type=pa.int32()),
                     'vector': pa.array(x.tolist(), type=pa.list_(pa.float32(), D))})

def oracle_topk(versions, vec, x, rep, k=10):
    scored = []
    for a, v in versions.items():
        vec_a = x[a] if v == 1 else rep[a]
        scored.append((float(np.sum((vec.astype(np.float64) - vec_a.astype(np.float64)) ** 2)), a, v))
    return [(a, v) for _, a, v in sorted(scored)[:k]]

def lance_rows(handle):
    return {(int(r['row']), int(r['version'])) for r in handle.to_arrow().to_pylist()}

cells = []
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    x = rng.random((N, D), dtype=np.float32)
    q = rng.random((16, D), dtype=np.float32)
    rep = x[rng.permutation(N)] + rng.normal(0, .02, x.shape).astype(np.float32)

    root = args.output_dir / 'db' / f's{seed}' / datetime.datetime.now().strftime('%H%M%S%f')
    root.mkdir(parents=True)
    db = lancedb.connect(str(root / 'lance'))

    sql = sqlite3.connect(root / 'source.sqlite')
    sql.execute('pragma journal_mode=WAL')
    sql.execute('create table history(row integer,epoch integer,version integer,deleted integer,payload text,primary key(row,epoch))')
    sql.execute('create table meta(epoch integer)')
    sql.execute('insert into meta values(0)')
    sql.executemany('insert into history values(?,0,1,0,?)', [(i, f'{i}:1') for i in range(N)])
    sql.commit()

    table = db.create_table('vec', arrow(np.arange(N), np.ones(N, dtype=int), x))
    table.create_index(index_type='IVF_HNSW_FLAT', metric='l2', num_partitions=4, m=16, ef_construction=64)

    def cur_version():
        return table.version() if callable(table.version) else table.version

    src = {i: 1 for i in range(N)}
    truth = {}
    version_at = {}
    version_at[0] = version_at[1] = cur_version()
    truth[0] = truth[1] = dict(src)

    def commit(epoch, changes):
        if changes:
            sql.executemany('insert into history values(?,?,?,?,?)', changes)
        sql.execute('update meta set epoch=?', (epoch,))
        sql.commit()
        for row, _ep, ver, deleted, _p in changes:
            if deleted:
                src.pop(row, None)
            else:
                src[row] = ver

    perm = [int(i) for i in rng.permutation(N)]
    g1 = perm[:G1N]
    g2 = perm[G1N:G1N + G2N]
    g1s, g2s = set(g1), set(g2)

    # cut 2: G1 replace v2
    commit(2, [(i, 2, 2, 0, f'{i}:2') for i in g1])
    table.delete('row IN (' + ','.join(map(str, g1)) + ')')
    table.add(arrow(np.array(g1), np.full(G1N, 2), rep[g1]))
    truth[2] = dict(src); version_at[2] = cur_version()

    # cut 3: G2 v2
    commit(3, [(i, 3, 2, 0, f'{i}:2') for i in g2])
    table.delete('row IN (' + ','.join(map(str, g2)) + ')')
    table.add(arrow(np.array(g2), np.full(G2N, 2), rep[g2]))
    truth[3] = dict(src); version_at[3] = cur_version()

    # cut 4: G1 tombstone
    commit(4, [(i, 4, 3, 1, 'deleted') for i in g1])
    table.delete('row IN (' + ','.join(map(str, g1)) + ')')
    truth[4] = dict(src); version_at[4] = cur_version()

    # cut 5: G2 v3
    commit(5, [(i, 5, 3, 0, f'{i}:3') for i in g2])
    table.delete('row IN (' + ','.join(map(str, g2)) + ')')
    table.add(arrow(np.array(g2), np.full(G2N, 3), rep[g2]))
    truth[5] = dict(src); version_at[5] = cur_version()

    # cut 6: G1 reinsert v4
    commit(6, [(i, 6, 4, 0, f'{i}:4') for i in g1])
    table.add(arrow(np.array(g1), np.full(G1N, 4), rep[g1]))
    truth[6] = dict(src); version_at[6] = cur_version()

    # cut 7: G2 v4
    commit(7, [(i, 7, 4, 0, f'{i}:4') for i in g2])
    table.delete('row IN (' + ','.join(map(str, g2)) + ')')
    table.add(arrow(np.array(g2), np.full(G2N, 4), rep[g2]))
    truth[7] = dict(src); version_at[7] = cur_version()
    truth[8] = dict(src); version_at[8] = version_at[7]

    def handle_at(cut):
        h = db.open_table('vec'); h.checkout(version_at[cut]); return h

    def query_k(handle, vec, k=10):
        return [(int(a['row']), int(a['version']))
                for a in handle.search(vec.tolist()).nprobes(4).ef(128).limit(40).to_arrow().to_pylist()][:k]

    # SCR arm: retained population at checkpoints {0,1,4,7,8}
    pop = set()
    for c in (0, 4, 7):
        h = handle_at(c)
        pop |= {(r, v) for r, v in lance_rows(h) if (r, v) in {('x',)} or True}
    pop = {(r, v) for r, v in pop}
    # (union over checkpoints of each row's latest version there)
    pop = set()
    for c in (0, 1, 4, 7, 8):
        h = handle_at(c)
        pop |= lance_rows(h)

    hist = sql.execute('select row,epoch,version,deleted from history order by row,epoch').fetchall()
    by_row = {}
    for row, epoch, ver, deleted in hist:
        by_row.setdefault(row, []).append((epoch, ver, deleted))
    lifetimes = []
    for row, recs in by_row.items():
        for idx, (ep, ver, deleted) in enumerate(recs):
            if deleted:
                continue
            nxt = recs[idx + 1][0] if idx + 1 < len(recs) else 9
            lifetimes.append((row, ver, ep, nxt))
    Sg = set(range(9))
    for row, ver, a, b in lifetimes:
        if (row, ver) not in pop:
            for s in range(a, b):
                Sg.discard(s)

    # per-cut evaluation
    per_cut = []
    for cut in range(9):
        T = truth[cut]
        h_cur = handle_at(8)   # latest version
        h_at = handle_at(cut)  # version at cut time
        row = {'cut': cut, 'Sg*': cut in Sg,
               'armB_scr': 'admit' if cut in Sg else 'reject'}
        for name, h in (('armA1_latest_version', h_cur), ('armA2_version_at_cut', h_at)):
            P = lance_rows(h)
            cov = len(P & set(T.items())) / max(1, len(T))
            recs = []
            for vec in q:
                ans = query_k(h, vec)
                ot = oracle_topk(T, vec, x, rep)
                recs.append(len(set(ans) & set(ot)) / 10)
            row[name] = {'coverage': round(cov, 4),
                         'recall@10_mean': round(float(np.mean(recs)), 4),
                         'recall@10_min': round(float(np.min(recs)), 4)}
        per_cut.append(row)

    # Zero-retention cleanup tests whether the old native version can reopen.
    table.optimize(cleanup_older_than=datetime.timedelta(seconds=0), delete_unverified=True)
    reopen_old_ok = True
    try:
        h = db.open_table('vec'); h.checkout(version_at[2]); h.to_arrow()
    except Exception:
        reopen_old_ok = False

    cells.append({'seed': seed, 'N': N, 'per_cut': per_cut,
                  'Sg*': sorted(Sg), 'holes': sorted(set(range(9)) - Sg),
                  'lance_versions_by_cut': {str(k): v for k, v in version_at.items()},
                  'versions_vs_cuts': 'LanceDB versions do not map 1:1 to source cuts',
                  'old_version_reopen_after_cleanup': reopen_old_ok})
    print(f'seed {seed}: S_g*={sorted(Sg)} holes={sorted(set(range(9))-Sg)} '
          f'versions={list(version_at.values())} old_reopen_after_cleanup={reopen_old_ok}', flush=True)
    sql.close()

out = {'scope': 'head-to-head LanceDB native versioning vs SCR coverage certification '
                '(real SQLite WAL + LanceDB 0.40.0; paper 9-cut disconnected schedule)',
       'cells': cells,
       'sha': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
output = args.output_dir / 'h2h_lance_vs_scr.json'
output.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
print('WROTE h2h_lance_vs_scr.json')
