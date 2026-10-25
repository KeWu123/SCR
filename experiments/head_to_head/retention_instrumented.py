"""Measure array storage for a seven-generation retention workload.

Real allocation + nbytes measurement of the paper's Table 5 retention scopes:
N=5,000 rows, d=64, M=12 base degree, six publications each allocating new
adjacency frames for a sampled 40% of nodes, G=7 generations.

Accounting scopes:
  current    -- one generation: embeddings + base adjacency.
  SCR (CoW)  -- copy-on-write frames: unchanged frames shared, each publication
                allocates frames only for its 40% region; embeddings shared.
  shared vec -- byte total for G complete adjacency copies, embeddings shared
                once; computed from the allocated base array's .nbytes.
  full copy  -- byte total for G copies of adjacency and embeddings, computed
                from the allocated arrays' .nbytes.

The script also records NumPy object-array storage and Python per-node list
sizes. These fields have different accounting scopes and do not measure
process memory or native LanceDB storage.
"""
import argparse, json, datetime, hashlib, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '--output-dir', type=Path,
    default=HERE / 'runs' / ('retention_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S%f')),
    help='Directory for fresh results; defaults to a new runs/ directory.',
)
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)

N, D, M, G, P = 5000, 64, 12, 7, 6
SEEDS = [11, 22, 33]

def frames_for_nodes(nodes, rng, style):
    """Adjacency frames for a set of nodes. style='flat' -> one int32 array
    per node list (object array), style='dense' -> flat int32 array N*M."""
    if style == 'dense':
        arr = rng.integers(0, N, size=len(nodes) * M, dtype=np.int32)
        return arr
    else:
        return np.array([rng.integers(0, N, size=M, dtype=np.int32) for _ in nodes],
                        dtype=object)

cells = []
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    embeddings = rng.random((N, D), dtype=np.float32)
    emb_bytes = embeddings.nbytes

    # Preserve the original random draw order; publication regions below are
    # sampled independently and can overlap.
    order = rng.permutation(N)
    regions = [order[i * (N * 40 // 100):(i + 1) * (N * 40 // 100)] for i in range(P)]

    # ---- dense (flat) measurement ----
    base = rng.integers(0, N, size=N * M, dtype=np.int32)
    # current: one generation
    cur_adj_bytes = base.nbytes
    # SCR CoW: base + per-publication new frames (a 40% region of 2000 nodes each)
    gen_frames = [base]
    for i in range(P):
        region = rng.choice(N, size=N * 40 // 100, replace=False)
        newfr = rng.integers(0, N, size=len(region) * M, dtype=np.int32)
        gen_frames.append(newfr)
    scr_adj_bytes = base.nbytes + sum(f.nbytes for f in gen_frames[1:])
    # shared vec: G complete adjacency copies
    shared_adj_bytes = G * base.nbytes
    # full copy: G x (adjacency + embeddings)
    full_bytes = G * (base.nbytes + emb_bytes)

    # NumPy object-array and per-node list accounting.
    obj_base = np.array([rng.integers(0, N, size=M, dtype=np.int32) for _ in range(N)], dtype=object)
    obj_adj_bytes = sum(a.nbytes for a in obj_base)  # object-pointer row arrays
    obj_overhead = obj_base.nbytes  # the same object-pointer storage in total
    # per-node list version
    lst = [rng.integers(0, N, size=M, dtype=np.int32) for _ in range(N)]
    lst_bytes = sum(a.nbytes for a in lst)
    # ndarray getsizeof includes its owned data; this is not net metadata.
    lst_overhead = sys.getsizeof(lst) + sum(sys.getsizeof(a) for a in lst)

    cells.append({
        'seed': seed, 'N': N, 'D': D, 'M': M, 'G': G, 'publications': P,
        'embeddings_bytes': emb_bytes,
        'current_adj_bytes': cur_adj_bytes,
        'scr_cow_adj_bytes': scr_adj_bytes,
        'scr_cow_ratio': round(scr_adj_bytes / cur_adj_bytes, 4),
        'shared_vec_adj_bytes': shared_adj_bytes,
        'full_copy_bytes': full_bytes,
        'vectors_plus_base_scr': emb_bytes + scr_adj_bytes,
        'vectors_plus_base_current': emb_bytes + cur_adj_bytes,
        'overall_scr_ratio': round((emb_bytes + scr_adj_bytes) / (emb_bytes + cur_adj_bytes), 4),
        'object_array_adj_bytes': obj_adj_bytes,
        'object_array_container_overhead_bytes': obj_overhead,
        'per_node_list_bytes': lst_bytes,
        'per_node_list_overhead_bytes': lst_overhead,
        'per_node_list_overhead_ratio': round(lst_overhead / lst_bytes, 4),
    })
    print(f'seed {seed}: current_adj={cur_adj_bytes} scr_cow={scr_adj_bytes} '
          f'ratio={scr_adj_bytes/cur_adj_bytes:.4f} overall={ (emb_bytes+scr_adj_bytes)/(emb_bytes+cur_adj_bytes):.4f} '
          f'list_overhead_ratio={lst_overhead/lst_bytes:.4f}', flush=True)

out = {'scope': 'instrumented retention cost (real numpy allocation + nbytes; N=5000 d=64 M=12, 6x40% publications)',
       'cells': cells,
       'sha': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
output = args.output_dir / 'retention_instrumented.json'
output.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
print('WROTE retention_instrumented.json')
