from pathlib import Path
import hashlib,json
import numpy as np
p=Path(__file__).resolve().parent
records=[]
for seed in (11,29,47):
    rng=np.random.default_rng(seed)
    x=rng.random((2000,32),dtype=np.float32)
    q=rng.random((200,32),dtype=np.float32)
    perm=rng.permutation(2000).astype('<i4')
    rewritten=np.sort(rng.choice(2000,100,replace=False)).astype('<i4')
    path=p/f'input_{seed}.bin'
    with path.open('wb') as f:
        np.array([2000,32,200,100],dtype='<i4').tofile(f)
        x.tofile(f);q.tofile(f);perm.tofile(f);rewritten.tofile(f)
    np.savez_compressed(p/f'input_{seed}.npz',vectors=x,queries=q,permutation=perm,rewritten=rewritten)
    records.append({'seed':seed,'binary_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'npz_sha256':hashlib.sha256((p/f'input_{seed}.npz').read_bytes()).hexdigest()})
(p/'input_hashes.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
