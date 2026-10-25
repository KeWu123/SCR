"""Check historical identities together with the declared released file bytes."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent


def matches(recorded, path):
    path = Path(path).resolve()
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if recorded == actual:
        return True
    if not path.is_relative_to(ROOT):
        return False
    relative = path.relative_to(ROOT).as_posix()
    entries = json.loads((ROOT / 'PROVENANCE.json').read_text(encoding='utf-8'))['files']
    return any(entry['path'] == relative
               and entry['historical_sha256'] == recorded
               and entry['release_sha256'] == actual for entry in entries)
