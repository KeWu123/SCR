"""Resolve retained evidence paths without altering measurement records."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[3]


def evidence_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _relocate(value):
    if isinstance(value, dict):
        return {key: _relocate(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_relocate(item) for item in value]
    if isinstance(value, str) and value.startswith('experiments/'):
        path = evidence_path(value).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Evidence path escapes the repository')
        return str(path)
    return value


def load_record(path):
    return _relocate(json.loads(evidence_path(path).read_text(encoding='utf-8')))


def require_database(path):
    if not evidence_path(path).is_file():
        raise FileNotFoundError(
            'Physical database snapshots are excluded from this source repository. '
            'Run run_native_intervals.py (or spectrum_scaling.py for the scaling '
            'checker) first, then rerun this physical checker. The recorded JSON '
            'and input checks are available in verify_recorded_support.py.'
        )
