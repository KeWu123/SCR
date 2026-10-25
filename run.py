"""Run experiments and independent checks from the repository root."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
SUITES = {
    'core': [
        'experiments/verify_protocol.py',
        'experiments/verify_async.py',
        'experiments/completion/verify_support_spectrum.py',
        'experiments/completion/native_interval/verify_recorded_support.py',
    ],
    'native': [
        'reference/verify_strict_broker.py',
        'reference/verify_retention_guards.py',
        'experiments/completion/native_interval/domain_key_hook_regressions.py',
    ],
    'evidence': [
        'experiments/verify_cross_store.py',
        'experiments/native_trace/verify_native.py',
    ],
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def inventory(root=ROOT):
    files = []
    for line in (root / 'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        expected, name = line.split('  ', 1)
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or digest(path) != expected:
            raise RuntimeError('Inventory mismatch: ' + name)
        files.append(name)
    print(f'PASS: {len(files)} release file hashes', flush=True)
    return files


def execute(root, name, arguments=()):
    root = root.resolve()
    script = (root / name).resolve()
    if not script.is_relative_to(root) or not script.is_file() or script.suffix != '.py':
        raise ValueError('Expected a Python script inside the repository: ' + name)
    env = os.environ.copy()
    env['PYTHONUTF8'] = '1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONPATH'] = os.pathsep.join(
        [str(root / 'paper/proto'), str(root / 'experiments'), env.get('PYTHONPATH', '')])
    subprocess.run([sys.executable, str(script), *arguments], cwd=root, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('inventory', help='Verify the released file hashes')
    verify = sub.add_parser('verify', help='Run independent checks in a temporary copy')
    verify.add_argument('--suite', choices=[*SUITES, 'all'], default='core')
    run = sub.add_parser('run', help='Run one script; its outputs are written in this checkout')
    run.add_argument('script')
    run.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == 'inventory':
        inventory()
    elif args.command == 'run':
        execute(ROOT, args.script, args.arguments)
    else:
        files = inventory()
        suites = list(SUITES) if args.suite == 'all' else [args.suite]
        with tempfile.TemporaryDirectory(prefix='scr-check-') as temporary:
            scratch = (Path(temporary) / 'SCR').resolve()
            for name in [*files, 'SHA256SUMS.txt']:
                target = scratch / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, target)
            for suite in suites:
                for script in SUITES[suite]:
                    print('CHECK:', script, flush=True)
                    execute(scratch, script)
        print('PASS: requested verification suite', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print('FAIL:', error, file=sys.stderr)
        sys.exit(1)
