"""Check the complete distribution against its SHA256 manifest (stdlib only)."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

ROOT = Path(__file__).resolve().parents[1]
IGNORED_DIRS = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', 'reproduced_figures'}
IGNORED_FILES = {'.DS_Store', 'Thumbs.db'}


def check(root=ROOT):
    root = root.resolve()
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    failures = []
    entries = manifest['files']
    if manifest.get('format_version') != 1 or manifest.get('algorithm') != 'sha256' or not entries:
        raise ValueError('Unsupported or empty distribution manifest')
    for name, entry in entries.items():
        rel = PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts or '\\' in name or ':' in name:
            failures.append(f'Unsafe manifest path: {name}')
            continue
        path = root.joinpath(*rel.parts)
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            failures.append(f'Unsafe file: {name}')
        elif not path.is_file():
            failures.append(f'Missing file: {name}')
        else:
            data = path.read_bytes()
            if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
                failures.append(f'Modified file: {name}')
    for path in root.rglob('*'):
        rel = path.relative_to(root)
        if any(part in IGNORED_DIRS for part in rel.parts) or path.name in IGNORED_FILES:
            continue
        if path.is_file() and rel.as_posix() != 'manifest.json' and rel.as_posix() not in entries:
            failures.append(f'Unexpected file: {rel.as_posix()}')
    return entries, failures


if __name__ == '__main__':
    entries, failures = check()
    if failures:
        print('\n'.join(failures), file=sys.stderr)
        raise SystemExit(1)
    print(f'PASS: {len(entries)} files match the distribution manifest.')
