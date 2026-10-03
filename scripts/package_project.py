"""Create and verify a portable project archive without private credentials."""
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
EXPORTS = ROOT / 'exports'
TOP = 'PrecisionRAG'
EXCLUDED_PARTS = {'__pycache__', '.pytest_cache', '.git', 'node_modules', '.venv', '.venv-gpu'}


def selected_files():
    roots = ['app', 'eval', 'scripts', 'tests', 'docs', 'reports', 'Work Done till now', '.streamlit']
    files = [p for p in ROOT.iterdir() if p.is_file() and
             (p.suffix in {'.md', '.pdf', '.yaml', '.yml', '.txt'} or p.name in {'.env.example', '.gitignore'})]
    for name in roots:
        files.extend((ROOT / name).rglob('*'))
    data = ROOT / 'data'
    files.extend(data / name for name in ['corpus.parquet', 'manifest.json', 'qrels.json',
                                        'eval_queries.jsonl', 'tuning_queries.jsonl', 'bm25_stats.json'])
    for name in ['checks', 'models', 'hf_cache']:
        files.extend((data / name).rglob('*'))
    return sorted({p for p in files if p.is_file() and not any(x in EXCLUDED_PARTS for x in p.parts)
                   and p.suffix not in {'.pyc', '.pyo', '.tmp', '.lock', '.log'}
                   and (not p.name.startswith('.env') or p.name == '.env.example')
                   and p.name not in {'secrets.toml', '.DS_Store'}}, key=lambda p: p.relative_to(ROOT).as_posix())


def private_values():
    # Values are checked in memory only and never printed or written to the archive.
    values = []
    path = ROOT / '.env'
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            name, separator, value = line.partition('=')
            value = value.strip().strip('\"\'')
            if separator and any(word in name.upper() for word in ['KEY', 'TOKEN', 'SECRET', 'PASSWORD']) and len(value) >= 12:
                values.append(value.encode())
    return values


def main():
    EXPORTS.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = EXPORTS / f'PrecisionRAG-full-project-{stamp}.zip'
    secrets = private_values()
    entries = []
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
        for path in selected_files():
            rel = path.relative_to(ROOT).as_posix()
            content = path.read_bytes()
            if any(value in content for value in secrets) or re.search(rb'gsk_[A-Za-z0-9]{30,}', content):
                raise ValueError(f'Credential detected; refusing to package {rel}')
            archive.writestr(f'{TOP}/{rel}', content)
            entries.append({'path': rel, 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()})
        # A checkpoint from another machine must not block rebuilding an empty database.
        # Preserve it only as historical evidence, outside the active data/checkpoint path.
        checkpoint = ROOT / 'data' / 'ingest_msmarco_v21_100k.json'
        if checkpoint.exists():
            content = checkpoint.read_bytes()
            rel = 'reports/packaged_source_ingest_checkpoint.json'
            archive.writestr(f'{TOP}/{rel}', content)
            entries.append({'path': rel, 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()})
        manifest = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'files': entries,
                    'excluded': ['private .env and secrets', 'Python environments', 'live qdrant_storage',
                                 'temporary files/installers/logs', 'active ingestion checkpoint'],
                    'evaluation_note': 'Point-in-time snapshot. Missing quality reports mean evaluation is incomplete.'}
        archive.writestr(f'{TOP}/PACKAGE_MANIFEST.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f'ZIP integrity failed: {bad}')
        for entry in entries:
            content = archive.read(f'{TOP}/{entry["path"]}')
            if hashlib.sha256(content).hexdigest() != entry['sha256']:
                raise RuntimeError(f'Archive checksum mismatch: {entry["path"]}')
        names = set(archive.namelist())
        assert f'{TOP}/.env' not in names
        assert f'{TOP}/data/ingest_msmarco_v21_100k.json' not in names
        data_manifest = json.loads(archive.read(f'{TOP}/data/manifest.json'))
        for name, expected in data_manifest['files'].items():
            assert hashlib.sha256(archive.read(f'{TOP}/data/{name}')).hexdigest() == expected, name
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix('.zip.sha256').write_text(f'{digest}  {output.name}\n', encoding='ascii')
    print(json.dumps({'zip': str(output), 'files': len(entries) + 1,
                      'bytes': output.stat().st_size, 'sha256': digest,
                      'verified': 'ZIP CRC, per-file SHA256, frozen dataset hashes, credential exclusion'}, indent=2))


if __name__ == '__main__':
    main()
