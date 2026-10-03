"""Atomic artifacts and explicit recalculation checkpoints."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(temp, path)


def checkpoint(directory: Path, stage: str, calculated: dict) -> dict:
    result = {'stage': stage, 'recalculated_at_utc': datetime.now(timezone.utc).isoformat(), **calculated}
    write_json(directory / f'{stage}.json', result)
    return result


def environment() -> dict:
    import psutil
    packages = {}
    for name in ['qdrant-client', 'fastembed', 'fastembed-gpu', 'datasets', 'onnxruntime',
                 'onnxruntime-gpu', 'numpy', 'ragas', 'langchain-groq']:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'processor': platform.processor(), 'cpu_logical': os.cpu_count(),
            'ram_bytes': psutil.virtual_memory().total, 'packages': packages,
            'devices': {name: os.getenv(f'PRAG_{name.upper()}_DEVICE', 'cpu') for name in ['dense', 'rerank']}}
