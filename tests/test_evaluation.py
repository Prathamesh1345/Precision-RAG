import sys
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from app.artifacts import read_json, read_jsonl, write_json
from app.retriever import Retriever
from app.store import Store
from eval import run_all
from scripts.build_corpus import build_from_streams
from scripts.ingest import ingest
from tests.conftest import FakeModels, row


def test_full_development_report_is_explicitly_incomplete(cfg, limiter, monkeypatch):
    data, reports = Path(cfg['data_dir']), Path(cfg['reports_dir'])
    build_from_streams([], [row(i, ['shared evidence']) for i in range(100)], data, 1, 100, 0,
                       limiter, {'revision': 'test-fixture'})
    client = QdrantClient(':memory:')
    store = Store(cfg, client)
    bundle = FakeModels()
    ingest(cfg, store, bundle)
    service = Retriever(cfg, store, bundle)
    monkeypatch.setattr(run_all, 'load_config', lambda: cfg)
    monkeypatch.setattr(run_all, 'get_retriever', lambda: service)
    monkeypatch.setattr(sys, 'argv', ['run_all', '--skip-ragas'])
    run_all.main()
    latest = read_json(reports / 'latest_run.json')
    assert latest['ragas_complete'] is False
    directory = Path(latest['path'])
    summary = read_json(directory / 'summary.json')
    assert len(summary) == 6
    assert all(r['context_precision'] is None for r in summary)
    assert all(r['latency_queries'] == 100 for r in summary)
    assert 'NOT MEASURED' in (directory / 'benchmark_report.md').read_text()
    assert (directory / 'ablation.png').exists()
    original = read_json(directory / 'latency_dense.json')
    original['qids'] = ['same'] * 100
    write_json(directory / 'latency_dense.json', original)
    with pytest.raises(ValueError, match='distinct query'):
        run_all.recalculate(directory, ['dense'])
    client.close()
