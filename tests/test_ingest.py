from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from app.artifacts import read_json
from app.store import Store
from scripts.build_corpus import build_from_streams
from scripts.ingest import ingest
from tests.conftest import FakeModels, row


def test_resume_after_acknowledged_write(cfg, limiter):
    data = Path(cfg['data_dir'])
    build_from_streams([row(2, ['third', 'fourth'])], [row(1, ['first', 'second'])], data, 4, 1, 0, limiter, {})
    client = QdrantClient(':memory:')
    store = Store(cfg, client)
    original = store.verify_ids
    calls = []

    def interrupted(ids, build_id=None):
        calls.append(ids)
        if len(calls) == 2:
            raise RuntimeError('simulated crash after second acknowledged batch')
        return original(ids, build_id)

    store.verify_ids = interrupted
    with pytest.raises(RuntimeError, match='simulated crash'):
        ingest(cfg, store, FakeModels())
    assert read_json(data / 'ingest_test.json')['next'] == 2
    assert store.count() == 4
    store.verify_ids = original
    result = ingest(cfg, store, FakeModels())
    assert result['passages'] == 4 and result['points'] == 4
    assert read_json(data / 'ingest_test.json')['complete']
    cfg['models']['dense'] = 'changed-model'
    with pytest.raises(ValueError, match='configuration changed'):
        ingest(cfg, store, FakeModels())
    client.close()
