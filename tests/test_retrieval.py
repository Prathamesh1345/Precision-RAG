import pytest

from app.api import create_app
from app.retriever import weighted_fusion
from fastapi.testclient import TestClient
from types import SimpleNamespace


@pytest.mark.parametrize('mode,fusion', [('dense', 'rrf'), ('hybrid', 'rrf'), ('hybrid', 'dbsf'),
                                        ('hybrid', 'weighted'), ('hybrid_rerank', 'rrf')])
def test_filters_and_updates(service, mode, fusion):
    service.upsert(1, 'dog normal temperature', 'vet.org', 'numeric')
    service.upsert(2, 'dog temperature fever', 'vet.org', 'description')
    service.upsert(3, 'cat normal temperature', 'cats.org', 'numeric')
    result = service.search('dog temperature', mode=mode, fusion=fusion, source='vet.org', category='numeric')
    assert [h['pid'] for h in result['hits']] == [1]
    assert result['timings_ms']['total'] > 0
    assert service.search('dog', mode=mode, fusion=fusion, category='missing')['hits'] == []
    service.delete(1)
    assert service.search('dog temperature', mode=mode, fusion=fusion, source='vet.org', category='numeric')['hits'] == []


def test_cache_is_copied_bypassed_and_invalidated(service):
    service.upsert(1, 'alpha evidence')
    first = service.search('alpha', mode='dense')
    first['hits'][0]['text'] = 'client mutation'
    calls = service.models.query_calls
    cached = service.search('alpha', mode='dense')
    assert cached['hits'][0]['text'] == 'alpha evidence'
    assert 'cache_hit' in cached['timings_ms'] and cached['timings_ms']['total'] > 0
    service.search('alpha', mode='dense', use_cache=False)
    assert service.models.query_calls == calls + 1
    service.upsert(1, 'replacement evidence')
    assert service.search('alpha', mode='dense')['hits'][0]['text'] == 'replacement evidence'
    assert service.delete(123)['status'] == 'not_found'


def test_weighted_missing_branch_and_ties():
    point = lambda pid, score: SimpleNamespace(id=pid, score=score, payload={'text': str(pid)})
    dense, sparse = [point(1, 5), point(2, 1)], [point(2, 10), point(3, 0)]
    assert weighted_fusion(dense, sparse, .6)[0]['pid'] == 1
    assert weighted_fusion(dense, sparse, 0)[0]['pid'] == 2
    assert weighted_fusion([point(3, 1), point(1, 1)], [], 1)[0]['pid'] == 1
    assert weighted_fusion([], [], .5) == []


def test_api_contract_and_validation(service):
    with TestClient(create_app(service)) as client:
        assert client.get('/health').status_code == 200
        assert client.post('/search', json={'query': '   '}).status_code == 422
        assert client.post('/search', json={'query': 'dog', 'mode': 'bad'}).status_code == 422
        assert client.post('/search', json={'query': 'dog', 'top_k': 0}).status_code == 422
        saved = client.post('/passages', json={'pid': 42, 'text': 'dog evidence'})
        assert saved.status_code == 200 and saved.json()['recalculation']['points'] == 1
        result = client.post('/search', json={'query': 'dog', 'mode': 'hybrid_rerank'}).json()
        assert result['hits'][0]['pid'] == 42
        assert client.delete('/passages/42').json()['status'] == 'deleted'
        assert client.delete('/passages/42').json()['status'] == 'not_found'


def test_presentation_keeps_frozen_collection_read_only(service, monkeypatch):
    service.upsert(42, 'existing evidence')
    monkeypatch.setenv('PRAG_READ_ONLY', '1')
    with TestClient(create_app(service)) as client:
        assert client.get('/stats').json()['read_only'] is True
        assert client.post('/passages', json={'pid': 42, 'text': 'changed'}).status_code == 409
        assert client.delete('/passages/42').status_code == 409
        result = client.post('/search', json={'query': 'evidence', 'mode': 'dense'}).json()
        assert result['hits'][0]['text'] == 'existing evidence'
