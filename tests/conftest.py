from copy import deepcopy
import hashlib
import re
from types import SimpleNamespace

import numpy as np
import pytest
from qdrant_client import QdrantClient
from tokenizers import Tokenizer, models, pre_tokenizers

from app.config import load_config
from app.retriever import Retriever
from app.store import Store
from scripts.build_corpus import TokenLimiter


class FakeModels:
    """Only unit-test fixtures use these vectors; never used by benchmark commands."""
    def __init__(self):
        self.query_calls = 0

    def encode(self, text):
        words = re.findall(r'\w+', text.lower())
        sparse = {}
        dense = np.zeros(8, dtype=np.float32)
        for word in words:
            key = int(hashlib.sha256(word.encode()).hexdigest()[:6], 16)
            sparse[key] = sparse.get(key, 0) + 1
            dense[key % 8] += 1
        dense /= np.linalg.norm(dense)
        return dense, SimpleNamespace(indices=np.array(sorted(sparse)),
                                      values=np.array([sparse[k] for k in sorted(sparse)], dtype=float))

    def documents(self, texts):
        vectors = [self.encode(t) for t in texts]
        return np.stack([v[0] for v in vectors]), [v[1] for v in vectors]

    def query(self, text, include_sparse=True):
        self.query_calls += 1
        d, s = self.encode(text)
        return d.tolist(), s if include_sparse else None

    def rerank(self, query, texts):
        return [float(len(set(query.split()) & set(t.split()))) for t in texts]

    def warmup(self, mode='hybrid_rerank'):
        pass


@pytest.fixture
def cfg(tmp_path):
    config = deepcopy(load_config())
    config['models']['dimension'] = 8
    config['data_dir'] = str(tmp_path / 'data')
    config['reports_dir'] = str(tmp_path / 'reports')
    config['index']['batch_size'] = 2
    config['collection'] = 'test'
    return config


@pytest.fixture
def service(cfg):
    client = QdrantClient(':memory:')
    store = Store(cfg, client)
    store.ensure_collection()
    result = Retriever(cfg, store, FakeModels())
    yield result
    client.close()


@pytest.fixture
def limiter():
    tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return TokenLimiter(tokenizer, 10)


def row(qid, texts, selected=None, category='description'):
    return {'query_id': qid, 'query': f'question {qid}', 'query_type': category, 'answers': ['reference answer'],
            'passages': {'passage_text': texts, 'url': ['https://www.example.org/p'] * len(texts),
                         'is_selected': selected or [1] + [0] * (len(texts) - 1)}}
