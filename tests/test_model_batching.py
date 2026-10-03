from types import SimpleNamespace

import numpy as np

from app.models import ModelBundle


def test_length_bucketing_preserves_passage_vector_alignment(cfg):
    bundle = ModelBundle(cfg)
    texts = ['long passage with several words', 'short', 'medium text']
    calls = []

    class Dense:
        def passage_embed(self, documents, batch_size):
            calls.append((documents, batch_size))
            for text in documents:
                vector = np.zeros(8, dtype=np.float32)
                vector[texts.index(text)] = 1
                yield vector

    class Sparse:
        def passage_embed(self, documents, batch_size):
            for text in documents:
                yield SimpleNamespace(indices=[texts.index(text)], values=[1.0])

    bundle.__dict__['dense'] = Dense()
    bundle.__dict__['sparse'] = Sparse()
    dense, sparse = bundle.documents(texts)
    assert calls == [(['short', 'medium text', 'long passage with several words'], 32)]
    assert list(dense.argmax(axis=1)) == [0, 1, 2]
    assert [s.indices[0] for s in sparse] == [0, 1, 2]
