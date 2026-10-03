"""Lazy, reusable ONNX models. Imports never initiate model downloads."""
from functools import cached_property
from pathlib import Path

import numpy as np

from app.config import project_path
from app.artifacts import checkpoint, read_json, sha256
from app.runtime import providers, verify_device


class ModelBundle:
    def __init__(self, cfg):
        self.cfg = cfg

    @property
    def kwargs(self):
        return {'cache_dir': str(project_path('data/models')), 'threads': self.cfg['models']['threads']}

    @cached_property
    def dense(self):
        from fastembed import TextEmbedding
        return verify_device(TextEmbedding(self.cfg['models']['dense'],
                                           providers=providers('dense'), **self.kwargs), 'dense')

    @cached_property
    def sparse(self):
        from fastembed import SparseTextEmbedding
        stats_file = project_path(self.cfg['data_dir']) / 'bm25_stats.json'
        stats = read_json(stats_file) if stats_file.exists() else {}
        manifest_file = project_path(self.cfg['data_dir']) / 'manifest.json'
        if stats and manifest_file.exists() and stats['corpus_sha256'] != read_json(manifest_file)['files']['corpus.parquet']:
            raise ValueError('BM25 calibration belongs to a different corpus; rerun ingestion')
        return SparseTextEmbedding(self.cfg['models']['sparse'], k=self.cfg['models'].get('bm25_k', 1.2),
                                   b=self.cfg['models'].get('bm25_b', .75),
                                   avg_len=stats.get('average_document_length', 256.0), **self.kwargs)

    @cached_property
    def passage_tokenizer(self):
        from tokenizers import Tokenizer
        # Clone instead of disabling truncation on the shared inference tokenizer.
        tok = Tokenizer.from_str(self.dense.model.tokenizer.to_str())
        tok.no_padding()
        tok.no_truncation()
        return tok

    def validate_passage_length(self, text):
        if len(self.passage_tokenizer.encode(text, add_special_tokens=False).ids) > 510:
            raise ValueError('Live passage exceeds 510 BGE tokens; split it into smaller passages')

    def calibrate(self, corpus: Path):
        """Recalculate BM25 avgdl with exactly the pinned encoder's stemmed tokens."""
        import pyarrow.parquet as pq
        from fastembed.sparse.bm25 import remove_non_alphanumeric
        model = self.sparse.model
        count, total = 0, 0
        for batch in pq.ParquetFile(corpus).iter_batches(batch_size=4096, columns=['text']):
            for text in batch.column(0).to_pylist():
                total += len(model._stem(model.tokenizer.tokenize(remove_non_alphanumeric(text))))
                count += 1
        avg = total / count if count else 0
        if avg <= 0:
            raise ValueError('Cannot calibrate BM25 on an empty-token corpus')
        model.avg_len = avg
        return checkpoint(corpus.parent, 'bm25_stats', {'corpus_sha256': sha256(corpus), 'documents': count,
                          'stemmed_tokens': total, 'average_document_length': avg,
                          'policy': 'avgdl frozen at bulk build; IDF remains dynamic in Qdrant'})

    def provenance(self, include_reranker=False):
        selected = {'dense': self.dense, 'sparse': self.sparse}
        if include_reranker:
            selected['reranker'] = self.reranker
        out = {}
        for name, instance in selected.items():
            root = Path(instance.model._model_dir)
            out[name] = {'snapshot_directory': root.name,
                         'files': {p.relative_to(root).as_posix(): sha256(p) for p in root.rglob('*') if p.is_file()}}
        return out

    @cached_property
    def reranker(self):
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        return verify_device(TextCrossEncoder(self.cfg['models']['reranker'],
                                              providers=providers('rerank'), **self.kwargs), 'rerank')

    def documents(self, texts):
        # Long random batches can allocate several GB for padded attention tensors.
        # Bound inference independently of upload size and group similar lengths.
        batch = min(self.cfg['models']['batch_size'], 32)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        ordered = [texts[i] for i in order]
        encoded = list(self.dense.passage_embed(ordered, batch_size=batch))
        dense = np.empty((len(texts), self.cfg['models']['dimension']), dtype=np.float32)
        for index, vector in zip(order, encoded):
            dense[index] = vector
        if len(encoded) != len(texts):
            raise ValueError('Dense model did not return one vector per passage')
        sparse = list(self.sparse.passage_embed(texts, batch_size=batch))
        validate_vectors(dense, sparse, len(texts), self.cfg['models']['dimension'])
        return dense, sparse

    def query(self, text, include_sparse=True):
        d = next(self.dense.query_embed([text]))
        s = next(self.sparse.query_embed([text])) if include_sparse else None
        if len(d) != self.cfg['models']['dimension'] or not np.isfinite(d).all() or np.linalg.norm(d) == 0:
            raise ValueError('Invalid query embedding')
        return d.tolist(), s

    def rerank(self, query, texts):
        scores = list(self.reranker.rerank(query, texts, batch_size=min(self.cfg['models']['batch_size'], 16)))
        if len(scores) != len(texts) or not np.isfinite(scores).all():
            raise ValueError('Invalid reranker output')
        return [float(s) for s in scores]

    def warmup(self, mode='hybrid_rerank'):
        self.query('warmup', include_sparse=mode != 'dense')
        if mode == 'hybrid_rerank':
            self.rerank('warmup', ['model warmup'])


def validate_vectors(dense, sparse, count, dimension):
    a = np.asarray(dense)
    if a.shape != (count, dimension) or not np.isfinite(a).all():
        raise ValueError('Dense vector count, dimension, or finiteness check failed')
    norms = np.linalg.norm(a, axis=1)
    if not np.allclose(norms, 1, atol=0.01):
        raise ValueError('Expected unit-normalized dense passage vectors')
    if len(sparse) != count:
        raise ValueError('Sparse vector count mismatch')
    for s in sparse:
        indices, values = list(s.indices), list(s.values)
        if len(indices) != len(values) or len(set(indices)) != len(indices):
            raise ValueError('Invalid sparse vector indices')
        if any(i < 0 for i in indices) or not np.isfinite(values).all() or any(v < 0 for v in values):
            raise ValueError('Invalid BM25 vector values')
    return {'vectors': count, 'dimension': dimension, 'norm_min': float(norms.min()),
            'norm_max': float(norms.max()), 'sparse_nonzeros': sum(len(s.indices) for s in sparse)}
