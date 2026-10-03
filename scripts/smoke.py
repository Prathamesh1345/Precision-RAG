"""Exercise REAL models and Qdrant's embedded API on a small frozen-corpus slice.

This is a functional check, never evidence of HNSW scale or the 300 ms SLA.
"""
from copy import deepcopy

import pyarrow.parquet as pq
from qdrant_client import QdrantClient

from app.artifacts import checkpoint, read_jsonl
from app.config import load_config, project_path
from app.models import ModelBundle, validate_vectors
from app.retriever import Retriever
from app.store import Store, make_point


def main():
    cfg = deepcopy(load_config())
    cfg['collection'] = 'functional_smoke_only'
    client = QdrantClient(':memory:')
    store = Store(cfg, client)
    store.ensure_collection()
    models = ModelBundle(cfg)
    data = project_path(cfg['data_dir'])
    models.calibrate(data / 'corpus.parquet')
    query = read_jsonl(data / 'eval_queries.jsonl')[0]
    rows = []
    wanted = set(query['relevant_pids'])
    for batch in pq.ParquetFile(data / 'corpus.parquet').iter_batches(batch_size=512):
        for p in batch.to_pylist():
            if len(rows) < 64 or p['pid'] in wanted:
                rows.append(p)
    dense, sparse = models.documents([p['text'] for p in rows])
    checked = validate_vectors(dense, sparse, len(rows), cfg['models']['dimension'])
    store.upsert([make_point(p, d, s, 'functional-smoke') for p, d, s in zip(rows, dense, sparse)])
    service = Retriever(cfg, store, models)
    results = {}
    models.warmup()
    for mode in ['dense', 'hybrid', 'hybrid_rerank']:
        result = service.search(query['query'], mode=mode, use_cache=False)
        assert result['hits'] and all(h['text'] for h in result['hits'])
        results[mode] = {'pids': [h['pid'] for h in result['hits']], 'timings_ms': result['timings_ms']}
    for fusion in ['weighted', 'dbsf']:
        assert service.search(query['query'], mode='hybrid', fusion=fusion)['hits']
    service.upsert(900000001, 'PrecisionRAG combines dense embeddings with BM25 keyword retrieval.', 'custom', 'custom')
    assert service.search('PrecisionRAG BM25', category='custom')['hits'][0]['pid'] == 900000001
    service.delete(900000001)
    assert service.search('PrecisionRAG BM25', category='custom')['hits'] == []
    checkpoint(project_path(cfg['reports_dir']) / 'checks', 'real_model_smoke', {
        'status': 'passed', 'purpose': 'functional verification only, embedded Qdrant has no HNSW',
        'vectors': checked, 'models': cfg['models'], 'model_provenance': models.provenance(include_reranker=True),
        'query': query['query'], 'retrieval': results})
    client.close()
    print('Real BGE, BM25, cross-encoder, fusion, filters, upsert and delete smoke checks passed.')


if __name__ == '__main__':
    main()
