from __future__ import annotations

import time
from urllib.parse import urlsplit, urlunsplit

from qdrant_client import QdrantClient, models


def transport_url(url):
    """Avoid repeated IPv6 connection timeouts for our IPv4-only Docker binding."""
    parts = urlsplit(url)
    if parts.hostname == 'localhost' and parts.username is None:
        host = '127.0.0.1' + (f':{parts.port}' if parts.port else '')
        return urlunsplit(parts._replace(netloc=host))
    return url


def build_filter(category=None, source=None, build_id=None):
    conditions = []
    for key, value in [('categories', category), ('sources', source), ('_build_id', build_id)]:
        if value:
            conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
    return models.Filter(must=conditions) if conditions else None


def sparse_vector(value):
    return models.SparseVector(indices=[int(i) for i in value.indices], values=[float(v) for v in value.values])


def make_point(passage, dense, sparse, build_id):
    return models.PointStruct(id=int(passage['pid']),
                              vector={'dense': list(map(float, dense)), 'bm25': sparse_vector(sparse)},
                              payload={**passage, '_build_id': build_id})


class Store:
    def __init__(self, cfg, client=None):
        self.cfg, self.name = cfg, cfg['collection']
        self.client = client or QdrantClient(url=transport_url(cfg['qdrant_url']), timeout=120)

    def ensure_collection(self, bulk=False):
        if self.client.collection_exists(self.name):
            info = self.client.get_collection(self.name)
            vectors = info.config.params.vectors
            if not isinstance(vectors, dict) or 'dense' not in vectors:
                raise ValueError('Collection lacks named dense vectors; choose another collection')
            dense = vectors['dense']
            sparse = info.config.params.sparse_vectors or {}
            if (dense.size != self.cfg['models']['dimension'] or dense.distance != models.Distance.COSINE
                    or 'bm25' not in sparse or sparse['bm25'].modifier != models.Modifier.IDF):
                raise ValueError('Incompatible collection schema; choose another collection')
            return
        idx = self.cfg['index']
        quant = models.ScalarQuantization(scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8, quantile=0.99, always_ram=True)) if idx['quantization'] == 'int8' else None
        self.client.create_collection(
            self.name,
            vectors_config={'dense': models.VectorParams(size=self.cfg['models']['dimension'],
                                                        distance=models.Distance.COSINE, on_disk=idx['on_disk_vectors'])},
            sparse_vectors_config={'bm25': models.SparseVectorParams(modifier=models.Modifier.IDF)},
            hnsw_config=models.HnswConfigDiff(m=idx['hnsw_m'], ef_construct=idx['hnsw_ef_construct']),
            quantization_config=quant,
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0 if bulk else idx['indexing_threshold']))
        for field in ['categories', 'sources', '_build_id']:
            self.client.create_payload_index(self.name, field, models.PayloadSchemaType.KEYWORD, wait=True)

    def count(self, query_filter=None):
        return self.client.count(self.name, count_filter=query_filter, exact=True).count

    def stats(self):
        info = self.client.get_collection(self.name)
        return {'points': self.count(), 'status': info.status.value,
                'indexed_vectors': info.indexed_vectors_count,
                'optimizer_status': str(info.optimizer_status), 'collection': self.name}

    def finish_bulk(self):
        self.client.update_collection(self.name, optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=self.cfg['index']['indexing_threshold']))
        deadline = time.monotonic() + self.cfg['index']['wait_timeout_seconds']
        needs_hnsw = self.count() >= 100000
        while time.monotonic() < deadline:
            info = self.client.get_collection(self.name)
            if (info.status == models.CollectionStatus.GREEN and info.optimizer_status == 'ok'
                    and (not needs_hnsw or (info.indexed_vectors_count or 0) > 0)):
                return self.stats()
            if info.status == models.CollectionStatus.RED:
                raise RuntimeError(f'Qdrant optimizer failed: {info.optimizer_status}')
            time.sleep(2)
        raise TimeoutError('Qdrant optimization did not finish before the configured timeout')

    def verify_ids(self, ids, build_id=None):
        records = self.client.retrieve(self.name, ids=ids, with_payload=True, with_vectors=False)
        if {r.id for r in records} != set(ids):
            raise RuntimeError('Qdrant read-back found missing IDs; checkpoint was not advanced')
        if build_id and any(r.payload.get('_build_id') != build_id for r in records):
            raise RuntimeError('Qdrant contains points from a different corpus/model build')
        return records

    def upsert(self, points):
        self.client.upsert(self.name, points=points, wait=True)

    def delete(self, pid):
        before = self.client.retrieve(self.name, ids=[pid], with_payload=False)
        self.client.delete(self.name, points_selector=models.PointIdsList(points=[pid]), wait=True)
        if self.client.retrieve(self.name, ids=[pid], with_payload=False):
            raise RuntimeError('Deleted passage is still present on read-back')
        return bool(before)
