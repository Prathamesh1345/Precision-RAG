from __future__ import annotations

import time
from functools import lru_cache
from threading import RLock

from qdrant_client import models

from app.cache import LRU
from app.config import load_config
from app.models import ModelBundle
from app.store import Store, build_filter, make_point, sparse_vector


def hits_from(points):
    return [{**{k: v for k, v in (p.payload or {}).items() if not k.startswith('_')},
             'pid': p.id, 'score': float(p.score)} for p in points]


def weighted_fusion(dense, sparse, alpha):
    def normalized(points):
        if not points:
            return {}
        lo, hi = min(p.score for p in points), max(p.score for p in points)
        return {p.id: ((p.score - lo) / (hi - lo) if hi > lo else 1.0, p) for p in points}
    d, s = normalized(dense), normalized(sparse)
    merged = []
    for pid in d.keys() | s.keys():
        ds, dp = d.get(pid, (0.0, None))
        ss, sp = s.get(pid, (0.0, None))
        hit = hits_from([dp or sp])[0]
        hit['score'] = alpha * ds + (1 - alpha) * ss
        merged.append(hit)
    return sorted(merged, key=lambda h: (-h['score'], h['pid']))


class Retriever:
    def __init__(self, cfg, store=None, models_bundle=None):
        self.cfg = cfg
        self.store = store or Store(cfg)
        self.models = models_bundle or ModelBundle(cfg)
        c = cfg['cache']
        self.results = LRU(c['max_items'], c['ttl_seconds'])
        self.embeddings = LRU(c['max_items'], c['ttl_seconds'])
        # Serializes mutations with search/cache insertion. Serve with ONE API worker.
        self.lock = RLock()

    def search(self, query, mode='hybrid_rerank', category=None, source=None, top_k=None,
               use_cache=True, fusion=None, overrides=None):
        started = time.perf_counter()
        with self.lock:
            return self._search(started, query, mode, category, source, top_k, use_cache, fusion, overrides)

    def _search(self, started, query, mode, category, source, top_k, use_cache, fusion, overrides):
        query = ' '.join(query.split())
        if not query or mode not in {'dense', 'hybrid', 'hybrid_rerank'}:
            raise ValueError('Nonempty query and a supported retrieval mode are required')
        r = {**self.cfg['retrieval'], **(overrides or {})}
        fusion = fusion or r['fusion']
        k = r['top_k'] if top_k is None else top_k
        if fusion not in {'rrf', 'weighted', 'dbsf'} or not 1 <= k <= 100:
            raise ValueError('Invalid fusion or top_k (1..100)')
        category = category.strip().lower() if category else None
        source = source.strip().lower().removeprefix('www.') if source else None
        use_cache = use_cache and self.cfg['cache']['enabled']
        key = (query, mode, category, source, k, fusion, tuple(sorted(r.items())))
        cached = self.results.get(key) if use_cache else None
        if cached is not None:
            cached['timings_ms'] = {'cache_hit': 1.0, 'total': (time.perf_counter() - started) * 1000}
            return cached
        timing = {}
        t = time.perf_counter()
        embedding_key = (query, mode != 'dense')
        encoded = self.embeddings.get(embedding_key) if use_cache else None
        if encoded is None:
            encoded = self.models.query(query, include_sparse=mode != 'dense')
            if use_cache:
                self.embeddings.put(embedding_key, encoded)
        d, s = encoded
        timing['encode'] = (time.perf_counter() - t) * 1000
        flt = build_filter(category, source)
        params = models.SearchParams(hnsw_ef=r['hnsw_ef'], quantization=models.QuantizationSearchParams(
            rescore=r['rescore'], oversampling=r['oversampling']))
        client, name = self.store.client, self.store.name
        t = time.perf_counter()
        if mode == 'dense':
            points = client.query_points(name, query=d, using='dense', query_filter=flt,
                                         search_params=params, limit=k, with_payload=True).points
            hits = hits_from(points)
        else:
            out = max(k, r['rerank_candidates']) if mode == 'hybrid_rerank' else k
            prefetch = max(out, r['prefetch_limit'])
            sv = sparse_vector(s)
            if fusion == 'weighted':
                dp = client.query_points(name, query=d, using='dense', query_filter=flt,
                                         search_params=params, limit=prefetch, with_payload=True).points
                sp = client.query_points(name, query=sv, using='bm25', query_filter=flt,
                                         limit=prefetch, with_payload=True).points
                hits = weighted_fusion(dp, sp, r['weighted_alpha'])[:out]
            else:
                fused = (models.RrfQuery(rrf=models.Rrf(k=r['rrf_k'])) if fusion == 'rrf'
                         else models.FusionQuery(fusion=models.Fusion.DBSF))
                points = client.query_points(name, prefetch=[
                    models.Prefetch(query=d, using='dense', filter=flt, params=params, limit=prefetch),
                    models.Prefetch(query=sv, using='bm25', filter=flt, limit=prefetch)],
                    query=fused, query_filter=flt, limit=out, with_payload=True).points
                hits = hits_from(points)
        timing['vector_db'] = (time.perf_counter() - t) * 1000
        if mode == 'hybrid_rerank' and hits:
            t = time.perf_counter()
            scores = self.models.rerank(query, [h['text'] for h in hits])
            if len(scores) != len(hits):
                raise ValueError('Rerank score count mismatch')
            for hit, score in zip(hits, scores):
                hit['fusion_score'], hit['score'] = hit['score'], score
            hits.sort(key=lambda h: (-h['score'], h['pid']))
            timing['rerank'] = (time.perf_counter() - t) * 1000
        timing['total'] = (time.perf_counter() - started) * 1000
        result = {'query': query, 'mode': mode, 'fusion': None if mode == 'dense' else fusion,
                  'filter': {'category': category, 'source': source}, 'hits': hits[:k], 'timings_ms': timing}
        if use_cache:
            self.results.put(key, result)
        return result

    def upsert(self, pid, text, source='custom', category='custom'):
        text, source, category = ' '.join(text.split()), source.strip().lower(), category.strip().lower()
        if not text or not source or not category:
            raise ValueError('Passage, source and category cannot be blank')
        source = source.removeprefix('www.')
        p = {'pid': pid, 'text': text, 'source': source, 'category': category,
             'sources': [source], 'categories': [category], 'url': ''}
        with self.lock:
            if hasattr(self.models, 'validate_passage_length'):
                self.models.validate_passage_length(text)
            dense, sparse = self.models.documents([text])
            self.results.clear()
            try:
                self.store.upsert([make_point(p, dense[0], sparse[0], 'live-update')])
                rows = self.store.verify_ids([pid], 'live-update')
                if rows[0].payload['text'] != text:
                    raise RuntimeError('Live update read-back mismatch')
                return {'status': 'upserted', 'pid': pid, 'recalculation': self.store.stats()}
            finally:
                self.results.clear()

    def delete(self, pid):
        with self.lock:
            self.results.clear()
            try:
                existed = self.store.delete(pid)
                return {'status': 'deleted' if existed else 'not_found', 'pid': pid,
                        'recalculation': self.store.stats()}
            finally:
                self.results.clear()


@lru_cache(maxsize=1)
def get_retriever():
    return Retriever(load_config())


def search(*args, **kwargs):
    return get_retriever().search(*args, **kwargs)
