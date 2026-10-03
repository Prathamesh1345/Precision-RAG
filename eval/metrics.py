"""Rank-based binary relevance metrics, computed directly from saved rankings."""
import math

import numpy as np


def ir_metrics(ranked_ids, relevant_ids):
    ids = [str(p) for p in ranked_ids]
    relevant = {str(p) for p in relevant_ids}
    if len(ids) != len(set(ids)) or not relevant:
        raise ValueError('Rankings must have unique IDs and qrels must be nonempty')
    mrr = next((1.0 / (i + 1) for i, pid in enumerate(ids[:10]) if pid in relevant), 0.0)
    recall = len(set(ids[:5]) & relevant) / len(relevant)
    dcg = sum(1 / math.log2(i + 2) for i, pid in enumerate(ids[:10]) if pid in relevant)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(10, len(relevant))))
    return {'mrr@10': mrr, 'recall@5': recall, 'ndcg@10': dcg / ideal}


def mean_metrics(rows):
    if not rows:
        raise ValueError('Cannot average an empty evaluation')
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def latency_summary(raw_ms, stages):
    a = np.asarray(raw_ms, dtype=float)
    if not len(a) or not np.isfinite(a).all() or (a < 0).any() or len(stages) != len(a):
        raise ValueError('Latency logs are empty, incomplete or nonfinite')
    return {'n': len(a), 'p50': float(np.percentile(a, 50)), 'p95': float(np.percentile(a, 95)),
            'p99': float(np.percentile(a, 99)), 'mean': float(a.mean()),
            'stage_mean_ms': {key: float(np.mean([s.get(key, 0) for s in stages]))
                              for key in sorted({k for s in stages for k in s})}}
