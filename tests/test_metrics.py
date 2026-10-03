import math

import pytest

from app.cache import LRU
from eval.metrics import ir_metrics, latency_summary


def test_metrics_against_hand_calculation():
    result = ir_metrics([9, 2, 3, 4, 5, 1], [1, 2])
    assert result['mrr@10'] == .5
    assert result['recall@5'] == .5
    assert result['ndcg@10'] == pytest.approx((1 / math.log2(3) + 1 / math.log2(7)) / (1 + 1 / math.log2(3)))
    assert ir_metrics([], [1]) == {'mrr@10': 0, 'recall@5': 0, 'ndcg@10': 0}
    with pytest.raises(ValueError):
        ir_metrics([1, 1], [1])


def test_latency_recalculation():
    result = latency_summary(list(range(1, 101)), [{'encode': 10}] * 100)
    assert result['p95'] == pytest.approx(95.05)
    assert result['p50'] == 50.5 and result['stage_mean_ms']['encode'] == 10
    with pytest.raises(ValueError):
        latency_summary([float('nan')], [{}])


def test_lru_eviction_and_expiry():
    c = LRU(2)
    c.put('a', 1)
    c.put('b', 2)
    assert c.get('a') == 1
    c.put('c', 3)
    assert c.get('b') is None
    c.ttl = 0
    assert c.get('a') is None
