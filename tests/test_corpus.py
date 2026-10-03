from pathlib import Path

import pyarrow.parquet as pq
import pytest

from app.artifacts import read_jsonl, write_jsonl
from scripts.build_corpus import audit_corpus, build_from_streams, domain, pid_of
from tests.conftest import row


def test_trim_preserve_qrels_and_merge_metadata(tmp_path, limiter):
    val = [row(1, ['alpha answer', 'common text'], category='numeric'),
           row(2, ['beta answer', 'common text'], category='entity')]
    train = [row(3, ['common text', 'gamma answer', 'one two three four five six seven eight nine ten eleven'])]
    m = build_from_streams(train, val, tmp_path, 5, 1, 1, limiter, {'revision': 'fixture'})
    assert m['passages'] == 5
    assert m['recalculation']['qrel_coverage'] == 1
    corpus = pq.read_table(tmp_path / 'corpus.parquet').to_pylist()
    assert next(p for p in corpus if p['text'] == 'common text')['categories'] == ['description', 'entity', 'numeric']
    assert max(p['token_count'] for p in corpus) == 10
    assert sum(p['trimmed'] for p in corpus) == 1
    assert m['recalculation']['token_counts_recomputed']
    q = read_jsonl(tmp_path / 'eval_queries.jsonl')[0]
    assert q['relevant_pids'] == [pid_of('alpha answer')]
    assert pid_of(' a  b ') == pid_of('a b')
    assert domain('https://www.EXAMPLE.org/test') == 'example.org'


def test_cap_cannot_remove_ground_truth(tmp_path, limiter):
    with pytest.raises(ValueError, match='cap'):
        build_from_streams([], [row(1, ['a', 'b', 'c'])], tmp_path, 2, 1, 0, limiter, {})


def test_skip_truncated_positive(tmp_path, limiter):
    val = [row(1, [' '.join(['long'] * 12)]), row(2, ['short relevant'])]
    build_from_streams([], val, tmp_path, 1, 1, 0, limiter, {})
    assert read_jsonl(tmp_path / 'eval_queries.jsonl')[0]['qid'] == '2'


def test_recalculation_detects_missing_label(tmp_path, limiter):
    build_from_streams([], [row(1, ['evidence'])], tmp_path, 1, 1, 0, limiter, {})
    qs = read_jsonl(tmp_path / 'eval_queries.jsonl')
    qs[0]['relevant_pids'] = [999]
    write_jsonl(tmp_path / 'eval_queries.jsonl', qs)
    with pytest.raises(ValueError, match='ground-truth'):
        audit_corpus(tmp_path)


def test_refuses_partial_dataset(tmp_path, limiter):
    with pytest.raises(ValueError, match='exactly'):
        build_from_streams([], [row(1, ['evidence'])], tmp_path, 100, 1, 0, limiter, {})
