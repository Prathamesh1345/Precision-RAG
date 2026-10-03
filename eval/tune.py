"""Measure candidate/ef trade-offs on tuning queries only; never overwrite config."""
from itertools import product
import time

from app.artifacts import checkpoint, read_jsonl
from app.config import load_config, project_path
from app.retriever import get_retriever
from eval.metrics import ir_metrics, latency_summary, mean_metrics


def main():
    cfg = load_config()
    queries = read_jsonl(project_path(cfg['data_dir']) / 'tuning_queries.jsonl')
    if not queries:
        raise ValueError('A separate frozen tuning set is required')
    service = get_retriever()
    service.models.warmup()
    results = []
    for candidates, ef in product([15, 20, 30], [64, 128]):
        overrides = {'rerank_candidates': candidates, 'hnsw_ef': ef}
        rows, raw, stages = [], [], []
        for q in queries:
            start = time.perf_counter()
            r = service.search(q['query'], mode='hybrid_rerank', top_k=10, use_cache=False, overrides=overrides)
            raw.append((time.perf_counter() - start) * 1000)
            stages.append(r['timings_ms'])
            rows.append(ir_metrics([h['pid'] for h in r['hits']], q['relevant_pids']))
        results.append({'parameters': overrides, 'metrics': mean_metrics(rows),
                        'latency': latency_summary(raw, stages), 'raw_ms': raw, 'per_query_ir': rows})
        checkpoint(project_path(cfg['reports_dir']) / 'checks', 'tuning', {
            'workload': 'separate tuning set, top-10; not submission latency evidence', 'results': results})
    print('Compare reports/checks/tuning.json, choose config values, then run the frozen evaluation once.')


if __name__ == '__main__':
    main()
