from __future__ import annotations

import argparse
import time

from app.artifacts import read_json, read_jsonl, write_json
from app.config import load_config, project_path
from app.retriever import get_retriever
from eval.metrics import latency_summary


def run(service, queries, mode, fusion, output, n=100, filtered=False, overrides=None):
    if n < 100 or len(queries) < n:
        raise ValueError('Submission latency runs require at least 100 distinct frozen queries')
    selected = queries[:n]
    service.models.warmup(mode)
    for q in selected[:5]:
        service.search(q['query'], mode=mode, fusion=fusion, top_k=5, use_cache=False,
                       category=q['category'] if filtered else None, overrides=overrides)
    raw, stages, qids = [], [], []
    # Keep writes, recalculation and plotting outside the consecutive-query timed loop.
    for q in selected:
        started = time.perf_counter()
        result = service.search(q['query'], mode=mode, fusion=fusion, top_k=5, use_cache=False,
                                category=q['category'] if filtered else None, overrides=overrides)
        raw.append((time.perf_counter() - started) * 1000)
        stages.append(result['timings_ms'])
        qids.append(q['qid'])
    write_json(output, {'raw_ms': raw, 'stages': stages, 'qids': qids})
    saved = read_json(output)
    summary = {**latency_summary(saved['raw_ms'], saved['stages']), 'mode': mode,
               'fusion': fusion, 'cache_enabled': False, 'scope': 'in-process retrieval, including query encoding; excludes HTTP and LLM generation',
               'under_300ms': latency_summary(saved['raw_ms'], saved['stages'])['p95'] < 300}
    write_json(output, {**saved, 'summary': summary})
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['dense', 'hybrid', 'hybrid_rerank'], default='hybrid_rerank')
    args = ap.parse_args()
    cfg = load_config()
    tag = args.mode + '_' + cfg['retrieval']['fusion']
    print(run(get_retriever(), read_jsonl(project_path(cfg['data_dir']) / 'eval_queries.jsonl'),
              args.mode, cfg['retrieval']['fusion'], project_path(cfg['reports_dir']) / f'latency_{tag}.json'))
