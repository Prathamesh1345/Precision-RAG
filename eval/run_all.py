"""Run dense first, preserve raw artifacts, then recalculate the benchmark report."""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

from app.artifacts import checkpoint, environment, fingerprint, read_json, read_jsonl, write_json
from app.config import load_config, project_path
from app.retriever import get_retriever
from app.runtime import device
from app.store import build_filter
from eval import ir_eval, latency_bench, ragas_eval
from eval.metrics import ir_metrics, latency_summary, mean_metrics
from scripts.ingest import build_identity, verify_manifest


def assert_frozen_index(service, cfg, manifest):
    expected = manifest['passages']
    build_id = build_identity(cfg, manifest)
    if service.store.count() != expected or service.store.count(build_filter(build_id=build_id)) != expected:
        raise ValueError('Benchmark requires the unmodified frozen corpus. Rebuild into a separate collection after live demos.')
    info = service.store.stats()
    if info['status'] != 'green' or info['optimizer_status'] != 'ok':
        raise ValueError('Qdrant indexing/optimization must finish before benchmarking')
    # A large server collection with no HNSW vectors is not ready for ANN benchmarking.
    if expected >= 100000 and not info['indexed_vectors']:
        raise ValueError('No indexed HNSW vectors reported; wait for optimization before benchmarking')
    return info


def recalculate(directory, tags):
    rows = []
    for tag in tags:
        rankings = read_jsonl(directory / f'ir_{tag}.jsonl')
        ir = mean_metrics([ir_metrics([h['pid'] for h in r['hits']], r['relevant_pids']) for r in rankings])
        latency = read_json(directory / f'latency_{tag}.json')
        lat = latency_summary(latency['raw_ms'], latency['stages'])
        if len(set(latency['qids'])) != lat['n'] or lat['n'] < 100:
            raise ValueError('Latency log must contain 100 or more distinct query IDs')
        row = {'mode': tag, **ir, **{k: lat[k] for k in ['p50', 'p95', 'p99']},
               'latency_queries': lat['n'], 'context_precision': None, 'context_recall': None, 'ragas_queries': 0}
        path = directory / f'ragas_{tag}.json'
        if path.exists():
            ragas = read_json(path)['rows']
            if len(ragas) < 20:
                raise ValueError('Incomplete RAGAS run')
            row.update({key: sum(r[key] for r in ragas) / len(ragas)
                        for key in ['context_precision', 'context_recall']})
            row['ragas_queries'] = len(ragas)
        rows.append(row)
    write_json(directory / 'summary.json', rows)
    return rows


def make_report(directory, rows, context):
    def number(value, digits=3):
        return 'NOT MEASURED' if value is None else f'{value:.{digits}f}'
    baseline = rows[0]
    phase2 = [r for r in rows if r['mode'] != 'dense' and 'filtered' not in r['mode']]
    qualifies = lambda r: (r['context_precision'] is not None and r['context_precision'] > .75
                           and r['context_recall'] > .70 and r['p95'] < 300
                           and baseline['context_precision'] is not None
                           and r['context_precision'] > baseline['context_precision']
                           and r['context_recall'] > baseline['context_recall'])
    status = ('Measured quality/latency targets met by at least one unfiltered Phase 2 mode'
              if any(qualifies(r) for r in phase2) else 'Incomplete or one or more quality/latency targets not met')
    write_json(directory / 'acceptance.json', {
        'minimum_index_scale': context['index']['points'] >= 100000,
        'indexing_under_two_hours': context['ingestion']['under_two_hours'] if context.get('ingestion') else None,
        'baseline_ragas_recorded': baseline['context_precision'] is not None and baseline['ragas_queries'] >= 20,
        'unfiltered_phase2_meets_quality_latency_and_improvement': {r['mode']: qualifies(r) for r in phase2},
        'latency_scope': 'in-process retrieval, not HTTP; see report',
        'publishing_and_live_demo': 'must be demonstrated separately'})
    lines = ['# PrecisionRAG benchmark', '', status + '.', '',
             f"Corpus: {context['manifest']['passages']:,} passages. Dataset revision: `{context['manifest']['revision']}`.",
             f"Judge: `{context['config']['llm']['judge_model']}`. Run ID: `{context['run_id']}`.", '',
             '| Mode | Context precision | Context recall | MRR@10 | Recall@5 | nDCG@10 | p50 ms | p95 ms | p99 ms |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| ' + row['mode'] + ' | ' + ' | '.join(number(row[k]) for k in
                     ['context_precision', 'context_recall', 'mrr@10', 'recall@5', 'ndcg@10', 'p50', 'p95', 'p99']) + ' |')
    lines += ['', '## Measurement protocol', '',
              f"- Same frozen validation queries and qrels for every mode; {context['config']['evaluation']['ragas_queries']} RAGAS queries per measured mode.",
              '- 100 consecutive top-5 queries after warmup; result AND embedding caches disabled.',
              '- Latency includes encoding, DB access, fusion and reranking; excludes HTTP transport and answer generation.',
              '- IR is recalculated from saved top-10 rankings. RAGAS uses the first five of those same rankings.',
              '- HNSW/index/vector parameters, hardware, package versions and corpus hashes are recorded in run_context.json.',
              '- Both RAGAS means and latency percentiles are recalculated from saved per-query artifacts.', '',
              '## Interpretation and limitations', '',
              'This is a sampled MS MARCO QnA v2.1 corpus, not the official passage-ranking leaderboard. '
              'All selected validation evidence is reserved before train distractors are added. Labels are sparse; '
              'unjudged retrieved passages may be useful despite receiving zero binary relevance credit. '
              'Exact normalized duplicates are merged; semantic near-duplicates are not removed. '
              'Overlength selected evidence is excluded from evaluation, and training passages are clipped to 510 BGE tokens. '
              'The cross-encoder was trained on MS MARCO; this is an in-domain benchmark.', '',
              'The filtered ablation uses the dataset query-type tag as an explicit scope constraint. '
              'It is a separate, favourable scoped workload, not evidence of general unfiltered improvement. '
              'Source/category arrays retain labels from observed duplicate occurrences. '
              'LLM judge variance and free-tier rate limits remain limitations. Missing scores are never replaced with estimates.', '']
    ingestion = context.get('ingestion')
    if ingestion:
        lines += ['## Indexing', '', f"Measured pipeline time: {ingestion['seconds']:.1f} s; "
                  f"throughput: {ingestion['passages_per_sec']:.1f} passages/s. "
                  f"Under two hours: {ingestion['under_two_hours']}. "
                  f"Index disk bytes observed: {ingestion['index_disk_bytes']} (0 if storage is outside this workspace).", '']
    (directory / 'benchmark_report.md').write_text('\n'.join(lines), encoding='utf-8')
    plot(directory, rows)


def plot(directory, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(10, 4))
    measured = [r for r in rows if r['context_precision'] is not None]
    if measured:
        x = np.arange(len(measured))
        ax.bar(x - .18, [r['context_precision'] for r in measured], .36, label='Context precision')
        ax.bar(x + .18, [r['context_recall'] for r in measured], .36, label='Context recall')
        ax.set_xticks(x, [r['mode'] for r in measured], rotation=20, ha='right')
        ax.set_ylim(0, 1)
        ax.legend()
    else:
        ax.text(.5, .5, 'RAGAS not measured', ha='center', va='center', transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(directory / 'ablation.png', dpi=160)
    plt.close(fig)
    for row in rows:
        data = read_json(directory / f'latency_{row["mode"]}.json')
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.hist(data['raw_ms'], bins=25)
        ax.axvline(300, color='red', linestyle='--', label='300 ms target')
        ax.axvline(row['p95'], color='black', label=f"p95 {row['p95']:.1f} ms")
        ax.set(xlabel='Milliseconds', ylabel='Queries', title=row['mode'])
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory / f'latency_{row["mode"]}.png', dpi=160)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase', choices=['baseline', 'core', 'all'], default='all',
                    help='core measures the dense baseline and hybrid+rerank; all adds fusion/filter ablations')
    ap.add_argument('--ragas-queries', type=int, help='Number of frozen queries to judge per mode (minimum 20)')
    ap.add_argument('--skip-ragas', action='store_true', help='Produce an explicitly incomplete development report')
    args = ap.parse_args()
    cfg = load_config()
    if args.ragas_queries is not None:
        if args.ragas_queries < 20:
            ap.error('--ragas-queries must be at least 20')
        cfg['evaluation']['ragas_queries'] = args.ragas_queries
    if not args.skip_ragas and not os.getenv('GROQ_API_KEY'):
        ap.error('Set GROQ_API_KEY, install requirements-eval.txt, or use --skip-ragas for development only')
    data, reports = project_path(cfg['data_dir']), project_path(cfg['reports_dir'])
    manifest = verify_manifest(data)
    service = get_retriever()
    status = assert_frozen_index(service, cfg, manifest)
    queries = read_jsonl(data / 'eval_queries.jsonl')
    if len(queries) < cfg['evaluation']['latency_queries']:
        ap.error('Not enough frozen queries for the requested latency benchmark')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = reports / 'runs' / run_id
    output.mkdir(parents=True)
    context = {'run_id': run_id, 'config': cfg, 'config_fingerprint': fingerprint(cfg), 'manifest': manifest,
               'environment': environment(), 'index': status,
               'ingestion': read_json(reports / 'ingest_stats.json') if (reports / 'ingest_stats.json').exists() else None}
    if context['ingestion'] and context['ingestion'].get('dense_device', 'cpu') != device('dense'):
        raise ValueError('Query embedding device differs from this index build; use the matching device')
    if hasattr(service.models, 'provenance'):
        model_provenance = service.models.provenance(include_reranker=args.phase != 'baseline')
        context['model_provenance'] = model_provenance
        if context['ingestion']:
            for branch, recorded in context['ingestion'].get('model_provenance', {}).items():
                if model_provenance.get(branch) != recorded:
                    raise ValueError('Evaluation models differ from the models used to index this corpus')
    write_json(output / 'run_context.json', context)
    modes = [('dense', 'dense', 'rrf', False)]
    if args.phase == 'core':
        modes += [('hybrid_rerank_rrf', 'hybrid_rerank', 'rrf', False)]
    if args.phase == 'all':
        modes += [('hybrid_rrf', 'hybrid', 'rrf', False), ('hybrid_weighted', 'hybrid', 'weighted', False),
                  ('hybrid_dbsf', 'hybrid', 'dbsf', False), ('hybrid_rerank_rrf', 'hybrid_rerank', 'rrf', False),
                  ('hybrid_rerank_filtered', 'hybrid_rerank', 'rrf', True)]
    tags = []
    for tag, mode, fusion, filtered in modes:
        service.models.warmup(mode)
        ir_eval.run(service, queries, mode, fusion, output / f'ir_{tag}.jsonl', filtered)
        latency_bench.run(service, queries, mode, fusion, output / f'latency_{tag}.json',
                          n=cfg['evaluation']['latency_queries'], filtered=filtered)
        if not args.skip_ragas:
            ragas_eval.run(output / f'ir_{tag}.jsonl', cfg, output / f'ragas_{tag}.json', reports / 'ragas_cache')
        assert_frozen_index(service, cfg, manifest)
        tags.append(tag)
        rows = recalculate(output, tags)
        checkpoint(output / 'checks', f'06_{tag}_recalculation', {'metrics': rows[-1]})
        make_report(output, rows, context)
        print(f'Finished {tag}; raw results: {output}', flush=True)
    # Publish only one coherent completed run; the UI follows this manifest.
    write_json(reports / 'latest_run.json', {'run_id': run_id, 'path': str(output), 'phase': args.phase,
                                          'ragas_complete': not args.skip_ragas})
    write_json(reports / 'summary.json', rows)
    (reports / 'benchmark_report.md').write_text((output / 'benchmark_report.md').read_text(encoding='utf-8'), encoding='utf-8')
    print(output / 'benchmark_report.md')


if __name__ == '__main__':
    main()
