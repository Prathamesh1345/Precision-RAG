# PrecisionRAG benchmark

Incomplete or one or more quality/latency targets not met.

Corpus: 100,000 passages. Dataset revision: `a47ee7aae8d7d466ba15f9f0bfac3b3681087b3a`.
Judge: `openai/gpt-oss-120b`. Run ID: `20261003T144417380800Z`.

| Mode | Context precision | Context recall | MRR@10 | Recall@5 | nDCG@10 | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 0.814 | 0.900 | 0.611 | 0.870 | 0.693 | 10.693 | 30.026 | 30.867 |
| hybrid_rerank_rrf | 0.864 | 0.900 | 0.680 | 0.885 | 0.748 | 49.277 | 65.185 | 65.754 |

## Measurement protocol

- Same frozen validation queries and qrels for every mode; 20 RAGAS queries per measured mode.
- 100 consecutive top-5 queries after warmup; result AND embedding caches disabled.
- Latency includes encoding, DB access, fusion and reranking; excludes HTTP transport and answer generation.
- IR is recalculated from saved top-10 rankings. RAGAS uses the first five of those same rankings.
- HNSW/index/vector parameters, hardware, package versions and corpus hashes are recorded in run_context.json.
- Both RAGAS means and latency percentiles are recalculated from saved per-query artifacts.

## Interpretation and limitations

This is a sampled MS MARCO QnA v2.1 corpus, not the official passage-ranking leaderboard. All selected validation evidence is reserved before train distractors are added. Labels are sparse; unjudged retrieved passages may be useful despite receiving zero binary relevance credit. Exact normalized duplicates are merged; semantic near-duplicates are not removed. Overlength selected evidence is excluded from evaluation, and training passages are clipped to 510 BGE tokens. The cross-encoder was trained on MS MARCO; this is an in-domain benchmark.

The filtered ablation uses the dataset query-type tag as an explicit scope constraint. It is a separate, favourable scoped workload, not evidence of general unfiltered improvement. Source/category arrays retain labels from observed duplicate occurrences. LLM judge variance and free-tier rate limits remain limitations. Missing scores are never replaced with estimates.
