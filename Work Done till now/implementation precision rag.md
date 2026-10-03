# PrecisionRAG: implemented pipeline and corrections

Team eyaaduhcaam - ADROSONIC BUILD 2026 - PS1

The runnable implementation is now at the repository root. The original design is retained in docs/implementation-original-reference.md for historical reference; its snippets are not the current execution instructions. Use the root README and config.yaml.

## What was retained

- The existing Streamlit design, search/compare/update/evaluation tabs and API contract.
- Qdrant, named BGE dense and BM25 sparse vectors, dense baseline, hybrid retrieval and cross-encoder reranking.
- RRF/weighted/DBSF fusion, database-level metadata filters, caching and live mutations.
- CPU inference with optional CUDA acceleration for BGE and MiniLM, open-source tools and Groq answers/judging.

## Corrections after reading both PDFs

1. Use MS MARCO QnA v2.1, as specified in the proposal. The earlier draft used v1.1. Default to a qualifying 100K corpus with a separate 500K scaling path.
2. Count unique passages, not query rows. Disk-backed deduplication detects hash collisions and reserves evaluation evidence before applying the corpus cap.
3. Hash deduplication removes exact normalized duplicates. The proposal's phrase "near-duplicates by hash" was inaccurate; semantic near-duplicate removal is not claimed.
4. Enforce 510 BGE content tokens plus two special tokens using tokenizer offsets. Skip evaluation rows whose labelled evidence would be clipped. Hashes and qrels identify the final stored text.
5. Freeze 100 evaluation and 50 separate tuning queries. RAGAS defaults to 50 evaluation queries; the core comparison supports the required minimum of 20 with `--phase core --ragas-queries 20`. Latency always uses at least 100. Retain every labelled passage and merge observed source/category memberships for duplicate text.
6. Recalculate BM25 average document length with the actual corpus and the encoder's stemmed tokens. The default 256 is not used after calibration. Qdrant maintains dynamic IDF; avgdl is frozen to support incremental updates.
7. Acknowledge writes before checkpoints. The draft used wait=False and could checkpoint uncommitted batches. Now wait=True, read-back and exact count verification precede atomic checkpoint writes. Collections are never automatically deleted on restart.
8. Bind resume state to corpus hashes, configuration, FastEmbed version and actual model file hashes. Final verification rereads every expected ID and passage text.
9. Distinguish encoder quantization from index quantization. FastEmbed 0.7.4 maps BGE-small to its quantized ONNX artifact. Qdrant scalar INT8 is separate, with original vectors on disk and explicit rescoring. MiniLM's ONNX artifact is not claimed to be INT8.
10. Pin Qdrant 1.16.2 for configurable RRF k. The proposal's generic v1.10+ Query API requirement is insufficient. Document Qdrant's zero-based RRF: sum(1 / (k + r)).
11. Dense baseline encodes only the dense query. Both query-embedding and result caches are disabled during evaluation.
12. Cache results are defensively copied. Search and mutation/cache invalidation share a lock; serve one API worker. Shared invalidation is required before scaling to multiple workers or external writers.
13. Preserve raw logs and recalculate. IR comes from saved rankings, percentiles from actual samples, and RAGAS means from saved query scores. Missing/nonfinite judge results cannot silently become partial means. Judgments are cached by sample, evidence, metric and judge version.
14. Remove fabricated evaluation placeholders from the UI. Sample search mode stays labelled. The scale check accepts exactly 100,000, and a text passage-ID input preserves 63-bit integers.
15. Keep targets separate from verified results. Embedded Qdrant tests have no HNSW. The real 100K server build and complete read-back are now finished. The original CPU run took about 3 hours 13 minutes, exceeding the two-hour target. See current runtime and benchmark reports for measured latency and quality.
16. Cap neural inference batches (32 documents, 16 reranking pairs) and group document lengths while restoring original vector/ID order. This reduces padded attention allocations. On Windows, resolve the local Docker endpoint to IPv4 to avoid a measured two-second IPv6 fallback on each request.
17. Keep CPU and CUDA packages in separate environments. Record actual kernel placement, model file hashes and numerical comparisons. CPU-built indexes use CPU dense query encoding; the live application accelerates reranking on CUDA. A device change during indexing requires a new collection/checkpoint. See `docs/gpu.md`.

## Execution order

Run from the repository root with the Python environment activated:

    docker compose up -d qdrant
    python -m scripts.build_corpus --n 100000
    python -m scripts.ingest
    python -m eval.run_all --phase baseline
    python -m eval.tune
    python -m eval.run_all
    python -m uvicorn app.api:app --host 127.0.0.1 --port 8000 --workers 1

Run the UI in another terminal:

    python -m streamlit run "Work Done till now/ui/streamlit_app.py"

The 100K corpus is already built in this workspace. Do not rebuild it unless intentionally selecting a new output directory. Set the Groq key in .env before the RAGAS commands. Use --skip-ragas only for an explicitly incomplete development report.

## Recalculation checkpoints

| Major process | Recalculated evidence |
|---|---|
| Query selection | Unique queries, disjoint tuning/evaluation sets, retained selected passages |
| Cleaning | SQL count, duplicate counters, normalized hashes, token lengths |
| Export | Parquet count, qrel coverage and exact file checksums |
| BM25 preparation | Stemmed-token total, document count and average length |
| Embedding | Batch count, dimension, finite values, unit norms, sparse indices |
| Upsert | Acknowledgement, read-back IDs and exact collection count |
| Index completion | Optimizer readiness, every expected ID and passage text |
| Live update/delete | Read-back state, exact count and cache invalidation |
| Evaluation | Rank metrics, RAGAS means and percentiles from saved logs |

See docs/requirements-map.md for the requirement-to-code mapping and reports/verification.md for checks actually performed.

