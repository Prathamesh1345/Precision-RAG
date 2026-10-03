# Local verification - 3 October 2026

This is a development verification record, not the required final retrieval benchmark.

## Data actually prepared

- Dataset: microsoft/ms_marco, QnA v2.1.
- Dataset revision: a47ee7aae8d7d466ba15f9f0bfac3b3681087b3a.
- 100,000 unique passages, 100 frozen evaluation queries, 50 disjoint tuning queries.
- Every labelled relevant passage is present: recalculated qrel coverage 100%.
- 848 exact duplicate occurrences merged; duplicate metadata memberships retained.
- Maximum passage length: 372 BGE content tokens. No passage needed clipping in this sample.
- Parquet size: 21,038,268 bytes.
- Corpus-building function time: 88.57 seconds, excluding earlier tokenizer/metadata setup. This is NOT embedding or indexing time.
- BM25 average stemmed-token length recalculated over the entire corpus: 34.04294 (3,404,294 tokens / 100,000 documents).
- Independent recalculation reopened all passages and recomputed token counts, hashes, uniqueness and qrel coverage successfully.

## Code and real-model checks

- 24 automated tests passed after GPU/demo changes: corpus caps, trimming, deduplication and metadata, qrel integrity, interrupted ingestion/replay, cache bypass/invalidation, retrieval/fusion modes, filtering, live mutations, API validation, hand-calculated metrics, development report generation, bounded batching/vector alignment, IPv4 transport, explicit GPU failure and read-only presentation behavior.
- Real BGE-small ONNX, BM25 and MiniLM cross-encoder smoke test passed on 65 actual corpus passages using in-memory Qdrant.
- Real dense vectors were 384-dimensional, finite and unit-normalized within floating-point tolerance.
- Real-model smoke checks included RRF, weighted and DBSF fusion, reranking, scoped retrieval, upsert and delete.
- Model file SHA-256 hashes and downloaded snapshot IDs are in checks/real_model_smoke.json.
- Streamlit loads successfully and handles absent RAGAS values and a full 63-bit passage ID.
- RAGAS 0.2.15 and the Groq LangChain wrapper imported; both required metric objects initialized successfully without making a judge call.
- Python compilation and pip dependency checks passed. Full tested dependency versions are recorded in requirements-lock-windows-py312.txt.

## Real server and GPU execution

Docker Desktop was installed and Qdrant 1.16.2 started. All 100,000 passages are now indexed. Every ID and text passed read-back; the optimizer reports OK and the collection is green. Recorded cumulative CPU ingestion time is 11,584.9 seconds, which fails the two-hour target. This timing is retained in `ingest_stats.json`.

A Groq key is configured privately. Both a live RAGAS integration check and a real API answer-generation check passed using `openai/gpt-oss-120b` on Groq. The core 20-query-per-mode RAGAS comparison is running; incomplete scores are not reported as final quality results.

The separate CUDA environment successfully executed both models on the RTX 3060. Kernel profiling confirms GPU work. The preliminary component probe and numerical comparisons are in `checks/device_benchmark.json`. The actual application uses CPU dense queries with CUDA reranking, and its Streamlit rendering check passed with no errors.

The first full-index dense timing run measured p95 35.2 ms on 100 distinct frozen queries, with caches disabled. See the current run directory under `reports/runs` for raw latency samples and eventual recalculated quality results. Fusion/filter ablations and a fresh fully CUDA-built indexing-time experiment remain additional work. Publishing to the team's repository requires a supplied destination.
