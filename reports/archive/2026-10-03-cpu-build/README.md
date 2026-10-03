# Archived measurements: first CPU build (3 October 2026)

Measured on a Ryzen 7 6800HS laptop (16 logical cores, 16 GB RAM), Windows 11, CPU-only ONNX, Qdrant 1.16.2 in Docker.
Kept as evidence of what was tried. These numbers are NOT the submission benchmark.

- `ingest_stats.json`: 100,000 passages indexed and fully read back. Cumulative ingestion time
  **11,585 s (3 h 13 min)**, about 8.6 passages/s. This **misses the 2-hour target**; see the README for
  what to change before the next build (thread count, batch size, or CUDA embedding).
- `run_dense_only/`: dense-only retrieval on the full 100K index, 100 frozen queries, caches off.
  p50 11.7 ms, **p95 35.2 ms**, p99 36.4 ms. No RAGAS scores were completed for this run.

`run_context.json` contains the exact config, package versions and model file hashes used.
