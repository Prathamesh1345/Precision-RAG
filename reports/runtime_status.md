# Runtime status

Updated 2026-10-03 during the live invigilation setup.

Docker Desktop and Qdrant 1.16.2 are running. **All 100,000 passages are indexed and the full ID/text read-back passed.** The checkpoint is complete, Qdrant reports green/optimizer OK and 100,000 indexed vectors, and the final HNSW settings are M=16, ef_construct=128, indexing_threshold=20000. Recorded cumulative ingestion time is 11,584.9 seconds (about 3 hours 13 minutes), so the original CPU run did not meet the two-hour target. See `reports/ingest_stats.json`.

The live presentation runs at http://127.0.0.1:8501, with the API at http://127.0.0.1:8000/docs. It uses CPU BGE queries and RTX 3060 CUDA MiniLM reranking. Groq answer generation was verified. The collection is read-only through this demo API.

A separate `.venv-gpu` contains FastEmbed GPU 0.7.4 and ONNX Runtime GPU 1.23.2 with local CUDA/cuDNN libraries. The CPU environment remains available. `reports/checks/device_benchmark.json` records timing, numerical comparisons and actual kernel provider counts. This is a preliminary component benchmark, not the final 100-query benchmark.

All 24 regression tests passed after the GPU/runtime changes; the real Streamlit app rendered without errors against the live API. Follow `reports/invigilation_walkthrough.md` for presentation steps. Final RAGAS and full-corpus retrieval benchmarks are still pending. The next core comparison judges the first 20 frozen queries per mode, meeting the stated minimum, and measures 100 distinct latency queries per mode.
