# PrecisionRAG: live progress demonstration

## Open these

- Application: http://127.0.0.1:8501
- API documentation: http://127.0.0.1:8000/docs
- Database dashboard: http://127.0.0.1:6333/dashboard

## A 60-second explanation

“Our project implements high-precision retrieval for RAG over a reproducible subset of MS MARCO QnA v2.1. We have downloaded and prepared 100,000 unique passages, with 100 evaluation queries and 50 separate tuning queries. We preserve the labelled evidence for every evaluation query.

“The pipeline normalizes and deduplicates passages, enforces the embedding token limit, and creates BGE dense vectors alongside BM25 sparse vectors. Qdrant stores both representations and supports metadata filtering. At query time, we combine dense and lexical results, rerank 30 candidates with MiniLM, and return the top five. Groq generates an answer using those retrieved passages.

“Each major stage has a recalculation check: corpus hashes and counts, vector dimensions and norms, acknowledged writes and read-back, and evaluation metrics recalculated from raw logs. We are currently completing the full index and then will run the frozen evaluation. Final precision, recall and p95 targets are not yet verified.”

## Show the real workflow

1. Open the application. Confirm the sidebar is connected to the live API and **Use sample data** is off. The passage count is the actual database count; click **Recheck connection** to refresh it.
2. Search **what does BANT stand for in sales** using **Hybrid + rerank**. Enable **Write an answer with citations**. Show the retrieved passages, source metadata, passage IDs, and grounded answer.
3. Open **Compare** and run the same question. Explain why dense retrieval, lexical retrieval and reranking can produce different rankings. Individual displayed scores are model/retrieval scores, not measured precision.
4. Open the Qdrant dashboard and show the named dense and BM25 vector configuration, collection count, and stored passage payloads.
5. Open `data/checks/03_corpus_recalculation.json` and `data/manifest.json` for the prepared dataset and provenance. Show `data/bm25_stats.json` for independently recalculated average document length.
6. Use `README.md` for the architecture diagram and `Work Done till now/implementation precision rag.md` for the implementation narrative. Keep `.env` closed.

## Current measured evidence

- 100,000 prepared passages; 100 evaluation queries and 50 disjoint tuning queries.
- No clipping was needed in this frozen sample; maximum passage length is 372 BGE content tokens.
- Real API retrieval and a Groq-generated answer succeeded. Saved evidence: `reports/checks/live_demo.json`.
- All 100,000 passages are now indexed; final ID/text read-back passed and Qdrant reports healthy. The original recorded CPU ingestion time is about 3 hours 13 minutes.
- 24 tests passed after the latest demo/GPU changes, and the live Streamlit rendering check passed.
- GPU execution was confirmed by ONNX kernel profiling. A small concurrent-load microbenchmark measured 128 embeddings at 4.337 seconds CPU / 0.068 seconds CUDA, and 30-passage reranking at 0.300 seconds CPU / 0.023 seconds CUDA. These are component measurements, not a complete retrieval benchmark or guaranteed speedup.
- Dense CPU/CUDA outputs differ slightly; the current demo keeps CPU query embeddings consistent with the CPU-built index and uses CUDA for reranking.
- The original CPU ingestion has exceeded two hours. Do not claim that acceptance target is met by this run.
- Final RAGAS precision/recall and 100-query p95 results are pending. A single search time does not establish p95.

Live update/delete code is implemented and covered by functional tests. The current presentation is read-only so it preserves the frozen evaluation corpus.
