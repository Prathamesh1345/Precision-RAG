# Problem statement and proposal mapping

| Requirement | Implemented in | Evidence still required at submission |
|---|---|---|
| FR-1 >=100K passages, Hugging Face model, metadata, CPU indexing <2h | `scripts/build_corpus.py`, `scripts/ingest.py`, `app/models.py`, `app/store.py` | Full server ingest log; 100K corpus preparation has already passed |
| FR-2 dense top-5 plus RAGAS baseline >=20 queries | `app/retriever.py`, `eval/run_all.py --phase baseline` | Groq-judged baseline run |
| FR-3 BM25 hybrid with configurable/documented fusion | `app/retriever.py`, `config.yaml`, README | Actual Phase 2 improvement |
| FR-4 pre-retrieval filters | `app/store.py`, dense/sparse Prefetch filters | Server demo |
| FR-5 live upsert/delete | `app/retriever.py`, `app/api.py`, retained UI | Server demo; embedded tests already pass |
| FR-6 query UI, mode toggle, scores | `ui/streamlit_app.py` | UI is retained and smoke-checked |
| NFR-1/2 quality thresholds | `eval/ragas_eval.py` | >0.75 precision, >0.70 recall measured |
| NFR-3 100 queries/p95 <300ms | `eval/latency_bench.py` | Full 100K server measurement |
| NFR-4 scale | `--n 100000`, optional `--n 500000` | Index count, not just corpus count |
| NFR-5 free tier only | Local OSS stack and explicit Groq adapter | Use a free Groq account |
| NFR-6 reproduction/GitHub | README, pinned core versions, environment lock, Compose | Publish to the team's own repository |
| NFR-7 benchmark report | `eval/run_all.py` | Measured final report |
| Proposal reranking/caching/citations | MiniLM, two LRU caches, `app/generator.py` | Full-scale quality/performance and optional generation demo |

The repository has not been published and no results are fabricated. Thresholds are targets until logs demonstrate them.
