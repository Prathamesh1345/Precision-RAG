# Demo script (about 4 minutes)

Open before you start: http://localhost:8501 (app), http://localhost:8000/docs (API), http://localhost:6333/dashboard (Qdrant). Confirm the sidebar says **Connected** and shows the real passage count.

1. **Why** (20 s). Dense search returns passages that are similar but wrong, which is where hallucinations come from. We built a precision-first retrieval stack on free tools.
2. **Compare tab** (60 s). Ask a question with an exact name or number, for example *what is form 1099-r used for* or *what does BANT stand for in sales*. Show dense missing or misranking it, hybrid catching the exact term, and rerank moving it to the top. Point at the per-stage timings. Say plainly that the displayed scores are model scores, not measured precision.
3. **Filter** (30 s). Same question with a category or source filter. Explain that the filter runs inside the HNSW and BM25 searches, not after them.
4. **Update index** (40 s). Save a passage about ADROSONIC BUILD, check with a search (found), delete it, check again (gone). No re-indexing.
5. **Answer with citations** (20 s). Turn on the Groq answer; show the `[pid:...]` citations and that generation time is reported separately from retrieval.
6. **Results tab** (60 s). Phase 1 vs Phase 2 precision/recall, p95 against the 300 ms line, indexing time and passage count. Only show numbers that `eval.run_all` produced.
7. **Close** (10 s). Everything is free tier and reproducible from `README.md`; fusion is configurable; every stage has a recalculation check.

If something fails live, the Qdrant dashboard (collection with named `dense` and `bm25` vectors, payload indexes) and `reports/benchmark_report.md` still tell the story.
