# PrecisionRAG: Implementation Guide

**Team eyaaduhcaam · ADROSONIC BUILD 2026 · PS1 (Vector Database Design for Large-Scale Precision Retrieval in RAG Systems)**

This is the build guide for the 24-hour hack. It follows the Round 1 proposal: dense + BM25 hybrid search in Qdrant, cross-encoder reranking, metadata filters, live upsert and delete, a Streamlit UI, and a RAGAS benchmark comparing Phase 1 with Phase 2.

> **Hard rule: free and free-tier only.** Every tool below is open source and runs locally, or is a permitted free tier (Groq, Hugging Face). No paid APIs, no OpenAI keys, and no cloud vector DB plans. Before adding any dependency, check it against §1.

---

## 0. Quick start (target: under 10 minutes on a fresh laptop)

```bash
git clone https://github.com/<you>/precisionrag && cd precisionrag
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # add your free GROQ_API_KEY
docker compose up -d qdrant                            # Qdrant on :6333 (dashboard at /dashboard)

python -m scripts.build_corpus --n 500000             # stream MS MARCO → data/corpus.parquet + eval set
python -m scripts.ingest                              # embed + index (dense + BM25) into Qdrant
uvicorn app.api:app --port 8000                       # REST API
streamlit run ui/streamlit_app.py                     # UI on :8501
python -m eval.run_all                                # RAGAS + IR metrics + latency → reports/
```

---

## 1. Free-tier compliance

| Need | Choice | Cost | Notes |
|---|---|---|---|
| Vector DB | **Qdrant** OSS (Docker `qdrant/qdrant`) | Free, local | Hybrid search, filtered HNSW, quantization. Don't use Qdrant Cloud. |
| Dataset | MS MARCO via Hugging Face `datasets` | Free | `microsoft/ms_marco`, streamed |
| Dense embeddings | `BAAI/bge-small-en-v1.5` via **fastembed** (ONNX, CPU) | Free, local | 384-d |
| Sparse / BM25 | `Qdrant/bm25` via fastembed | Free, local | Qdrant applies IDF server-side |
| Reranker | `Xenova/ms-marco-MiniLM-L-6-v2` via fastembed `TextCrossEncoder` | Free, local | ONNX version of the MS MARCO cross-encoder |
| LLM (answers + RAGAS judge) | **Groq free tier** | Free, rate-limited | Explicitly permitted by the PS. Watch the rate limits (§8.4). |
| API / UI | FastAPI, Streamlit | Free OSS | |
| IR metrics | `ranx` | Free OSS | |
| Hosting | **Run locally**, no deployment | Free | Not required by the PS |
| Repo | GitHub public repo | Free | |

**Not allowed:** OpenAI or Anthropic API keys, Cohere rerank, Pinecone, Weaviate Cloud, Qdrant Cloud paid tiers, or any trial that asks for a card.

---

## 2. Repository layout

```
precisionrag/
├── docker-compose.yml
├── requirements.txt
├── .env.example                 # GROQ_API_KEY=
├── config.yaml                  # every tunable lives here (fusion, k, limits, models)
├── README.md
├── app/
│   ├── config.py                # loads config.yaml + .env
│   ├── models.py                # cached fastembed models (loaded once)
│   ├── store.py                 # Qdrant client, collection schema, upsert/delete
│   ├── retriever.py             # dense / hybrid / hybrid+rerank, filters, timing
│   ├── generator.py             # Groq answer with citations (bonus)
│   ├── cache.py                 # LRU query cache
│   └── api.py                   # FastAPI endpoints
├── scripts/
│   ├── build_corpus.py          # MS MARCO → corpus.parquet + eval_queries.jsonl + qrels
│   └── ingest.py                # batch embed + upload, resumable
├── ui/
│   └── streamlit_app.py
├── eval/
│   ├── ragas_eval.py
│   ├── ir_eval.py               # MRR@10, Recall@5, nDCG@10 with ranx
│   ├── latency_bench.py         # 100 consecutive queries → p50/p95/p99
│   └── run_all.py               # runs every mode, writes reports/
├── reports/                     # benchmark_report.md, *.json logs, charts
├── data/                        # gitignored
└── tests/
    └── test_api.py
```

---

## 3. Dependencies

`requirements.txt` (pin versions once everything works, so judges can reproduce):

```
qdrant-client>=1.16
fastembed>=0.5
datasets
pandas
pyarrow
pyyaml
python-dotenv
fastapi
uvicorn[standard]
pydantic>=2
streamlit
requests
groq
ragas==0.2.15
langchain-groq
ranx
numpy
tqdm
matplotlib
pytest
httpx
```

> RAGAS changes its API between minor versions. **Pin the version that works for you** and don't upgrade during the hack. The snippets below use the 0.2.x API (`EvaluationDataset`, `LLMContextPrecisionWithReference`, `LLMContextRecall`).

`docker-compose.yml`:

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest          # needs >= v1.16 for configurable RRF k
    ports: ["6333:6333", "6334:6334"]
    volumes: ["./qdrant_storage:/qdrant/storage"]
    restart: unless-stopped
```

---

## 4. Configuration (`config.yaml`)

The PS requires the fusion method and weights to be **documented and configurable**. All of them live here.

```yaml
collection: msmarco
qdrant_url: http://localhost:6333

models:
  dense: BAAI/bge-small-en-v1.5
  sparse: Qdrant/bm25
  reranker: Xenova/ms-marco-MiniLM-L-6-v2

index:
  hnsw_m: 16
  hnsw_ef_construct: 128
  quantization: int8            # int8 | none
  on_disk_vectors: false

retrieval:
  top_k: 5                      # passages returned to user/LLM
  prefetch_limit: 50            # candidates per branch (dense, sparse)
  rerank_candidates: 30         # fused candidates sent to cross-encoder
  hnsw_ef: 128                  # search-time ef
  fusion: rrf                   # rrf | dbsf | weighted
  rrf_k: 60
  weighted_alpha: 0.6           # weight on dense when fusion=weighted

cache:
  enabled: true
  max_items: 2048

llm:
  provider: groq
  model: llama-3.3-70b-versatile    # check console.groq.com/docs/models; fallback llama-3.1-8b-instant
  judge_model: llama-3.3-70b-versatile
  temperature: 0
```

---

## 5. Phase 1: data and baseline

### 5.1 Building the corpus (`scripts/build_corpus.py`)

MS MARCO QnA rows look like `{query, query_id, query_type, answers, passages: {is_selected[], passage_text[], url[]}, wellFormedAnswers}`. That shape gives us three things directly:

- **Passages**: about 10 per query. Flatten them and dedupe by text hash.
- **Metadata**: `source` is the URL domain, and `category` is the `query_type` (DESCRIPTION / NUMERIC / ENTITY / LOCATION / PERSON) of the query the passage came from.
- **Ground truth**: `is_selected == 1` passages are the relevant ones (qrels), and `answers[0]` is the reference answer for RAGAS recall.

**Pull the eval queries from the `validation` split and make sure their passages are in the index**, otherwise recall is capped by construction.

```python
# scripts/build_corpus.py
import argparse, hashlib, json, os
from urllib.parse import urlparse
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

def pid_of(text: str) -> int:
    # stable 63-bit int id (Qdrant accepts unsigned ints or UUIDs)
    return int(hashlib.blake2b(text.encode(), digest_size=8).hexdigest(), 16) >> 1

def domain(url: str) -> str:
    d = urlparse(url).netloc.lower()
    return d[4:] if d.startswith("www.") else d

def rows_to_passages(row, seen, out):
    qt = (row["query_type"] or "unknown").lower()
    p = row["passages"]
    for text, url in zip(p["passage_text"], p["url"]):
        text = " ".join(text.split())
        pid = pid_of(text)
        if pid in seen:
            continue
        seen.add(pid)
        out.append({"pid": pid, "text": text, "source": domain(url), "category": qt, "url": url})

def main(n: int, n_eval: int, config: str):
    os.makedirs("data", exist_ok=True)
    seen, passages = set(), []

    # 1) eval queries first (validation split), so their passages are guaranteed in the corpus
    val = load_dataset("microsoft/ms_marco", config, split="validation", streaming=True)
    eval_rows = []
    for row in val:
        sel = [t for t, s in zip(row["passages"]["passage_text"], row["passages"]["is_selected"]) if s == 1]
        ans = [a for a in row["answers"] if a and a != "No Answer Present."]
        if not sel or not ans:
            continue
        rows_to_passages(row, seen, passages)
        eval_rows.append({
            "qid": str(row["query_id"]), "query": row["query"], "category": row["query_type"].lower(),
            "reference": ans[0],
            "relevant_pids": [pid_of(" ".join(t.split())) for t in sel],
        })
        if len(eval_rows) >= n_eval:
            break

    # 2) fill the rest from train
    train = load_dataset("microsoft/ms_marco", config, split="train", streaming=True)
    for row in tqdm(train, desc="train"):
        rows_to_passages(row, seen, passages)
        if len(passages) >= n:
            break

    pd.DataFrame(passages[:n]).to_parquet("data/corpus.parquet", index=False)
    with open("data/eval_queries.jsonl", "w") as f:
        for r in eval_rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(passages[:n])} passages, {len(eval_rows)} eval queries")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500_000)
    ap.add_argument("--n_eval", type=int, default=100)   # 100 for latency bench; RAGAS uses a subset
    ap.add_argument("--config", default="v1.1")          # v1.1 train ≈ 82K queries × ~8 passages ≈ 650K+
    a = ap.parse_args()
    main(a.n, a.n_eval, a.config)
```

> **Do this before the event clock starts, if the rules allow it**, or as the very first job. Streaming takes a while on venue Wi-Fi. Keep `corpus.parquet` on a USB drive or a teammate's laptop as a backup.

### 5.2 Loading models once (`app/models.py`)

```python
from functools import lru_cache
from fastembed import TextEmbedding, SparseTextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from app.config import CFG

@lru_cache
def dense():  return TextEmbedding(CFG["models"]["dense"])
@lru_cache
def sparse(): return SparseTextEmbedding(CFG["models"]["sparse"])
@lru_cache
def reranker(): return TextCrossEncoder(CFG["models"]["reranker"])

def warmup():
    list(dense().query_embed(["warmup"])); list(sparse().query_embed(["warmup"]))
    list(reranker().rerank("warmup", ["warmup doc"]))
```

`app/config.py`:

```python
import os, yaml
from dotenv import load_dotenv
load_dotenv()
CFG = yaml.safe_load(open(os.getenv("PRAG_CONFIG", "config.yaml")))
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
```

### 5.3 Collection schema (`app/store.py`)

One collection with **named vectors**, `dense` and `bm25`, plus payload indexes so filters run inside HNSW (FR-4: pre-retrieval).

```python
from qdrant_client import QdrantClient, models
from app.config import CFG
from app import models as M

C = CFG["collection"]
client = QdrantClient(url=CFG["qdrant_url"], timeout=60)

def create_collection(recreate=False):
    if recreate and client.collection_exists(C):
        client.delete_collection(C)
    if client.collection_exists(C):
        return
    q = (models.ScalarQuantization(scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8, quantile=0.99, always_ram=True))
         if CFG["index"]["quantization"] == "int8" else None)
    client.create_collection(
        collection_name=C,
        vectors_config={"dense": models.VectorParams(
            size=384, distance=models.Distance.COSINE,
            on_disk=CFG["index"]["on_disk_vectors"])},
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        hnsw_config=models.HnswConfigDiff(m=CFG["index"]["hnsw_m"],
                                          ef_construct=CFG["index"]["hnsw_ef_construct"]),
        quantization_config=q,
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),  # bulk load: build HNSW after
    )
    for field in ("category", "source"):
        client.create_payload_index(C, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD)

def finish_bulk_load():
    # re-enable indexing so HNSW gets built after the bulk upload
    client.update_collection(C, optimizers_config=models.OptimizersConfigDiff(indexing_threshold=20000))

def make_point(pid, text, payload, dvec, svec):
    return models.PointStruct(
        id=pid,
        vector={"dense": dvec.tolist(),
                "bm25": models.SparseVector(indices=svec.indices.tolist(), values=svec.values.tolist())},
        payload={"text": text, **payload},
    )

# ---- FR-5: live updates ----
def upsert_passage(pid: int, text: str, source="custom", category="custom"):
    dvec = next(M.dense().embed([text]))
    svec = next(M.sparse().embed([text]))
    client.upsert(C, points=[make_point(pid, text, {"source": source, "category": category}, dvec, svec)],
                  wait=True)

def delete_passage(pid: int):
    client.delete(C, points_selector=models.PointIdsList(points=[pid]), wait=True)
```

### 5.4 Ingestion (`scripts/ingest.py`): resumable and checkpointed

```python
import json, os, time
import pandas as pd
from tqdm import tqdm
from app import models as M
from app.store import client, create_collection, finish_bulk_load, make_point, C

BATCH, CKPT = 1024, "data/ingest.ckpt"

def main():
    df = pd.read_parquet("data/corpus.parquet")
    create_collection(recreate=not os.path.exists(CKPT))
    start = json.load(open(CKPT))["next"] if os.path.exists(CKPT) else 0
    t0 = time.time()
    for i in tqdm(range(start, len(df), BATCH)):
        b = df.iloc[i:i + BATCH]
        texts = b.text.tolist()
        dv = list(M.dense().embed(texts, batch_size=256, parallel=0))   # parallel=0 → all cores
        sv = list(M.sparse().embed(texts, batch_size=256))
        pts = [make_point(int(r.pid), r.text, {"source": r.source, "category": r.category, "url": r.url}, d, s)
               for r, d, s in zip(b.itertuples(), dv, sv)]
        client.upsert(C, points=pts, wait=False)
        json.dump({"next": i + BATCH}, open(CKPT, "w"))
    finish_bulk_load()
    secs = time.time() - t0
    json.dump({"passages": len(df), "seconds": secs, "passages_per_sec": len(df) / secs},
              open("reports/ingest_stats.json", "w"), indent=2)
    os.remove(CKPT)

if __name__ == "__main__":
    main()
```

**Indexing-time checklist (FR-1 requires under 2 hours on a CPU):**
- Test with `--n 20000` first and work out passages/sec. Then decide between 100K and 500K.
- If it's too slow, embed on the fastest laptop in the team and **share a Qdrant snapshot** (`client.create_snapshot(C)`, then copy the file across).
- Log `reports/ingest_stats.json`. The time and throughput go into the report as scale evidence.

### 5.5 Dense baseline: Phase 1 mode

In `app/retriever.py` (full file in §6.3), mode `dense` is plain cosine HNSW search returning the top 5. **Record the baseline RAGAS scores before writing any Phase 2 code**, and commit `reports/phase1_*.json`.

---

## 6. Phase 2: precision stack

### 6.1 Filters (FR-4)

```python
def build_filter(category: str | None = None, source: str | None = None):
    must = []
    if category: must.append(models.FieldCondition(key="category", match=models.MatchValue(value=category)))
    if source:   must.append(models.FieldCondition(key="source",   match=models.MatchValue(value=source)))
    return models.Filter(must=must) if must else None
```

Pass the filter **into every `Prefetch`** as well as the top-level `query_filter`, so it's applied inside the HNSW and sparse searches and never post-hoc.

### 6.2 Fusion options (FR-3: configurable and documented)

| `fusion` | Where it runs | How |
|---|---|---|
| `rrf` (default) | Qdrant server | `models.RrfQuery(rrf=models.Rrf(k=rrf_k))`, configurable k needs Qdrant ≥ 1.16 (older: `FusionQuery(fusion=Fusion.RRF)`, k fixed) |
| `dbsf` | Qdrant server | `models.FusionQuery(fusion=models.Fusion.DBSF)`, distribution-based score normalisation |
| `weighted` | Our code | Run dense and sparse separately, min-max normalise, `α·dense + (1−α)·bm25` |

RRF score: `score(d) = Σ_branches 1 / (k + rank_branch(d))`. Because it fuses by rank, the different score ranges of cosine and BM25 don't matter.

### 6.3 Retriever (`app/retriever.py`)

```python
import time
from qdrant_client import models
from app.config import CFG
from app import models as M
from app.store import client, C
from app.cache import cache

R = CFG["retrieval"]

def build_filter(category=None, source=None):
    must = []
    if category: must.append(models.FieldCondition(key="category", match=models.MatchValue(value=category)))
    if source:   must.append(models.FieldCondition(key="source",   match=models.MatchValue(value=source)))
    return models.Filter(must=must) if must else None

def _encode(q):
    d = next(M.dense().query_embed([q])).tolist()
    s = next(M.sparse().query_embed([q]))
    return d, models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())

def _hits(points):
    return [{"pid": p.id, "score": float(p.score), **(p.payload or {})} for p in points]

def _weighted(dense_pts, sparse_pts, alpha):
    def norm(pts):
        if not pts: return {}
        lo, hi = min(p.score for p in pts), max(p.score for p in pts)
        return {p.id: ((p.score - lo) / (hi - lo) if hi > lo else 1.0, p) for p in pts}
    nd, ns = norm(dense_pts), norm(sparse_pts)
    out = {}
    for pid in set(nd) | set(ns):
        sd, p = nd.get(pid, (0.0, None)); ss, p2 = ns.get(pid, (0.0, None))
        out[pid] = (alpha * sd + (1 - alpha) * ss, p or p2)
    ranked = sorted(out.values(), key=lambda x: -x[0])
    return [{"pid": p.id, "score": s, **(p.payload or {})} for s, p in ranked]

def search(query: str, mode: str = "hybrid_rerank", category=None, source=None,
           top_k: int | None = None, use_cache: bool = True):
    """mode: dense | hybrid | hybrid_rerank"""
    top_k = top_k or R["top_k"]
    key = (query, mode, category, source, top_k, R["fusion"])
    if use_cache and CFG["cache"]["enabled"] and key in cache:
        res = dict(cache[key]); res["timings_ms"] = {"cache_hit": 0.0}; return res

    t = {}; t0 = time.perf_counter()
    dvec, svec = _encode(query)
    t["encode"] = (time.perf_counter() - t0) * 1000

    flt = build_filter(category, source)
    params = models.SearchParams(hnsw_ef=R["hnsw_ef"])
    t1 = time.perf_counter()

    if mode == "dense":
        pts = client.query_points(C, query=dvec, using="dense", query_filter=flt,
                                  limit=top_k, search_params=params, with_payload=True).points
        hits = _hits(pts)
    else:
        n_out = R["rerank_candidates"] if mode == "hybrid_rerank" else top_k
        if R["fusion"] == "weighted":
            dp = client.query_points(C, query=dvec, using="dense", query_filter=flt,
                                     limit=R["prefetch_limit"], search_params=params, with_payload=True).points
            sp = client.query_points(C, query=svec, using="bm25", query_filter=flt,
                                     limit=R["prefetch_limit"], with_payload=True).points
            hits = _weighted(dp, sp, R["weighted_alpha"])[:n_out]
        else:
            fusion_q = (models.RrfQuery(rrf=models.Rrf(k=R["rrf_k"])) if R["fusion"] == "rrf"
                        else models.FusionQuery(fusion=models.Fusion.DBSF))
            pts = client.query_points(
                C,
                prefetch=[
                    models.Prefetch(query=dvec, using="dense", limit=R["prefetch_limit"],
                                    filter=flt, params=params),
                    models.Prefetch(query=svec, using="bm25", limit=R["prefetch_limit"], filter=flt),
                ],
                query=fusion_q, query_filter=flt, limit=n_out, with_payload=True,
            ).points
            hits = _hits(pts)
    t["vector_db"] = (time.perf_counter() - t1) * 1000

    if mode == "hybrid_rerank" and hits:
        t2 = time.perf_counter()
        scores = list(M.reranker().rerank(query, [h["text"] for h in hits]))
        for h, s in zip(hits, scores):
            h["fusion_score"], h["score"] = h["score"], float(s)
        hits = sorted(hits, key=lambda h: -h["score"])[:top_k]
        t["rerank"] = (time.perf_counter() - t2) * 1000

    t["total"] = (time.perf_counter() - t0) * 1000
    res = {"query": query, "mode": mode, "fusion": R["fusion"] if mode != "dense" else None,
           "filter": {"category": category, "source": source}, "hits": hits[:top_k], "timings_ms": t}
    if use_cache and CFG["cache"]["enabled"]:
        cache[key] = res
    return res
```

`app/cache.py`:

```python
from collections import OrderedDict
from app.config import CFG

class LRU(OrderedDict):
    def __init__(self, n): super().__init__(); self.n = n
    def __getitem__(self, k): v = super().__getitem__(k); self.move_to_end(k); return v
    def __setitem__(self, k, v):
        super().__setitem__(k, v); self.move_to_end(k)
        if len(self) > self.n: self.popitem(last=False)

cache = LRU(CFG["cache"]["max_items"])
```

### 6.4 Answer generation with citations (bonus, Groq free tier)

```python
# app/generator.py
from groq import Groq
from app.config import CFG, GROQ_API_KEY

_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
SYSTEM = ("Answer ONLY from the numbered passages. Cite every claim like [1], [2]. "
          "If the passages do not contain the answer, say you don't know.")

def answer(query: str, hits: list[dict]) -> str:
    if not _client: return "(GROQ_API_KEY not set)"
    ctx = "\n\n".join(f"[{i+1}] {h['text']}" for i, h in enumerate(hits))
    r = _client.chat.completions.create(
        model=CFG["llm"]["model"], temperature=CFG["llm"]["temperature"], max_tokens=300,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"Passages:\n{ctx}\n\nQuestion: {query}"}])
    return r.choices[0].message.content
```

Generation isn't counted in the retrieval latency budget, so report it separately.

---

## 7. API and UI

### 7.1 FastAPI (`app/api.py`)

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app import retriever, store, generator
from app.models import warmup
from app.cache import cache

app = FastAPI(title="PrecisionRAG")

@app.on_event("startup")
def _startup(): store.create_collection(); warmup()

class SearchReq(BaseModel):
    query: str
    mode: str = "hybrid_rerank"          # dense | hybrid | hybrid_rerank
    category: str | None = None
    source: str | None = None
    top_k: int = 5
    generate: bool = False

class Passage(BaseModel):
    pid: int
    text: str
    source: str = "custom"
    category: str = "custom"

@app.post("/search")
def search(r: SearchReq):
    if r.mode not in {"dense", "hybrid", "hybrid_rerank"}: raise HTTPException(400, "bad mode")
    res = retriever.search(r.query, r.mode, r.category, r.source, r.top_k)
    if r.generate: res["answer"] = generator.answer(r.query, res["hits"])
    return res

@app.post("/passages")
def upsert(p: Passage):
    store.upsert_passage(p.pid, p.text, p.source, p.category); cache.clear()
    return {"status": "upserted", "pid": p.pid}

@app.delete("/passages/{pid}")
def delete(pid: int):
    store.delete_passage(pid); cache.clear()
    return {"status": "deleted", "pid": pid}

@app.get("/stats")
def stats():
    info = store.client.get_collection(store.C)
    return {"points": info.points_count, "status": str(info.status)}
```

> Remember `cache.clear()` after an upsert or delete. Otherwise the demo shows stale results right after a live update, and judges will notice.

### 7.2 Streamlit UI (`ui/streamlit_app.py`), covering FR-6

What it must show: a query box, a **mode toggle** (Dense / Hybrid / Hybrid + Rerank), top-5 passages with **scores**, the **method used**, filters, and an upsert/delete panel. Add a "compare all modes" view for the demo.

```python
import json, requests, streamlit as st, pandas as pd
API = "http://localhost:8000"
st.set_page_config(page_title="PrecisionRAG", layout="wide")
st.title("PrecisionRAG: hybrid, reranked retrieval")

tab_search, tab_compare, tab_update, tab_eval = st.tabs(["Search", "Compare modes", "Live updates", "Evaluation"])
CATS = ["", "description", "numeric", "entity", "location", "person"]

with tab_search:
    q = st.text_input("Ask a question", "what is the normal body temperature of a dog")
    c1, c2, c3, c4 = st.columns(4)
    mode = c1.radio("Retrieval mode", ["dense", "hybrid", "hybrid_rerank"], index=2,
                    format_func={"dense": "Phase 1: Dense", "hybrid": "Phase 2: Hybrid (BM25+dense)",
                                 "hybrid_rerank": "Phase 2: Hybrid + Rerank"}.get)
    cat = c2.selectbox("Category filter", CATS) or None
    src = c3.text_input("Source domain filter (optional)") or None
    gen = c4.checkbox("Generate answer (Groq)")
    if st.button("Search", type="primary") and q:
        r = requests.post(f"{API}/search", json={"query": q, "mode": mode, "category": cat,
                                                  "source": src, "generate": gen}).json()
        st.caption(f"Method: **{r['mode']}** · fusion: {r['fusion']} · timings (ms): "
                   + ", ".join(f"{k}={v:.1f}" for k, v in r["timings_ms"].items()))
        if gen: st.success(r.get("answer", ""))
        for i, h in enumerate(r["hits"], 1):
            with st.container(border=True):
                st.markdown(f"**[{i}] score {h['score']:.4f}** · `{h.get('category')}` · {h.get('source')} · pid {h['pid']}")
                st.write(h["text"])

with tab_compare:
    q2 = st.text_input("Question to compare", key="cmp")
    if st.button("Compare") and q2:
        cols = st.columns(3)
        for col, m in zip(cols, ["dense", "hybrid", "hybrid_rerank"]):
            r = requests.post(f"{API}/search", json={"query": q2, "mode": m}).json()
            col.subheader(m); col.caption(f"{r['timings_ms'].get('total', 0):.0f} ms")
            for h in r["hits"]: col.markdown(f"- ({h['score']:.3f}) {h['text'][:180]}…")

with tab_update:
    pid = st.number_input("Passage ID", min_value=1, value=999_000_001, step=1)
    text = st.text_area("Passage text", "ADROSONIC BUILD is a 24-hour hackathon held at BIT Mesra in October 2026.")
    a, b = st.columns(2)
    if a.button("Upsert"): st.json(requests.post(f"{API}/passages", json={"pid": int(pid), "text": text}).json())
    if b.button("Delete"): st.json(requests.delete(f"{API}/passages/{int(pid)}").json())
    st.info("Demo: upsert → search 'when is ADROSONIC BUILD' → found → delete → search again → gone.")

with tab_eval:
    try:
        st.dataframe(pd.read_json("reports/summary.json"))
        st.image("reports/latency_hist.png")
        st.image("reports/ablation.png")
    except Exception:
        st.warning("Run `python -m eval.run_all` first.")
```

---

## 8. Evaluation and benchmarking

### 8.1 Protocol (decide it now and never change it mid-hack)

- **Frozen query set** from `data/eval_queries.jsonl`. Use the **same queries for every mode**.
- **RAGAS:** 30–50 queries (the PS minimum is 20). Use fewer if Groq rate limits bite (§8.4).
- **IR metrics:** all 100 eval queries, using the `is_selected` qrels. No LLM needed, so it's free and has no rate limits.
- **Latency:** 100 consecutive queries, **cache disabled**, after warmup.
- **Ablation ladder:** `dense` → `hybrid (rrf)` → `hybrid (dbsf)` / `weighted` → `hybrid_rerank`.

### 8.2 RAGAS (`eval/ragas_eval.py`)

```python
import json
from ragas import evaluate, EvaluationDataset, RunConfig
from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq
from app.config import CFG, GROQ_API_KEY
from app.retriever import search

def run(mode: str, n: int = 30, fusion: str | None = None):
    if fusion: CFG["retrieval"]["fusion"] = fusion
    qs = [json.loads(l) for l in open("data/eval_queries.jsonl")][:n]
    samples = []
    for q in qs:
        r = search(q["query"], mode=mode, use_cache=False)
        samples.append({"user_input": q["query"],
                        "retrieved_contexts": [h["text"] for h in r["hits"]],
                        "reference": q["reference"]})
    ds = EvaluationDataset.from_list(samples)
    judge = LangchainLLMWrapper(ChatGroq(model=CFG["llm"]["judge_model"], temperature=0, api_key=GROQ_API_KEY))
    res = evaluate(ds, metrics=[LLMContextPrecisionWithReference(), LLMContextRecall()], llm=judge,
                   run_config=RunConfig(max_workers=2, max_retries=10, max_wait=60, timeout=120))
    df = res.to_pandas()
    tag = f"{mode}_{fusion or CFG['retrieval']['fusion']}"
    df.to_json(f"reports/ragas_{tag}.json", orient="records", indent=2)
    return {"mode": tag,
            "context_precision": float(df["llm_context_precision_with_reference"].mean()),
            "context_recall": float(df["context_recall"].mean())}
```

> Column names in `to_pandas()` can differ between RAGAS versions. Print `df.columns` once and adjust.

### 8.3 IR metrics with human labels (`eval/ir_eval.py`)

This is the cross-check that shows the improvement doesn't depend on the judge LLM.

```python
import json
from ranx import Qrels, Run, evaluate
from app.retriever import search

def run(mode: str):
    qs = [json.loads(l) for l in open("data/eval_queries.jsonl")]
    qrels = Qrels({q["qid"]: {str(p): 1 for p in q["relevant_pids"]} for q in qs})
    run = Run({q["qid"]: {str(h["pid"]): h["score"] for h in search(q["query"], mode=mode, top_k=10, use_cache=False)["hits"]}
               for q in qs})
    return {"mode": mode, **evaluate(qrels, run, ["mrr@10", "recall@5", "ndcg@10"])}
```

> Caveat: MS MARCO labels are sparse (usually one relevant passage), so absolute numbers look low. **The relative gain between modes is what matters.** Say so in the report.

### 8.4 Groq free-tier budget

Context precision makes about one LLM call per retrieved context, and recall about one per query. So **30 queries × 5 contexts ≈ 180 calls per mode**, or roughly 700+ calls for four modes. Groq's free tier has per-minute and per-day request and token limits. Check the current limits in the Groq console before you start.

- Run RAGAS with `max_workers=2` and generous retries.
- **Save every result to JSON and never re-run a finished mode.**
- If you hit the daily cap on the 70B model, switch the judge to the 8B instant model for **all** modes. Never mix judges across modes.
- Each teammate can create their own free Groq key. Give each person a different mode to evaluate, all on the same judge model.

### 8.5 Latency benchmark (`eval/latency_bench.py`)

```python
import json, time, numpy as np
import matplotlib.pyplot as plt
from app.retriever import search
from app.models import warmup

def run(mode="hybrid_rerank", n=100):
    warmup()
    qs = [json.loads(l)["query"] for l in open("data/eval_queries.jsonl")][:n]
    for q in qs[:5]: search(q, mode=mode, use_cache=False)          # extra warmup
    lat, stages = [], []
    for q in qs:
        t0 = time.perf_counter()
        r = search(q, mode=mode, use_cache=False)
        lat.append((time.perf_counter() - t0) * 1000); stages.append(r["timings_ms"])
    a = np.array(lat)
    out = {"mode": mode, "n": n, "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
           "p99": float(np.percentile(a, 99)), "mean": float(a.mean()),
           "stage_mean_ms": {k: float(np.mean([s.get(k, 0) for s in stages])) for k in stages[0]}}
    json.dump({"summary": out, "raw_ms": lat}, open(f"reports/latency_{mode}.json", "w"), indent=2)
    plt.figure(figsize=(6, 3)); plt.hist(a, bins=30); plt.axvline(300, ls="--", c="r", label="300 ms SLA")
    plt.axvline(out["p95"], c="k", label=f"p95={out['p95']:.0f} ms"); plt.legend(); plt.xlabel("ms")
    plt.title(f"Latency: {mode} (100 queries)"); plt.tight_layout(); plt.savefig("reports/latency_hist.png", dpi=150)
    return out
```

**If p95 is over 300 ms, apply these in order:** lower `rerank_candidates` from 30 to 20 to 15 → lower `hnsw_ef` from 128 to 64 → set `prefetch_limit` from 50 to 30 → make sure the models are warm and Qdrant is local → as a last resort, switch the reranker to `jinaai/jina-reranker-v1-tiny-en`. **Log the precision/latency trade-off** for each change; it makes a good chart for the report.

### 8.6 `eval/run_all.py`

It runs `ir_eval`, `ragas_eval` and `latency_bench` for every mode, then writes:
- `reports/summary.json`, one row per mode with precision, recall, MRR@10, Recall@5, nDCG@10, p50 and p95
- `reports/ablation.png`, grouped bars of precision and recall per mode
- `reports/latency_hist.png`
- `reports/benchmark_report.md`, using the template in §10

---

## 9. 24-hour execution plan

Split into four parallel tracks and agree on the interfaces first (`search()` signature, `config.yaml` keys, eval file format).

| Hours | Track A: Data and index | Track B: Retrieval | Track C: API and UI | Track D: Evaluation and report |
|---|---|---|---|---|
| 0–1 | Repo, docker-compose, Qdrant up | config.yaml, models.py | FastAPI skeleton | eval set format, Groq keys |
| 1–4 | build_corpus (20K test → 100K), ingest | dense search | /search + Streamlit search tab | RAGAS + IR on dense → **Phase 1 baseline committed** |
| 4–8 | Scale to 500K (background) | sparse + RRF hybrid, filters | filters + upsert/delete UI | IR eval for hybrid |
| 8–12 | ingest stats, Qdrant snapshot | rerank, cache, DBSF/weighted | compare-modes tab | latency bench, first p95 check |
| 12–16 | help tune hnsw_ef / limits | latency tuning | Groq answer + citations | RAGAS for all modes (rate-limited) |
| 16–20 | README, reproduction test on a 2nd laptop | bug fixes | eval dashboard tab | benchmark_report.md + charts |
| 20–24 | **Freeze code**, record a backup demo video, rehearse the demo twice | | | |

**Checkpoints (don't skip these):**
- **Hour 4:** Phase 1 works end to end, with baseline scores logged and committed. If anything breaks later, you still have a valid submission.
- **Hour 12:** all FRs work at 100K, even if 500K isn't finished.
- **Hour 20:** feature freeze. Only docs, the report and demo prep after this.

---

## 10. Deliverables checklist

The PS requires all of these:

- [ ] **GitHub repo**, public, with a README (setup, architecture, how to run eval, config options and fusion documentation)
- [ ] **Architecture diagram** (reuse the one from the proposal, put it in `docs/architecture.png`, and embed it in the README)
- [ ] **Working demo**: Streamlit UI with the dense ↔ hybrid toggle, scores, method shown, a filter demo, and upsert/delete
- [ ] **Phase 1 RAGAS baseline** logged (`reports/ragas_dense_*.json`)
- [ ] **Benchmark report** (`reports/benchmark_report.md` or PDF):
  - Setup (hardware, corpus size, models, versions)
  - Phase 1 vs Phase 2 table: Context Precision, Context Recall (+ MRR@10, Recall@5)
  - Ablation chart
  - Latency: p50/p95/p99 over 100 queries, histogram, per-stage breakdown
  - Indexing time and throughput
  - Fusion method and weights used, and why
  - Honest limitations (sparse qrels, LLM-judge variance, Groq limits)
- [ ] NFR check: precision > 0.75 · recall > 0.70 · p95 < 300 ms · ≥ 100K passages (500K bonus) · free tier only

**Report results table template:**

| Mode | Context Precision | Context Recall | MRR@10 | Recall@5 | p95 (ms) |
|---|---|---|---|---|---|
| Phase 1: Dense | | | | | |
| Phase 2: Hybrid (RRF k=60) | | | | | |
| Phase 2: Hybrid (DBSF) | | | | | |
| Phase 2: Hybrid + Rerank | | | | | |

Fill it in **only with measured numbers**. Never estimate.

---

## 11. Demo script (about 4 minutes)

1. **Context (20 s):** "Dense retrieval returns similar-but-wrong passages, and that's where hallucinations come from. We built a precision-first retrieval stack."
2. **Compare tab (60 s):** use a query with an exact term or number (an entity or numeric query works best). Show dense missing it, hybrid catching it, and rerank putting it at #1. Point out the scores and timings.
3. **Filter (30 s):** run the same query with `category=numeric`, and explain that the filter is applied inside HNSW.
4. **Live update (40 s):** upsert the ADROSONIC passage, search for it and find it, delete it, search again and show it's gone.
5. **Answer with citations (20 s):** turn on Groq generation and show the [1][2] citations.
6. **Evaluation tab (60 s):** the Phase 1 → Phase 2 gains, p95 against the 300 ms line, and 500K indexed in X minutes.
7. **Close (10 s):** everything is free tier, reproducible with one command, and the fusion is configurable.

---

## 12. Troubleshooting

| Symptom | Fix |
|---|---|
| `Rrf(k=...)` error | Qdrant server or client is older than 1.16. Upgrade the image, or use `FusionQuery(fusion=Fusion.RRF)` |
| Search slow right after ingest | HNSW is still building. Wait until the collection status is green (`/stats`) |
| Filtered queries slow or empty | Payload index missing. Check that `create_payload_index` ran, and that filter values are lowercase |
| RAGAS 429 errors | Lower `max_workers` to 1, raise retries, use the per-mode result cache, or split modes across teammates' keys |
| RAGAS scores look noisy | Use `temperature=0`, the same judge model for every mode, the same queries, and report the IR metrics alongside |
| Out of RAM at 500K | Keep int8 quantization with `always_ram=True` and set `on_disk_vectors: true` |
| fastembed downloads stall | Pre-download models before the event (run `warmup()` once), since the HF cache persists |
| Upserted passage not found | The cache wasn't cleared, or `wait=True` is missing on the upsert |
