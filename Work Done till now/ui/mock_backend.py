"""Mock backend so the UI works before the real FastAPI + Qdrant backend exists.

Mirrors the API contract in implementation.md:
  POST   /search    -> {query, mode, fusion, filter, hits[], timings_ms, answer?}
  POST   /passages  -> {status, pid}
  DELETE /passages/{pid}
  GET    /stats     -> {points, status}

Everything here is FAKE demo data. Never put these numbers in the report.
"""
import random
import re
import time

SAMPLE_PASSAGES = [
    (101, "The normal body temperature for dogs is between 101 and 102.5 degrees Fahrenheit, which is higher than the human average of 97.6 to 99.6 F.", "akc.org", "numeric"),
    (102, "A dog's temperature can be taken rectally with a digital thermometer. Readings above 103 F indicate a fever and warrant a call to the vet.", "petmd.com", "description"),
    (103, "Cats have a normal body temperature range of 100.5 to 102.5 degrees Fahrenheit.", "vcahospitals.com", "numeric"),
    (104, "Hyperthermia in dogs occurs when body temperature rises above normal, often caused by heat stroke during hot weather.", "petmd.com", "description"),
    (105, "Form 1099-B reports proceeds from broker and barter exchange transactions, such as sales of stocks and bonds.", "irs.gov", "entity"),
    (106, "Form 1099-R is used to report distributions from pensions, annuities, retirement plans, IRAs and insurance contracts.", "irs.gov", "entity"),
    (107, "Ranchi is the capital of the Indian state of Jharkhand and is known as the City of Waterfalls.", "wikipedia.org", "location"),
    (108, "Mesra is a village near Ranchi, Jharkhand, best known as the home of the Birla Institute of Technology.", "wikipedia.org", "location"),
    (109, "Retrieval-Augmented Generation (RAG) combines a retriever that fetches relevant documents with a generator that writes an answer grounded in them.", "arxiv.org", "description"),
    (110, "BM25 is a ranking function used by search engines to estimate the relevance of documents to a search query based on term frequency and inverse document frequency.", "wikipedia.org", "description"),
    (111, "Reciprocal Rank Fusion combines ranked lists by summing 1/(k + rank) for each document across lists, with k typically set to 60.", "elastic.co", "description"),
    (112, "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity.", "britannica.com", "person"),
    (113, "Marie Curie was the first woman to win a Nobel Prize and the only person to win Nobel Prizes in two scientific fields.", "nobelprize.org", "person"),
    (114, "The average cost of a kitchen remodel in the US ranges from $14,000 to $40,000 depending on materials and size.", "homeadvisor.com", "numeric"),
    (115, "Mount Everest, at 8,849 metres, is Earth's highest mountain above sea level, located in the Mahalangur Himal of the Himalayas.", "nationalgeographic.com", "location"),
    (116, "Cross-encoders process the query and document together through a transformer, producing more accurate relevance scores than bi-encoders at higher cost.", "sbert.net", "description"),
]

FUSION = "rrf"


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9\-]+", s.lower()))


class MockBackend:
    is_mock = True

    def __init__(self):
        self.store = {pid: {"pid": pid, "text": t, "source": src, "category": cat, "url": f"https://{src}/..."}
                      for pid, t, src, cat in SAMPLE_PASSAGES}

    # ---------- API surface ----------
    def stats(self):
        return {"points": len(self.store), "status": "mock"}

    def search(self, query, mode="hybrid_rerank", category=None, source=None, top_k=5, generate=False):
        rng = random.Random(hash((query, mode)) & 0xFFFF)
        t0 = time.perf_counter()
        q = _tokens(query)
        cands = [p for p in self.store.values()
                 if (not category or p["category"] == category) and (not source or p["source"] == source)]

        scored = []
        for p in cands:
            d = _tokens(p["text"])
            overlap = len(q & d) / (len(q) or 1)
            if mode == "dense":
                # dense = fuzzy: semantic-ish noise, weaker on exact tokens
                s = 0.55 + 0.30 * overlap + rng.uniform(-0.12, 0.12)
            elif mode == "hybrid":
                s = 1 / (60 + 1) * (0.6 + overlap) + rng.uniform(-0.002, 0.002)
            else:
                s = -4 + 12 * overlap + rng.uniform(-0.8, 0.8)   # cross-encoder logits
            scored.append({**p, "score": round(s, 4)})
        hits = sorted(scored, key=lambda h: -h["score"])[:top_k]

        timings = {"encode": rng.uniform(12, 22), "vector_db": rng.uniform(8, 30) if mode == "dense" else rng.uniform(20, 45)}
        if mode == "hybrid_rerank":
            timings["rerank"] = rng.uniform(90, 170)
        timings["total"] = sum(timings.values()) + (time.perf_counter() - t0) * 1000

        res = {"query": query, "mode": mode, "fusion": None if mode == "dense" else FUSION,
               "filter": {"category": category, "source": source}, "hits": hits, "timings_ms": timings}
        if generate:
            res["answer"] = ("(mock answer) " + hits[0]["text"][:160] + " [1]") if hits else "I don't know."
        return res

    def upsert(self, pid, text, source="custom", category="custom"):
        self.store[int(pid)] = {"pid": int(pid), "text": text, "source": source, "category": category, "url": ""}
        return {"status": "upserted", "pid": int(pid)}

    def delete(self, pid):
        existed = self.store.pop(int(pid), None) is not None
        return {"status": "deleted" if existed else "not_found", "pid": int(pid)}
