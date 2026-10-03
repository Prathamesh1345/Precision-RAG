"""Talks to the real FastAPI backend; falls back to the mock when it isn't running."""
import os
import requests

from ui.mock_backend import MockBackend

API_URL = os.getenv("PRAG_API_URL", "http://localhost:8000")


class ApiBackend:
    is_mock = False

    def __init__(self, url=API_URL):
        self.url = url.rstrip("/")

    def stats(self):
        return requests.get(f"{self.url}/stats", timeout=3).json()

    def search(self, query, mode="hybrid_rerank", category=None, source=None, top_k=5, generate=False):
        r = requests.post(f"{self.url}/search", timeout=60, json={
            "query": query, "mode": mode, "category": category, "source": source,
            "top_k": top_k, "generate": generate})
        r.raise_for_status()
        return r.json()

    def upsert(self, pid, text, source="custom", category="custom"):
        r = requests.post(f"{self.url}/passages", timeout=30,
                          json={"pid": int(pid), "text": text, "source": source, "category": category})
        r.raise_for_status()
        return r.json()

    def delete(self, pid):
        r = requests.delete(f"{self.url}/passages/{int(pid)}", timeout=30)
        r.raise_for_status()
        return r.json()


def api_alive(url=API_URL) -> bool:
    try:
        return requests.get(f"{url.rstrip('/')}/stats", timeout=1.5).ok
    except requests.RequestException:
        return False
