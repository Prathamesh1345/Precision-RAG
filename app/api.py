from __future__ import annotations

import time
import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel, Field, field_validator

from app.generator import answer
from app.retriever import get_retriever


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    mode: Literal['dense', 'hybrid', 'hybrid_rerank'] = 'hybrid_rerank'
    fusion: Literal['rrf', 'weighted', 'dbsf'] | None = None
    category: str | None = Field(default=None, max_length=100)
    source: str | None = Field(default=None, max_length=255)
    top_k: int = Field(default=5, ge=1, le=100)
    generate: bool = False
    use_cache: bool = True

    @field_validator('query')
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError('Query cannot be blank')
        return value


class Passage(BaseModel):
    pid: int = Field(ge=0, le=2**63 - 1)
    text: str = Field(min_length=1, max_length=20000)
    source: str = Field(default='custom', min_length=1, max_length=255)
    category: str = Field(default='custom', min_length=1, max_length=100)


def create_app(service=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.retriever = service or get_retriever()
        app.state.retriever.store.ensure_collection()
        app.state.retriever.models.warmup()
        yield

    application = FastAPI(title='PrecisionRAG', lifespan=lifespan)

    def backend():
        return application.state.retriever

    @application.get('/health')
    def health():
        return {'status': 'ok'}

    @application.get('/stats')
    def stats():
        return {**backend().store.stats(), 'read_only': os.getenv('PRAG_READ_ONLY') == '1'}

    @application.post('/search')
    def search(request: SearchRequest):
        try:
            b = backend()
            result = b.search(**request.model_dump(exclude={'generate'}))
            if request.generate:
                started = time.perf_counter()
                try:
                    result['answer'] = answer(request.query, result['hits'], b.cfg)
                except Exception as exc:
                    result['generation_error'] = str(exc)
                result['generation_ms'] = (time.perf_counter() - started) * 1000
            return result
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @application.post('/passages')
    def upsert(passage: Passage):
        if os.getenv('PRAG_READ_ONLY') == '1':
            raise HTTPException(409, 'This demo preserves the frozen benchmark index; restart without PRAG_READ_ONLY to edit a separate demo collection.')
        try:
            return backend().upsert(**passage.model_dump())
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @application.delete('/passages/{pid}')
    def delete(pid: int = Path(ge=0, le=2**63 - 1)):
        if os.getenv('PRAG_READ_ONLY') == '1':
            raise HTTPException(409, 'This demo preserves the frozen benchmark index.')
        return backend().delete(pid)

    return application


app = create_app()
