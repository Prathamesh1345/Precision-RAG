from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / '.env')


def project_path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


@lru_cache(maxsize=1)
def load_config() -> dict:
    path = project_path(os.getenv('PRAG_CONFIG', 'config.yaml'))
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    cfg['qdrant_url'] = os.getenv('PRAG_QDRANT_URL', cfg['qdrant_url'])
    r = cfg['retrieval']
    if not 1 <= r['top_k'] <= r['rerank_candidates'] <= r['prefetch_limit']:
        raise ValueError('Require 1 <= top_k <= rerank_candidates <= prefetch_limit')
    if r['fusion'] not in {'rrf', 'dbsf', 'weighted'} or not 0 <= r['weighted_alpha'] <= 1:
        raise ValueError('Invalid fusion method or weighted_alpha')
    if r['rrf_k'] < 1 or cfg['index']['quantization'] not in {'int8', 'none'}:
        raise ValueError('Invalid RRF constant or index quantization')
    if not 0 <= cfg['models'].get('bm25_b', .75) <= 1 or cfg['models'].get('bm25_k', 1.2) <= 0:
        raise ValueError('Invalid BM25 k1 or b')
    if any(cfg['models'][key] < 1 for key in ['dimension', 'threads', 'batch_size']):
        raise ValueError('Model dimension, threads and batch size must be positive')
    return cfg
