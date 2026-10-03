"""Bounded-memory, acknowledged, resumable dense + BM25 ingestion."""
from __future__ import annotations

import argparse
import time
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

import pyarrow.parquet as pq
import psutil

from app.artifacts import checkpoint, environment, fingerprint, read_json, sha256, write_json
from app.config import load_config, project_path
from app.models import ModelBundle, validate_vectors
from app.runtime import device
from app.store import Store, build_filter, make_point
from scripts.build_corpus import audit_corpus


def build_identity(cfg, manifest):
    try:
        fastembed_version = version('fastembed')
    except PackageNotFoundError:
        fastembed_version = version('fastembed-gpu')
    return fingerprint({'files': manifest['files'], 'models': cfg['models'], 'index': cfg['index'],
                        'schema': 2, 'fastembed_version': fastembed_version,
                        'collection': cfg['collection'], 'qdrant_url': cfg['qdrant_url']})


def verify_manifest(data):
    manifest = read_json(data / 'manifest.json')
    for name, digest in manifest['files'].items():
        if sha256(data / name) != digest:
            raise ValueError(f'{name} changed since the frozen corpus was created')
    return manifest


def ingest(cfg, store=None, models_bundle=None):
    run_started = time.perf_counter()
    data, reports = project_path(cfg['data_dir']), project_path(cfg['reports_dir'])
    manifest = verify_manifest(data)
    audit_corpus(data, expected=manifest['passages'])
    build_id = build_identity(cfg, manifest)
    progress_path = data / f'ingest_{cfg["collection"]}.json'
    progress = read_json(progress_path) if progress_path.exists() else {
        'build_id': build_id, 'next': 0, 'elapsed_seconds': 0, 'complete': False}
    if progress['build_id'] != build_id:
        raise ValueError('Corpus/model/index configuration changed; use a new collection and checkpoint')
    if progress['next'] and progress.get('dense_device', 'cpu') != device('dense'):
        raise ValueError('Dense inference device changed; use a new collection rather than mixing embeddings')
    store = store or Store(cfg)
    existed = store.client.collection_exists(store.name)
    if progress['next'] and not existed:
        raise ValueError('Checkpoint exists but collection is missing. Use a new collection name to start again.')
    store.ensure_collection(bulk=True)
    count = store.count()
    own_count = store.count(build_filter(build_id=build_id))
    if count != own_count:
        raise ValueError('Collection contains unrelated or live-edited points; choose a clean collection for benchmarking')
    if count < progress['next'] or count > manifest['passages']:
        raise ValueError('Collection count is inconsistent with the checkpoint')
    # A crash after an acknowledged write can leave up to one uncheckpointed batch.
    # Replaying its stable IDs is safe; never delete a collection automatically.
    bundle = models_bundle or ModelBundle(cfg)
    model_provenance = bundle.provenance() if hasattr(bundle, 'provenance') else {'fixture': True}
    if progress.get('model_provenance') and progress['model_provenance'] != model_provenance:
        raise ValueError('Downloaded model files changed; refusing to mix embeddings on resume')
    if hasattr(bundle, 'calibrate'):
        bundle.calibrate(data / 'corpus.parquet')
    process = psutil.Process()
    peak_rss = max(progress.get('sampled_python_rss_peak_bytes', 0), process.memory_info().rss)
    batch_size = cfg['index']['batch_size']
    offset, session_started = 0, run_started
    prior_seconds = progress['elapsed_seconds']
    for batch in pq.ParquetFile(data / 'corpus.parquet').iter_batches(batch_size=batch_size):
        rows = batch.to_pylist()
        end = offset + len(rows)
        if end <= progress['next']:
            store.verify_ids([p['pid'] for p in rows], build_id)
            offset = end
            continue
        rows = rows[max(0, progress['next'] - offset):]
        batch_started = time.perf_counter()
        d, s = bundle.documents([p['text'] for p in rows])
        encoding_seconds = time.perf_counter() - batch_started
        peak_rss = max(peak_rss, process.memory_info().rss)
        validation = validate_vectors(d, s, len(rows), cfg['models']['dimension'])
        store.upsert([make_point(p, dv, sv, build_id) for p, dv, sv in zip(rows, d, s)])
        store.verify_ids([p['pid'] for p in rows], build_id)
        recalculated = store.count(build_filter(build_id=build_id))
        if recalculated != end:
            raise RuntimeError(f'Index count recalculation failed: expected {end}, found {recalculated}')
        progress = {'build_id': build_id, 'next': end, 'complete': False,
                    'elapsed_seconds': prior_seconds + time.perf_counter() - session_started,
                    'last_batch': validation, 'recalculated_points': recalculated, 'model_provenance': model_provenance,
                    'last_batch_encoding_seconds': encoding_seconds,
                    'last_batch_total_seconds': time.perf_counter() - batch_started,
                    'effective_dense_batch_size': min(cfg['models']['batch_size'], 32),
                    'dense_device': device('dense'),
                    'sampled_python_rss_peak_bytes': peak_rss}
        write_json(progress_path, progress)
        checkpoint(reports / 'checks', '04_embedding_and_upsert', progress)
        offset = end
        print(f'Indexed and read back {end:,}/{manifest["passages"]:,} passages', flush=True)
    status = store.finish_bulk()
    # Re-read every expected ID, not merely the server's approximate points_count.
    for batch in pq.ParquetFile(data / 'corpus.parquet').iter_batches(batch_size=512, columns=['pid', 'text']):
        rows = batch.to_pylist()
        records = store.verify_ids([p['pid'] for p in rows], build_id)
        expected = {p['pid']: p['text'] for p in rows}
        if any(r.payload['text'] != expected[r.id] for r in records):
            raise ValueError('Indexed passage text differs from the frozen corpus')
    elapsed = prior_seconds if progress.get('complete') else prior_seconds + time.perf_counter() - session_started
    if status['points'] != manifest['passages']:
        raise RuntimeError('Final exact index count does not equal the corpus count')
    stats = {**status, 'build_id': build_id, 'passages': manifest['passages'], 'seconds': elapsed,
             'dense_device': device('dense'),
             'model_provenance': model_provenance,
             'sampled_python_rss_peak_bytes': peak_rss,
             'passages_per_sec': manifest['passages'] / elapsed, 'under_two_hours': elapsed < 7200,
             'environment': environment(), 'corpus_bytes': (data / 'corpus.parquet').stat().st_size,
             'index_disk_bytes': sum(p.stat().st_size for p in project_path('qdrant_storage').rglob('*') if p.is_file()),
             'notes': 'Elapsed includes model loading, encoding, writes, validation and HNSW optimization; interrupted downtime excluded.'}
    checkpoint(reports / 'checks', '05_index_recalculation', stats)
    write_json(reports / 'ingest_stats.json', stats)
    write_json(progress_path, {**progress, 'complete': True, 'elapsed_seconds': elapsed})
    return stats


if __name__ == '__main__':
    import portalocker
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()
    cfg = load_config()
    lock_path = project_path(cfg['data_dir']) / f'ingest_{cfg["collection"]}.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with portalocker.Lock(str(lock_path), timeout=0):
        print(ingest(cfg))
