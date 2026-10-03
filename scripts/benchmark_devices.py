"""Measure the same frozen ONNX files on CPU/CUDA and audit actual kernel placement."""
import gc
import os
import time
from collections import Counter

import numpy as np
import pyarrow.parquet as pq

from app.artifacts import checkpoint, environment, read_json
from app.config import load_config, project_path
from app.models import ModelBundle
from app.runtime import providers


def benchmark():
    import onnxruntime as ort
    cfg = load_config()
    report_dir = project_path(cfg['reports_dir']) / 'checks'
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = next(pq.ParquetFile(project_path(cfg['data_dir']) / 'corpus.parquet').iter_batches(128)).to_pylist()
    texts = [p['text'] for p in sorted(rows, key=lambda p: len(p['text']))]
    query = 'What causes high blood pressure?'
    results, outputs = {}, {}
    for component in ['dense', 'rerank']:
        results[component], outputs[component] = {}, {}
        for target in ['cpu', 'cuda']:
            os.environ[f'PRAG_{component.upper()}_DEVICE'] = target
            bundle = ModelBundle(cfg)
            instance = bundle.dense if component == 'dense' else bundle.reranker
            model_path = instance.model.model._model_path
            run = (lambda: np.array(list(instance.passage_embed(texts, batch_size=32)))) if component == 'dense' else (
                lambda: np.array(list(instance.rerank(query, texts[:30], batch_size=16))))
            run()
            durations = []
            for _ in range(3):
                start = time.perf_counter()
                values = run()
                durations.append(time.perf_counter() - start)
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite inference output')
            outputs[component][target] = values
            # A provider appearing in get_providers() does not prove GPU execution.
            # Profile a separate untimed call and count actual kernel assignments.
            instance.model.model = None
            gc.collect()
            options = ort.SessionOptions()
            options.intra_op_num_threads = cfg['models']['threads']
            options.inter_op_num_threads = cfg['models']['threads']
            options.enable_profiling = True
            options.profile_file_prefix = str(report_dir / f'ort_{component}_{target}')
            instance.model.model = ort.InferenceSession(model_path, sess_options=options, providers=providers(component))
            run()
            profile = instance.model.model.end_profiling()
            events = read_json(project_path(profile))
            counts = Counter(e.get('args', {}).get('provider') for e in events
                             if e.get('cat') == 'Node' and e.get('args', {}).get('provider'))
            result = {'seconds': durations, 'median_seconds': float(np.median(durations)),
                      'items': len(values), 'session_providers': instance.model.model.get_providers(),
                      'kernel_events_by_provider': dict(counts), 'profile': profile,
                      'model_file': model_path}
            results[component][target] = result
            print(component, target, result, flush=True)
            del run, instance, bundle
            gc.collect()
        cpu, gpu = outputs[component]['cpu'], outputs[component]['cuda']
        results[component]['comparison'] = {
            'speedup': results[component]['cpu']['median_seconds'] / results[component]['cuda']['median_seconds'],
            'max_absolute_difference': float(np.max(np.abs(cpu - gpu))),
            'allclose_rtol_1e-3_atol_1e-4': bool(np.allclose(cpu, gpu, rtol=1e-3, atol=1e-4)),
            'identical_order': bool(np.array_equal(np.argsort(-cpu), np.argsort(-gpu))) if component == 'rerank' else None}
    return checkpoint(report_dir, 'device_benchmark', {'results': results, 'environment': environment(),
        'scope': 'Local microbenchmark, 3 warm repeats; concurrent ingestion may affect timings. Not the 100-query acceptance benchmark.'})


if __name__ == '__main__':
    print(benchmark())
