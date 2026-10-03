"""Explicit ONNX device selection; a requested GPU may not silently fall back."""
import os
from functools import lru_cache


def device(component):
    value = os.getenv(f'PRAG_{component.upper()}_DEVICE', 'cpu').lower()
    if value not in {'cpu', 'cuda'}:
        raise ValueError(f'Invalid {component} device: {value}; use cpu or cuda')
    return value


@lru_cache(maxsize=1)
def preload_cuda():
    import onnxruntime as ort
    if 'CUDAExecutionProvider' not in ort.get_available_providers():
        raise RuntimeError('CUDA requires the separate GPU environment; see docs/gpu.md')
    ort.preload_dlls(directory='')


def providers(component):
    if device(component) == 'cpu':
        return ['CPUExecutionProvider']
    preload_cuda()
    return [('CUDAExecutionProvider', {'device_id': 0, 'gpu_mem_limit': 2 * 1024**3,
                                       'arena_extend_strategy': 'kSameAsRequested', 'use_tf32': 0}),
            'CPUExecutionProvider']


def verify_device(instance, component):
    actual = instance.model.model.get_providers()
    if device(component) == 'cuda' and 'CUDAExecutionProvider' not in actual:
        raise RuntimeError(f'{component}: requested CUDA failed to initialize; providers={actual}')
    return instance
