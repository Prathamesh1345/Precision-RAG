from types import SimpleNamespace

import pytest

from app.runtime import device, verify_device


def test_requested_gpu_cannot_silently_fall_back(monkeypatch):
    monkeypatch.setenv('PRAG_RERANK_DEVICE', 'cuda')
    instance = SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(
        get_providers=lambda: ['CPUExecutionProvider'])))
    with pytest.raises(RuntimeError, match='failed to initialize'):
        verify_device(instance, 'rerank')


def test_invalid_device_is_rejected(monkeypatch):
    monkeypatch.setenv('PRAG_DENSE_DEVICE', 'invalid')
    with pytest.raises(ValueError, match='Invalid dense device'):
        device('dense')
