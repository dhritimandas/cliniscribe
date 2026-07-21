"""Tests for src/model_registry.py — process-lifetime residency for the fast
engine (latency Wave 2). No real models load: pyannote's Pipeline.from_pretrained
and l3_5_normalize._EmbeddingBackend are both stubbed with cheap sentinels.
"""

import pytest

from src import model_registry


@pytest.fixture(autouse=True)
def _reset_registry():
    """Every test starts and ends with a clean registry — global module state
    would otherwise leak a resident "model" (a test double) across tests."""
    model_registry.reset()
    yield
    model_registry.reset()


class _FakePipeline:
    def __init__(self):
        self.to_calls = []

    def to(self, device):
        self.to_calls.append(device)


def test_get_pyannote_pipeline_loads_once_and_caches(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    calls = []

    def fake_from_pretrained(model_id, token=None):
        calls.append((model_id, token))
        return _FakePipeline()

    monkeypatch.setattr("pyannote.audio.Pipeline.from_pretrained", fake_from_pretrained)

    first = model_registry.get_pyannote_pipeline()
    second = model_registry.get_pyannote_pipeline()

    assert first is second
    assert len(calls) == 1  # loaded once, not once per call


def test_get_pyannote_pipeline_requires_hf_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(EnvironmentError):
        model_registry.get_pyannote_pipeline()


class _FakeEmbeddingBackend:
    _instances = 0

    def __init__(self):
        _FakeEmbeddingBackend._instances += 1
        self.released = False

    def release(self):
        self.released = True


def test_get_embedding_backend_loads_once_and_caches(monkeypatch):
    _FakeEmbeddingBackend._instances = 0
    monkeypatch.setattr(
        "src.l3_5_normalize._EmbeddingBackend", _FakeEmbeddingBackend
    )

    first = model_registry.get_embedding_backend()
    second = model_registry.get_embedding_backend()

    assert first is second
    assert _FakeEmbeddingBackend._instances == 1


def test_reset_releases_embedding_backend_and_clears_cache(monkeypatch):
    monkeypatch.setattr(
        "src.l3_5_normalize._EmbeddingBackend", _FakeEmbeddingBackend
    )

    backend = model_registry.get_embedding_backend()
    model_registry.reset()

    assert backend.released is True


def test_reset_forces_reload_on_next_get(monkeypatch):
    _FakeEmbeddingBackend._instances = 0
    monkeypatch.setattr(
        "src.l3_5_normalize._EmbeddingBackend", _FakeEmbeddingBackend
    )

    first = model_registry.get_embedding_backend()
    model_registry.reset()
    second = model_registry.get_embedding_backend()

    assert first is not second
    assert _FakeEmbeddingBackend._instances == 2
