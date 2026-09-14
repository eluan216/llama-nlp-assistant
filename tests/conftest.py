"""Shared fixtures for unit tests. No network / no HF model downloads."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FakeEmbeddingModel:
    """Deterministic bag-of-characters embedding for offline retriever tests."""

    def __init__(self, dimension: int = 8):
        self.dimension = dimension
        self.model_name = "fake-embedding"

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        if not text:
            return vec
        for i, ch in enumerate(text.encode("utf-8", errors="ignore")):
            vec[i % self.dimension] += (ch % 31) / 31.0
        vec[0] += len(text) * 0.01
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)

    def embed(self, texts, batch_size: int = 32) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return np.vstack([self._vector(t) for t in texts]).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self._vector(query)


@pytest.fixture
def fake_embedder() -> FakeEmbeddingModel:
    return FakeEmbeddingModel(dimension=8)
