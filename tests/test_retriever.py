"""Unit tests for VectorStore with FakeEmbeddingModel (no HF downloads)."""

import pytest

from src.retriever import VectorStore
from tests.conftest import FakeEmbeddingModel


def test_empty_store_search(fake_embedder):
    vs = VectorStore(embedding_model=fake_embedder)
    assert vs.search("anything") == []
    assert vs.get_context("anything") == ""
    assert vs.size == 0


def test_build_rejects_empty(fake_embedder):
    vs = VectorStore(embedding_model=fake_embedder)
    with pytest.raises(ValueError, match="empty"):
        vs.build([])


def test_build_and_search(fake_embedder):
    vs = VectorStore(embedding_model=fake_embedder)
    chunks = [
        "alpha beta gamma",
        "delta epsilon",
        "alpha alpha alpha",
    ]
    vs.build(chunks)
    assert vs.size == 3
    results = vs.search("alpha", top_k=2)
    assert len(results) == 2
    assert all(isinstance(c, str) and isinstance(s, float) for c, s in results)


def test_top_k_respected(fake_embedder):
    vs = VectorStore(embedding_model=fake_embedder)
    vs.build([f"chunk number {i}" for i in range(10)])
    assert len(vs.search("chunk", top_k=3)) == 3
    assert len(vs.search("chunk", top_k=1)) == 1


def test_ordering_descending_scores(fake_embedder):
    vs = VectorStore(embedding_model=fake_embedder)
    vs.build(["zzzz unique", "query query query", "other"])
    results = vs.search("query query", top_k=3)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_add_to_existing(fake_embedder):
    vs = VectorStore(embedding_model=fake_embedder)
    vs.build(["first document"])
    vs.add(["second document"])
    assert vs.size == 2
    results = vs.search("second", top_k=2)
    assert len(results) >= 1


def test_get_context_joins(fake_embedder):
    vs = VectorStore(embedding_model=fake_embedder)
    vs.build(["part one content", "part two content"])
    ctx = vs.get_context("content", top_k=2)
    assert "---" in ctx or len(ctx) > 0


def test_repeated_build_replaces(fake_embedder):
    vs = VectorStore(embedding_model=fake_embedder)
    vs.build(["old"])
    vs.build(["new only"])
    assert vs.size == 1
    assert vs.chunks == ["new only"]


def test_dimension_matches_fake(fake_embedder):
    vs = VectorStore(embedding_model=fake_embedder)
    assert vs.dimension == fake_embedder.dimension
    vs.build(["x"])
    assert vs.index.d == fake_embedder.dimension
