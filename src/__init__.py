"""LLM Document Q&A and Summarization package.

Heavy dependencies are imported lazily so lightweight modules can be tested
without downloading models.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "load_document",
    "chunk_text",
    "EmbeddingModel",
    "VectorStore",
    "LLMPipeline",
    "evaluate_relevance",
]


def __getattr__(name: str) -> Any:
    if name == "load_document":
        from .data_loader import load_document
        return load_document
    if name == "chunk_text":
        from .chunking import chunk_text
        return chunk_text
    if name == "EmbeddingModel":
        from .embeddings import EmbeddingModel
        return EmbeddingModel
    if name == "VectorStore":
        from .retriever import VectorStore
        return VectorStore
    if name == "LLMPipeline":
        from .llm_pipeline import LLMPipeline
        return LLMPipeline
    if name == "evaluate_relevance":
        from .evaluation import evaluate_relevance
        return evaluate_relevance
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
