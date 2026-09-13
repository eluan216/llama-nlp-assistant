"""FAISS-based vector store and retriever."""

from typing import List, Optional, Tuple

import faiss
import numpy as np

from .embeddings import EmbeddingModel


class VectorStore:
    """Simple FAISS IndexFlatIP (inner product) store with metadata."""

    def __init__(self, embedding_model: Optional[EmbeddingModel] = None):
        self.embedding_model = embedding_model or EmbeddingModel()
        self.index: Optional[faiss.Index] = None
        self.chunks: List[str] = []
        self.dimension = self.embedding_model.dimension

    def build(self, chunks: List[str]) -> None:
        """Embed chunks and build the FAISS index.

        Args:
            chunks: List of text chunks to index.
        """
        if not chunks:
            raise ValueError("Cannot build vector store from empty chunk list.")

        self.chunks = chunks
        embeddings = self.embedding_model.embed(chunks)

        # Use inner product on normalized vectors = cosine similarity
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

    def add(self, new_chunks: List[str]) -> None:
        """Add more chunks to an existing index."""
        if not new_chunks:
            return

        if self.index is None:
            self.build(new_chunks)
            return

        embeddings = self.embedding_model.embed(new_chunks)
        self.index.add(embeddings)
        self.chunks.extend(new_chunks)

    def search(
        self, query: str, top_k: int = 4
    ) -> List[Tuple[str, float]]:
        """Retrieve the most relevant chunks for a query.

        Args:
            query: User question or search text.
            top_k: Number of results to return.

        Returns:
            List of (chunk_text, score) tuples sorted by descending relevance.
        """
        if self.index is None or not self.chunks:
            return []

        query_vec = self.embedding_model.embed_query(query).reshape(1, -1)
        scores, indices = self.index.search(query_vec, min(top_k, len(self.chunks)))

        results: List[Tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for missing
                continue
            results.append((self.chunks[idx], float(score)))

        return results

    def get_context(self, query: str, top_k: int = 4) -> str:
        """Return concatenated top-k chunks as a single context string."""
        results = self.search(query, top_k=top_k)
        if not results:
            return ""
        return "\n\n---\n\n".join(chunk for chunk, _ in results)

    @property
    def size(self) -> int:
        return len(self.chunks)
