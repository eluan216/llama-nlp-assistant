"""Embedding model wrapper using sentence-transformers."""

from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Thin wrapper around a sentence-transformers model."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Args:
            model_name: Hugging Face model id. Default is a fast, high-quality
                        384-dimensional model (~90 MB).
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """Compute embeddings for one or more texts.

        Args:
            texts: Single string or list of strings.
            batch_size: Batch size for encoding.

        Returns:
            Numpy array of shape (n_texts, embedding_dim).
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 50,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine similarity friendly
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string (returns 1-D vector)."""
        return self.embed(query)[0]
