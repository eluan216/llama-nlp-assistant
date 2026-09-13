"""Text chunking utilities for RAG."""

from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: List[str] | None = None,
) -> List[str]:
    """Split text into overlapping chunks using a recursive character approach.

    This is a lightweight pure-Python implementation inspired by
    LangChain's RecursiveCharacterTextSplitter.

    Args:
        text: Full document text.
        chunk_size: Target maximum characters per chunk.
        chunk_overlap: Number of characters to overlap between consecutive chunks.
        separators: Ordered list of separators to try (largest to smallest).

    Returns:
        List of text chunks.
    """
    if not text or not text.strip():
        return []

    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    def _split(text: str, seps: List[str]) -> List[str]:
        if not seps:
            return [
                text[i : i + chunk_size]
                for i in range(0, len(text), max(1, chunk_size - chunk_overlap))
            ]

        sep = seps[0]
        remaining_seps = seps[1:]

        if sep == "":
            return [
                text[i : i + chunk_size]
                for i in range(0, len(text), max(1, chunk_size - chunk_overlap))
            ]

        parts = text.split(sep)
        chunks: List[str] = []
        current = ""

        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                if len(part) > chunk_size:
                    chunks.extend(_split(part, remaining_seps))
                    current = ""
                else:
                    current = part

        if current.strip():
            chunks.append(current.strip())

        return chunks

    raw_chunks = _split(text.strip(), separators)

    if chunk_overlap <= 0 or len(raw_chunks) <= 1:
        return [c for c in raw_chunks if c]

    final_chunks: List[str] = []
    for i, chunk in enumerate(raw_chunks):
        if i == 0:
            final_chunks.append(chunk)
            continue

        prev = raw_chunks[i - 1]
        overlap_text = prev[-chunk_overlap:] if len(prev) > chunk_overlap else prev
        combined = (overlap_text + " " + chunk).strip()
        final_chunks.append(combined)

    return [c for c in final_chunks if c]


def get_chunk_stats(chunks: List[str]) -> dict:
    """Return basic statistics about the produced chunks."""
    if not chunks:
        return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}

    lengths = [len(c) for c in chunks]
    return {
        "count": len(chunks),
        "avg_len": sum(lengths) / len(lengths),
        "min_len": min(lengths),
        "max_len": max(lengths),
    }
