"""Simple evaluation helpers for RAG quality."""

from typing import List, Tuple
import re


def tokenize(text: str) -> List[str]:
    """Very lightweight tokenizer for overlap metrics."""
    return re.findall(r"\w+", text.lower())


def jaccard_similarity(a: str, b: str) -> float:
    """Jaccard similarity between two texts based on word tokens."""
    set_a = set(tokenize(a))
    set_b = set(tokenize(b))
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def evaluate_relevance(
    query: str, retrieved_chunks: List[Tuple[str, float]], threshold: float = 0.3
) -> dict:
    """Basic relevance evaluation of retrieved chunks.

    Args:
        query: The user question.
        retrieved_chunks: List of (chunk, score) from the retriever.
        threshold: Minimum score considered "relevant".

    Returns:
        Dictionary with simple metrics.
    """
    if not retrieved_chunks:
        return {
            "num_retrieved": 0,
            "avg_score": 0.0,
            "max_score": 0.0,
            "num_above_threshold": 0,
            "relevance_ratio": 0.0,
        }

    scores = [score for _, score in retrieved_chunks]
    above = sum(1 for s in scores if s >= threshold)

    return {
        "num_retrieved": len(retrieved_chunks),
        "avg_score": sum(scores) / len(scores),
        "max_score": max(scores),
        "num_above_threshold": above,
        "relevance_ratio": above / len(scores),
    }


def answer_overlap(answer: str, context: str) -> float:
    """Measure how much of the answer is grounded in the context (token overlap)."""
    return jaccard_similarity(answer, context)


def simple_rouge_l(reference: str, candidate: str) -> float:
    """Extremely simplified ROUGE-L approximation using longest common subsequence length ratio."""
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)

    if not ref_tokens or not cand_tokens:
        return 0.0

    # LCS length via DP
    m, n = len(ref_tokens), len(cand_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == cand_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[m][n]
    precision = lcs / n if n else 0.0
    recall = lcs / m if m else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
