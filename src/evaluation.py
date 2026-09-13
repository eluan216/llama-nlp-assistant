"""Evaluation helpers for RAG quality."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
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
    """Basic relevance evaluation of retrieved chunks."""
    if not retrieved_chunks:
        return {
            "num_retrieved": 0,
            "avg_score": 0.0,
            "max_score": 0.0,
            "min_score": 0.0,
            "num_above_threshold": 0,
            "relevance_ratio": 0.0,
            "query_token_coverage": 0.0,
        }

    scores = [score for _, score in retrieved_chunks]
    above = sum(1 for s in scores if s >= threshold)

    query_tokens = set(tokenize(query))
    covered = 0
    if query_tokens:
        chunk_tokens = set()
        for chunk, _ in retrieved_chunks:
            chunk_tokens |= set(tokenize(chunk))
        covered = len(query_tokens & chunk_tokens) / len(query_tokens)

    return {
        "num_retrieved": len(retrieved_chunks),
        "avg_score": sum(scores) / len(scores),
        "max_score": max(scores),
        "min_score": min(scores),
        "num_above_threshold": above,
        "relevance_ratio": above / len(scores),
        "query_token_coverage": covered,
    }


def answer_overlap(answer: str, context: str) -> float:
    """Measure how much of the answer is grounded in the context (token overlap)."""
    return jaccard_similarity(answer, context)


def simple_rouge_l(reference: str, candidate: str) -> float:
    """Simplified ROUGE-L approximation using LCS length ratio."""
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)

    if not ref_tokens or not cand_tokens:
        return 0.0

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


def evaluate_answer(
    question: str,
    answer: str,
    context: str,
    retrieved_chunks: List[Tuple[str, float]] | None = None,
) -> Dict[str, Any]:
    """Aggregate retrieval + grounding metrics for a single Q&A turn."""
    retrieved_chunks = retrieved_chunks or []
    relevance = evaluate_relevance(question, retrieved_chunks)
    grounding = answer_overlap(answer, context)
    answer_len = len(tokenize(answer))
    context_len = len(tokenize(context))

    return {
        "retrieval": relevance,
        "grounding_jaccard": grounding,
        "answer_token_count": answer_len,
        "context_token_count": context_len,
        "answer_too_short": answer_len < 5,
        "likely_ungrounded": grounding < 0.05 and answer_len > 10,
    }


def evaluate_chat_history(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize metrics across multi-turn chat history."""
    qa_turns = [t for t in history if t.get("role") == "assistant" and "metrics" in t]
    if not qa_turns:
        return {"turns": 0}

    groundings = [t["metrics"].get("grounding_jaccard", 0.0) for t in qa_turns]
    avg_retrieval = [
        t["metrics"].get("retrieval", {}).get("avg_score", 0.0) for t in qa_turns
    ]

    return {
        "turns": len(qa_turns),
        "avg_grounding": sum(groundings) / len(groundings),
        "avg_retrieval_score": sum(avg_retrieval) / len(avg_retrieval),
        "ungrounded_answers": sum(
            1 for t in qa_turns if t["metrics"].get("likely_ungrounded")
        ),
    }
