"""Unit tests for src.evaluation heuristics."""

from src.evaluation import (
    jaccard_similarity,
    simple_rouge_l,
    evaluate_relevance,
    evaluate_answer,
    evaluate_chat_history,
    answer_overlap,
)


def test_jaccard_empty_both():
    assert jaccard_similarity("", "") == 1.0


def test_jaccard_identical():
    assert jaccard_similarity("hello world", "hello world") == 1.0


def test_jaccard_disjoint():
    score = jaccard_similarity("alpha beta", "gamma delta")
    assert score == 0.0


def test_jaccard_partial():
    score = jaccard_similarity("the quick brown", "the slow brown")
    assert 0.0 < score < 1.0


def test_jaccard_deterministic():
    a, b = "one two three", "two three four"
    assert jaccard_similarity(a, b) == jaccard_similarity(a, b)


def test_rouge_l_identical():
    assert simple_rouge_l("a b c", "a b c") == 1.0


def test_rouge_l_empty():
    assert simple_rouge_l("", "x") == 0.0
    assert simple_rouge_l("x", "") == 0.0


def test_evaluate_relevance_empty():
    result = evaluate_relevance("query", [])
    assert result["num_retrieved"] == 0
    assert result["avg_score"] == 0.0
    assert "query_token_coverage" in result


def test_evaluate_relevance_structure():
    chunks = [("the cat sat", 0.9), ("dog ran", 0.2)]
    result = evaluate_relevance("cat", chunks, threshold=0.3)
    assert result["num_retrieved"] == 2
    assert result["max_score"] == 0.9
    assert result["min_score"] == 0.2
    assert result["num_above_threshold"] == 1
    assert 0.0 <= result["relevance_ratio"] <= 1.0


def test_evaluate_answer_structure():
    out = evaluate_answer(
        "What color?",
        "The cat is blue",
        "The cat is blue and soft",
        [("The cat is blue and soft", 0.8)],
    )
    assert "retrieval" in out
    assert "grounding_jaccard" in out
    assert "answer_token_count" in out
    assert "likely_ungrounded" in out
    assert isinstance(out["grounding_jaccard"], float)


def test_answer_overlap_matches_jaccard():
    a, c = "shared tokens here", "shared tokens elsewhere"
    assert answer_overlap(a, c) == jaccard_similarity(a, c)


def test_evaluate_chat_history_empty():
    assert evaluate_chat_history([]) == {"turns": 0}


def test_evaluate_chat_history_with_turns():
    history = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "a",
            "metrics": {
                "grounding_jaccard": 0.5,
                "retrieval": {"avg_score": 0.7},
                "likely_ungrounded": False,
            },
        },
    ]
    summary = evaluate_chat_history(history)
    assert summary["turns"] == 1
    assert summary["avg_grounding"] == 0.5
    assert summary["avg_retrieval_score"] == 0.7
