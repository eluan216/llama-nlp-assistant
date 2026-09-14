"""Unit tests for src.chunking — documents existing behavior; does not alter it."""

from src.chunking import chunk_text, get_chunk_stats


def test_empty_input():
    assert chunk_text("") == []
    assert chunk_text("   \n\t  ") == []


def test_short_text_single_chunk():
    text = "Hello world."
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    assert "Hello" in chunks[0]


def test_text_smaller_than_chunk_size():
    text = "abc"
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 1
    assert chunks[0] == "abc"


def test_multiple_chunks_produced():
    text = "Paragraph one. " * 40
    chunks = chunk_text(text, chunk_size=80, chunk_overlap=10)
    assert len(chunks) >= 2


def test_deterministic_output():
    text = "Line A\n\nLine B\n\nLine C " * 20
    a = chunk_text(text, chunk_size=60, chunk_overlap=5)
    b = chunk_text(text, chunk_size=60, chunk_overlap=5)
    assert a == b


def test_overlap_behavior_when_multiple_chunks():
    text = "AAAA " * 50 + "BBBB " * 50
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 2


def test_zero_overlap():
    text = "word " * 100
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=0)
    assert len(chunks) >= 1
    for c in chunks:
        assert c.strip()


def test_separator_paragraph_breaks():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = chunk_text(text, chunk_size=40, chunk_overlap=0)
    assert len(chunks) >= 1
    joined = " ".join(chunks)
    assert "First" in joined
    assert "Third" in joined


def test_get_chunk_stats_empty():
    stats = get_chunk_stats([])
    assert stats["count"] == 0
    assert stats["avg_len"] == 0


def test_get_chunk_stats_nonempty():
    chunks = ["aa", "bbbb"]
    stats = get_chunk_stats(chunks)
    assert stats["count"] == 2
    assert stats["min_len"] == 2
    assert stats["max_len"] == 4
    assert stats["avg_len"] == 3.0


def test_pathological_tiny_chunk_size():
    text = "abcdefghij"
    chunks = chunk_text(text, chunk_size=1, chunk_overlap=0)
    assert isinstance(chunks, list)
    assert all(isinstance(c, str) for c in chunks)
