"""Unit tests for src.data_loader — offline fixtures only."""

import pytest

from src.data_loader import load_from_bytes, load_txt, load_document, MAX_UPLOAD_BYTES


def test_txt_from_bytes_utf8():
    content = "Héllo café — UTF-8 \u2713".encode("utf-8")
    text = load_from_bytes(content, "note.txt")
    assert "Héllo" in text
    assert "café" in text


def test_txt_round_trip(tmp_path):
    path = tmp_path / "sample.txt"
    original = "Line one\nLine two"
    path.write_text(original, encoding="utf-8")
    assert load_txt(path) == original
    assert load_document(path) == original


def test_empty_txt_bytes_raises():
    with pytest.raises(ValueError, match="empty|no readable"):
        load_from_bytes(b"", "empty.txt")


def test_whitespace_only_txt_bytes_raises():
    with pytest.raises(ValueError, match="empty|no readable"):
        load_from_bytes(b"   \n\t  ", "blank.txt")


def test_unsupported_extension():
    with pytest.raises(ValueError, match="Unsupported"):
        load_from_bytes(b"data", "file.docx")


def test_empty_bytes_none_like():
    with pytest.raises(ValueError, match="empty"):
        load_from_bytes(b"", "x.pdf")


def _minimal_pdf_with_text(text: str) -> bytes:
    """Build a minimal PDF bytes fixture with a simple text drawing operator."""
    return b"""%PDF-1.4
1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj
2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 >>endobj
3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R /Resources<< /Font<< /F1 5 0 R >> >> >>endobj
4 0 obj<< /Length 44 >>stream
BT /F1 12 Tf 50 150 Td (Hello) Tj ET
endstream
endobj
5 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000266 00000 n 
0000000361 00000 n 
trailer<< /Size 6 /Root 1 0 R >>
startxref
440
%%EOF
"""


def test_minimal_pdf_extracts_or_errors_cleanly():
    """Minimal PDF fixture: either extracts text or raises ValueError (no crash)."""
    pdf_bytes = _minimal_pdf_with_text("Hello")
    try:
        text = load_from_bytes(pdf_bytes, "mini.pdf")
        assert isinstance(text, str)
        assert len(text) > 0
    except ValueError as e:
        assert "extractable" in str(e).lower() or "parse" in str(e).lower() or "pdf" in str(e).lower()


def test_malformed_pdf_raises():
    with pytest.raises(ValueError, match="parse|PDF|extractable"):
        load_from_bytes(b"not-a-pdf-at-all", "bad.pdf")


def test_oversized_upload_guard():
    big = b"x" * (MAX_UPLOAD_BYTES + 1)
    with pytest.raises(ValueError, match="maximum|size"):
        load_from_bytes(big, "huge.txt")
