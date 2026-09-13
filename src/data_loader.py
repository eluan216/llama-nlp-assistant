"""Document loading utilities for PDF and plain text files."""

from pathlib import Path
from typing import Union

from pypdf import PdfReader


def load_pdf(file_path: Union[str, Path]) -> str:
    """Extract text from a PDF file.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Extracted text as a single string.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())

    if not pages:
        raise ValueError(f"No extractable text found in PDF: {path}")

    return "\n\n".join(pages)


def load_txt(file_path: Union[str, Path]) -> str:
    """Load plain text from a .txt file.

    Args:
        file_path: Path to the text file.

    Returns:
        File contents as a string.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {path}")

    return path.read_text(encoding="utf-8", errors="ignore").strip()


def load_document(file_path: Union[str, Path]) -> str:
    """Load a document based on its file extension.

    Supports `.pdf` and `.txt`.

    Args:
        file_path: Path to the document.

    Returns:
        Extracted / loaded text.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(path)
    if suffix == ".txt":
        return load_txt(path)

    raise ValueError(
        f"Unsupported file type: {suffix}. Supported types: .pdf, .txt"
    )


def load_from_bytes(file_bytes: bytes, filename: str) -> str:
    """Load document content from in-memory bytes (useful for Streamlit uploads).

    Args:
        file_bytes: Raw file content.
        filename: Original filename (used to determine type).

    Returns:
        Extracted text.
    """
    suffix = Path(filename).suffix.lower()

    if suffix == ".txt":
        return file_bytes.decode("utf-8", errors="ignore").strip()

    if suffix == ".pdf":
        from io import BytesIO

        reader = PdfReader(BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())

        if not pages:
            raise ValueError("No extractable text found in the uploaded PDF.")

        return "\n\n".join(pages)

    raise ValueError(f"Unsupported file type: {suffix}. Use .pdf or .txt")
