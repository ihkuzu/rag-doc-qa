"""Read PDF files into page-level text records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class Page:
    """Text of a single PDF page, with enough info to cite it later."""

    source: str  # file name, e.g. "manual.pdf"
    number: int  # 1-based page number
    text: str


def load_pdf(path: str | Path) -> list[Page]:
    """Extract the text of every page in a PDF.

    Pages without extractable text (blank pages, scans without OCR) are skipped.
    """
    path = Path(path)
    reader = PdfReader(str(path))
    pages: list[Page] = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(Page(source=path.name, number=number, text=text))
    return pages


def load_directory(directory: str | Path) -> list[Page]:
    """Load all PDFs in a directory (non-recursive), sorted by file name."""
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a directory")
    pages: list[Page] = []
    for pdf in sorted(directory.glob("*.pdf")):
        pages.extend(load_pdf(pdf))
    return pages
