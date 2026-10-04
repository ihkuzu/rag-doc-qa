from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class Page:
    source: str  # file name
    number: int  # starts at 1
    text: str


def load_pdf(path: str | Path) -> list[Page]:
    path = Path(path)
    reader = PdfReader(str(path))
    pages: list[Page] = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        # blank pages and scans without a text layer are dropped
        if text:
            pages.append(Page(source=path.name, number=number, text=text))
    return pages


def load_directory(directory: str | Path) -> list[Page]:
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a directory")
    pages: list[Page] = []
    for pdf in sorted(directory.glob("*.pdf")):
        pages.extend(load_pdf(pdf))
    return pages


def load_path(path: str | Path) -> list[Page]:
    target = Path(path)
    return load_pdf(target) if target.is_file() else load_directory(target)
