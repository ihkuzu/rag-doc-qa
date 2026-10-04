from __future__ import annotations

import re
from dataclasses import dataclass

from .loader import Page

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    source: str
    page: int
    index: int  # position within the page, from 0
    text: str


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_sentences(text: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        sentence = sentence.strip()
        # an over-long sentence is cut at the last space before the limit
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            pieces.append(sentence)
    return pieces


def _overlap_tail(sentences: list[str], overlap_chars: int) -> list[str]:
    tail: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        cost = len(sentence) + (1 if tail else 0)
        if total + cost > overlap_chars:
            break
        tail.insert(0, sentence)
        total += cost
    return tail


def chunk_text(text: str, max_chars: int = 800, overlap_chars: int = 150) -> list[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not 0 <= overlap_chars < max_chars:
        raise ValueError("overlap_chars must be >= 0 and smaller than max_chars")

    sentences = _split_sentences(_normalize(text), max_chars)
    chunks: list[str] = []
    current: list[str] = []

    for sentence in sentences:
        if current and len(" ".join(current + [sentence])) > max_chars:
            chunks.append(" ".join(current))
            current = _overlap_tail(current, overlap_chars)
            # the carried-over tail must leave room for the new sentence
            while current and len(" ".join(current + [sentence])) > max_chars:
                current.pop(0)
        current.append(sentence)

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_pages(
    pages: list[Page], max_chars: int = 800, overlap_chars: int = 150
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        for index, text in enumerate(chunk_text(page.text, max_chars, overlap_chars)):
            chunks.append(Chunk(source=page.source, page=page.number, index=index, text=text))
    return chunks
