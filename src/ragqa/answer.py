from __future__ import annotations

import re
from dataclasses import dataclass

from .embedders import Embedder
from .ingest import retrieve
from .llm import LLM
from .store import PgVectorStore

SYSTEM_PROMPT = (
    "Answer the question using only the numbered passages. "
    "Cite the passages you used like [1] or [2]. "
    "If the passages do not contain the answer, say that you could not find it."
)

_CITATION = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class Source:
    number: int
    source: str
    page: int
    score: float
    text: str
    cited: bool


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[Source]


def build_prompt(question: str, passages: list[tuple[int, str, int, str]]) -> str:
    blocks = [f"[{n}] ({source}, page {page})\n{text}" for n, source, page, text in passages]
    return "Passages:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


def answer(
    question: str,
    store: PgVectorStore,
    embedder: Embedder,
    llm: LLM,
    k: int = 4,
) -> Answer:
    hits = retrieve(question, store, embedder, k)
    if not hits:
        return Answer("No documents are indexed yet.", [])

    passages = [(i, h.chunk.source, h.chunk.page, h.chunk.text) for i, h in enumerate(hits, 1)]
    text = llm.complete(SYSTEM_PROMPT, build_prompt(question, passages))

    # numbers outside the retrieved range are ignored
    cited = {int(n) for n in _CITATION.findall(text)} & set(range(1, len(hits) + 1))
    sources = [
        Source(i, h.chunk.source, h.chunk.page, h.score, h.chunk.text, i in cited)
        for i, h in enumerate(hits, 1)
    ]
    return Answer(text, sources)
