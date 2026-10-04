from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .embedders import Embedder
from .store import PgVectorStore


@dataclass(frozen=True)
class Question:
    text: str
    source: str
    page: int
    kind: str = "keyword"


@dataclass(frozen=True)
class Result:
    question: Question
    rank: int | None  # 1-based position of the expected page, None if it was not retrieved
    top: str  # what was ranked first, e.g. "handbook.pdf p.2"


def load_questions(path: str | Path) -> list[Question]:
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Question(item["question"], item["source"], item["page"], item.get("kind", "keyword"))
        for item in items
    ]


def evaluate(
    questions: Sequence[Question], store: PgVectorStore, embedder: Embedder, k: int = 5
) -> list[Result]:
    vectors = embedder.embed([q.text for q in questions])
    results: list[Result] = []
    for question, vector in zip(questions, vectors, strict=True):
        hits = store.search(vector, k)
        rank = next(
            (
                position
                for position, hit in enumerate(hits, start=1)
                if hit.chunk.source == question.source and hit.chunk.page == question.page
            ),
            None,
        )
        top = f"{hits[0].chunk.source} p.{hits[0].chunk.page}" if hits else "-"
        results.append(Result(question, rank, top))
    return results


def hit_rate(results: Sequence[Result], k: int) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.rank is not None and r.rank <= k) / len(results)


def mrr(results: Sequence[Result]) -> float:
    if not results:
        return 0.0
    return sum(1 / r.rank for r in results if r.rank) / len(results)


def format_report(results: Sequence[Result], k: int = 5) -> str:
    ks = sorted({n for n in (1, 3, k) if n <= k})

    def row(label: str, subset: Sequence[Result]) -> str:
        cells = "  ".join(f"hit@{n} {hit_rate(subset, n):.2f}" for n in ks)
        return f"{label:<11} n={len(subset):<3} {cells}  mrr {mrr(subset):.2f}"

    lines = [row("all", results)]
    for kind in sorted({r.question.kind for r in results}):
        lines.append(row(kind, [r for r in results if r.question.kind == kind]))

    misses = [r for r in results if r.rank != 1]
    if misses:
        lines.append("")
        lines.append("not ranked first:")
    for r in misses:
        found = f"rank {r.rank}" if r.rank else f"not in top {k}"
        lines.append(
            f"  - {r.question.text} (expected {r.question.source} p.{r.question.page}, "
            f"{found}, top: {r.top})"
        )
    return "\n".join(lines)
