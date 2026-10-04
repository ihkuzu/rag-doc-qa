from pathlib import Path

import pytest

from ragqa.embedders import HashingEmbedder
from ragqa.evaluation import (
    Question,
    Result,
    evaluate,
    format_report,
    hit_rate,
    load_questions,
    mrr,
)
from ragqa.ingest import ingest
from ragqa.loader import load_path
from ragqa.store import PgVectorStore

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "sample"
QUESTIONS = ROOT / "data" / "eval" / "questions.json"


def result(rank, kind="keyword"):
    return Result(Question("q", "a.pdf", 1, kind), rank, "a.pdf p.1")


def test_hit_rate_counts_ranks_within_k():
    results = [result(1), result(3), result(5), result(None)]

    assert hit_rate(results, 1) == 0.25
    assert hit_rate(results, 3) == 0.5
    assert hit_rate(results, 5) == 0.75


def test_mrr_averages_reciprocal_ranks():
    assert mrr([result(1), result(2), result(None)]) == pytest.approx((1 + 0.5 + 0) / 3)


def test_empty_results_give_zero():
    assert hit_rate([], 3) == 0.0
    assert mrr([]) == 0.0


def test_report_splits_by_kind_and_lists_misses():
    report = format_report([result(1), result(2, "paraphrase"), result(None, "paraphrase")], k=3)

    assert "all" in report
    assert "keyword" in report and "paraphrase" in report
    assert "rank 2" in report
    assert "not in top 3" in report


def test_evaluate_finds_the_expected_page(clean_db, pdf_factory, tmp_path):
    embedder = HashingEmbedder()
    pdf_factory("handbook.pdf", ["Backups run nightly.", "Employees receive thirty vacation days."])
    with PgVectorStore(clean_db, embedder.dimension, embedder.name) as store:
        ingest(tmp_path, store, embedder)
        results = evaluate(
            [
                Question("How many vacation days do employees receive?", "handbook.pdf", 2),
                Question("When do backups run?", "handbook.pdf", 2),
            ],
            store,
            embedder,
            k=2,
        )

    assert results[0].rank == 1
    assert results[1].rank == 2


def test_every_question_points_at_an_existing_page():
    pages = {(p.source, p.number) for p in load_path(SAMPLE)}
    questions = load_questions(QUESTIONS)

    assert len(questions) >= 30
    assert all((q.source, q.page) in pages for q in questions)
    assert {q.kind for q in questions} == {"keyword", "paraphrase"}


def test_keyword_baseline_does_not_regress(clean_db):
    embedder = HashingEmbedder()
    with PgVectorStore(clean_db, embedder.dimension, embedder.name) as store:
        ingest(SAMPLE, store, embedder)
        results = evaluate(load_questions(QUESTIONS), store, embedder, k=5)

    keyword = [r for r in results if r.question.kind == "keyword"]
    assert hit_rate(keyword, 3) >= 0.8
