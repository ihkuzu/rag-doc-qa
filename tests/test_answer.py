import pytest

from conftest import FakeLLM
from ragqa.answer import answer
from ragqa.embedders import HashingEmbedder
from ragqa.ingest import ingest
from ragqa.store import PgVectorStore

PAGES = [
    "The hydraulic pump must be inspected every 200 operating hours.",
    "Employees receive thirty vacation days per year.",
    "Backups run nightly at 02:00 and are kept for ninety days.",
]


@pytest.fixture
def indexed(clean_db, pdf_factory, tmp_path):
    embedder = HashingEmbedder()
    pdf_factory("handbook.pdf", PAGES)
    with PgVectorStore(clean_db, embedder.dimension, embedder.name) as store:
        ingest(tmp_path, store, embedder)
        yield store, embedder


def test_prompt_contains_the_passages_and_the_question(indexed):
    store, embedder = indexed
    llm = FakeLLM()

    answer("How many vacation days do employees receive?", store, embedder, llm, k=2)

    system, prompt = llm.calls[0]
    assert "only the numbered passages" in system
    assert "[1] (handbook.pdf, page 2)" in prompt
    assert "thirty vacation days" in prompt
    assert prompt.endswith("Question: How many vacation days do employees receive?")


def test_cited_sources_are_marked(indexed):
    store, embedder = indexed
    llm = FakeLLM("Thirty days [1].")

    result = answer("How many vacation days do employees receive?", store, embedder, llm, k=3)

    assert result.text == "Thirty days [1]."
    assert [s.cited for s in result.sources] == [True, False, False]
    assert result.sources[0].page == 2


def test_citations_outside_the_retrieved_range_are_ignored(indexed):
    store, embedder = indexed

    result = answer("vacation days", store, embedder, FakeLLM("See [1] and [9]."), k=2)

    assert [s.number for s in result.sources if s.cited] == [1]


def test_empty_index_skips_the_model(clean_db):
    embedder = HashingEmbedder()
    llm = FakeLLM()
    with PgVectorStore(clean_db, embedder.dimension, embedder.name) as store:
        result = answer("anything", store, embedder, llm)

    assert result.sources == []
    assert llm.calls == []
