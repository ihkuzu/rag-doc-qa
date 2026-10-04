import pytest

from ragqa.embedders import HashingEmbedder
from ragqa.ingest import ingest, retrieve
from ragqa.store import PgVectorStore

PAGES = [
    "The hydraulic pump must be inspected every 200 operating hours and the seals replaced yearly.",
    "Employees receive thirty vacation days per year and may carry over five days.",
    "Backups run nightly at 02:00 and are kept for ninety days.",
]


@pytest.fixture
def store(clean_db):
    embedder = HashingEmbedder()
    with PgVectorStore(clean_db, embedder.dimension, embedder.name) as s:
        yield s


def test_ingest_indexes_every_chunk(store, pdf_factory, tmp_path):
    pdf_factory("handbook.pdf", PAGES)

    added = ingest(tmp_path, store, HashingEmbedder())

    assert added == 3
    assert store.count() == 3


def test_a_question_finds_the_right_page(store, pdf_factory, tmp_path):
    pdf_factory("handbook.pdf", PAGES)
    embedder = HashingEmbedder()
    ingest(tmp_path, store, embedder)

    hits = retrieve("How many vacation days do employees receive?", store, embedder, k=3)

    assert hits[0].chunk.page == 2
    assert hits[0].chunk.source == "handbook.pdf"


def test_ingesting_twice_does_not_duplicate(store, pdf_factory, tmp_path):
    pdf_factory("handbook.pdf", PAGES)
    embedder = HashingEmbedder()

    ingest(tmp_path, store, embedder)
    ingest(tmp_path, store, embedder)

    assert store.count() == 3


def test_reingesting_a_shorter_file_drops_the_old_pages(store, pdf_factory, tmp_path):
    embedder = HashingEmbedder()
    pdf_factory("handbook.pdf", PAGES)
    ingest(tmp_path, store, embedder)

    pdf_factory("handbook.pdf", PAGES[:1])
    ingest(tmp_path, store, embedder)

    assert store.count() == 1
