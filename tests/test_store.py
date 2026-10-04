import pytest

from ragqa.chunking import Chunk
from ragqa.store import PgVectorStore


def chunk(source, page, text, index=0):
    return Chunk(source=source, page=page, index=index, text=text)


def test_search_returns_the_closest_chunk_first(clean_db):
    with PgVectorStore(clean_db, dimension=3, embedder_name="test-3") as store:
        store.add(
            [chunk("a.pdf", 1, "x axis"), chunk("a.pdf", 2, "y axis"), chunk("a.pdf", 3, "z axis")],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        )
        hits = store.search([0.9, 0.1, 0], k=2)

    assert [h.chunk.page for h in hits] == [1, 2]
    assert hits[0].score > hits[1].score
    assert hits[0].score == pytest.approx(0.994, abs=0.01)


def test_adding_the_same_chunk_twice_updates_it(clean_db):
    with PgVectorStore(clean_db, dimension=3, embedder_name="test-3") as store:
        store.add([chunk("a.pdf", 1, "old text")], [[1, 0, 0]])
        store.add([chunk("a.pdf", 1, "new text")], [[1, 0, 0]])

        assert store.count() == 1
        assert store.search([1, 0, 0], k=1)[0].chunk.text == "new text"


def test_delete_source_only_removes_that_file(clean_db):
    with PgVectorStore(clean_db, dimension=3, embedder_name="test-3") as store:
        store.add(
            [chunk("a.pdf", 1, "a"), chunk("b.pdf", 1, "b")],
            [[1, 0, 0], [0, 1, 0]],
        )
        store.delete_source("a.pdf")

        assert store.count() == 1
        assert store.search([0, 1, 0], k=5)[0].chunk.source == "b.pdf"


def test_mismatched_lengths_are_rejected(clean_db):
    with PgVectorStore(clean_db, dimension=3, embedder_name="test-3") as store:
        with pytest.raises(ValueError):
            store.add([chunk("a.pdf", 1, "a")], [[1, 0, 0], [0, 1, 0]])


def test_switching_embedder_on_an_existing_index_fails(clean_db):
    PgVectorStore(clean_db, dimension=3, embedder_name="model-a-3").close()

    with pytest.raises(RuntimeError, match="model-a-3"):
        PgVectorStore(clean_db, dimension=3, embedder_name="model-b-3")
