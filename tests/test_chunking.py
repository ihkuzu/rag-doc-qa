import pytest

from ragqa.chunking import chunk_pages, chunk_text
from ragqa.loader import Page


def sentences(n: int) -> str:
    return " ".join(f"This is sentence number {i}." for i in range(n))


def test_short_text_is_a_single_chunk():
    assert chunk_text("One short sentence.", max_chars=100, overlap_chars=10) == [
        "One short sentence."
    ]


def test_empty_text_gives_no_chunks():
    assert chunk_text("   \n  ") == []


def test_chunks_never_exceed_max_chars():
    chunks = chunk_text(sentences(60), max_chars=200, overlap_chars=50)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_consecutive_chunks_share_text():
    chunks = chunk_text(sentences(60), max_chars=200, overlap_chars=60)
    for previous, following in zip(chunks, chunks[1:]):
        last_sentence = previous.split(". ")[-1]
        assert last_sentence in following


def test_no_overlap_means_no_repeated_sentences():
    chunks = chunk_text(sentences(30), max_chars=150, overlap_chars=0)
    joined = " ".join(chunks)
    assert joined == sentences(30)


def test_no_content_is_lost():
    text = sentences(40)
    chunks = chunk_text(text, max_chars=180, overlap_chars=40)
    for i in range(40):
        assert any(f"number {i}." in c for c in chunks)


def test_a_sentence_longer_than_max_chars_is_split():
    long_sentence = "word " * 100  # 500 chars, no sentence end
    chunks = chunk_text(long_sentence, max_chars=120, overlap_chars=20)
    assert all(len(c) <= 120 for c in chunks)
    assert len(chunks) >= 4


@pytest.mark.parametrize(
    "max_chars, overlap",
    [(0, 0), (-5, 0), (100, 100), (100, 150), (100, -1)],
)
def test_invalid_sizes_raise(max_chars, overlap):
    with pytest.raises(ValueError):
        chunk_text("text", max_chars=max_chars, overlap_chars=overlap)


def test_chunk_pages_keeps_source_and_page():
    pages = [
        Page("a.pdf", 1, sentences(20)),
        Page("a.pdf", 2, "Tiny page."),
    ]
    chunks = chunk_pages(pages, max_chars=150, overlap_chars=30)

    assert {c.page for c in chunks} == {1, 2}
    assert all(c.source == "a.pdf" for c in chunks)
    page_one = [c for c in chunks if c.page == 1]
    assert [c.index for c in page_one] == list(range(len(page_one)))
