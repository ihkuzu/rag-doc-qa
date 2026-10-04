import pytest

from ragqa.loader import load_directory, load_pdf


def test_load_pdf_returns_text_with_page_numbers(pdf_factory):
    path = pdf_factory("manual.pdf", ["Alpha page one.", "Beta page two."])
    pages = load_pdf(path)

    assert [p.number for p in pages] == [1, 2]
    assert pages[0].source == "manual.pdf"
    assert "Alpha" in pages[0].text
    assert "Beta" in pages[1].text


def test_blank_pages_are_skipped_but_numbering_is_kept(pdf_factory):
    path = pdf_factory("gap.pdf", ["First.", "", "Third."])
    pages = load_pdf(path)

    assert [p.number for p in pages] == [1, 3]


def test_load_directory_reads_only_pdfs_in_sorted_order(pdf_factory, tmp_path):
    pdf_factory("b.pdf", ["From b."])
    pdf_factory("a.pdf", ["From a."])
    (tmp_path / "notes.txt").write_text("ignore me")

    pages = load_directory(tmp_path)

    assert [p.source for p in pages] == ["a.pdf", "b.pdf"]


def test_load_directory_rejects_a_file_path(tmp_path):
    file = tmp_path / "x.txt"
    file.write_text("x")
    with pytest.raises(NotADirectoryError):
        load_directory(file)
