from ragqa.cli import main


def test_chunks_command_prints_statistics(pdf_factory, tmp_path, capsys):
    pdf_factory("doc.pdf", ["Hello world. " * 80])

    exit_code = main(["chunks", str(tmp_path), "--max-chars", "200", "--overlap", "40"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "doc.pdf:" in out
    assert "total:" in out


def test_chunks_command_reports_empty_input(pdf_factory, tmp_path, capsys):
    pdf_factory("blank.pdf", [""])

    assert main(["chunks", str(tmp_path)]) == 1
    assert "No extractable text" in capsys.readouterr().out
