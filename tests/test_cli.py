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


def test_ingest_then_search(pdf_factory, tmp_path, clean_db, capsys):
    pdf_factory(
        "handbook.pdf",
        ["Backups run nightly at 02:00.", "Employees receive thirty vacation days."],
    )

    assert main(["ingest", str(tmp_path), "--database-url", clean_db]) == 0
    assert "indexed 2 chunks" in capsys.readouterr().out

    assert main(["search", "vacation days for employees", "-k", "1", "--database-url", clean_db]) == 0
    out = capsys.readouterr().out
    assert "handbook.pdf p.2" in out


def test_search_without_a_database_url_exits(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    try:
        main(["search", "anything"])
    except SystemExit as exit_:
        assert "DATABASE_URL" in str(exit_)
    else:
        raise AssertionError("expected SystemExit")


def test_ask_prints_answer_and_marks_cited_sources(pdf_factory, tmp_path, clean_db, capsys, monkeypatch):
    from conftest import FakeLLM

    pdf_factory("handbook.pdf", ["Employees receive thirty vacation days.", "Backups run nightly."])
    main(["ingest", str(tmp_path), "--database-url", clean_db])
    capsys.readouterr()
    monkeypatch.setattr("ragqa.cli.get_llm", lambda *args: FakeLLM("Thirty days [1]."))

    assert main(["ask", "vacation days for employees", "-k", "2", "--database-url", clean_db]) == 0

    out = capsys.readouterr().out
    assert "Thirty days [1]." in out
    assert " * [1] handbook.pdf p.1" in out


def test_ask_reports_model_errors(pdf_factory, tmp_path, clean_db, capsys, monkeypatch):
    from ragqa.llm import LLMError

    class Broken:
        def complete(self, system, prompt):
            raise LLMError("model is down")

    pdf_factory("handbook.pdf", ["Employees receive thirty vacation days."])
    main(["ingest", str(tmp_path), "--database-url", clean_db])
    capsys.readouterr()
    monkeypatch.setattr("ragqa.cli.get_llm", lambda *args: Broken())

    assert main(["ask", "vacation", "--database-url", clean_db]) == 1
    assert "model is down" in capsys.readouterr().out


def test_eval_prints_a_report(pdf_factory, tmp_path, clean_db, capsys):
    import json

    pdf_factory("handbook.pdf", ["Backups run nightly.", "Employees receive thirty vacation days."])
    main(["ingest", str(tmp_path), "--database-url", clean_db])
    questions = tmp_path / "questions.json"
    questions.write_text(
        json.dumps([{"question": "vacation days for employees", "source": "handbook.pdf", "page": 2}])
    )
    capsys.readouterr()

    assert main(["eval", str(questions), "-k", "3", "--database-url", clean_db]) == 0

    out = capsys.readouterr().out
    assert "embedder: hashing-384" in out
    assert "hit@1 1.00" in out
