import os
from pathlib import Path

import psycopg
import pytest
from reportlab.pdfgen import canvas


def make_pdf(path: Path, pages: list[str]) -> Path:
    pdf = canvas.Canvas(str(path))
    for text in pages:
        y = 780
        for line in text.splitlines():
            pdf.drawString(72, y, line)
            y -= 16
        pdf.showPage()
    pdf.save()
    return path


@pytest.fixture
def pdf_factory(tmp_path):
    def _make(name: str, pages: list[str]) -> Path:
        return make_pdf(tmp_path / name, pages)

    return _make


@pytest.fixture(scope="session")
def db_url(tmp_path_factory):
    # CI can point this at a real server, locally a throwaway one is started
    url = os.getenv("TEST_DATABASE_URL")
    if url:
        yield url
        return
    pgserver = pytest.importorskip("pgserver")
    server = pgserver.get_server(tmp_path_factory.mktemp("pg"))
    yield server.get_uri()
    server.cleanup()


@pytest.fixture
def clean_db(db_url):
    with psycopg.connect(db_url, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS chunks, meta")
    return db_url
