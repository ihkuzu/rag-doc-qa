from pathlib import Path

import pytest
from reportlab.pdfgen import canvas


def make_pdf(path: Path, pages: list[str]) -> Path:
    """Write a simple PDF; an empty string produces a blank page."""
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
