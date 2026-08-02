"""PyMuPdfLoader is a thin adapter delegating to PyMuPdfExtractor
(already covered by test_pymupdf_extractor.py). These tests only
confirm the adapter wiring itself - that load() produces the same
result as PyMuPdfExtractor.extract() - not PyMuPDF behavior in
general.
"""

from pathlib import Path

from app.infrastructure.pdf.pymupdf_extractor import PyMuPdfExtractor
from app.infrastructure.pdf.pymupdf_loader import PyMuPdfLoader
from tests.support.pdf_factory import build_pdf


def test_load_delegates_to_pymupdf_extractor(tmp_path: Path) -> None:
    pdf_path = build_pdf(tmp_path / "sample.pdf", ["First page", "Second page"])

    loader_pages = PyMuPdfLoader().load(pdf_path)
    extractor_pages = PyMuPdfExtractor().extract(pdf_path)

    assert loader_pages == extractor_pages


def test_page_number_is_one_indexed(tmp_path: Path) -> None:
    pdf_path = build_pdf(tmp_path / "multi.pdf", ["First", "Second"])

    pages = PyMuPdfLoader().load(pdf_path)

    assert [page.page_number for page in pages] == [1, 2]


def test_empty_page_is_preserved(tmp_path: Path) -> None:
    pdf_path = build_pdf(tmp_path / "with_blank.pdf", ["First page", ""])

    pages = PyMuPdfLoader().load(pdf_path)

    assert len(pages) == 2
    assert pages[1].text == ""
