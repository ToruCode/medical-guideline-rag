"""PypdfExtractor is a thin adapter delegating to PypdfLoader (the
production implementation, already covered by test_pypdf_loader.py).
These tests only confirm the adapter wiring itself - that extract()
produces the same result as PypdfLoader.load() - not pypdf behavior
in general.
"""

from pathlib import Path

from app.infrastructure.pdf.pypdf_extractor import PypdfExtractor
from app.infrastructure.pdf.pypdf_loader import PypdfLoader
from tests.support.pdf_factory import build_pdf


def test_extract_delegates_to_pypdf_loader(tmp_path: Path) -> None:
    pdf_path = build_pdf(tmp_path / "sample.pdf", ["First page", "Second page"])

    extractor_pages = PypdfExtractor().extract(pdf_path)
    loader_pages = PypdfLoader().load(pdf_path)

    assert extractor_pages == loader_pages


def test_page_number_is_one_indexed(tmp_path: Path) -> None:
    pdf_path = build_pdf(tmp_path / "multi.pdf", ["First", "Second"])

    pages = PypdfExtractor().extract(pdf_path)

    assert [page.page_number for page in pages] == [1, 2]


def test_empty_page_is_preserved(tmp_path: Path) -> None:
    pdf_path = build_pdf(tmp_path / "with_blank.pdf", ["First page", ""])

    pages = PypdfExtractor().extract(pdf_path)

    assert len(pages) == 2
    assert pages[1].text == ""
