import hashlib
from pathlib import Path

import pytest
from app.domain.exceptions.document import (
    DocumentNotFoundError,
    EncryptedDocumentError,
    InvalidPdfError,
    UnsupportedDocumentTypeError,
)
from app.infrastructure.pdf.pymupdf_extractor import PyMuPdfExtractor
from tests.support.pdf_factory import (
    build_corrupted_pdf,
    build_encrypted_pdf,
    build_non_pdf_file,
    build_pdf,
)


def test_extracts_single_page_pdf(tmp_path: Path) -> None:
    pdf_path = build_pdf(tmp_path / "single.pdf", ["Page one content"])

    pages = PyMuPdfExtractor().extract(pdf_path)

    assert len(pages) == 1
    assert "Page one content" in pages[0].text


def test_preserves_multi_page_order(tmp_path: Path) -> None:
    pdf_path = build_pdf(tmp_path / "multi.pdf", ["First page", "Second page", "Third page"])

    pages = PyMuPdfExtractor().extract(pdf_path)

    assert len(pages) == 3
    assert "First page" in pages[0].text
    assert "Second page" in pages[1].text
    assert "Third page" in pages[2].text


def test_page_number_is_one_indexed(tmp_path: Path) -> None:
    pdf_path = build_pdf(tmp_path / "multi.pdf", ["First", "Second"])

    pages = PyMuPdfExtractor().extract(pdf_path)

    assert [page.page_number for page in pages] == [1, 2]


def test_source_name_and_source_path_are_preserved(tmp_path: Path) -> None:
    pdf_path = build_pdf(tmp_path / "sample.pdf", ["Content"])

    pages = PyMuPdfExtractor().extract(pdf_path)

    assert pages[0].source_name == "sample.pdf"
    assert pages[0].source_path == str(pdf_path)


def test_same_content_produces_same_document_id(tmp_path: Path) -> None:
    first_path = build_pdf(tmp_path / "a.pdf", ["Same content"])
    second_path = tmp_path / "b.pdf"
    second_path.write_bytes(first_path.read_bytes())

    first_pages = PyMuPdfExtractor().extract(first_path)
    second_pages = PyMuPdfExtractor().extract(second_path)

    assert first_pages[0].document_id == second_pages[0].document_id
    assert first_pages[0].document_id == hashlib.sha256(first_path.read_bytes()).hexdigest()
    assert len(first_pages[0].document_id) == 64


def test_different_content_produces_different_document_id(tmp_path: Path) -> None:
    first_path = build_pdf(tmp_path / "a.pdf", ["Content A"])
    second_path = build_pdf(tmp_path / "b.pdf", ["Content B, which differs"])

    first_pages = PyMuPdfExtractor().extract(first_path)
    second_pages = PyMuPdfExtractor().extract(second_path)

    assert first_pages[0].document_id != second_pages[0].document_id


def test_pdf_title_is_extracted(tmp_path: Path) -> None:
    pdf_path = build_pdf(tmp_path / "titled.pdf", ["Content"], title="Sample Guideline")

    pages = PyMuPdfExtractor().extract(pdf_path)

    assert pages[0].title == "Sample Guideline"


def test_empty_page_is_preserved(tmp_path: Path) -> None:
    pdf_path = build_pdf(tmp_path / "with_blank.pdf", ["First page", ""])

    pages = PyMuPdfExtractor().extract(pdf_path)

    assert len(pages) == 2
    assert pages[1].text == ""


def test_missing_file_raises_document_not_found_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "does_not_exist.pdf"

    with pytest.raises(DocumentNotFoundError):
        PyMuPdfExtractor().extract(missing_path)


def test_non_pdf_extension_raises_unsupported_document_type_error(tmp_path: Path) -> None:
    non_pdf_path = build_non_pdf_file(tmp_path / "notes.txt")

    with pytest.raises(UnsupportedDocumentTypeError):
        PyMuPdfExtractor().extract(non_pdf_path)


def test_corrupted_pdf_raises_invalid_pdf_error(tmp_path: Path) -> None:
    corrupted_path = build_corrupted_pdf(tmp_path / "corrupted.pdf")

    with pytest.raises(InvalidPdfError):
        PyMuPdfExtractor().extract(corrupted_path)


def test_encrypted_pdf_raises_encrypted_document_error(tmp_path: Path) -> None:
    encrypted_path = build_encrypted_pdf(tmp_path / "encrypted.pdf", password="secret")

    with pytest.raises(EncryptedDocumentError):
        PyMuPdfExtractor().extract(encrypted_path)


def test_log_output_does_not_contain_page_text(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    pdf_path = build_pdf(tmp_path / "secret.pdf", ["This exact sentence must not be logged"])

    with caplog.at_level("INFO"):
        PyMuPdfExtractor().extract(pdf_path)

    assert "This exact sentence must not be logged" not in caplog.text
