"""Unit tests for scripts/pdf_extraction_comparison_core.py.

Uses only synthetic, self-authored content (tests/support/pdf_factory.py
and inline strings) - no real guideline PDF is needed or used.
"""

from pathlib import Path

import pytest
from app.domain.models.chunk import Chunk
from app.domain.models.document import DocumentPage
from app.infrastructure.pdf.pymupdf_extractor import PyMuPdfExtractor
from app.infrastructure.pdf.pypdf_extractor import PypdfExtractor
from scripts.pdf_extraction_comparison_core import (
    compute_average_chars_per_chunk,
    compute_extraction_stats,
    count_suspicious_chars,
    get_extractor,
    has_abnormal_japanese_char_ratio,
    has_abnormal_symbol_run,
    is_suspicious_page,
)
from tests.support.pdf_factory import build_pdf


def _page(text: str, page_number: int = 1) -> DocumentPage:
    return DocumentPage(
        document_id="doc",
        source_name="doc.pdf",
        source_path="/doc.pdf",
        page_number=page_number,
        text=text,
        title=None,
    )


def _chunk(text: str, page_number: int = 1, chunk_index: int = 0) -> Chunk:
    return Chunk(
        document_id="doc",
        source_name="doc.pdf",
        source_path="/doc.pdf",
        page_number=page_number,
        chunk_index=chunk_index,
        text=text,
        title=None,
    )


# --- extractor selection ---


def test_get_extractor_returns_pypdf_extractor() -> None:
    assert isinstance(get_extractor("pypdf"), PypdfExtractor)


def test_get_extractor_returns_pymupdf_extractor() -> None:
    assert isinstance(get_extractor("pymupdf"), PyMuPdfExtractor)


def test_get_extractor_raises_for_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown extractor"):
        get_extractor("docling")


@pytest.mark.parametrize("extractor_name", ["pypdf", "pymupdf"])
def test_page_numbers_are_preserved_across_extractors(tmp_path: Path, extractor_name: str) -> None:
    pdf_path = build_pdf(tmp_path / "multi.pdf", ["First", "Second", "Third"])

    pages = get_extractor(extractor_name).extract(pdf_path)

    assert [page.page_number for page in pages] == [1, 2, 3]


@pytest.mark.parametrize("extractor_name", ["pypdf", "pymupdf"])
def test_empty_pages_are_handled_safely(tmp_path: Path, extractor_name: str) -> None:
    pdf_path = build_pdf(tmp_path / "with_blank.pdf", ["Content", "", "More content"])

    pages = get_extractor(extractor_name).extract(pdf_path)

    assert len(pages) == 3
    assert pages[1].text == ""
    stats = compute_extraction_stats(extractor_name, pages)
    assert stats.empty_pages == 1
    assert stats.pages == 3


# --- garbled-text heuristic ---


def test_normal_japanese_text_is_not_suspicious() -> None:
    text = "これは自己作成のサンプル文書です。ガイドラインの本文はここに入ります。"
    assert is_suspicious_page(text) is False


def test_short_normal_english_text_is_not_suspicious() -> None:
    # Kept under the Japanese-ratio heuristic's minimum length: that
    # heuristic assumes Japanese content (this tool's intended use
    # case) and is documented as saying nothing about longer
    # legitimately non-Japanese text - see
    # test_low_japanese_ratio_long_text_is_suspicious below.
    assert is_suspicious_page("This is a normal sentence.") is False


def test_replacement_character_is_suspicious() -> None:
    assert count_suspicious_chars("some text �� here") == 2
    assert is_suspicious_page("some text �� here") is True


def test_unnatural_control_character_is_suspicious() -> None:
    assert count_suspicious_chars("text\x00\x01more") == 2
    assert is_suspicious_page("text\x00\x01more") is True


def test_newline_and_tab_are_not_counted_as_suspicious_control_chars() -> None:
    assert count_suspicious_chars("line one\nline two\tindented") == 0


def test_private_use_area_characters_are_suspicious() -> None:
    text = "prefix" + chr(0xE000) * 3 + "suffix"
    assert count_suspicious_chars(text) == 3
    assert is_suspicious_page(text) is True


def test_abnormal_ascii_symbol_run_is_suspicious() -> None:
    text = "normal text !@#$%^&*() more normal text"
    assert has_abnormal_symbol_run(text) is True
    assert is_suspicious_page(text) is True


def test_short_symbol_run_is_not_suspicious() -> None:
    assert has_abnormal_symbol_run("a (b) c - d!") is False


def test_low_japanese_ratio_long_text_is_suspicious() -> None:
    text = "abcdefghijklmnopqrstuvwxyz" * 3
    assert has_abnormal_japanese_char_ratio(text) is True
    assert is_suspicious_page(text) is True


def test_short_non_japanese_text_is_not_flagged_by_ratio_check() -> None:
    assert has_abnormal_japanese_char_ratio("short text") is False


def test_blank_page_is_not_suspicious() -> None:
    assert is_suspicious_page("") is False
    assert is_suspicious_page("   \n\t") is False


# --- extraction stats aggregation ---


def test_compute_extraction_stats_aggregates_correctly() -> None:
    pages = [
        _page("normal content here", page_number=1),
        _page("", page_number=2),
        _page("more � garbled � content", page_number=3),
    ]

    stats = compute_extraction_stats("pypdf", pages)

    assert stats.extractor_name == "pypdf"
    assert stats.pages == 3
    assert stats.empty_pages == 1
    assert stats.total_chars == sum(len(page.text) for page in pages)
    assert stats.suspicious_pages == 1
    assert stats.suspicious_ratio == pytest.approx(1 / 3)
    assert stats.average_chars_per_page == pytest.approx(stats.total_chars / 3)


def test_compute_extraction_stats_handles_zero_pages() -> None:
    stats = compute_extraction_stats("pypdf", [])

    assert stats.pages == 0
    assert stats.empty_pages == 0
    assert stats.total_chars == 0
    assert stats.suspicious_pages == 0
    assert stats.suspicious_ratio == 0.0
    assert stats.average_chars_per_page == 0.0
    assert stats.representative_text_preview == ""


def test_compute_extraction_stats_representative_preview_picks_longest_page() -> None:
    pages = [
        _page("short", page_number=1),
        _page("a much longer page of representative content", page_number=2),
    ]

    stats = compute_extraction_stats("pypdf", pages)

    assert stats.representative_text_preview == "a much longer page of representative content"


# --- average_chars_per_chunk ---


def test_compute_average_chars_per_chunk() -> None:
    chunks = [_chunk("a" * 10), _chunk("b" * 20), _chunk("c" * 30)]

    assert compute_average_chars_per_chunk(chunks) == pytest.approx(20.0)


def test_compute_average_chars_per_chunk_handles_no_chunks() -> None:
    assert compute_average_chars_per_chunk([]) == 0.0
