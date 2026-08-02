import argparse

import pytest
from scripts.compare_pdf_extractors import _parse_extractors
from scripts.pdf_extraction_comparison_core import (
    ExtractionStats,
    ExtractorComparisonResult,
    markdown_comparison_table,
)
from scripts.retrieval_baseline_core import Aggregate, ConfigurationRun, RunConfig


def _make_result(extractor_name: str, recall_at_1: float) -> ExtractorComparisonResult:
    config = RunConfig(
        chunk_size=1000,
        chunk_overlap=200,
        top_k=5,
        embedding_model_name="intfloat/multilingual-e5-base",
        query_prefix="query: ",
        passage_prefix="passage: ",
        case_count=4,
        indexed_page_count=10,
        indexed_chunk_count=12,
        measured_at="2026-08-02",
    )
    aggregate = Aggregate(recall_at_1=recall_at_1, recall_at_3=1.0, recall_at_5=1.0, mrr=1.0)
    retrieval_run = ConfigurationRun(config=config, case_results=[], aggregate=aggregate)
    extraction_stats = ExtractionStats(
        extractor_name=extractor_name,
        pages=10,
        empty_pages=1,
        total_chars=5000,
        suspicious_pages=2,
        suspicious_ratio=0.2,
        average_chars_per_page=500.0,
        representative_text_preview="representative sample text",
    )
    return ExtractorComparisonResult(
        extractor_name=extractor_name,
        extraction_stats=extraction_stats,
        average_chars_per_chunk=416.7,
        retrieval_run=retrieval_run,
    )


def test_parse_extractors_splits_names() -> None:
    assert _parse_extractors("pypdf,pymupdf") == ["pypdf", "pymupdf"]


def test_parse_extractors_strips_whitespace() -> None:
    assert _parse_extractors(" pypdf , pymupdf ") == ["pypdf", "pymupdf"]


def test_parse_extractors_rejects_empty_string() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_extractors("")


def test_parse_extractors_rejects_unknown_name() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="Unknown extractor"):
        _parse_extractors("pypdf,docling")


def test_markdown_comparison_table_includes_one_row_per_extractor() -> None:
    results = [
        _make_result("pypdf", recall_at_1=0.5),
        _make_result("pymupdf", recall_at_1=0.75),
    ]

    table = markdown_comparison_table("Guideline A", results)

    assert "Document: Guideline A" in table
    assert "chunk_size=1000, chunk_overlap=200, top_k=5" in table
    assert (
        "| pypdf | 10 | 1 | 5000 | 500.0 | 2 | 0.200 | 416.7 | 0.50 | 1.00 | 1.00 | 1.00 |" in table
    )
    assert (
        "| pymupdf | 10 | 1 | 5000 | 500.0 | 2 | 0.200 | 416.7 | 0.75 | 1.00 | 1.00 | 1.00 |"
        in table
    )


def test_markdown_comparison_table_never_includes_question_text() -> None:
    results = [_make_result("pypdf", recall_at_1=0.5)]

    table = markdown_comparison_table("Guideline A", results)

    assert "case_results" not in table
    assert "representative sample text" not in table
