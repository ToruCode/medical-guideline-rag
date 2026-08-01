import argparse

import pytest
from scripts.compare_chunk_sizes import _markdown_comparison_table, _parse_chunk_sizes
from scripts.retrieval_baseline_core import Aggregate, ConfigurationRun, RunConfig


def _make_run(chunk_size: int, indexed_chunk_count: int, recall_at_1: float) -> ConfigurationRun:
    config = RunConfig(
        chunk_size=chunk_size,
        chunk_overlap=200,
        top_k=5,
        embedding_model_name="intfloat/multilingual-e5-base",
        query_prefix="query: ",
        passage_prefix="passage: ",
        case_count=4,
        indexed_page_count=10,
        indexed_chunk_count=indexed_chunk_count,
        measured_at="2026-08-02",
    )
    aggregate = Aggregate(recall_at_1=recall_at_1, recall_at_3=1.0, recall_at_5=1.0, mrr=1.0)
    return ConfigurationRun(config=config, case_results=[], aggregate=aggregate)


def test_parse_chunk_sizes_splits_and_converts_to_ints() -> None:
    assert _parse_chunk_sizes("300,500,700") == [300, 500, 700]


def test_parse_chunk_sizes_strips_whitespace() -> None:
    assert _parse_chunk_sizes(" 300 , 500 ") == [300, 500]


def test_parse_chunk_sizes_rejects_non_integer_values() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_chunk_sizes("300,abc")


def test_parse_chunk_sizes_rejects_empty_string() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_chunk_sizes("")


def test_parse_chunk_sizes_rejects_non_positive_values() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_chunk_sizes("300,0")


def test_markdown_comparison_table_includes_one_row_per_run() -> None:
    runs = [
        _make_run(chunk_size=300, indexed_chunk_count=42, recall_at_1=0.5),
        _make_run(chunk_size=1000, indexed_chunk_count=15, recall_at_1=0.75),
    ]

    table = _markdown_comparison_table("Guideline A", runs)

    assert "Document: Guideline A" in table
    assert "chunk_overlap=200, top_k=5" in table
    assert "| 300 | 42 | 0.50 | 1.00 | 1.00 | 1.00 |" in table
    assert "| 1000 | 15 | 0.75 | 1.00 | 1.00 | 1.00 |" in table


def test_markdown_comparison_table_never_includes_question_text() -> None:
    runs = [_make_run(chunk_size=300, indexed_chunk_count=42, recall_at_1=0.5)]

    table = _markdown_comparison_table("Guideline A", runs)

    assert "case_results" not in table
