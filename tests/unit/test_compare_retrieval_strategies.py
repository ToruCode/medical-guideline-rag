import argparse

import pytest
from scripts.compare_retrieval_strategies import _parse_strategies
from scripts.hybrid_retrieval_core import (
    StrategyAggregate,
    StrategyComparisonResult,
    StrategyRunConfig,
    markdown_comparison_table,
)


def _make_result(
    strategy: str, alpha: float | None, recall_at_1: float
) -> StrategyComparisonResult:
    config = StrategyRunConfig(
        strategy=strategy,
        alpha=alpha,
        chunk_size=1000,
        chunk_overlap=200,
        top_k=5,
        dense_candidate_k=20 if alpha is not None else None,
        bm25_candidate_k=20 if alpha is not None else None,
        embedding_model_name="intfloat/multilingual-e5-base",
        query_prefix="query: ",
        passage_prefix="passage: ",
        case_count=4,
        indexed_page_count=10,
        indexed_chunk_count=12,
        measured_at="2026-08-03",
    )
    aggregate = StrategyAggregate(
        recall_at_1=recall_at_1, recall_at_3=1.0, recall_at_5=1.0, mrr=1.0
    )
    return StrategyComparisonResult(config=config, case_results=[], aggregate=aggregate)


def test_parse_strategies_splits_names() -> None:
    assert _parse_strategies("dense,hybrid") == ["dense", "hybrid"]


def test_parse_strategies_strips_whitespace() -> None:
    assert _parse_strategies(" dense , hybrid ") == ["dense", "hybrid"]


def test_parse_strategies_rejects_empty_string() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_strategies("")


def test_parse_strategies_rejects_unknown_name() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="Unknown strategy"):
        _parse_strategies("dense,rrf")


def test_markdown_comparison_table_includes_one_row_per_strategy() -> None:
    results = [
        _make_result("dense", alpha=None, recall_at_1=0.583),
        _make_result("hybrid", alpha=0.7, recall_at_1=0.75),
    ]

    table = markdown_comparison_table("Guideline A", results)

    assert "Document: Guideline A" in table
    assert "chunk_size=1000, chunk_overlap=200, top_k=5" in table
    assert "| dense |  | 0.58 | 1.00 | 1.00 | 1.00 |" in table
    assert "| hybrid | 0.7 | 0.75 | 1.00 | 1.00 | 1.00 |" in table


def test_markdown_comparison_table_never_includes_question_text() -> None:
    results = [_make_result("dense", alpha=None, recall_at_1=0.5)]

    table = markdown_comparison_table("Guideline A", results)

    assert "case_results" not in table
    assert "ranked_results" not in table
