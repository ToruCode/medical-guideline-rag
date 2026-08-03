import argparse

import pytest
from scripts.chunking_comparison_core import ChunkStats, markdown_comparison_table
from scripts.compare_chunking_strategies import _parse_strategies
from scripts.reranking_core import RerankAggregate, RerankComparisonResult, RerankRunConfig


def _make_result(
    strategy: str, recall_at_1: float, total_chunks: int
) -> tuple[RerankComparisonResult, ChunkStats]:
    config = RerankRunConfig(
        strategy="hybrid_rerank",
        alpha=0.7,
        reranker_model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        device="cpu",
        batch_size=8,
        chunk_size=1000,
        chunk_overlap=200,
        top_k=5,
        dense_candidate_k=20,
        bm25_candidate_k=20,
        reranker_candidate_k=5,
        embedding_model_name="intfloat/multilingual-e5-base",
        query_prefix="query: ",
        passage_prefix="passage: ",
        case_count=10,
        indexed_page_count=29,
        indexed_chunk_count=total_chunks,
        measured_at="2026-08-03",
    )
    aggregate = RerankAggregate(
        recall_at_1=recall_at_1,
        recall_at_3=1.0,
        recall_at_5=1.0,
        mrr=1.0,
        avg_retrieval_latency_ms=10.0,
        avg_reranking_latency_ms=500.0,
        avg_total_latency_ms=510.0,
    )
    rerank_result = RerankComparisonResult(config=config, case_results=[], aggregate=aggregate)
    stats = ChunkStats(
        strategy=strategy,
        total_chunks=total_chunks,
        avg_chars=800.0,
        median_chars=850.0,
        min_chars=100,
        max_chars=1000,
        short_chunk_count=2,
        numeric_symbol_only_count=0,
        table_block_count=5,
        table_chunk_with_title_or_heading_count=5,
        column_header_duplicated_chunk_count=1,
        cross_page_chunk_count=0,
        chunking_time_ms=12.3,
        table_row_count_distribution={10: 4, 5: 1},
        split_by_max_chars_count=1,
        split_by_row_group_size_count=0,
        exceeded_max_chars_after_header_count=0,
    )
    return rerank_result, stats


def test_parse_strategies_splits_names() -> None:
    assert _parse_strategies("fixed,table_aware") == ["fixed", "table_aware"]


def test_parse_strategies_strips_whitespace() -> None:
    assert _parse_strategies(" fixed , table_aware ") == ["fixed", "table_aware"]


def test_parse_strategies_rejects_empty_string() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_strategies("")


def test_parse_strategies_rejects_unknown_name() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="Unknown strategy"):
        _parse_strategies("fixed,semantic")


def test_markdown_comparison_table_includes_one_row_per_strategy() -> None:
    results = {
        "fixed": _make_result("fixed", recall_at_1=0.617, total_chunks=50),
        "table_aware": _make_result("table_aware", recall_at_1=0.75, total_chunks=48),
    }

    table = markdown_comparison_table("Guideline A", case_count=30, top_k=5, results=results)

    assert "Document: Guideline A" in table
    assert "| fixed | 50 | 800.0 | 2 | 5 | 0.62 | 1.00 | 1.00 | 1.00 | 510.0 |" in table
    assert "| table_aware | 48 | 800.0 | 2 | 5 | 0.75 | 1.00 | 1.00 | 1.00 | 510.0 |" in table


def test_markdown_comparison_table_never_includes_question_text() -> None:
    results = {"fixed": _make_result("fixed", recall_at_1=0.5, total_chunks=50)}

    table = markdown_comparison_table("Guideline A", case_count=30, top_k=5, results=results)

    assert "case_results" not in table
    assert "text_preview" not in table
