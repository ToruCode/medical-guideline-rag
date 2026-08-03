"""Unit tests for scripts/chunking_comparison_core.py.

Uses only self-authored, fictional synthetic text (via DocumentPage
built directly - no real PDF, no PyMuPDF extraction, no text_preview
from any real document).
"""

from app.application.services.chunk_document import ChunkDocumentService
from app.domain.models.document import DocumentPage
from app.infrastructure.chunking.fixed_size_text_splitter import FixedSizeTextSplitter
from scripts.chunking_comparison_core import (
    CASE_BOTH_FAIL,
    CASE_BOTH_SUCCESS,
    CASE_FIXED_ONLY_SUCCESS,
    CASE_TABLE_AWARE_ONLY_SUCCESS,
    RANK_IMPROVED,
    RANK_UNCHANGED,
    RANK_WORSENED,
    ChunkingCaseComparison,
    build_chunks_fixed,
    build_chunks_table_aware,
    compare_fixed_to_table_aware,
    compute_chunk_stats,
    count_table_blocks,
)
from scripts.reranking_core import RerankCaseResult


def _page(text: str, page_number: int) -> DocumentPage:
    return DocumentPage(
        document_id="doc",
        source_name="doc.pdf",
        source_path="/doc.pdf",
        page_number=page_number,
        text=text,
        title=None,
    )


def _case_result(reciprocal_rank: float) -> RerankCaseResult:
    return RerankCaseResult(
        question="q",
        expected=[(1, None)],
        ranked_locations=[],
        recall_at_1=0.0,
        recall_at_3=0.0,
        recall_at_5=1.0 if reciprocal_rank > 0 else 0.0,
        reciprocal_rank=reciprocal_rank,
        retrieval_latency_ms=0.0,
        reranking_latency_ms=0.0,
        total_latency_ms=0.0,
    )


# --- build_chunks_fixed: must not diverge from calling FixedSizeTextSplitter directly ---


def test_build_chunks_fixed_matches_existing_splitter_directly() -> None:
    pages = [
        _page("これはテスト用の説明文です。" * 50, page_number=1),
        _page("短い文章。", page_number=2),
    ]

    via_helper = build_chunks_fixed(pages, chunk_size=100, chunk_overlap=20)
    via_direct = ChunkDocumentService(
        FixedSizeTextSplitter(chunk_size=100, chunk_overlap=20)
    ).execute(pages)

    assert [c.text for c in via_helper] == [c.text for c in via_direct]
    assert [c.page_number for c in via_helper] == [c.page_number for c in via_direct]
    assert [c.chunk_index for c in via_helper] == [c.chunk_index for c in via_direct]


# --- build_chunks_table_aware: page_number / chunk_index / empty page ---


def test_page_number_is_preserved_in_table_aware_chunks() -> None:
    pages = [
        _page("表1　サンプル基準表\n項目\n基準値\n0.01\n0.02", page_number=3),
        _page("通常の説明文です。", page_number=4),
    ]

    chunks, metadata = build_chunks_table_aware(
        pages, chunk_size=1000, chunk_overlap=200, table_max_chars=1000, table_row_group_size=20
    )

    assert {c.page_number for c in chunks} == {3, 4}
    assert len(metadata) == len(chunks)


def test_chunk_index_is_stable_and_sequential_per_page() -> None:
    text = "表1　基準表\n項目\n基準値\n0.01\n0.02\n通常の説明文がここに続きます。"
    pages = [_page(text, page_number=1)]

    chunks, _metadata = build_chunks_table_aware(
        pages, chunk_size=1000, chunk_overlap=200, table_max_chars=1000, table_row_group_size=20
    )

    indices = [c.chunk_index for c in chunks if c.page_number == 1]
    assert indices == list(range(len(indices)))


def test_empty_page_produces_no_chunks() -> None:
    pages = [_page("", page_number=1)]

    chunks, metadata = build_chunks_table_aware(
        pages, chunk_size=1000, chunk_overlap=200, table_max_chars=1000, table_row_group_size=20
    )

    assert chunks == []
    assert metadata == {}


def test_count_table_blocks_counts_across_pages() -> None:
    pages = [
        _page("表1　基準表\n項目\n基準値\n0.01\n0.02", page_number=1),
        _page("通常の説明文のみのページです。", page_number=2),
        _page("表2　別の基準表\n項目\n基準値\n1.0\n2.0", page_number=3),
    ]

    assert count_table_blocks(pages) == 2


# --- compute_chunk_stats: aggregation correctness ---


def test_compute_chunk_stats_basic_counts() -> None:
    text = "表1　基準表\n項目\n基準値\n0.01\n0.02\n通常の説明文がここに続きます。"
    pages = [_page(text, page_number=1)]
    chunks, metadata = build_chunks_table_aware(
        pages, chunk_size=1000, chunk_overlap=200, table_max_chars=1000, table_row_group_size=20
    )

    stats = compute_chunk_stats(
        "table_aware", chunks, metadata, chunking_time_ms=1.0, table_block_count=1
    )

    assert stats.total_chunks == len(chunks)
    assert stats.table_block_count == 1
    assert stats.cross_page_chunk_count == 0
    assert stats.min_chars == min(len(c.text) for c in chunks)
    assert stats.max_chars == max(len(c.text) for c in chunks)


def test_compute_chunk_stats_handles_no_chunks() -> None:
    stats = compute_chunk_stats("fixed", [], None, chunking_time_ms=0.0, table_block_count=0)

    assert stats.total_chunks == 0
    assert stats.avg_chars == 0.0
    assert stats.min_chars == 0
    assert stats.max_chars == 0


# --- fixed vs table_aware per-question comparison ---


def test_compare_fixed_to_table_aware_fixed_only_success() -> None:
    comparison = compare_fixed_to_table_aware(_case_result(1.0), _case_result(0.0))
    assert comparison.category == CASE_FIXED_ONLY_SUCCESS


def test_compare_fixed_to_table_aware_table_aware_only_success() -> None:
    comparison = compare_fixed_to_table_aware(_case_result(0.0), _case_result(1.0))
    assert comparison.category == CASE_TABLE_AWARE_ONLY_SUCCESS


def test_compare_fixed_to_table_aware_both_fail() -> None:
    comparison = compare_fixed_to_table_aware(_case_result(0.0), _case_result(0.0))
    assert comparison.category == CASE_BOTH_FAIL


def test_compare_fixed_to_table_aware_improved() -> None:
    comparison = compare_fixed_to_table_aware(_case_result(1 / 3), _case_result(1.0))
    assert comparison.category == CASE_BOTH_SUCCESS
    assert comparison.rank_change == RANK_IMPROVED


def test_compare_fixed_to_table_aware_worsened() -> None:
    comparison = compare_fixed_to_table_aware(_case_result(1.0), _case_result(1 / 3))
    assert comparison.category == CASE_BOTH_SUCCESS
    assert comparison.rank_change == RANK_WORSENED


def test_compare_fixed_to_table_aware_unchanged() -> None:
    comparison = compare_fixed_to_table_aware(_case_result(0.5), _case_result(0.5))
    assert comparison.category == CASE_BOTH_SUCCESS
    assert comparison.rank_change == RANK_UNCHANGED
    assert isinstance(comparison, ChunkingCaseComparison)
