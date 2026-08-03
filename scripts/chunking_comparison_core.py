"""Shared library for comparing fixed-size vs table-aware chunking under
Hybrid+Cross-Encoder-Rerank retrieval (Issue #30).

Not a CLI entry point - imported by
scripts/compare_chunking_strategies.py. Builds on
scripts/reranking_core.py (Issue #25) and scripts/hybrid_retrieval_core.py
(Issue #24): only the chunk-production step differs between "fixed"
(the existing, unmodified FixedSizeTextSplitter/ChunkDocumentService)
and "table_aware" (scripts/table_aware_chunking.py, new in this issue);
everything downstream (embedding, indexing, Dense/BM25/Hybrid scoring,
Cross Encoder reranking) is the same evaluate_hybrid_rerank() call used
for both, against a fixed "hybrid_rerank" configuration (per Issue #29's
recommended settings).

This is comparison/measurement only: app/domain, app/application,
app/infrastructure, and app/api/dependencies.py are all untouched, and
production chunking (FixedSizeTextSplitter) is not changed or replaced.

Never commit a dataset file, the PDF it points to, or this script's
--save-report output; data/eval/ is gitignored for exactly this reason.
See docs/evaluation-dataset-format.md.
"""

import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from app.application.services.chunk_document import ChunkDocumentService
from app.application.services.embed_chunks import EmbedChunksService
from app.domain.models.chunk import Chunk
from app.domain.models.document import DocumentPage
from app.domain.ports.embedder import Embedder
from app.infrastructure.chunking.fixed_size_text_splitter import FixedSizeTextSplitter
from app.infrastructure.vector_store.in_memory_vector_store import InMemoryVectorStore
from scripts.bm25 import Bm25Index
from scripts.hybrid_retrieval_core import IndexedCorpus
from scripts.hybrid_scorer import ScoreFuser
from scripts.japanese_tokenizer import tokenize_japanese_text
from scripts.reranker import Reranker
from scripts.reranking_core import RerankCaseResult, RerankComparisonResult, evaluate_hybrid_rerank
from scripts.retrieval_baseline_core import DatasetCase, DatasetDocument, location_key
from scripts.table_aware_chunking import (
    TableAwareChunk,
    TableAwareTextSplitter,
    TableBlock,
    TableBlockDetector,
)

STRATEGY_FIXED = "fixed"
STRATEGY_TABLE_AWARE = "table_aware"
STRATEGIES: tuple[str, ...] = (STRATEGY_FIXED, STRATEGY_TABLE_AWARE)

_SHORT_CHUNK_MAX_CHARS = 100
_NUMERIC_SYMBOL_ONLY_ALPHA_RATIO_THRESHOLD = 0.1


# --- Chunk production ---


def build_chunks_fixed(
    pages: list[DocumentPage], *, chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    """The existing, unmodified production chunking path."""
    return ChunkDocumentService(
        FixedSizeTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    ).execute(pages)


def build_chunks_table_aware(
    pages: list[DocumentPage],
    *,
    chunk_size: int,
    chunk_overlap: int,
    table_max_chars: int,
    table_row_group_size: int,
) -> tuple[list[Chunk], dict[str, TableAwareChunk]]:
    """Returns (chunks, metadata) - chunks is the same production Chunk
    type used everywhere else (so the rest of the retrieval pipeline is
    unaffected), and metadata maps chunk_id -> TableAwareChunk (the
    comparison-only is_table_chunk/heading_context/table_title/etc.
    fields Chunk itself does not and must not carry).
    """
    splitter = TableAwareTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        table_max_chars=table_max_chars,
        table_row_group_size=table_row_group_size,
    )
    chunks: list[Chunk] = []
    metadata: dict[str, TableAwareChunk] = {}
    for page in pages:
        for index, aware_chunk in enumerate(splitter.split_page(page.text)):
            chunk = Chunk(
                document_id=page.document_id,
                source_name=page.source_name,
                source_path=page.source_path,
                page_number=page.page_number,
                chunk_index=index,
                text=aware_chunk.text,
                title=page.title,
            )
            chunks.append(chunk)
            metadata[chunk.chunk_id] = aware_chunk
    return chunks, metadata


def count_table_blocks(pages: list[DocumentPage]) -> int:
    detector = TableBlockDetector()
    return sum(
        1
        for page in pages
        if page.text
        for block in detector.detect(page.text.split("\n"))
        if isinstance(block, TableBlock)
    )


def build_corpus_from_chunks(chunks: list[Chunk], passage_embedder: Embedder) -> IndexedCorpus:
    """Embeds and indexes an already-produced chunk list - the chunk-
    production step (fixed vs table_aware) is fully decoupled from
    corpus building, so both strategies share identical embedding/
    indexing logic.
    """
    embedded_chunks = EmbedChunksService(passage_embedder).execute(chunks)
    vector_store = InMemoryVectorStore()
    vector_store.upsert(embedded_chunks)
    bm25_index = Bm25Index(
        [(chunk.chunk_id, tokenize_japanese_text(chunk.text)) for chunk in chunks]
    )
    page_count = len({chunk.page_number for chunk in chunks})
    return IndexedCorpus(
        vector_store=vector_store, chunks=chunks, bm25_index=bm25_index, page_count=page_count
    )


# --- Chunk statistics ---


@dataclass(frozen=True, slots=True)
class ChunkStats:
    strategy: str
    total_chunks: int
    avg_chars: float
    median_chars: float
    min_chars: int
    max_chars: int
    short_chunk_count: int
    numeric_symbol_only_count: int
    table_block_count: int
    table_chunk_with_title_or_heading_count: int
    column_header_duplicated_chunk_count: int
    cross_page_chunk_count: int
    chunking_time_ms: float
    table_row_count_distribution: dict[int, int]
    split_by_max_chars_count: int
    split_by_row_group_size_count: int
    exceeded_max_chars_after_header_count: int


def _is_numeric_symbol_only(text: str) -> bool:
    non_space = [ch for ch in text if not ch.isspace()]
    if not non_space:
        return False
    alpha_count = sum(1 for ch in non_space if ch.isalpha())
    return (alpha_count / len(non_space)) < _NUMERIC_SYMBOL_ONLY_ALPHA_RATIO_THRESHOLD


def compute_chunk_stats(
    strategy: str,
    chunks: list[Chunk],
    metadata: dict[str, TableAwareChunk] | None,
    chunking_time_ms: float,
    table_block_count: int,
) -> ChunkStats:
    lengths = [len(chunk.text) for chunk in chunks]
    table_meta = [m for m in (metadata or {}).values() if m.is_table_chunk]

    row_distribution: dict[int, int] = {}
    for m in table_meta:
        if m.row_count is not None:
            row_distribution[m.row_count] = row_distribution.get(m.row_count, 0) + 1

    return ChunkStats(
        strategy=strategy,
        total_chunks=len(chunks),
        avg_chars=statistics.fmean(lengths) if lengths else 0.0,
        median_chars=float(statistics.median(lengths)) if lengths else 0.0,
        min_chars=min(lengths) if lengths else 0,
        max_chars=max(lengths) if lengths else 0,
        short_chunk_count=sum(1 for length in lengths if length < _SHORT_CHUNK_MAX_CHARS),
        numeric_symbol_only_count=sum(1 for chunk in chunks if _is_numeric_symbol_only(chunk.text)),
        table_block_count=table_block_count,
        table_chunk_with_title_or_heading_count=sum(
            1 for m in table_meta if m.table_title is not None or m.heading_context is not None
        ),
        column_header_duplicated_chunk_count=sum(1 for m in table_meta if m.is_header_duplicate),
        cross_page_chunk_count=0,
        chunking_time_ms=chunking_time_ms,
        table_row_count_distribution=row_distribution,
        split_by_max_chars_count=sum(1 for m in table_meta if m.split_trigger == "max_chars"),
        split_by_row_group_size_count=sum(
            1 for m in table_meta if m.split_trigger == "row_group_size"
        ),
        exceeded_max_chars_after_header_count=sum(
            1 for m in table_meta if m.exceeded_max_chars_after_header
        ),
    )


# --- Strategy evaluation ---


def evaluate_chunking_strategy(
    strategy: str,
    pages: list[DocumentPage],
    cases: list[DatasetCase],
    passage_embedder: Embedder,
    query_embedder: Embedder,
    scorer: ScoreFuser,
    reranker: Reranker,
    *,
    chunk_size: int,
    chunk_overlap: int,
    table_max_chars: int,
    table_row_group_size: int,
    top_k: int,
    dense_candidate_k: int,
    bm25_candidate_k: int,
    reranker_candidate_k: int,
    alpha: float,
    reranker_model_name: str,
    device: str,
    batch_size: int,
    embedding_model_name: str,
) -> tuple[RerankComparisonResult, ChunkStats, dict[str, TableAwareChunk] | None]:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy!r} (available: {list(STRATEGIES)})")

    chunk_start = time.perf_counter()
    metadata: dict[str, TableAwareChunk] | None
    table_block_count = 0
    if strategy == STRATEGY_FIXED:
        chunks = build_chunks_fixed(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        metadata = None
    else:
        chunks, metadata = build_chunks_table_aware(
            pages,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            table_max_chars=table_max_chars,
            table_row_group_size=table_row_group_size,
        )
        table_block_count = count_table_blocks(pages)
    chunking_time_ms = (time.perf_counter() - chunk_start) * 1000

    corpus = build_corpus_from_chunks(chunks, passage_embedder)
    stats = compute_chunk_stats(strategy, chunks, metadata, chunking_time_ms, table_block_count)

    rerank_result = evaluate_hybrid_rerank(
        corpus,
        cases,
        query_embedder,
        scorer,
        reranker,
        top_k=top_k,
        dense_candidate_k=dense_candidate_k,
        bm25_candidate_k=bm25_candidate_k,
        reranker_candidate_k=reranker_candidate_k,
        alpha=alpha,
        reranker_model_name=reranker_model_name,
        device=device,
        batch_size=batch_size,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model_name=embedding_model_name,
    )
    return rerank_result, stats, metadata


# --- Per-question fixed vs table_aware comparison ---

CASE_FIXED_ONLY_SUCCESS = "fixed_only_success"
CASE_TABLE_AWARE_ONLY_SUCCESS = "table_aware_only_success"
CASE_BOTH_SUCCESS = "both_success"
CASE_BOTH_FAIL = "both_fail"

RANK_IMPROVED = "improved"
RANK_WORSENED = "worsened"
RANK_UNCHANGED = "unchanged"


@dataclass(frozen=True, slots=True)
class ChunkingCaseComparison:
    category: str
    rank_change: str | None
    fixed_rank: int | None
    table_aware_rank: int | None


def _first_hit_rank(reciprocal_rank: float) -> int | None:
    if reciprocal_rank == 0.0:
        return None
    return round(1.0 / reciprocal_rank)


def compare_fixed_to_table_aware(
    fixed_case: RerankCaseResult, table_aware_case: RerankCaseResult
) -> ChunkingCaseComparison:
    fixed_rank = _first_hit_rank(fixed_case.reciprocal_rank)
    table_aware_rank = _first_hit_rank(table_aware_case.reciprocal_rank)

    if fixed_rank is not None and table_aware_rank is None:
        return ChunkingCaseComparison(CASE_FIXED_ONLY_SUCCESS, None, fixed_rank, table_aware_rank)
    if fixed_rank is None and table_aware_rank is not None:
        return ChunkingCaseComparison(
            CASE_TABLE_AWARE_ONLY_SUCCESS, None, fixed_rank, table_aware_rank
        )
    if fixed_rank is None and table_aware_rank is None:
        return ChunkingCaseComparison(CASE_BOTH_FAIL, None, fixed_rank, table_aware_rank)

    assert fixed_rank is not None and table_aware_rank is not None
    if table_aware_rank < fixed_rank:
        rank_change = RANK_IMPROVED
    elif table_aware_rank > fixed_rank:
        rank_change = RANK_WORSENED
    else:
        rank_change = RANK_UNCHANGED
    return ChunkingCaseComparison(CASE_BOTH_SUCCESS, rank_change, fixed_rank, table_aware_rank)


def summarize_case_comparisons(
    fixed_result: RerankComparisonResult, table_aware_result: RerankComparisonResult
) -> dict[str, int]:
    counts: dict[str, int] = {
        CASE_FIXED_ONLY_SUCCESS: 0,
        CASE_TABLE_AWARE_ONLY_SUCCESS: 0,
        CASE_BOTH_SUCCESS: 0,
        CASE_BOTH_FAIL: 0,
        RANK_IMPROVED: 0,
        RANK_WORSENED: 0,
        RANK_UNCHANGED: 0,
    }
    for fixed_case, table_aware_case in zip(
        fixed_result.case_results, table_aware_result.case_results, strict=True
    ):
        comparison = compare_fixed_to_table_aware(fixed_case, table_aware_case)
        counts[comparison.category] += 1
        if comparison.rank_change is not None:
            counts[comparison.rank_change] += 1
    return counts


# --- Best-effort failure-cause auxiliary tagging (item 15) ---
#
# Mechanically derived only: candidate-pool membership and rerank rank
# movement come directly from RerankCaseResult.ranked_results (whose
# rank_before_rerank/rank_after_rerank fully reflect the candidate pool
# here, since Issue #30 fixes reranker_candidate_k == final_top_k == 5 -
# nothing is cut between reranking and the final result). For
# table_aware, missing title/heading/column-header context on the
# expected page's own winning chunk is read directly from its
# TableAwareChunk metadata. Semantic causes (duplicate_or_near_duplicate_content,
# query_document_vocabulary_gap, ambiguous_question, row/unit/note
# detachment) are NOT auto-detected - they require the same manual,
# per-question review done in Issue #29's failure analysis. Treat every
# tag from this function as a starting hypothesis, not a conclusion.

TAG_CORRECT_PAGE_NOT_IN_POOL = "correct_page_not_in_candidate_pool"
TAG_RERANKER_MISRANKING = "reranker_misranking"
TAG_TABLE_TITLE_MISSING = "table_title_missing"
TAG_HEADING_CONTEXT_MISSING = "heading_context_missing"
TAG_COLUMN_HEADER_MISSING = "column_header_missing"
TAG_OTHER = "other"


def auto_tag_failure_causes(
    case_result: RerankCaseResult,
    expected_pages: list[int],
    document_id: str,
    metadata: dict[str, TableAwareChunk] | None,
) -> list[str]:
    matching = [r for r in case_result.ranked_results if r.page_number in expected_pages]
    if not matching:
        return [TAG_CORRECT_PAGE_NOT_IN_POOL]

    best = min(matching, key=lambda r: r.rank_after_rerank)
    tags: list[str] = []
    if best.rank_after_rerank > best.rank_before_rerank:
        tags.append(TAG_RERANKER_MISRANKING)

    if metadata is not None:
        chunk_id = f"{document_id}:{best.page_number}:{best.chunk_index}"
        meta = metadata.get(chunk_id)
        if meta is not None and meta.is_table_chunk:
            if meta.table_title is None:
                tags.append(TAG_TABLE_TITLE_MISSING)
            if meta.heading_context is None:
                tags.append(TAG_HEADING_CONTEXT_MISSING)
            if not meta.has_header_lines:
                tags.append(TAG_COLUMN_HEADER_MISSING)

    return tags or [TAG_OTHER]


# --- Reporting ---

ChunkingResults = dict[str, tuple[RerankComparisonResult, ChunkStats]]

_TABLE_HEADER = (
    f"{'strategy':>12} | {'chunks':>6} | {'avg_chars':>9} | {'short':>6} | {'table':>6} | "
    f"{'Recall@1':>8} | {'Recall@3':>8} | {'Recall@5':>8} | {'MRR@k':>6} | {'avg_latency_ms':>14}"
)


def print_comparison_table(results: ChunkingResults) -> None:
    print(f"\n{_TABLE_HEADER}")
    print("-" * len(_TABLE_HEADER))
    for strategy in STRATEGIES:
        if strategy not in results:
            continue
        rerank_result, stats = results[strategy]
        aggregate = rerank_result.aggregate
        print(
            f"{strategy:>12} | {stats.total_chunks:>6} | {stats.avg_chars:>9.1f} | "
            f"{stats.short_chunk_count:>6} | {stats.table_block_count:>6} | "
            f"{aggregate.recall_at_1:>8.3f} | {aggregate.recall_at_3:>8.3f} | "
            f"{aggregate.recall_at_5:>8.3f} | {aggregate.mrr:>6.3f} | "
            f"{aggregate.avg_total_latency_ms:>14.1f}"
        )


def markdown_comparison_table(
    label: str, case_count: int, top_k: int, results: ChunkingResults
) -> str:
    header = (
        f"## Table-aware chunking comparison ({date.today().isoformat()})\n\n"
        f"- Document: {label}\n"
        f"- Cases: {case_count}\n"
        f"- top_k={top_k}\n\n"
        "| strategy | chunks | avg_chars | short_chunks | table_chunks | Recall@1 | "
        f"Recall@3 | Recall@5 | MRR@{top_k} | avg_latency_ms |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = ""
    for strategy in STRATEGIES:
        if strategy not in results:
            continue
        rerank_result, stats = results[strategy]
        aggregate = rerank_result.aggregate
        rows += (
            f"| {strategy} | {stats.total_chunks} | {stats.avg_chars:.1f} | "
            f"{stats.short_chunk_count} | {stats.table_block_count} | "
            f"{aggregate.recall_at_1:.2f} | {aggregate.recall_at_3:.2f} | "
            f"{aggregate.recall_at_5:.2f} | {aggregate.mrr:.2f} | "
            f"{aggregate.avg_total_latency_ms:.1f} |\n"
        )
    return header + rows


def print_markdown_comparison_table(
    label: str, case_count: int, top_k: int, results: ChunkingResults
) -> None:
    print("\n" + "=" * 70)
    print("Markdown snippet for docs/table-aware-chunking-comparison-results.md")
    print("(review before pasting - confirm nothing identifying leaks):")
    print("=" * 70)
    print(markdown_comparison_table(label, case_count, top_k, results))


def print_case_comparison_summary(counts: dict[str, int]) -> None:
    print("\nfixed vs table_aware per-question breakdown:")
    print(f"  fixed only success: {counts[CASE_FIXED_ONLY_SUCCESS]}")
    print(f"  table_aware only success: {counts[CASE_TABLE_AWARE_ONLY_SUCCESS]}")
    print(f"  both success: {counts[CASE_BOTH_SUCCESS]}")
    print(f"  both fail: {counts[CASE_BOTH_FAIL]}")
    print(f"    (of both-success) rank improved: {counts[RANK_IMPROVED]}")
    print(f"    (of both-success) rank worsened: {counts[RANK_WORSENED]}")
    print(f"    (of both-success) rank unchanged: {counts[RANK_UNCHANGED]}")


def print_verbose_strategy_results(
    strategy: str,
    result: RerankComparisonResult,
    cases: list[DatasetCase],
    metadata: dict[str, TableAwareChunk] | None,
    document_id: str,
) -> None:
    print(f"\n{'=' * 70}\nstrategy: {strategy}\n{'=' * 70}")
    for index, case_result in enumerate(result.case_results):
        qid = f"Q{index + 1}"
        expected_pages = sorted({page for page, _ in case_result.expected})
        retrieved_pages = [r.page_number for r in case_result.ranked_results]
        print(f"\n{qid}: expected_page={expected_pages} strategy={strategy}")
        print(f"  retrieved_pages: {retrieved_pages}")
        for ranked in case_result.ranked_results:
            chunk_id = f"{document_id}:{ranked.page_number}:{ranked.chunk_index}"
            meta = (metadata or {}).get(chunk_id)
            is_table_chunk = meta.is_table_chunk if meta is not None else False
            heading_context = meta.heading_context if meta is not None else None
            table_title = meta.table_title if meta is not None else None
            print(
                f"    rank_after_rerank={ranked.rank_after_rerank} "
                f"page={ranked.page_number} chunk_index={ranked.chunk_index} "
                f"dense_score={ranked.dense_score:.4f} bm25_score={ranked.bm25_score:.4f} "
                f"hybrid_score={ranked.hybrid_score:.4f} "
                f"reranker_score={ranked.reranker_score:.4f} "
                f"is_table_chunk={is_table_chunk} heading_context={heading_context!r} "
                f"table_title={table_title!r}"
            )
            print(f"    text_preview: {ranked.text_preview}")


def resolve_chunking_report_path(save_report_arg: str, dataset_path: Path) -> Path:
    if save_report_arg != "__default__":
        return Path(save_report_arg)
    today = date.today().isoformat()
    return Path("data/eval/results") / f"{dataset_path.stem}_chunking_comparison_{today}.json"


def write_chunking_comparison_report(
    path: Path,
    document: DatasetDocument,
    results: ChunkingResults,
    case_comparison_counts: dict[str, int],
    failure_tags: dict[str, dict[str, list[str]]],
) -> None:
    payload: dict[str, object] = {
        "document_label": document.label,
        "strategies": [
            {
                "strategy": strategy,
                "config": asdict(rerank_result.config),
                "aggregate": asdict(rerank_result.aggregate),
                "chunk_stats": asdict(stats),
                "cases": [
                    {
                        "question_id": f"Q{i + 1}",
                        "expected": sorted(
                            location_key(page, idx) for page, idx in case_result.expected
                        ),
                        "ranked_locations": case_result.ranked_locations,
                        "recall_at_1": case_result.recall_at_1,
                        "recall_at_3": case_result.recall_at_3,
                        "recall_at_5": case_result.recall_at_5,
                        "reciprocal_rank": case_result.reciprocal_rank,
                        "retrieval_latency_ms": case_result.retrieval_latency_ms,
                        "reranking_latency_ms": case_result.reranking_latency_ms,
                        "total_latency_ms": case_result.total_latency_ms,
                        "ranked_results": [asdict(r) for r in case_result.ranked_results],
                        "failure_tags": failure_tags.get(strategy, {}).get(f"Q{i + 1}", []),
                    }
                    for i, case_result in enumerate(rerank_result.case_results)
                ],
            }
            for strategy, (rerank_result, stats) in results.items()
        ],
        "case_comparison_counts": case_comparison_counts,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved detailed local report to {path}")
    print("Reminder: this file may contain guideline-derived content - never commit it.")
