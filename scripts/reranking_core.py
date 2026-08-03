"""Shared library for comparing Hybrid Search with and without a Cross
Encoder reranking step (Issue #25).

Not a CLI entry point - imported by
scripts/compare_reranking_strategies.py. Builds on
scripts/hybrid_retrieval_core.py (Issue #24): "dense" and "hybrid"
strategies are evaluated via that module's unmodified
build_indexed_corpus()/dense_search()/evaluate_strategy(); this module
adds only the "hybrid_rerank" strategy, which reranks Hybrid's
candidates (rank_hybrid_candidates(), *before* Hybrid's own final_top_k
cut - see docs/adr/0020-cross-encoder-reranker-comparison.md,
requirement 8) with a Cross Encoder (scripts/reranker.py's Reranker
Protocol).

This is comparison/measurement only: app/domain, app/application,
app/infrastructure, and app/api/dependencies.py are all untouched, and
production retrieval is not changed.

Never commit a dataset file, the PDF it points to, or this script's
--save-report output; data/eval/ is gitignored for exactly this
reason. See docs/evaluation-dataset-format.md.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from app.domain.ports.embedder import Embedder
from scripts.hybrid_retrieval_core import (
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    STRATEGY_DENSE,
    STRATEGY_HYBRID,
    IndexedCorpus,
    StrategyCaseResult,
    StrategyComparisonResult,
    rank_hybrid_candidates,
)
from scripts.hybrid_scorer import ScoreFuser
from scripts.japanese_tokenizer import tokenize_japanese_text
from scripts.reranker import Reranker
from scripts.retrieval_baseline_core import (
    DatasetCase,
    DatasetDocument,
    location_key,
    truncate_text,
)
from tests.support.evaluation.metrics import mean, recall_at_k, reciprocal_rank

STRATEGY_HYBRID_RERANK = "hybrid_rerank"

ALL_STRATEGIES: tuple[str, ...] = (STRATEGY_DENSE, STRATEGY_HYBRID, STRATEGY_HYBRID_RERANK)


# --- Per-query rerank search ---


@dataclass(frozen=True, slots=True)
class RerankedChunkResult:
    """One retrieved chunk after Cross Encoder reranking, for
    --verbose terminal output and --save-report JSON only.

    rank_before_rerank is the candidate's position (1-based) within the
    reranker_candidate_k slice of Hybrid's own ranking, *before*
    Hybrid's final_top_k cut (requirement 8) - not necessarily <=
    final_top_k. rank_after_rerank is its position after reranking and
    the final_top_k cut, i.e. what is actually returned.
    """

    rank_before_rerank: int
    rank_after_rerank: int
    page_number: int
    chunk_index: int
    dense_score: float
    bm25_score: float
    hybrid_score: float
    reranker_score: float
    text_preview: str


def hybrid_rerank_search(
    query_vector: list[float],
    query_tokens: list[str],
    query_text: str,
    corpus: IndexedCorpus,
    scorer: ScoreFuser,
    reranker: Reranker,
    *,
    dense_candidate_k: int,
    bm25_candidate_k: int,
    reranker_candidate_k: int,
    final_top_k: int,
) -> tuple[list[RerankedChunkResult], float, float]:
    """Returns (ranked_results, retrieval_latency_ms, reranking_latency_ms).

    Candidates are Hybrid's full Dense/BM25 union
    (rank_hybrid_candidates()), cut to the top reranker_candidate_k by
    hybrid_score - *not* Hybrid's final_top_k - before Cross Encoder
    reranking (requirement 8: reranking before a 5-item cut has more
    room to change the outcome). The Cross Encoder receives each
    candidate's full chunk text (chunk.text - never the truncated
    text_preview, and never page number or other metadata; requirement
    9). Ties in reranker_score keep each candidate's original Hybrid
    rank: Python's sort() is stable, and candidates arrive from
    rank_hybrid_candidates() already in Hybrid rank order, so no
    explicit tie-break key is needed.
    """
    retrieval_start = time.perf_counter()
    candidates = rank_hybrid_candidates(
        query_vector,
        query_tokens,
        corpus,
        scorer,
        dense_candidate_k=dense_candidate_k,
        bm25_candidate_k=bm25_candidate_k,
    )[:reranker_candidate_k]
    retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

    if not candidates:
        return [], retrieval_latency_ms, 0.0

    texts = [candidate.chunk.text for candidate in candidates]
    reranking_start = time.perf_counter()
    reranker_scores = reranker.score(query_text, texts)
    reranking_latency_ms = (time.perf_counter() - reranking_start) * 1000

    ranked_before = list(enumerate(zip(candidates, reranker_scores, strict=True), start=1))
    ranked_after = sorted(ranked_before, key=lambda item: -item[1][1])[:final_top_k]

    results = [
        RerankedChunkResult(
            rank_before_rerank=rank_before,
            rank_after_rerank=rank_after,
            page_number=candidate.chunk.page_number,
            chunk_index=candidate.chunk.chunk_index,
            dense_score=candidate.dense_score,
            bm25_score=candidate.bm25_score,
            hybrid_score=candidate.hybrid_score,
            reranker_score=reranker_score,
            text_preview=truncate_text(candidate.chunk.text),
        )
        for rank_after, (rank_before, (candidate, reranker_score)) in enumerate(
            ranked_after, start=1
        )
    ]
    return results, retrieval_latency_ms, reranking_latency_ms


# --- Case/aggregate evaluation ---


@dataclass(frozen=True, slots=True)
class RerankCaseResult:
    question: str
    expected: list[tuple[int, int | None]]
    ranked_locations: list[str]
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    reciprocal_rank: float
    retrieval_latency_ms: float
    reranking_latency_ms: float
    total_latency_ms: float
    ranked_results: list[RerankedChunkResult] = field(default_factory=list)


def _to_rerank_case_result(
    case: DatasetCase,
    ranked_results: list[RerankedChunkResult],
    retrieval_latency_ms: float,
    reranking_latency_ms: float,
) -> RerankCaseResult:
    if case.granularity == "chunk":
        ranked_ids = [
            location_key(result.page_number, result.chunk_index) for result in ranked_results
        ]
    else:
        ranked_ids = [location_key(result.page_number, None) for result in ranked_results]
    relevant_ids = {location_key(page, idx) for page, idx in case.expected_locations}

    return RerankCaseResult(
        question=case.question,
        expected=case.expected_locations,
        ranked_locations=ranked_ids,
        recall_at_1=recall_at_k(ranked_ids, relevant_ids, k=1),
        recall_at_3=recall_at_k(ranked_ids, relevant_ids, k=3),
        recall_at_5=recall_at_k(ranked_ids, relevant_ids, k=5),
        reciprocal_rank=reciprocal_rank(ranked_ids, relevant_ids),
        retrieval_latency_ms=retrieval_latency_ms,
        reranking_latency_ms=reranking_latency_ms,
        total_latency_ms=retrieval_latency_ms + reranking_latency_ms,
        ranked_results=ranked_results,
    )


@dataclass(frozen=True, slots=True)
class RerankAggregate:
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    avg_retrieval_latency_ms: float
    avg_reranking_latency_ms: float
    avg_total_latency_ms: float


def summarize_rerank(case_results: list[RerankCaseResult]) -> RerankAggregate:
    return RerankAggregate(
        recall_at_1=mean([c.recall_at_1 for c in case_results]),
        recall_at_3=mean([c.recall_at_3 for c in case_results]),
        recall_at_5=mean([c.recall_at_5 for c in case_results]),
        mrr=mean([c.reciprocal_rank for c in case_results]),
        avg_retrieval_latency_ms=mean([c.retrieval_latency_ms for c in case_results]),
        avg_reranking_latency_ms=mean([c.reranking_latency_ms for c in case_results]),
        avg_total_latency_ms=mean([c.total_latency_ms for c in case_results]),
    )


@dataclass(frozen=True, slots=True)
class RerankRunConfig:
    strategy: str
    alpha: float
    reranker_model_name: str
    device: str
    batch_size: int
    chunk_size: int
    chunk_overlap: int
    top_k: int
    dense_candidate_k: int
    bm25_candidate_k: int
    reranker_candidate_k: int
    embedding_model_name: str
    query_prefix: str
    passage_prefix: str
    case_count: int
    indexed_page_count: int
    indexed_chunk_count: int
    measured_at: str


@dataclass(frozen=True, slots=True)
class RerankComparisonResult:
    config: RerankRunConfig
    case_results: list[RerankCaseResult]
    aggregate: RerankAggregate


def evaluate_hybrid_rerank(
    corpus: IndexedCorpus,
    cases: list[DatasetCase],
    query_embedder: Embedder,
    scorer: ScoreFuser,
    reranker: Reranker,
    *,
    top_k: int,
    dense_candidate_k: int,
    bm25_candidate_k: int,
    reranker_candidate_k: int,
    alpha: float,
    reranker_model_name: str,
    device: str,
    batch_size: int,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model_name: str,
) -> RerankComparisonResult:
    """Runs every case in cases through Hybrid Search + Cross Encoder
    reranking against the already-built corpus (built the same way as
    for "dense"/"hybrid" - see
    scripts/hybrid_retrieval_core.build_indexed_corpus()), returning
    per-question and aggregate Recall@1/3/5/MRR plus latency.

    reranker must already have its model loaded (see
    scripts/reranker.load_cross_encoder()) - it is reused for every
    question, never reloaded per query.
    """
    case_results = []
    for case in cases:
        query_vector = query_embedder.embed([case.question])[0]
        query_tokens = tokenize_japanese_text(case.question)

        ranked_results, retrieval_latency_ms, reranking_latency_ms = hybrid_rerank_search(
            query_vector,
            query_tokens,
            case.question,
            corpus,
            scorer,
            reranker,
            dense_candidate_k=dense_candidate_k,
            bm25_candidate_k=bm25_candidate_k,
            reranker_candidate_k=reranker_candidate_k,
            final_top_k=top_k,
        )
        case_results.append(
            _to_rerank_case_result(case, ranked_results, retrieval_latency_ms, reranking_latency_ms)
        )

    config = RerankRunConfig(
        strategy=STRATEGY_HYBRID_RERANK,
        alpha=alpha,
        reranker_model_name=reranker_model_name,
        device=device,
        batch_size=batch_size,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        dense_candidate_k=dense_candidate_k,
        bm25_candidate_k=bm25_candidate_k,
        reranker_candidate_k=reranker_candidate_k,
        embedding_model_name=embedding_model_name,
        query_prefix=QUERY_PREFIX,
        passage_prefix=PASSAGE_PREFIX,
        case_count=len(cases),
        indexed_page_count=corpus.page_count,
        indexed_chunk_count=len(corpus.chunks),
        measured_at=date.today().isoformat(),
    )
    return RerankComparisonResult(
        config=config, case_results=case_results, aggregate=summarize_rerank(case_results)
    )


# --- Hybrid -> Hybrid+Rerank per-question categorization (requirement 18) ---

CATEGORY_HYBRID_CORRECT_RERANK_WORSE = "hybrid_correct_rerank_worse"
CATEGORY_HYBRID_INCORRECT_RERANK_BETTER = "hybrid_incorrect_rerank_better"
CATEGORY_RANK_IMPROVED = "rank_improved"
CATEGORY_RANK_WORSENED = "rank_worsened"
CATEGORY_UNCHANGED_CORRECT = "unchanged_correct"
CATEGORY_UNCHANGED_MISSING = "unchanged_missing"

_CATEGORIES: tuple[str, ...] = (
    CATEGORY_HYBRID_CORRECT_RERANK_WORSE,
    CATEGORY_HYBRID_INCORRECT_RERANK_BETTER,
    CATEGORY_RANK_IMPROVED,
    CATEGORY_RANK_WORSENED,
    CATEGORY_UNCHANGED_CORRECT,
    CATEGORY_UNCHANGED_MISSING,
)


@dataclass(frozen=True, slots=True)
class RerankCaseComparison:
    """One question's Hybrid -> Hybrid+Rerank comparison. No question
    text, expected pages, or text_preview here - this is a
    classification only, safe to print/aggregate without repeating
    guideline-derived content beyond what the underlying case results
    already carry.
    """

    category: str
    hybrid_rank: int | None
    rerank_rank: int | None


def _first_hit_rank(reciprocal_rank_value: float) -> int | None:
    if reciprocal_rank_value == 0.0:
        return None
    return round(1.0 / reciprocal_rank_value)


def compare_hybrid_to_rerank(
    hybrid_case: StrategyCaseResult, rerank_case: RerankCaseResult
) -> RerankCaseComparison:
    hybrid_rank = _first_hit_rank(hybrid_case.reciprocal_rank)
    rerank_rank = _first_hit_rank(rerank_case.reciprocal_rank)

    category: str
    if hybrid_rank is not None and rerank_rank is None:
        category = CATEGORY_HYBRID_CORRECT_RERANK_WORSE
    elif hybrid_rank is None and rerank_rank is not None:
        category = CATEGORY_HYBRID_INCORRECT_RERANK_BETTER
    elif hybrid_rank is not None and rerank_rank is not None:
        if rerank_rank < hybrid_rank:
            category = CATEGORY_RANK_IMPROVED
        elif rerank_rank > hybrid_rank:
            category = CATEGORY_RANK_WORSENED
        else:
            category = CATEGORY_UNCHANGED_CORRECT
    else:
        category = CATEGORY_UNCHANGED_MISSING

    return RerankCaseComparison(category=category, hybrid_rank=hybrid_rank, rerank_rank=rerank_rank)


def summarize_rerank_comparison(
    hybrid_result: StrategyComparisonResult, rerank_result: RerankComparisonResult
) -> dict[str, int]:
    """Counts each requirement-18 category across all cases."""
    counts: dict[str, int] = dict.fromkeys(_CATEGORIES, 0)
    for hybrid_case, rerank_case in zip(
        hybrid_result.case_results, rerank_result.case_results, strict=True
    ):
        comparison = compare_hybrid_to_rerank(hybrid_case, rerank_case)
        counts[comparison.category] += 1
    return counts


def print_rerank_comparison_summary(counts: dict[str, int]) -> None:
    print("\nHybrid -> Hybrid+Rerank per-question breakdown:")
    print(
        f"  hybrid correct, rerank worse (now missing): "
        f"{counts[CATEGORY_HYBRID_CORRECT_RERANK_WORSE]}"
    )
    print(
        f"  hybrid incorrect, rerank better (now correct): "
        f"{counts[CATEGORY_HYBRID_INCORRECT_RERANK_BETTER]}"
    )
    print(f"  rank improved (both correct): {counts[CATEGORY_RANK_IMPROVED]}")
    print(f"  rank worsened (both correct): {counts[CATEGORY_RANK_WORSENED]}")
    print(f"  unchanged, still correct: {counts[CATEGORY_UNCHANGED_CORRECT]}")
    print(f"  unchanged, still missing (top-k): {counts[CATEGORY_UNCHANGED_MISSING]}")


# --- Reporting: unified dense/hybrid/hybrid_rerank comparison table ---


@dataclass(frozen=True, slots=True)
class ComparisonRow:
    strategy: str
    alpha: float | None
    reranker_model_name: str | None
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    avg_latency_ms: float


def row_from_strategy_result(
    result: StrategyComparisonResult, avg_latency_ms: float
) -> ComparisonRow:
    return ComparisonRow(
        strategy=result.config.strategy,
        alpha=result.config.alpha,
        reranker_model_name=None,
        recall_at_1=result.aggregate.recall_at_1,
        recall_at_3=result.aggregate.recall_at_3,
        recall_at_5=result.aggregate.recall_at_5,
        mrr=result.aggregate.mrr,
        avg_latency_ms=avg_latency_ms,
    )


def row_from_rerank_result(result: RerankComparisonResult) -> ComparisonRow:
    return ComparisonRow(
        strategy=result.config.strategy,
        alpha=result.config.alpha,
        reranker_model_name=result.config.reranker_model_name,
        recall_at_1=result.aggregate.recall_at_1,
        recall_at_3=result.aggregate.recall_at_3,
        recall_at_5=result.aggregate.recall_at_5,
        mrr=result.aggregate.mrr,
        avg_latency_ms=result.aggregate.avg_total_latency_ms,
    )


_TABLE_HEADER = (
    f"{'strategy':>13} | {'alpha':>6} | {'reranker':>12} | {'Recall@1':>8} | "
    f"{'Recall@3':>8} | {'Recall@5':>8} | {'MRR@k':>6} | {'avg_latency_ms':>14}"
)


def _alpha_display(alpha: float | None) -> str:
    return f"{alpha:.1f}" if alpha is not None else "-"


def _reranker_display(reranker_model_name: str | None) -> str:
    return reranker_model_name if reranker_model_name is not None else "none"


def print_comparison_table(rows: list[ComparisonRow]) -> None:
    print(f"\n{_TABLE_HEADER}")
    print("-" * len(_TABLE_HEADER))
    for row in rows:
        print(
            f"{row.strategy:>13} | {_alpha_display(row.alpha):>6} | "
            f"{_reranker_display(row.reranker_model_name):>12} | "
            f"{row.recall_at_1:>8.3f} | {row.recall_at_3:>8.3f} | {row.recall_at_5:>8.3f} | "
            f"{row.mrr:>6.3f} | {row.avg_latency_ms:>14.1f}"
        )


def _markdown_row(row: ComparisonRow) -> str:
    alpha_cell = f"{row.alpha:.1f}" if row.alpha is not None else ""
    reranker_cell = row.reranker_model_name if row.reranker_model_name is not None else "none"
    return (
        f"| {row.strategy} | {alpha_cell} | {reranker_cell} | "
        f"{row.recall_at_1:.2f} | {row.recall_at_3:.2f} | {row.recall_at_5:.2f} | "
        f"{row.mrr:.2f} | {row.avg_latency_ms:.1f} |\n"
    )


def markdown_comparison_table(
    label: str, case_count: int, top_k: int, rows: list[ComparisonRow]
) -> str:
    header = (
        f"## Cross Encoder reranking comparison ({date.today().isoformat()})\n\n"
        f"- Document: {label}\n"
        f"- Cases: {case_count}\n"
        f"- top_k={top_k}\n\n"
        "| strategy | alpha | reranker | Recall@1 | Recall@3 | Recall@5 | "
        f"MRR@{top_k} | avg_latency_ms |\n"
        "|---|---:|---|---:|---:|---:|---:|---:|\n"
    )
    rows_markdown = "".join(_markdown_row(row) for row in rows)
    return header + rows_markdown


def print_markdown_comparison_table(
    label: str, case_count: int, top_k: int, rows: list[ComparisonRow]
) -> None:
    print("\n" + "=" * 70)
    print("Markdown snippet for docs/cross-encoder-reranker-comparison-results.md")
    print("(review before pasting - confirm nothing identifying leaks):")
    print("=" * 70)
    print(markdown_comparison_table(label, case_count, top_k, rows))


def print_verbose_rerank_results(result: RerankComparisonResult) -> None:
    """Per-question breakdown for the hybrid_rerank strategy (local use
    only - never paste anywhere committed): question, expected page(s),
    retrieved pages, and per-rank before/after rerank rank plus
    dense/bm25/hybrid/reranker score and text_preview.
    """
    strategy_label = f"{result.config.strategy} (alpha={result.config.alpha})"
    print(f"\n{'=' * 70}\nstrategy: {strategy_label}\n{'=' * 70}")
    for case_result in result.case_results:
        expected_pages = sorted({page for page, _ in case_result.expected})
        retrieved_pages = [r.page_number for r in case_result.ranked_results]
        print(f"\nQ: {case_result.question}")
        print(f"  expected page(s): {expected_pages}")
        print(f"  retrieved pages: {retrieved_pages}")
        for ranked in case_result.ranked_results:
            print(
                f"    rank_before_rerank={ranked.rank_before_rerank} "
                f"rank_after_rerank={ranked.rank_after_rerank} "
                f"dense_score={ranked.dense_score:.4f} bm25_score={ranked.bm25_score:.4f} "
                f"hybrid_score={ranked.hybrid_score:.4f} "
                f"reranker_score={ranked.reranker_score:.4f} strategy={result.config.strategy}"
            )
            print(f"    text_preview: {ranked.text_preview}")


def resolve_rerank_report_path(save_report_arg: str, dataset_path: Path) -> Path:
    if save_report_arg != "__default__":
        return Path(save_report_arg)
    today = date.today().isoformat()
    return Path("data/eval/results") / f"{dataset_path.stem}_reranking_comparison_{today}.json"


def write_rerank_comparison_report(
    path: Path,
    document: DatasetDocument,
    strategy_results: list[StrategyComparisonResult],
    rerank_result: RerankComparisonResult | None,
    comparison_counts: dict[str, int] | None,
) -> None:
    strategies_payload = [
        {
            "config": asdict(result.config),
            "aggregate": asdict(result.aggregate),
            "cases": [
                {
                    "question": case_result.question,
                    "expected": sorted(
                        location_key(page, idx) for page, idx in case_result.expected
                    ),
                    "ranked_locations": case_result.ranked_locations,
                    "recall_at_1": case_result.recall_at_1,
                    "recall_at_3": case_result.recall_at_3,
                    "recall_at_5": case_result.recall_at_5,
                    "reciprocal_rank": case_result.reciprocal_rank,
                    "ranked_results": [asdict(ranked) for ranked in case_result.ranked_results],
                }
                for case_result in result.case_results
            ],
        }
        for result in strategy_results
    ]
    if rerank_result is not None:
        strategies_payload.append(
            {
                "config": asdict(rerank_result.config),
                "aggregate": asdict(rerank_result.aggregate),
                "cases": [
                    {
                        "question": case_result.question,
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
                        "ranked_results": [asdict(ranked) for ranked in case_result.ranked_results],
                    }
                    for case_result in rerank_result.case_results
                ],
            }
        )

    payload: dict[str, object] = {
        "document_label": document.label,
        "strategies": strategies_payload,
    }
    if comparison_counts is not None:
        payload["hybrid_vs_hybrid_rerank_comparison"] = comparison_counts

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved detailed local report to {path}")
    print("Reminder: this file may contain guideline-derived content - never commit it.")
