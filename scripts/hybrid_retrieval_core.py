"""Shared library for comparing Dense-only vs Hybrid (Dense+BM25)
retrieval quality against a real, local retrieval evaluation dataset
(Issue #24).

Not a CLI entry point - imported by
scripts/compare_retrieval_strategies.py. This is comparison/measurement
tooling only, following the same pattern as
scripts/retrieval_baseline_core.py (Issue #18/#19) and
scripts/pdf_extraction_comparison_core.py (Issue #22): it does not
change production retrieval. app/domain, app/application,
app/infrastructure, and app/api/dependencies.py are all untouched by
this issue - the existing Dense path (InMemoryVectorStore,
SearchChunksService, RetrieveChunksService) is called directly (for
"dense") or composed alongside a new BM25 index (for "hybrid"), never
modified. See docs/adr/0019-hybrid-search-comparison.md.

Both strategies are evaluated against one shared IndexedCorpus (built
once via build_indexed_corpus(), PyMuPDF-only per this issue's fixed
conditions), so "dense" and "hybrid" are compared on identical indexed
content - only the retrieval step differs.

Never commit a dataset file, the PDF it points to, or this script's
--save-report output; data/eval/ is gitignored for exactly this
reason. See docs/evaluation-dataset-format.md.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from app.application.services.chunk_document import ChunkDocumentService
from app.application.services.embed_chunks import EmbedChunksService
from app.application.services.search_chunks import SearchChunksService
from app.domain.models.chunk import Chunk
from app.domain.ports.embedder import Embedder
from app.domain.ports.pdf_extractor import PdfExtractor
from app.infrastructure.chunking.fixed_size_text_splitter import FixedSizeTextSplitter
from app.infrastructure.pdf.pymupdf_extractor import PyMuPdfExtractor
from app.infrastructure.vector_store.in_memory_vector_store import InMemoryVectorStore
from scripts.bm25 import Bm25Index
from scripts.hybrid_scorer import ScoreFuser
from scripts.japanese_tokenizer import tokenize_japanese_text
from scripts.retrieval_baseline_core import (
    DatasetCase,
    DatasetDocument,
    location_key,
    truncate_text,
)
from tests.support.evaluation.metrics import mean, recall_at_k, reciprocal_rank

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

STRATEGY_DENSE = "dense"
STRATEGY_HYBRID = "hybrid"
STRATEGIES: tuple[str, ...] = (STRATEGY_DENSE, STRATEGY_HYBRID)


# --- Corpus indexing (shared by both strategies) ---


@dataclass(frozen=True, slots=True)
class IndexedCorpus:
    vector_store: InMemoryVectorStore
    chunks: list[Chunk]
    bm25_index: Bm25Index
    page_count: int


def build_indexed_corpus(
    document: DatasetDocument,
    passage_embedder: Embedder,
    *,
    chunk_size: int,
    chunk_overlap: int,
    pdf_extractor: PdfExtractor | None = None,
) -> IndexedCorpus:
    """Extracts, chunks, and embeds document.source_path once, storing
    the result in a fresh InMemoryVectorStore and a fresh Bm25Index -
    the single shared corpus both "dense" and "hybrid" strategies are
    evaluated against.

    pdf_extractor defaults to PyMuPdfExtractor() (this issue's fixed
    PDF extractor condition), mirroring
    pdf_extraction_comparison_core.py's PdfExtractor-based extraction.
    """
    extractor = pdf_extractor if pdf_extractor is not None else PyMuPdfExtractor()
    pages = extractor.extract(document.source_path)

    chunks = ChunkDocumentService(
        FixedSizeTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    ).execute(pages)

    embedded_chunks = EmbedChunksService(passage_embedder).execute(chunks)
    vector_store = InMemoryVectorStore()
    vector_store.upsert(embedded_chunks)

    bm25_index = Bm25Index(
        [(chunk.chunk_id, tokenize_japanese_text(chunk.text)) for chunk in chunks]
    )

    return IndexedCorpus(
        vector_store=vector_store, chunks=chunks, bm25_index=bm25_index, page_count=len(pages)
    )


# --- Per-query search ---


@dataclass(frozen=True, slots=True)
class RankedChunkResult:
    """One retrieved chunk for a single question/strategy, for
    --verbose terminal output and --save-report JSON only.

    dense_score/bm25_score are None for the "dense" strategy (BM25 is
    never computed for it - there is nothing to blend). For "hybrid",
    both are always populated: candidates are scored by both signals
    over the full corpus, not just within each signal's own top-k (see
    hybrid_search()).
    """

    rank: int
    page_number: int
    chunk_index: int
    final_score: float
    dense_score: float | None
    bm25_score: float | None
    text_preview: str


def dense_search(
    query_vector: list[float], corpus: IndexedCorpus, top_k: int
) -> list[RankedChunkResult]:
    """Dense-only search, via the existing, unmodified SearchChunksService
    against corpus.vector_store - identical to what RetrieveChunksService
    already does in production.
    """
    results = SearchChunksService(corpus.vector_store).execute(query_vector, top_k)
    return [
        RankedChunkResult(
            rank=rank,
            page_number=result.embedded_chunk.chunk.page_number,
            chunk_index=result.embedded_chunk.chunk.chunk_index,
            final_score=result.score,
            dense_score=result.score,
            bm25_score=None,
            text_preview=truncate_text(result.embedded_chunk.chunk.text),
        )
        for rank, result in enumerate(results, start=1)
    ]


def hybrid_search(
    query_vector: list[float],
    query_tokens: list[str],
    corpus: IndexedCorpus,
    scorer: ScoreFuser,
    *,
    dense_candidate_k: int,
    bm25_candidate_k: int,
    final_top_k: int,
) -> list[RankedChunkResult]:
    """Dense candidates (top dense_candidate_k by cosine similarity) and
    BM25 candidates (top bm25_candidate_k by BM25 score) are unioned,
    then every candidate in that union is scored by *both* signals
    against the full corpus (not only within each signal's own
    candidate list - this avoids inventing a placeholder score for a
    candidate that one signal's top-k missed but the other's did not),
    fused by scorer, and the top final_top_k by fused score is
    returned.

    When dense_candidate_k >= final_top_k, this produces exactly the
    same ranking as dense_search() at scorer alpha=1.0: min-max
    normalization is a monotonic transform of the raw Dense score, and
    the true top final_top_k by Dense score are always included in
    dense_candidate_k's slice by construction.
    """
    total_chunks = len(corpus.chunks)
    if total_chunks == 0:
        return []

    dense_results = corpus.vector_store.search(query_vector, top_k=total_chunks)
    dense_score_by_id = {
        result.embedded_chunk.chunk.chunk_id: result.score for result in dense_results
    }
    chunk_by_id = {
        result.embedded_chunk.chunk.chunk_id: result.embedded_chunk.chunk
        for result in dense_results
    }
    bm25_score_by_id = corpus.bm25_index.score_all(query_tokens)

    dense_candidate_ids = [
        result.embedded_chunk.chunk.chunk_id for result in dense_results[:dense_candidate_k]
    ]
    bm25_candidate_ids = [
        chunk_id for chunk_id, _ in corpus.bm25_index.top_k(query_tokens, bm25_candidate_k)
    ]
    union_ids = set(dense_candidate_ids) | set(bm25_candidate_ids)

    dense_scores = {chunk_id: dense_score_by_id[chunk_id] for chunk_id in union_ids}
    bm25_scores = {chunk_id: bm25_score_by_id[chunk_id] for chunk_id in union_ids}
    fused_scores = scorer.fuse(dense_scores, bm25_scores)

    ranked_ids = sorted(union_ids, key=lambda chunk_id: (-fused_scores[chunk_id], chunk_id))
    ranked_ids = ranked_ids[:final_top_k]

    return [
        RankedChunkResult(
            rank=rank,
            page_number=chunk_by_id[chunk_id].page_number,
            chunk_index=chunk_by_id[chunk_id].chunk_index,
            final_score=fused_scores[chunk_id],
            dense_score=dense_scores[chunk_id],
            bm25_score=bm25_scores[chunk_id],
            text_preview=truncate_text(chunk_by_id[chunk_id].text),
        )
        for rank, chunk_id in enumerate(ranked_ids, start=1)
    ]


# --- Case/aggregate evaluation ---


@dataclass(frozen=True, slots=True)
class StrategyCaseResult:
    question: str
    expected: list[tuple[int, int | None]]
    ranked_locations: list[str]
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    reciprocal_rank: float
    ranked_results: list[RankedChunkResult] = field(default_factory=list)


def _to_case_result(
    case: DatasetCase, ranked_results: list[RankedChunkResult]
) -> StrategyCaseResult:
    if case.granularity == "chunk":
        ranked_ids = [
            location_key(result.page_number, result.chunk_index) for result in ranked_results
        ]
    else:
        ranked_ids = [location_key(result.page_number, None) for result in ranked_results]
    relevant_ids = {location_key(page, idx) for page, idx in case.expected_locations}

    return StrategyCaseResult(
        question=case.question,
        expected=case.expected_locations,
        ranked_locations=ranked_ids,
        recall_at_1=recall_at_k(ranked_ids, relevant_ids, k=1),
        recall_at_3=recall_at_k(ranked_ids, relevant_ids, k=3),
        recall_at_5=recall_at_k(ranked_ids, relevant_ids, k=5),
        reciprocal_rank=reciprocal_rank(ranked_ids, relevant_ids),
        ranked_results=ranked_results,
    )


@dataclass(frozen=True, slots=True)
class StrategyAggregate:
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float


def summarize(case_results: list[StrategyCaseResult]) -> StrategyAggregate:
    return StrategyAggregate(
        recall_at_1=mean([c.recall_at_1 for c in case_results]),
        recall_at_3=mean([c.recall_at_3 for c in case_results]),
        recall_at_5=mean([c.recall_at_5 for c in case_results]),
        mrr=mean([c.reciprocal_rank for c in case_results]),
    )


@dataclass(frozen=True, slots=True)
class StrategyRunConfig:
    strategy: str
    alpha: float | None
    chunk_size: int
    chunk_overlap: int
    top_k: int
    dense_candidate_k: int | None
    bm25_candidate_k: int | None
    embedding_model_name: str
    query_prefix: str
    passage_prefix: str
    case_count: int
    indexed_page_count: int
    indexed_chunk_count: int
    measured_at: str


@dataclass(frozen=True, slots=True)
class StrategyComparisonResult:
    config: StrategyRunConfig
    case_results: list[StrategyCaseResult]
    aggregate: StrategyAggregate


def evaluate_strategy(
    strategy: str,
    corpus: IndexedCorpus,
    cases: list[DatasetCase],
    query_embedder: Embedder,
    *,
    top_k: int,
    dense_candidate_k: int,
    bm25_candidate_k: int,
    alpha: float,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model_name: str,
    scorer: ScoreFuser,
) -> StrategyComparisonResult:
    """Runs every case in cases through the named strategy ("dense" or
    "hybrid") against the already-built corpus, returning per-question
    and aggregate Recall@1/3/5/MRR.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy!r} (available: {list(STRATEGIES)})")

    case_results = []
    for case in cases:
        query_vector = query_embedder.embed([case.question])[0]
        if strategy == STRATEGY_DENSE:
            ranked_results = dense_search(query_vector, corpus, top_k)
        else:
            query_tokens = tokenize_japanese_text(case.question)
            ranked_results = hybrid_search(
                query_vector,
                query_tokens,
                corpus,
                scorer,
                dense_candidate_k=dense_candidate_k,
                bm25_candidate_k=bm25_candidate_k,
                final_top_k=top_k,
            )
        case_results.append(_to_case_result(case, ranked_results))

    is_hybrid = strategy == STRATEGY_HYBRID
    config = StrategyRunConfig(
        strategy=strategy,
        alpha=alpha if is_hybrid else None,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        dense_candidate_k=dense_candidate_k if is_hybrid else None,
        bm25_candidate_k=bm25_candidate_k if is_hybrid else None,
        embedding_model_name=embedding_model_name,
        query_prefix=QUERY_PREFIX,
        passage_prefix=PASSAGE_PREFIX,
        case_count=len(cases),
        indexed_page_count=corpus.page_count,
        indexed_chunk_count=len(corpus.chunks),
        measured_at=date.today().isoformat(),
    )
    return StrategyComparisonResult(
        config=config, case_results=case_results, aggregate=summarize(case_results)
    )


# --- Reporting ---

_TABLE_HEADER = (
    f"{'strategy':>8} | {'alpha':>6} | {'Recall@1':>8} | {'Recall@3':>8} | "
    f"{'Recall@5':>8} | {'MRR@k':>6}"
)


def _alpha_display(alpha: float | None) -> str:
    return f"{alpha:.1f}" if alpha is not None else "-"


def print_comparison_table(results: list[StrategyComparisonResult]) -> None:
    print(f"\n{_TABLE_HEADER}")
    print("-" * len(_TABLE_HEADER))
    for result in results:
        aggregate = result.aggregate
        print(
            f"{result.config.strategy:>8} | {_alpha_display(result.config.alpha):>6} | "
            f"{aggregate.recall_at_1:>8.3f} | {aggregate.recall_at_3:>8.3f} | "
            f"{aggregate.recall_at_5:>8.3f} | {aggregate.mrr:>6.3f}"
        )


def markdown_comparison_table(label: str, results: list[StrategyComparisonResult]) -> str:
    first_config = results[0].config
    header = (
        f"## Hybrid search comparison ({first_config.measured_at})\n\n"
        f"- Document: {label}\n"
        f"- Cases: {first_config.case_count}\n"
        f"- chunk_size={first_config.chunk_size}, chunk_overlap={first_config.chunk_overlap}, "
        f"top_k={first_config.top_k}\n"
        f"- Embedding: sentence_transformers / {first_config.embedding_model_name} "
        f'(query prefix: "{first_config.query_prefix}", '
        f'passage prefix: "{first_config.passage_prefix}")\n\n'
        "| strategy | alpha | Recall@1 | Recall@3 | Recall@5 | "
        f"MRR@{first_config.top_k} |\n"
        "|---|---:|---:|---:|---:|---:|\n"
    )
    rows = "".join(_markdown_row(result) for result in results)
    return header + rows


def _markdown_row(result: StrategyComparisonResult) -> str:
    alpha_cell = f"{result.config.alpha:.1f}" if result.config.alpha is not None else ""
    aggregate = result.aggregate
    return (
        f"| {result.config.strategy} | {alpha_cell} | "
        f"{aggregate.recall_at_1:.2f} | {aggregate.recall_at_3:.2f} | "
        f"{aggregate.recall_at_5:.2f} | {aggregate.mrr:.2f} |\n"
    )


def print_markdown_comparison_table(label: str, results: list[StrategyComparisonResult]) -> None:
    print("\n" + "=" * 70)
    print("Markdown snippet for docs/hybrid-search-comparison-results.md")
    print("(review before pasting - confirm nothing identifying leaks):")
    print("=" * 70)
    print(markdown_comparison_table(label, results))


def print_verbose_strategy_results(results: list[StrategyComparisonResult]) -> None:
    """Per-question breakdown for each strategy (local use only - never
    paste anywhere committed): question, expected page(s), retrieved
    pages, and per-rank final/dense/bm25 score plus text_preview.
    """
    for result in results:
        strategy_label = result.config.strategy
        if result.config.alpha is not None:
            strategy_label += f" (alpha={result.config.alpha})"
        print(f"\n{'=' * 70}\nstrategy: {strategy_label}\n{'=' * 70}")

        for case_result in result.case_results:
            expected_pages = sorted({page for page, _ in case_result.expected})
            retrieved_pages = [r.page_number for r in case_result.ranked_results]
            print(f"\nQ: {case_result.question}")
            print(f"  expected page(s): {expected_pages}")
            print(f"  retrieved pages: {retrieved_pages}")
            for ranked in case_result.ranked_results:
                dense_display = (
                    f"{ranked.dense_score:.4f}" if ranked.dense_score is not None else "-"
                )
                bm25_display = f"{ranked.bm25_score:.4f}" if ranked.bm25_score is not None else "-"
                print(
                    f"    rank={ranked.rank} final_score={ranked.final_score:.4f} "
                    f"dense_score={dense_display} bm25_score={bm25_display} "
                    f"strategy={result.config.strategy}"
                )
                print(f"    text_preview: {ranked.text_preview}")


def resolve_strategy_report_path(save_report_arg: str, dataset_path: Path) -> Path:
    if save_report_arg != "__default__":
        return Path(save_report_arg)
    today = date.today().isoformat()
    return Path("data/eval/results") / f"{dataset_path.stem}_hybrid_search_comparison_{today}.json"


def write_strategy_report(
    path: Path, document: DatasetDocument, results: list[StrategyComparisonResult]
) -> None:
    payload = {
        "document_label": document.label,
        "strategies": [
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
            for result in results
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved detailed local report to {path}")
    print("Reminder: this file may contain guideline-derived content - never commit it.")
