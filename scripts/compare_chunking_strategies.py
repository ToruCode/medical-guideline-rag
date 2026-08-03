"""Compare fixed-size vs table-aware chunking under Hybrid+Cross-Encoder-
Rerank retrieval, against the same real, local evaluation dataset
(Issue #30).

Background: Issue #29's 30-question failure analysis found no
evaluation-label errors, but did find duplicate_or_near_duplicate_content
as the most common retrieval-pipeline-side cause (Q11, Q29, Q30) -
tables sharing the same substance names/values across pages, with
fixed-size chunking sometimes separating a table's heading/title/column
headers from its data rows. This script measures whether a rule-based,
table-aware chunker (scripts/table_aware_chunking.py) that keeps that
context attached improves Recall@1/3/5/MRR@5 over the existing fixed-
size chunker, under the exact retrieval configuration Issue #29
recommended (hybrid_rerank, alpha=0.7, reranker_candidate_k=5). See
docs/adr/0021-table-aware-chunking-comparison.md for the full design
(detection heuristics and their limitations, table_max_chars vs
table_row_group_size priority, comparison methodology).

This is comparison/measurement only - production chunking
(app/infrastructure/chunking/fixed_size_text_splitter.py,
app/api/dependencies.py) is not changed.

Not a CLI entry point test - the real PDF and dataset exist only on the
operator's machine and are never committed; this is a one-off/
occasional measurement, not a CI gate.

Dataset format: see docs/evaluation-dataset-format.md. Never commit a
dataset file, the PDF it points to, or this script's --save-report
output; data/eval/ is gitignored for exactly this reason.

Usage (from the repo root):

    uv run python -m scripts.compare_chunking_strategies \\
        --dataset data/eval/my_guideline_qa.json --save-report

    # table_aware only, with a larger table_max_chars:
    uv run python -m scripts.compare_chunking_strategies \\
        --dataset data/eval/my_guideline_qa.json --strategies table_aware \\
        --table-max-chars 1500
"""

import argparse
from pathlib import Path

from app.infrastructure.embedding.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
    load_sentence_transformer_model,
)
from app.infrastructure.pdf.pymupdf_extractor import PyMuPdfExtractor
from scripts.chunking_comparison_core import (
    STRATEGIES,
    ChunkingResults,
    auto_tag_failure_causes,
    evaluate_chunking_strategy,
    print_case_comparison_summary,
    print_comparison_table,
    print_markdown_comparison_table,
    print_verbose_strategy_results,
    resolve_chunking_report_path,
    summarize_case_comparisons,
    write_chunking_comparison_report,
)
from scripts.hybrid_retrieval_core import PASSAGE_PREFIX, QUERY_PREFIX
from scripts.hybrid_scorer import HybridScorer
from scripts.reranker import CrossEncoderReranker, load_cross_encoder
from scripts.retrieval_baseline_core import load_dataset
from scripts.table_aware_chunking import TableAwareChunk

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TABLE_MAX_CHARS = 1000
DEFAULT_TABLE_ROW_GROUP_SIZE = 20
DEFAULT_TOP_K = 5
DEFAULT_DENSE_CANDIDATE_K = 20
DEFAULT_BM25_CANDIDATE_K = 20
DEFAULT_RERANKER_CANDIDATE_K = 5
DEFAULT_ALPHA = 0.7
DEFAULT_EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
DEFAULT_RERANKER_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
DEFAULT_DEVICE = "cpu"
DEFAULT_BATCH_SIZE = 8


def _parse_strategies(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise argparse.ArgumentTypeError("--strategies must contain at least one value")
    unknown = [name for name in names if name not in STRATEGIES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown strategy(ies): {unknown} (available: {list(STRATEGIES)})"
        )
    return names


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare fixed-size vs table-aware chunking under Hybrid+Cross-"
            "Encoder-Rerank retrieval accuracy (Recall@1/3/5, MRR) against a "
            "local, real evaluation dataset. See docs/evaluation-dataset-format.md."
        )
    )
    parser.add_argument(
        "--dataset", required=True, help="Path to a local dataset JSON file (never committed)."
    )
    parser.add_argument(
        "--strategies",
        type=_parse_strategies,
        default=list(STRATEGIES),
        metavar="NAME,NAME,...",
        help=f"Comma-separated strategies to compare (default: all of {list(STRATEGIES)}).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"chunk_size for prose (both strategies) (default: {DEFAULT_CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"chunk_overlap for prose (both strategies) (default: {DEFAULT_CHUNK_OVERLAP}).",
    )
    parser.add_argument(
        "--table-max-chars",
        type=int,
        default=DEFAULT_TABLE_MAX_CHARS,
        help=(
            "Max characters per table chunk (table_aware only); takes priority "
            f"over --table-row-group-size (default: {DEFAULT_TABLE_MAX_CHARS})."
        ),
    )
    parser.add_argument(
        "--table-row-group-size",
        type=int,
        default=DEFAULT_TABLE_ROW_GROUP_SIZE,
        help=(
            "Max table data rows per chunk before splitting (table_aware only); "
            "an initial value for this comparison, not a tuned optimum "
            f"(default: {DEFAULT_TABLE_ROW_GROUP_SIZE})."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Final retrieval depth; must be >= 5 (default: {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--dense-candidate-k",
        type=int,
        default=DEFAULT_DENSE_CANDIDATE_K,
        help=(
            f"Dense candidate depth for hybrid's union step (default: {DEFAULT_DENSE_CANDIDATE_K})."
        ),
    )
    parser.add_argument(
        "--bm25-candidate-k",
        type=int,
        default=DEFAULT_BM25_CANDIDATE_K,
        help=f"BM25 candidate depth for hybrid's union step (default: {DEFAULT_BM25_CANDIDATE_K}).",
    )
    parser.add_argument(
        "--reranker-candidate-k",
        type=int,
        default=DEFAULT_RERANKER_CANDIDATE_K,
        help=(
            f"Hybrid candidates passed to the Cross Encoder "
            f"(default: {DEFAULT_RERANKER_CANDIDATE_K})."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help=f"Dense weight in Hybrid's fused score, in [0.0, 1.0] (default: {DEFAULT_ALPHA}).",
    )
    parser.add_argument(
        "--embedding-model-name",
        default=DEFAULT_EMBEDDING_MODEL_NAME,
        help=f"sentence-transformers model (default: {DEFAULT_EMBEDDING_MODEL_NAME}).",
    )
    parser.add_argument(
        "--reranker-model-name",
        default=DEFAULT_RERANKER_MODEL_NAME,
        help=f"sentence-transformers CrossEncoder model (default: {DEFAULT_RERANKER_MODEL_NAME}).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default=DEFAULT_DEVICE,
        help=f"Device for the Cross Encoder model (default: {DEFAULT_DEVICE}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Cross Encoder inference batch size (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Also print each strategy's per-question breakdown, including "
            "rank/scores/is_table_chunk/heading_context/table_title/text_preview "
            "(local use only)."
        ),
    )
    parser.add_argument(
        "--save-report",
        nargs="?",
        const="__default__",
        default=None,
        metavar="PATH",
        help=(
            "Save detailed per-strategy, per-question results locally (never "
            "commit this file). Defaults to "
            "data/eval/results/<dataset-name>_chunking_comparison_<date>.json "
            "when given without a value."
        ),
    )
    return parser.parse_args()


def main() -> None:  # noqa: C901 - orchestration; each branch is a single call-out.
    args = _parse_args()
    dataset_path = Path(args.dataset)
    if args.top_k < 5:
        raise SystemExit("--top-k must be at least 5 (Recall@5 requires it).")
    if not 0.0 <= args.alpha <= 1.0:
        raise SystemExit("--alpha must be within [0.0, 1.0].")
    if args.dense_candidate_k < args.top_k:
        raise SystemExit("--dense-candidate-k must be >= --top-k.")
    if args.bm25_candidate_k < args.top_k:
        raise SystemExit("--bm25-candidate-k must be >= --top-k.")
    if args.reranker_candidate_k < args.top_k:
        raise SystemExit("--reranker-candidate-k must be >= --top-k.")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")
    if args.table_max_chars <= 0:
        raise SystemExit("--table-max-chars must be positive.")
    if args.table_row_group_size <= 0:
        raise SystemExit("--table-row-group-size must be positive.")

    document, cases = load_dataset(dataset_path)
    model = load_sentence_transformer_model(args.embedding_model_name)
    passage_embedder = SentenceTransformerEmbedder(model, prefix=PASSAGE_PREFIX)
    query_embedder = SentenceTransformerEmbedder(model, prefix=QUERY_PREFIX)

    print(f"\nExtracting {document.source_path} (extractor=pymupdf) ...")
    pages = PyMuPdfExtractor().extract(document.source_path)
    document_id = pages[0].document_id if pages else ""

    scorer = HybridScorer(alpha=args.alpha)

    print(f"\nLoading reranker model={args.reranker_model_name} (device={args.device}) ...")
    reranker_model = load_cross_encoder(args.reranker_model_name, args.device)
    reranker = CrossEncoderReranker(reranker_model, batch_size=args.batch_size)

    results: ChunkingResults = {}
    metadata_by_strategy: dict[str, dict[str, TableAwareChunk] | None] = {}
    for strategy in args.strategies:
        print(f"\nEvaluating strategy={strategy} ...")
        rerank_result, stats, metadata = evaluate_chunking_strategy(
            strategy,
            pages,
            cases,
            passage_embedder,
            query_embedder,
            scorer,
            reranker,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            table_max_chars=args.table_max_chars,
            table_row_group_size=args.table_row_group_size,
            top_k=args.top_k,
            dense_candidate_k=args.dense_candidate_k,
            bm25_candidate_k=args.bm25_candidate_k,
            reranker_candidate_k=args.reranker_candidate_k,
            alpha=args.alpha,
            reranker_model_name=args.reranker_model_name,
            device=args.device,
            batch_size=args.batch_size,
            embedding_model_name=args.embedding_model_name,
        )
        results[strategy] = (rerank_result, stats)
        metadata_by_strategy[strategy] = metadata

    if args.verbose:
        for strategy, (rerank_result, _stats) in results.items():
            print_verbose_strategy_results(
                strategy, rerank_result, cases, metadata_by_strategy[strategy], document_id
            )

    print_comparison_table(results)
    print_markdown_comparison_table(document.label, len(cases), args.top_k, results)

    case_comparison_counts = None
    if "fixed" in results and "table_aware" in results:
        case_comparison_counts = summarize_case_comparisons(
            results["fixed"][0], results["table_aware"][0]
        )
        print_case_comparison_summary(case_comparison_counts)

    if args.save_report is not None:
        failure_tags: dict[str, dict[str, list[str]]] = {}
        for strategy, (rerank_result, _stats) in results.items():
            per_case: dict[str, list[str]] = {}
            for index, case_result in enumerate(rerank_result.case_results):
                qid = f"Q{index + 1}"
                expected_pages = sorted({page for page, _ in case_result.expected})
                per_case[qid] = auto_tag_failure_causes(
                    case_result, expected_pages, document_id, metadata_by_strategy[strategy]
                )
            failure_tags[strategy] = per_case

        report_path = resolve_chunking_report_path(args.save_report, dataset_path)
        write_chunking_comparison_report(
            report_path,
            document,
            results,
            case_comparison_counts or {},
            failure_tags,
        )


if __name__ == "__main__":
    main()
