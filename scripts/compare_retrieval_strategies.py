"""Compare Dense-only vs Hybrid (Dense+BM25) retrieval quality against
the same real, local evaluation dataset (Issue #24).

Background: Issue #23 adopted PyMuPDF as the production PDF extractor,
substantially improving retrieval quality (see
docs/adr/0018-adopt-pymupdf-for-production-pdf-extraction.md). The
current production retrieval path is Dense-only. Medical guidelines
contain drug names, abbreviations, device names, numeric values, and
chemical names where exact term matching matters - this script measures
whether blending BM25 (lexical match) with Dense (semantic match) via
scripts/hybrid_retrieval_core.py improves Recall@1/3/5/MRR@5, without
replacing production Dense search. See
docs/adr/0019-hybrid-search-comparison.md for the full design
(BM25/tokenizer/normalization/fusion choices).

This is comparison/measurement only - production retrieval
(app/api/dependencies.py, RetrieveChunksService) is not changed.

Not a pytest test, for the same reason as
evaluate_retrieval_baseline.py / compare_pdf_extractors.py: the real
PDF and dataset exist only on the operator's machine and are never
committed; this is a one-off/occasional measurement, not a CI gate.

Dataset format: see docs/evaluation-dataset-format.md. Never commit a
dataset file, the PDF it points to, or this script's --save-report
output; data/eval/ is gitignored for exactly this reason.

Usage (from the repo root):

    uv run python -m scripts.compare_retrieval_strategies \\
        --dataset data/eval/my_guideline_qa.json --save-report

    # hybrid only, with a different alpha:
    uv run python -m scripts.compare_retrieval_strategies \\
        --dataset data/eval/my_guideline_qa.json --strategies hybrid --alpha 0.5

Prints a comparison table (strategy/alpha/Recall@1/Recall@3/Recall@5/
MRR@5) and a ready-to-review Markdown table for
docs/hybrid-search-comparison-results.md. Pass --verbose to also print,
per strategy and per question, the expected/retrieved pages and each
retrieved chunk's rank/final score/dense score/bm25 score/text_preview
(local use only, never committed).
"""

import argparse
from pathlib import Path

from app.infrastructure.embedding.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
    load_sentence_transformer_model,
)
from scripts.hybrid_retrieval_core import (
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    STRATEGIES,
    StrategyComparisonResult,
    build_indexed_corpus,
    evaluate_strategy,
    print_comparison_table,
    print_markdown_comparison_table,
    print_verbose_strategy_results,
    resolve_strategy_report_path,
    write_strategy_report,
)
from scripts.hybrid_scorer import HybridScorer
from scripts.retrieval_baseline_core import load_dataset

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 5
DEFAULT_DENSE_CANDIDATE_K = 20
DEFAULT_BM25_CANDIDATE_K = 20
DEFAULT_ALPHA = 0.7
DEFAULT_EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"


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
            "Compare Dense-only vs Hybrid (Dense+BM25) retrieval accuracy "
            "(Recall@1/3/5, MRR) against a local, real evaluation dataset. "
            "See docs/evaluation-dataset-format.md."
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
        help=f"chunk_size held fixed across all strategies (default: {DEFAULT_CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"chunk_overlap held fixed across all strategies (default: {DEFAULT_CHUNK_OVERLAP}).",
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
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help=(
            "Dense weight in hybrid_score = alpha*dense_normalized + "
            f"(1-alpha)*bm25_normalized, in [0.0, 1.0] (default: {DEFAULT_ALPHA})."
        ),
    )
    parser.add_argument(
        "--embedding-model-name",
        default=DEFAULT_EMBEDDING_MODEL_NAME,
        help=f"sentence-transformers model, used regardless of .env (default: "
        f"{DEFAULT_EMBEDDING_MODEL_NAME}).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Also print each strategy's per-question breakdown, including "
            "expected/retrieved pages and each retrieved chunk's rank/final "
            "score/dense score/bm25 score/text_preview (local use only)."
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
            "data/eval/results/<dataset-name>_hybrid_search_comparison_<date>.json "
            "when given without a value."
        ),
    )
    return parser.parse_args()


def main() -> None:
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

    document, cases = load_dataset(dataset_path)
    model = load_sentence_transformer_model(args.embedding_model_name)
    passage_embedder = SentenceTransformerEmbedder(model, prefix=PASSAGE_PREFIX)
    query_embedder = SentenceTransformerEmbedder(model, prefix=QUERY_PREFIX)

    print(f"\nIndexing {document.source_path} (extractor=pymupdf) ...")
    corpus = build_indexed_corpus(
        document,
        passage_embedder,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    scorer = HybridScorer(alpha=args.alpha)
    results: list[StrategyComparisonResult] = []
    for strategy in args.strategies:
        print(f"\nEvaluating strategy={strategy} ...")
        result = evaluate_strategy(
            strategy,
            corpus,
            cases,
            query_embedder,
            top_k=args.top_k,
            dense_candidate_k=args.dense_candidate_k,
            bm25_candidate_k=args.bm25_candidate_k,
            alpha=args.alpha,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            embedding_model_name=args.embedding_model_name,
            scorer=scorer,
        )
        results.append(result)

    if args.verbose:
        print_verbose_strategy_results(results)

    print_comparison_table(results)
    print_markdown_comparison_table(document.label, results)

    if args.save_report is not None:
        report_path = resolve_strategy_report_path(args.save_report, dataset_path)
        write_strategy_report(report_path, document, results)


if __name__ == "__main__":
    main()
