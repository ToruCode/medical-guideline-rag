"""Compare Dense / Hybrid / Hybrid+Cross-Encoder-Rerank retrieval quality
against the same real, local evaluation dataset (Issue #25).

Background: Issue #24 compared Dense-only vs Hybrid (Dense+BM25) search
and found Hybrid (alpha=0.7) improved Recall@3/MRR@5 over Dense with no
regressions (see docs/adr/0019-hybrid-search-comparison.md). This script
tests a further candidate: reranking Hybrid's retrieved candidates with
a Cross Encoder (query, chunk_text pairs scored jointly, unlike Dense/
BM25's independently-computed scores) before the final top_k cut. See
docs/adr/0020-cross-encoder-reranker-comparison.md for the full design
(candidate/rerank separation, model choice, latency measurement).

This is comparison/measurement only - production retrieval
(app/api/dependencies.py, RetrieveChunksService) is not changed. "dense"
and "hybrid" strategies reuse scripts/hybrid_retrieval_core.py's
unmodified evaluate_strategy() exactly as
scripts/compare_retrieval_strategies.py (Issue #24) does; only
"hybrid_rerank" (scripts/reranking_core.py) is new.

Not a CLI entry point test - the real PDF and dataset exist only on the
operator's machine and are never committed; this is a one-off/
occasional measurement, not a CI gate.

Dataset format: see docs/evaluation-dataset-format.md. Never commit a
dataset file, the PDF it points to, or this script's --save-report
output; data/eval/ is gitignored for exactly this reason.

Usage (from the repo root):

    uv run python -m scripts.compare_reranking_strategies \\
        --dataset data/eval/my_guideline_qa.json --save-report

    # hybrid_rerank only, with a different reranker model/device:
    uv run python -m scripts.compare_reranking_strategies \\
        --dataset data/eval/my_guideline_qa.json --strategies hybrid_rerank \\
        --reranker-model-name BAAI/bge-reranker-v2-m3 --device cpu

Prints a comparison table (strategy/alpha/reranker/Recall@1/Recall@3/
Recall@5/MRR@5/avg_latency_ms), a ready-to-review Markdown table for
docs/cross-encoder-reranker-comparison-results.md, and (when both
"hybrid" and "hybrid_rerank" are evaluated) a per-question Hybrid ->
Hybrid+Rerank category breakdown. Pass --verbose to also print, per
strategy and per question, the expected/retrieved pages and each
retrieved chunk's score detail (local use only, never committed).
"""

import argparse
import time
from pathlib import Path

from app.infrastructure.embedding.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
    load_sentence_transformer_model,
)
from scripts.hybrid_retrieval_core import (
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    STRATEGY_DENSE,
    STRATEGY_HYBRID,
    StrategyComparisonResult,
    build_indexed_corpus,
    evaluate_strategy,
    print_verbose_strategy_results,
)
from scripts.hybrid_scorer import HybridScorer
from scripts.reranker import CrossEncoderReranker, load_cross_encoder
from scripts.reranking_core import (
    ALL_STRATEGIES,
    ComparisonRow,
    RerankComparisonResult,
    evaluate_hybrid_rerank,
    print_comparison_table,
    print_markdown_comparison_table,
    print_rerank_comparison_summary,
    print_verbose_rerank_results,
    resolve_rerank_report_path,
    row_from_rerank_result,
    row_from_strategy_result,
    summarize_rerank_comparison,
    write_rerank_comparison_report,
)
from scripts.retrieval_baseline_core import load_dataset

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 5
DEFAULT_DENSE_CANDIDATE_K = 20
DEFAULT_BM25_CANDIDATE_K = 20
# 10, not 20: a candidate_k=5/10/20/30 sweep on a real 30-question
# dataset found 10 to be the best accuracy/latency tradeoff - it is the
# only tested value with a net Recall@5 improvement over candidate_k=5
# (0.950 -> 0.967); 20/30 cost far more reranking latency while
# introducing a new regression on a previously-correct question that
# more than offsets the same gain 10 already achieves. See
# docs/cross-encoder-reranker-comparison-results.md's
# "reranker_candidate_k sweep" entry for the full comparison.
DEFAULT_RERANKER_CANDIDATE_K = 10
DEFAULT_ALPHA = 0.7
DEFAULT_EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
DEFAULT_RERANKER_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
DEFAULT_DEVICE = "cpu"
DEFAULT_BATCH_SIZE = 8


def _parse_strategies(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise argparse.ArgumentTypeError("--strategies must contain at least one value")
    unknown = [name for name in names if name not in ALL_STRATEGIES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown strategy(ies): {unknown} (available: {list(ALL_STRATEGIES)})"
        )
    return names


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Dense / Hybrid / Hybrid+Cross-Encoder-Rerank retrieval "
            "accuracy (Recall@1/3/5, MRR, latency) against a local, real "
            "evaluation dataset. See docs/evaluation-dataset-format.md."
        )
    )
    parser.add_argument(
        "--dataset", required=True, help="Path to a local dataset JSON file (never committed)."
    )
    parser.add_argument(
        "--strategies",
        type=_parse_strategies,
        default=list(ALL_STRATEGIES),
        metavar="NAME,NAME,...",
        help=f"Comma-separated strategies to compare (default: all of {list(ALL_STRATEGIES)}).",
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
        "--reranker-candidate-k",
        type=int,
        default=DEFAULT_RERANKER_CANDIDATE_K,
        help=(
            "Number of Hybrid candidates (before Hybrid's own final_top_k cut) "
            f"passed to the Cross Encoder (default: {DEFAULT_RERANKER_CANDIDATE_K})."
        ),
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
        "--reranker-model-name",
        default=DEFAULT_RERANKER_MODEL_NAME,
        help=(
            "sentence-transformers CrossEncoder model for hybrid_rerank "
            f"(default: {DEFAULT_RERANKER_MODEL_NAME})."
        ),
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default=DEFAULT_DEVICE,
        help=(
            "Device for the Cross Encoder model. Defaults to 'cpu' for "
            "Windows+CPU reproducibility; pass 'cuda' explicitly to use a GPU, "
            "or 'auto' to use CUDA only if available (default: cpu)."
        ),
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
            "expected/retrieved pages and each retrieved chunk's score detail "
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
            "data/eval/results/<dataset-name>_reranking_comparison_<date>.json "
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
    strategy_results: list[StrategyComparisonResult] = []
    rows: list[ComparisonRow] = []
    hybrid_result: StrategyComparisonResult | None = None
    rerank_result: RerankComparisonResult | None = None

    for strategy in args.strategies:
        if strategy in (STRATEGY_DENSE, STRATEGY_HYBRID):
            print(f"\nEvaluating strategy={strategy} ...")
            start = time.perf_counter()
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
            elapsed_ms = (time.perf_counter() - start) * 1000
            avg_latency_ms = elapsed_ms / len(cases) if cases else 0.0
            strategy_results.append(result)
            rows.append(row_from_strategy_result(result, avg_latency_ms))
            if strategy == STRATEGY_HYBRID:
                hybrid_result = result
        else:
            print(f"\nLoading reranker model={args.reranker_model_name} (device={args.device}) ...")
            reranker_model = load_cross_encoder(args.reranker_model_name, args.device)
            reranker = CrossEncoderReranker(reranker_model, batch_size=args.batch_size)

            print(f"\nEvaluating strategy={strategy} ...")
            rerank_result = evaluate_hybrid_rerank(
                corpus,
                cases,
                query_embedder,
                scorer,
                reranker,
                top_k=args.top_k,
                dense_candidate_k=args.dense_candidate_k,
                bm25_candidate_k=args.bm25_candidate_k,
                reranker_candidate_k=args.reranker_candidate_k,
                alpha=args.alpha,
                reranker_model_name=args.reranker_model_name,
                device=args.device,
                batch_size=args.batch_size,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                embedding_model_name=args.embedding_model_name,
            )
            rows.append(row_from_rerank_result(rerank_result))

    if args.verbose:
        for result in strategy_results:
            print_verbose_strategy_results([result])
        if rerank_result is not None:
            print_verbose_rerank_results(rerank_result)

    print_comparison_table(rows)
    print_markdown_comparison_table(document.label, len(cases), args.top_k, rows)

    comparison_counts = None
    if hybrid_result is not None and rerank_result is not None:
        comparison_counts = summarize_rerank_comparison(hybrid_result, rerank_result)
        print_rerank_comparison_summary(comparison_counts)

    if args.save_report is not None:
        report_path = resolve_rerank_report_path(args.save_report, dataset_path)
        write_rerank_comparison_report(
            report_path, document, strategy_results, rerank_result, comparison_counts
        )


if __name__ == "__main__":
    main()
