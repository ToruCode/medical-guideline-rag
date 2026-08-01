"""Compare retrieval quality (Recall@1/3/5, MRR) across several
chunk_size values against the same real, local evaluation dataset.

Issue #19. Builds on scripts/evaluate_retrieval_baseline.py (Issue
#18): both share their core index-and-evaluate logic via
scripts/retrieval_baseline_core.py, but this script explicitly
overrides chunk_size/chunk_overlap/top_k/embedding_model_name via CLI
arguments instead of reading them from the current Settings (.env) -
the point here is a controlled, fixed-configuration sweep, independent
of whatever .env currently has configured. See
docs/adr/0015-chunk-size-comparison.md.

Not a pytest test, for the same reason as
evaluate_retrieval_baseline.py: the real PDF and dataset exist only on
the machine running this, are never committed, and there is no
pass/fail threshold - this issue is a comparison measurement only, no
retrieval changes are made.

The sentence-transformers model is loaded once and reused across every
chunk_size candidate (loading it is the expensive part); a fresh
InMemoryVectorStore is built per candidate so results never mix.

Dataset format: see docs/evaluation-dataset-format.md. Never commit a
dataset file, the PDF it points to, or this script's --save-report
output; data/eval/ is gitignored for exactly this reason.

Usage (from the repo root):

    uv run python -m scripts.compare_chunk_sizes \\
        --dataset data/eval/my_guideline_qa.json --save-report

    # non-default candidates:
    uv run python -m scripts.compare_chunk_sizes \\
        --dataset data/eval/my_guideline_qa.json --chunk-sizes 400,800,1200

Prints a compact comparison table (aggregate only) and a ready-to-review
Markdown table for docs/chunk-size-comparison.md. Pass --verbose to
also print each candidate's per-question breakdown (local use only).
"""

import argparse
from pathlib import Path

from app.infrastructure.embedding.sentence_transformer_embedder import (
    load_sentence_transformer_model,
)
from scripts.retrieval_baseline_core import (
    ConfigurationRun,
    evaluate_configuration,
    load_dataset,
    print_case_report,
    resolve_report_path,
    write_local_report,
)

DEFAULT_CHUNK_SIZES = [300, 500, 700, 1000, 1500]
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"


def _parse_chunk_sizes(value: str) -> list[int]:
    try:
        sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid --chunk-sizes value: {value!r}") from exc
    if not sizes:
        raise argparse.ArgumentTypeError("--chunk-sizes must contain at least one value")
    if any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("--chunk-sizes values must be positive integers")
    return sizes


def _print_comparison_table(runs: list[ConfigurationRun]) -> None:
    print(
        f"\n{'chunk_size':>10} | {'chunks':>6} | {'Recall@1':>8} | {'Recall@3':>8} | "
        f"{'Recall@5':>8} | {'MRR@k':>6}"
    )
    print("-" * 62)
    for run in runs:
        config, aggregate = run.config, run.aggregate
        print(
            f"{config.chunk_size:>10} | {config.indexed_chunk_count:>6} | "
            f"{aggregate.recall_at_1:>8.3f} | {aggregate.recall_at_3:>8.3f} | "
            f"{aggregate.recall_at_5:>8.3f} | {aggregate.mrr:>6.3f}"
        )


def _markdown_comparison_table(label: str, runs: list[ConfigurationRun]) -> str:
    first = runs[0].config
    header = (
        f"## Chunk size comparison ({first.measured_at})\n\n"
        f"- Document: {label}\n"
        f"- Cases: {first.case_count}\n"
        f"- chunk_overlap={first.chunk_overlap}, top_k={first.top_k}\n"
        f"- Embedding: sentence_transformers / {first.embedding_model_name} "
        f'(query prefix: "{first.query_prefix}", passage prefix: "{first.passage_prefix}")\n\n'
        f"| chunk_size | chunks | Recall@1 | Recall@3 | Recall@5 | MRR@{first.top_k} |\n"
        "|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = "".join(
        f"| {run.config.chunk_size} | {run.config.indexed_chunk_count} | "
        f"{run.aggregate.recall_at_1:.2f} | {run.aggregate.recall_at_3:.2f} | "
        f"{run.aggregate.recall_at_5:.2f} | {run.aggregate.mrr:.2f} |\n"
        for run in runs
    )
    return header + rows


def _print_markdown_comparison_table(label: str, runs: list[ConfigurationRun]) -> None:
    print("\n" + "=" * 70)
    print("Markdown snippet for docs/chunk-size-comparison.md")
    print("(review before pasting - confirm nothing identifying leaks):")
    print("=" * 70)
    print(_markdown_comparison_table(label, runs))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Recall@1/3/5 and MRR across several chunk_size values "
            "against a local, real evaluation dataset. See "
            "docs/evaluation-dataset-format.md."
        )
    )
    parser.add_argument(
        "--dataset", required=True, help="Path to a local dataset JSON file (never committed)."
    )
    parser.add_argument(
        "--chunk-sizes",
        type=_parse_chunk_sizes,
        default=DEFAULT_CHUNK_SIZES,
        metavar="N,N,...",
        help=f"Comma-separated chunk_size candidates (default: {DEFAULT_CHUNK_SIZES}).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"chunk_overlap held fixed across all candidates (default: {DEFAULT_CHUNK_OVERLAP}).",
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Retrieval depth; must be >= 5 (default: 5)."
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
        help="Also print each candidate's per-question breakdown (local use only).",
    )
    parser.add_argument(
        "--save-report",
        nargs="?",
        const="__default__",
        default=None,
        metavar="PATH",
        help=(
            "Save detailed per-question results for every candidate locally "
            "(never commit this file). Defaults to "
            "data/eval/results/<dataset-name>_chunk_size_comparison_<date>.json "
            "when given without a value."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_path = Path(args.dataset)
    if args.top_k < 5:
        raise SystemExit("--top-k must be at least 5 (Recall@5 requires it).")

    document, cases = load_dataset(dataset_path)
    model = load_sentence_transformer_model(args.embedding_model_name)

    runs: list[ConfigurationRun] = []
    for chunk_size in args.chunk_sizes:
        print(f"\nEvaluating chunk_size={chunk_size} ...")
        run = evaluate_configuration(
            document,
            cases,
            model,
            chunk_size=chunk_size,
            chunk_overlap=args.chunk_overlap,
            top_k=args.top_k,
            embedding_model_name=args.embedding_model_name,
        )
        if args.verbose:
            print_case_report(run.case_results)
        runs.append(run)

    _print_comparison_table(runs)
    _print_markdown_comparison_table(document.label, runs)

    if args.save_report is not None:
        report_path = resolve_report_path(args.save_report, dataset_path, "_chunk_size_comparison")
        write_local_report(report_path, document, runs)


if __name__ == "__main__":
    main()
