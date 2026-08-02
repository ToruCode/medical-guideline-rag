"""Compare Japanese PDF text-extraction quality and retrieval accuracy
across several extraction strategies (pypdf, PyMuPDF) against the same
real, local evaluation dataset.

Issue #22. Follows Issue #21's diagnosis that pypdf's Japanese text
extraction quality - not embedding choice, chunk_size, or top_k - is
suspected to be the main driver of low retrieval accuracy (see
docs/adr/0016-retrieval-quality-diagnosis.md). This script measures
whether that is true by running the identical retrieval evaluation
(scripts/retrieval_baseline_core.py's evaluate_configuration(), fixed
chunk_size/chunk_overlap/top_k/embedding_model_name) against pages
loaded by each extractor under comparison, alongside per-extractor
extraction-quality statistics (page/char counts and a garbled-text
heuristic - see scripts/pdf_extraction_comparison_core.py's module
docstring for why that heuristic is not a quality guarantee).

This is a comparison measurement only: it does not change production
PDF extraction (app/api/dependencies.py still wires up pypdf only).

Not a pytest test, for the same reason as
evaluate_retrieval_baseline.py / compare_chunk_sizes.py: the real PDF
and dataset exist only on the machine running this, are never
committed, and there is no pass/fail threshold.

Dataset format: see docs/evaluation-dataset-format.md. Never commit a
dataset file, the PDF it points to, or this script's --save-report
output; data/eval/ is gitignored for exactly this reason.

Usage (from the repo root):

    uv run python -m scripts.compare_pdf_extractors \\
        --dataset data/eval/my_guideline_qa.json --save-report

    # non-default extractor subset:
    uv run python -m scripts.compare_pdf_extractors \\
        --dataset data/eval/my_guideline_qa.json --extractors pypdf

Prints a comparison table (extractor/pages/empty_pages/total_chars/
average_chars_per_page/suspicious_pages/suspicious_ratio/
average_chars_per_chunk/Recall@1/Recall@3/Recall@5/MRR@5) and a
ready-to-review Markdown table for
docs/pdf-extraction-comparison-results.md. Pass --verbose to also
print each extractor's per-question breakdown, including each
retrieved chunk's rank/page/chunk_index/score/text_preview (local use
only).
"""

import argparse
from pathlib import Path

from app.infrastructure.embedding.sentence_transformer_embedder import (
    load_sentence_transformer_model,
)
from scripts.pdf_extraction_comparison_core import (
    ExtractorComparisonResult,
    available_extractor_names,
    evaluate_extractor,
    print_comparison_table,
    print_markdown_comparison_table,
    print_verbose_extractor_results,
    resolve_comparison_report_path,
    write_comparison_report,
)
from scripts.retrieval_baseline_core import load_dataset

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"


def _parse_extractors(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise argparse.ArgumentTypeError("--extractors must contain at least one value")
    available = available_extractor_names()
    unknown = [name for name in names if name not in available]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown extractor(s): {unknown} (available: {available})"
        )
    return names


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare PDF extraction quality and retrieval accuracy (Recall@1/3/5, "
            "MRR) across extraction strategies against a local, real evaluation "
            "dataset. See docs/evaluation-dataset-format.md."
        )
    )
    parser.add_argument(
        "--dataset", required=True, help="Path to a local dataset JSON file (never committed)."
    )
    parser.add_argument(
        "--extractors",
        type=_parse_extractors,
        default=available_extractor_names(),
        metavar="NAME,NAME,...",
        help=f"Comma-separated extractors to compare (default: all of "
        f"{available_extractor_names()}).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"chunk_size held fixed across all extractors (default: {DEFAULT_CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"chunk_overlap held fixed across all extractors (default: {DEFAULT_CHUNK_OVERLAP}).",
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
        help=(
            "Also print each extractor's per-question breakdown, including each "
            "retrieved chunk's rank/page/chunk_index/score/text_preview "
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
            "Save detailed per-question results for every extractor locally "
            "(never commit this file). Defaults to "
            "data/eval/results/<dataset-name>_pdf_extraction_comparison_<date>.json "
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

    results: list[ExtractorComparisonResult] = []
    for extractor_name in args.extractors:
        print(f"\nEvaluating extractor={extractor_name} ...")
        result = evaluate_extractor(
            document,
            cases,
            model,
            extractor_name,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            top_k=args.top_k,
            embedding_model_name=args.embedding_model_name,
        )
        results.append(result)

    if args.verbose:
        print_verbose_extractor_results(results)

    print_comparison_table(results)
    print_markdown_comparison_table(document.label, results)

    if args.save_report is not None:
        report_path = resolve_comparison_report_path(args.save_report, dataset_path)
        write_comparison_report(report_path, document, results)


if __name__ == "__main__":
    main()
