"""Manual retrieval-quality baseline measurement against a real, local
evaluation dataset (Recall@1/3/5, MRR).

Not a pytest test: unlike tests/integration/test_retrieval_evaluation.py
(Issue #17's synthetic, committed, CI-reproducible dataset), this reads
a real guideline PDF and a real question/expected-page dataset that
exist only on the machine running it - never committed, never
reproducible by another developer or CI. There is no pass/fail
threshold here; this issue (#18) is a baseline measurement only, not a
quality gate. See docs/adr/0014-real-data-retrieval-baseline.md.

Requires MEDICAL_RAG_EMBEDDING_PROVIDER=sentence_transformers (in
.env or the environment): FakeEmbedder is not semantically meaningful,
so a baseline measured with it would not reflect real retrieval
quality. Uses the real sentence-transformers model to index the real
PDF, then runs every dataset question through RetrieveChunksService
exactly as POST /questions/ask would.

chunk_size/chunk_overlap/embedding_model_name are read from the
current Settings (.env) - this measures "whatever the app is
currently configured to do". To compare several chunk_size values
directly against each other in one run, see
scripts/compare_chunk_sizes.py (Issue #19) instead, which explicitly
overrides these rather than reading them from Settings. Both scripts
share their core index-and-evaluate logic via
scripts/retrieval_baseline_core.py.

Dataset format: see docs/evaluation-dataset-format.md. Never commit a
dataset file, the PDF it points to, or this script's --save-report
output; data/eval/ is gitignored for exactly this reason.

Usage (from the repo root):

    uv run python -m scripts.evaluate_retrieval_baseline \\
        --dataset data/eval/my_guideline_qa.json --save-report

Prints a per-question breakdown (local use only), an aggregate
summary, and a ready-to-review Markdown snippet for
docs/baseline-retrieval-evaluation.md (aggregate numbers and
measurement configuration only - review it for anything identifying
before pasting).
"""

import argparse
from pathlib import Path

from app.core.config import get_settings
from app.infrastructure.embedding.sentence_transformer_embedder import (
    load_sentence_transformer_model,
)
from scripts.retrieval_baseline_core import (
    evaluate_configuration,
    load_dataset,
    print_aggregate_report,
    print_case_report,
    print_markdown_snippet,
    resolve_report_path,
    write_local_report,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure a retrieval baseline (Recall@1/3/5, MRR) against a local, "
            "real evaluation dataset. See docs/evaluation-dataset-format.md."
        )
    )
    parser.add_argument(
        "--dataset", required=True, help="Path to a local dataset JSON file (never committed)."
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Retrieval depth; must be >= 5 (default: 5)."
    )
    parser.add_argument(
        "--save-report",
        nargs="?",
        const="__default__",
        default=None,
        metavar="PATH",
        help=(
            "Save detailed per-question results locally (never commit this file). "
            "Defaults to data/eval/results/<dataset-name>_<date>.json when given "
            "without a value."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_path = Path(args.dataset)
    if args.top_k < 5:
        raise SystemExit("--top-k must be at least 5 (Recall@5 requires it).")

    settings = get_settings()
    if settings.embedding_provider != "sentence_transformers":
        raise SystemExit(
            "MEDICAL_RAG_EMBEDDING_PROVIDER must be 'sentence_transformers' for a "
            "meaningful baseline measurement (FakeEmbedder is not semantically "
            "meaningful - its vectors are derived from text length only)."
        )

    document, cases = load_dataset(dataset_path)
    model = load_sentence_transformer_model(settings.embedding_model_name)

    run = evaluate_configuration(
        document,
        cases,
        model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=args.top_k,
        embedding_model_name=settings.embedding_model_name,
    )

    print_case_report(run.case_results)
    print_aggregate_report(run.aggregate, run.config)
    print_markdown_snippet(document.label, run.aggregate, run.config)

    if args.save_report is not None:
        report_path = resolve_report_path(args.save_report, dataset_path)
        write_local_report(report_path, document, [run])


if __name__ == "__main__":
    main()
