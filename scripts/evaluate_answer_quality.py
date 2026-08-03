"""Manual answer-quality-and-citation-consistency measurement against a
real, local evaluation dataset (Issue #10).

Not a pytest test: like scripts/evaluate_retrieval_baseline.py, this
reads a real guideline PDF and a real question/expected-page dataset
that exist only on the machine running it. There is no pass/fail
threshold here - this is measurement/reporting only, matching the
retrieval baseline's convention (docs/adr/0014-real-data-retrieval-baseline.md).

Runs with a FakeLlm by default (--llm fake): no API key, no network
access, and no cost, per Issue #10's "default evaluation must not
require external services" requirement. Pass --llm openai (requires
MEDICAL_RAG_LLM_API_KEY) to measure against a real LLM instead - see
docs/adr/0023-answer-quality-and-citation-consistency-evaluation.md for
why this is opt-in rather than the default.

Dataset format: see docs/evaluation-dataset-format.md (expected_pages
via granularity/expected, plus the optional expected_answer_points and
expected_insufficient_evidence fields added for this issue). Never
commit a dataset file, the PDF it points to, or this script's
--save-report output; data/eval/ is gitignored for exactly this reason.

Usage (from the repo root):

    uv run python -m scripts.evaluate_answer_quality \\
        --dataset data/eval/my_guideline_qa.json --save-report

    # Opt-in: measure against the real OpenAI API (billable, network required)
    uv run python -m scripts.evaluate_answer_quality \\
        --dataset data/eval/my_guideline_qa.json --llm openai
"""

import argparse
from pathlib import Path

from app.core.config import get_settings
from app.domain.ports.llm import Llm
from app.infrastructure.embedding.sentence_transformer_embedder import (
    load_sentence_transformer_model,
)
from app.infrastructure.llm.fake_llm import FakeLlm
from app.infrastructure.llm.openai_llm import OpenAiLlm
from scripts.answer_quality_core import (
    evaluate_answer_configuration,
    print_aggregate_report,
    print_case_report,
    print_failure_analysis,
    write_local_report,
)
from scripts.evaluation_common import load_dataset, resolve_report_path


def _build_llm(provider: str) -> tuple[Llm, str | None]:
    settings = get_settings()
    if provider == "fake":
        return FakeLlm(), None
    if provider == "openai":
        if settings.llm_api_key is None:
            raise SystemExit("MEDICAL_RAG_LLM_API_KEY must be set to use --llm openai.")
        llm = OpenAiLlm(
            api_key=settings.llm_api_key.get_secret_value(),
            model=settings.llm_model_name,
            timeout=settings.llm_timeout_seconds,
        )
        return llm, settings.llm_model_name
    raise SystemExit(f"Unknown --llm provider: {provider!r} (must be 'fake' or 'openai')")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure answer quality and citation consistency against a local, "
            "real evaluation dataset. See docs/evaluation-dataset-format.md."
        )
    )
    parser.add_argument(
        "--dataset", required=True, help="Path to a local dataset JSON file (never committed)."
    )
    parser.add_argument(
        "--llm",
        choices=["fake", "openai"],
        default="fake",
        help=(
            "Llm provider to evaluate. 'fake' (default) needs no API key or "
            "network; 'openai' is opt-in and billable (requires "
            "MEDICAL_RAG_LLM_API_KEY)."
        ),
    )
    parser.add_argument("--top-k", type=int, default=5, help="Retrieval depth (default: 5).")
    parser.add_argument(
        "--save-report",
        nargs="?",
        const="__default__",
        default=None,
        metavar="PATH",
        help=(
            "Save detailed per-question results locally (never commit this file). "
            "Defaults to data/eval/results/<dataset-name>_answer_quality_<date>.json "
            "when given without a value."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_path = Path(args.dataset)

    settings = get_settings()
    if settings.embedding_provider != "sentence_transformers":
        raise SystemExit(
            "MEDICAL_RAG_EMBEDDING_PROVIDER must be 'sentence_transformers' for a "
            "meaningful measurement (FakeEmbedder is not semantically meaningful - "
            "its vectors are derived from text length only)."
        )

    document, cases = load_dataset(dataset_path)
    model = load_sentence_transformer_model(settings.embedding_model_name)
    llm, llm_model_name = _build_llm(args.llm)

    run = evaluate_answer_configuration(
        document,
        cases,
        model,
        llm,
        llm_provider=args.llm,
        llm_model_name=llm_model_name,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=args.top_k,
        context_max_chars=settings.llm_context_max_chars,
        embedding_model_name=settings.embedding_model_name,
    )

    print_case_report(run.case_results)
    print_aggregate_report(run.aggregate, run.config)
    print_failure_analysis(run.case_results)

    if args.save_report is not None:
        report_path = resolve_report_path(
            args.save_report, dataset_path, name_suffix="_answer_quality"
        )
        write_local_report(report_path, document, [run])


if __name__ == "__main__":
    main()
