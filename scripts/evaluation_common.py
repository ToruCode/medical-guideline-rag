"""Evaluation-type-agnostic building blocks shared by every scripts/*
evaluation tool: retrieval-only (scripts/retrieval_baseline_core.py),
answer-quality-and-citation-consistency (scripts/answer_quality_core.py,
Issue #10), and any future evaluation (e.g. an LLM-as-a-Judge pass) that
reads the same local, gitignored dataset format.

Kept deliberately free of any Recall/MRR- or answer-quality-specific
logic - those stay in their own core modules. Extracted from
scripts/retrieval_baseline_core.py (Issue #10) as a behavior-preserving
refactor: retrieval_baseline_core.py re-exports every name below so its
existing callers (scripts/compare_chunk_sizes.py,
scripts/compare_pdf_extractors.py, scripts/compare_reranking_strategies.py,
scripts/compare_retrieval_strategies.py,
scripts/compare_chunking_strategies.py, scripts/chunking_comparison_core.py,
scripts/hybrid_retrieval_core.py, scripts/pdf_extraction_comparison_core.py,
scripts/reranking_core.py, scripts/evaluate_retrieval_baseline.py) are
unaffected; verified by the existing test suite passing unchanged.

Never commit a dataset file, the PDF it points to, or any --save-report
output; data/eval/ is gitignored for exactly this reason. See
docs/evaluation-dataset-format.md.
"""

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

# Max length of a retrieved chunk's text_preview before truncation, for
# --verbose terminal output and --save-report JSON. Deliberately separate
# from CITATION_TEXT_PREVIEW_MAX_CHARS (app/core/constants.py): this
# tooling is for debugging evaluation results locally, not an API
# response, so it is implemented independently of
# app/api/v1/endpoints/questions.py rather than importing from the API
# layer.
TEXT_PREVIEW_MAX_CHARS = 300


def truncate_text(text: str, max_chars: int = TEXT_PREVIEW_MAX_CHARS) -> str:
    """Truncates text to at most max_chars, appending "..." when cut.

    The returned string's own length can exceed max_chars by the length of
    the "..." marker; callers relying on a hard cap should account for that.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


@dataclass(frozen=True, slots=True)
class DatasetDocument:
    source_path: Path
    label: str


@dataclass(frozen=True, slots=True)
class DatasetCase:
    """One evaluation question, plus every kind of ground truth a
    scripts/*_core.py evaluator might check against.

    expected_locations (page/chunk granularity) is required and used by
    every evaluator (retrieval-only and answer-quality alike).
    expected_answer_points and expected_insufficient_evidence are
    optional, answer-quality-specific ground truth (Issue #10, see
    docs/evaluation-dataset-format.md); they default to "no answer-point
    check" / "evidence is expected to be sufficient" so existing
    retrieval-only dataset files (with no such keys) keep parsing
    unchanged.

    category and difficulty are optional, free-form metadata (Issue
    #15, see docs/evaluation-dataset-format.md) used only for filtering
    in scripts/evaluation_dashboard_core.py; they default to None
    ("no metadata"), do not affect any metric, and are ignored by every
    existing evaluator.
    """

    question: str
    granularity: str  # "page" or "chunk"
    expected_locations: list[tuple[int, int | None]]
    expected_answer_points: list[str] = field(default_factory=list)
    expected_insufficient_evidence: bool = False
    category: str | None = None
    difficulty: str | None = None


def location_key(page_number: int, chunk_index: int | None) -> str:
    if chunk_index is None:
        return str(page_number)
    return f"{page_number}:{chunk_index}"


def _parse_case(case_raw: dict[str, Any]) -> DatasetCase:
    granularity = case_raw["granularity"]
    if granularity not in ("page", "chunk"):
        raise ValueError(f"Unknown granularity: {granularity!r} (must be 'page' or 'chunk')")

    expected_raw = case_raw["expected"]
    expected_locations: list[tuple[int, int | None]]
    if granularity == "page":
        expected_locations = [(int(page), None) for page in expected_raw]
    else:
        expected_locations = [(int(page), int(chunk_index)) for page, chunk_index in expected_raw]

    return DatasetCase(
        question=case_raw["question"],
        granularity=granularity,
        expected_locations=expected_locations,
        expected_answer_points=list(case_raw.get("expected_answer_points", [])),
        expected_insufficient_evidence=bool(case_raw.get("expected_insufficient_evidence", False)),
        category=case_raw.get("category"),
        difficulty=case_raw.get("difficulty"),
    )


def load_dataset(path: Path) -> tuple[DatasetDocument, list[DatasetCase]]:
    raw = json.loads(path.read_text(encoding="utf-8"))

    document_raw = raw["document"]
    document = DatasetDocument(
        source_path=Path(document_raw["source_path"]),
        label=document_raw.get("label", "Guideline A"),
    )

    cases = [_parse_case(case_raw) for case_raw in raw["cases"]]
    if not cases:
        raise ValueError(f"{path} contains no evaluation cases")
    return document, cases


def resolve_report_path(save_report_arg: str, dataset_path: Path, name_suffix: str = "") -> Path:
    if save_report_arg != "__default__":
        return Path(save_report_arg)
    today = date.today().isoformat()
    return Path("data/eval/results") / f"{dataset_path.stem}{name_suffix}_{today}.json"
