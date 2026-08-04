"""Loads local answer-quality evaluation reports (Issue #10's JSON format,
written by scripts/answer_quality_core.py::write_local_report) back into
the same dataclasses that produced them, for
scripts/evaluation_dashboard_core.py and app/ui/evaluation_dashboard.py
(Issue #15).

Never reads or writes anything under version control: every report this
loads lives under a local, gitignored directory (data/eval/results/ by
default, produced via `--save-report`) and may contain guideline-derived
content and generated answer text - see docs/evaluation-dataset-format.md.

data/eval/results/ is also used by other scripts/*_core.py tools
(e.g. scripts/retrieval_baseline_core.py's Recall/MRR reports,
chunk-size/PDF-extraction/reranking comparison reports), which share the
same outer runs/config/aggregate/cases envelope but different case
fields (no citation_precision, answer_point_coverage, etc. - those tools
never call an Llm). load_report() validates the case shape and raises
ReportFormatError for anything that is not an answer-quality report
instead of silently mis-parsing it; load_answer_quality_reports() uses
that to skip non-matching files in a mixed directory.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.answer_quality_core import (
    AnswerAggregate,
    AnswerCaseResult,
    AnswerConfigurationRun,
    AnswerRunConfig,
)

DEFAULT_REPORTS_DIR = Path("data/eval/results")

# Present on every answer-quality AnswerCaseResult but on no other
# scripts/*_core.py tool's case result - used to distinguish an
# answer-quality report from a same-shaped report produced by a
# different tool sharing data/eval/results/.
_ANSWER_QUALITY_CASE_KEYS = frozenset(
    {
        "citation_precision",
        "answer_point_coverage",
        "insufficient_evidence_correct",
        "citations_consistent",
    }
)


class ReportFormatError(ValueError):
    """Raised when a JSON file is not a well-formed answer-quality report."""


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """One loaded report file: its path, document label, and every run
    it contains (usually exactly one - see write_local_report - but the
    format supports more than one).
    """

    path: Path
    document_label: str
    runs: list[AnswerConfigurationRun]


def _parse_case(raw: dict[str, Any]) -> AnswerCaseResult:
    missing = _ANSWER_QUALITY_CASE_KEYS - raw.keys()
    if missing:
        raise ReportFormatError(
            f"Not an answer-quality evaluation report (case is missing field(s): {sorted(missing)})"
        )
    try:
        return AnswerCaseResult(
            question=raw["question"],
            expected_pages=list(raw["expected_pages"]),
            expected_answer_points=list(raw["expected_answer_points"]),
            expected_insufficient_evidence=bool(raw["expected_insufficient_evidence"]),
            is_insufficient_evidence=bool(raw["is_insufficient_evidence"]),
            insufficient_evidence_correct=bool(raw["insufficient_evidence_correct"]),
            cited_pages=list(raw["cited_pages"]),
            citation_precision=raw["citation_precision"],
            citation_recall=raw["citation_recall"],
            answer_point_coverage=raw["answer_point_coverage"],
            citations_consistent=bool(raw["citations_consistent"]),
            latency_seconds=float(raw["latency_seconds"]),
            answer_preview=raw["answer_preview"],
            category=raw.get("category"),
            difficulty=raw.get("difficulty"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ReportFormatError(f"Malformed answer-quality case: {exc}") from exc


def _parse_run(raw: dict[str, Any]) -> AnswerConfigurationRun:
    try:
        cases_raw = raw["cases"]
        config_raw = raw["config"]
        aggregate_raw = raw["aggregate"]
    except KeyError as exc:
        raise ReportFormatError(f"Report run is missing required key: {exc}") from exc

    case_results = [_parse_case(case_raw) for case_raw in cases_raw]

    try:
        config = AnswerRunConfig(**config_raw)
        aggregate = AnswerAggregate(**aggregate_raw)
    except TypeError as exc:
        raise ReportFormatError(f"Malformed answer-quality run config/aggregate: {exc}") from exc

    return AnswerConfigurationRun(config=config, case_results=case_results, aggregate=aggregate)


def load_report(path: Path) -> EvaluationReport:
    """Loads one report JSON file into an EvaluationReport.

    Raises ReportFormatError if path is not valid JSON, has no top-level
    "runs" key, or its cases are not shaped like an answer-quality
    report (see module docstring) - never partially parses a mismatched
    file.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReportFormatError(f"{path} is not valid JSON") from exc

    if not isinstance(raw, dict) or "runs" not in raw:
        raise ReportFormatError(f"{path} has no top-level 'runs' key")

    runs = [_parse_run(run_raw) for run_raw in raw["runs"]]
    return EvaluationReport(path=path, document_label=raw.get("document_label", ""), runs=runs)


def list_reports(reports_dir: Path = DEFAULT_REPORTS_DIR) -> list[Path]:
    """Lists every *.json file directly under reports_dir, most recently
    modified first.

    Returns an empty list if reports_dir does not exist yet, rather than
    raising - the dashboard's default view before any report has been
    saved.
    """
    if not reports_dir.is_dir():
        return []
    return sorted(reports_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)


def latest_report_path(reports_dir: Path = DEFAULT_REPORTS_DIR) -> Path | None:
    reports = list_reports(reports_dir)
    return reports[0] if reports else None


def load_answer_quality_reports(
    reports_dir: Path = DEFAULT_REPORTS_DIR,
) -> list[EvaluationReport]:
    """Loads every answer-quality-shaped report under reports_dir, most
    recently modified first, silently skipping files that are not valid
    JSON or do not match the answer-quality schema (see load_report()).
    """
    loaded = []
    for path in list_reports(reports_dir):
        try:
            loaded.append(load_report(path))
        except ReportFormatError:
            continue
    return loaded
