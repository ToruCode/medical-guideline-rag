import json
import os
from pathlib import Path
from typing import Any

from scripts.evaluation_report_loader import (
    ReportFormatError,
    latest_report_path,
    list_reports,
    load_answer_quality_reports,
    load_report,
)


def _answer_quality_case(**overrides: Any) -> dict[str, Any]:
    case: dict[str, Any] = {
        "question": "fictional dosage question",
        "expected_pages": [1],
        "expected_answer_points": ["500 mg"],
        "expected_insufficient_evidence": False,
        "is_insufficient_evidence": False,
        "insufficient_evidence_correct": True,
        "cited_pages": [1],
        "citation_precision": 1.0,
        "citation_recall": 1.0,
        "answer_point_coverage": 1.0,
        "citations_consistent": True,
        "latency_seconds": 0.123,
        "answer_preview": "fictional generated answer, never real guideline text",
        "category": "dosage",
        "difficulty": "easy",
    }
    case.update(overrides)
    return case


def _answer_quality_report(**case_overrides: Any) -> dict[str, Any]:
    return {
        "document_label": "Fictional Guideline",
        "runs": [
            {
                "config": {
                    "llm_provider": "fake",
                    "llm_model_name": None,
                    "chunk_size": 1000,
                    "chunk_overlap": 200,
                    "top_k": 5,
                    "context_max_chars": 6000,
                    "embedding_model_name": "fake-model",
                    "case_count": 1,
                    "indexed_page_count": 1,
                    "indexed_chunk_count": 1,
                    "measured_at": "2026-01-01",
                },
                "aggregate": {
                    "insufficient_evidence_accuracy": 1.0,
                    "mean_citation_precision": 1.0,
                    "mean_citation_recall": 1.0,
                    "mean_answer_point_coverage": 1.0,
                    "mean_latency_seconds": 0.123,
                    "citation_consistency_violations": 0,
                },
                "cases": [_answer_quality_case(**case_overrides)],
            }
        ],
    }


def _retrieval_only_report() -> dict[str, Any]:
    """Shaped like scripts/retrieval_baseline_core.py's report: same
    outer envelope, but cases have no answer-quality-specific fields.
    """
    return {
        "document_label": "Fictional Guideline",
        "runs": [
            {
                "config": {"chunk_size": 1000, "top_k": 5},
                "aggregate": {"recall_at_3": 1.0, "mrr": 1.0},
                "cases": [{"question": "q", "recall_at_3": 1.0, "reciprocal_rank": 1.0}],
            }
        ],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


# --- load_report ---


def test_load_report_parses_answer_quality_report(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "report.json", _answer_quality_report())

    report = load_report(path)

    assert report.document_label == "Fictional Guideline"
    assert len(report.runs) == 1
    run = report.runs[0]
    assert run.config.llm_provider == "fake"
    assert run.aggregate.mean_citation_precision == 1.0
    assert len(run.case_results) == 1
    assert run.case_results[0].question == "fictional dosage question"
    assert run.case_results[0].category == "dosage"
    assert run.case_results[0].difficulty == "easy"


def test_load_report_parses_case_with_no_category_or_difficulty(tmp_path: Path) -> None:
    """Older reports (written before Issue #15) have no category/difficulty
    keys at all - they must still load, with both defaulting to None.
    """
    payload = _answer_quality_report()
    del payload["runs"][0]["cases"][0]["category"]
    del payload["runs"][0]["cases"][0]["difficulty"]
    path = _write_json(tmp_path / "report.json", payload)

    report = load_report(path)

    assert report.runs[0].case_results[0].category is None
    assert report.runs[0].case_results[0].difficulty is None


def test_load_report_raises_for_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("not json", encoding="utf-8")

    try:
        load_report(path)
        raise AssertionError("expected ReportFormatError")
    except ReportFormatError:
        pass


def test_load_report_raises_for_missing_runs_key(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "report.json", {"document_label": "x"})

    try:
        load_report(path)
        raise AssertionError("expected ReportFormatError")
    except ReportFormatError:
        pass


def test_load_report_raises_for_retrieval_only_report_shape(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "report.json", _retrieval_only_report())

    try:
        load_report(path)
        raise AssertionError("expected ReportFormatError")
    except ReportFormatError:
        pass


def test_load_report_handles_empty_cases(tmp_path: Path) -> None:
    payload = _answer_quality_report()
    payload["runs"][0]["cases"] = []
    path = _write_json(tmp_path / "report.json", payload)

    report = load_report(path)

    assert report.runs[0].case_results == []


# --- list_reports / latest_report_path ---


def test_list_reports_returns_empty_list_for_missing_directory(tmp_path: Path) -> None:
    assert list_reports(tmp_path / "does_not_exist") == []


def test_list_reports_sorts_most_recently_modified_first(tmp_path: Path) -> None:
    older = _write_json(tmp_path / "older.json", _answer_quality_report())
    newer = _write_json(tmp_path / "newer.json", _answer_quality_report())
    now = 1_800_000_000
    os.utime(older, (now, now))
    os.utime(newer, (now + 100, now + 100))

    reports = list_reports(tmp_path)

    assert reports == [newer, older]


def test_latest_report_path_returns_none_when_no_reports(tmp_path: Path) -> None:
    assert latest_report_path(tmp_path) is None


def test_latest_report_path_returns_most_recently_modified(tmp_path: Path) -> None:
    older = _write_json(tmp_path / "older.json", _answer_quality_report())
    newer = _write_json(tmp_path / "newer.json", _answer_quality_report())
    now = 1_800_000_000
    os.utime(older, (now, now))
    os.utime(newer, (now + 100, now + 100))

    assert latest_report_path(tmp_path) == newer


# --- load_answer_quality_reports ---


def test_load_answer_quality_reports_skips_incompatible_and_invalid_files(
    tmp_path: Path,
) -> None:
    _write_json(tmp_path / "answer_quality.json", _answer_quality_report())
    _write_json(tmp_path / "retrieval_only.json", _retrieval_only_report())
    (tmp_path / "broken.json").write_text("not json", encoding="utf-8")

    reports = load_answer_quality_reports(tmp_path)

    assert len(reports) == 1
    assert reports[0].path.name == "answer_quality.json"


def test_load_answer_quality_reports_returns_empty_list_for_missing_directory(
    tmp_path: Path,
) -> None:
    assert load_answer_quality_reports(tmp_path / "does_not_exist") == []
