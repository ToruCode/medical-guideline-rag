import pytest
from scripts.answer_quality_core import AnswerAggregate, AnswerCaseResult
from scripts.evaluation_dashboard_core import (
    available_categories,
    available_difficulties,
    compare_aggregates,
    filter_case_results,
)


def _case(
    question: str = "q",
    *,
    category: str | None = None,
    difficulty: str | None = None,
    citations_consistent: bool = True,
    insufficient_evidence_correct: bool = True,
    citation_recall: float | None = 1.0,
    answer_point_coverage: float | None = 1.0,
) -> AnswerCaseResult:
    return AnswerCaseResult(
        question=question,
        expected_pages=[1],
        expected_answer_points=[],
        expected_insufficient_evidence=False,
        is_insufficient_evidence=False,
        insufficient_evidence_correct=insufficient_evidence_correct,
        cited_pages=[1],
        citation_precision=1.0,
        citation_recall=citation_recall,
        answer_point_coverage=answer_point_coverage,
        citations_consistent=citations_consistent,
        latency_seconds=0.1,
        answer_preview="preview",
        category=category,
        difficulty=difficulty,
    )


def _aggregate(**overrides: float | int | None) -> AnswerAggregate:
    values: dict[str, float | int | None] = {
        "insufficient_evidence_accuracy": 1.0,
        "mean_citation_precision": 1.0,
        "mean_citation_recall": 1.0,
        "mean_answer_point_coverage": 1.0,
        "mean_latency_seconds": 0.5,
        "citation_consistency_violations": 0,
    }
    values.update(overrides)
    return AnswerAggregate(**values)  # type: ignore[arg-type]


# --- filter_case_results ---


def test_filter_case_results_no_filters_returns_all() -> None:
    cases = [_case("a"), _case("b")]

    assert filter_case_results(cases) == cases


def test_filter_case_results_failures_only_keeps_only_failing_cases() -> None:
    passing = _case("passing")
    failing = _case("failing", citations_consistent=False)

    result = filter_case_results([passing, failing], failures_only=True)

    assert result == [failing]


def test_filter_case_results_by_category_is_case_insensitive() -> None:
    dosage = _case("dosage question", category="Dosage")
    other = _case("other question", category="side-effects")

    result = filter_case_results([dosage, other], category="dosage")

    assert result == [dosage]


def test_filter_case_results_by_category_excludes_cases_with_no_category() -> None:
    with_category = _case("a", category="dosage")
    without_category = _case("b", category=None)

    result = filter_case_results([with_category, without_category], category="dosage")

    assert result == [with_category]


def test_filter_case_results_by_difficulty() -> None:
    easy = _case("a", difficulty="easy")
    hard = _case("b", difficulty="hard")

    assert filter_case_results([easy, hard], difficulty="hard") == [hard]


def test_filter_case_results_by_question_substring_is_case_insensitive() -> None:
    dosage = _case("What is the DOSAGE?")
    other = _case("What are side effects?")

    result = filter_case_results([dosage, other], question_contains="dosage")

    assert result == [dosage]


def test_filter_case_results_combines_filters_with_and() -> None:
    match = _case("dosage question", category="dosage", citations_consistent=False)
    wrong_category = _case("dosage question", category="other", citations_consistent=False)
    passing = _case("dosage question", category="dosage", citations_consistent=True)

    result = filter_case_results(
        [match, wrong_category, passing], failures_only=True, category="dosage"
    )

    assert result == [match]


# --- available_categories / available_difficulties ---


def test_available_categories_deduplicates_and_sorts() -> None:
    cases = [_case(category="b"), _case(category="a"), _case(category="a"), _case(category=None)]

    assert available_categories(cases) == ["a", "b"]


def test_available_difficulties_deduplicates_and_sorts() -> None:
    cases = [_case(difficulty="hard"), _case(difficulty="easy"), _case(difficulty=None)]

    assert available_difficulties(cases) == ["easy", "hard"]


def test_available_categories_empty_when_none_defined() -> None:
    assert available_categories([_case(), _case()]) == []


# --- compare_aggregates ---


def test_compare_aggregates_flags_higher_precision_as_improved() -> None:
    comparisons = compare_aggregates(
        _aggregate(mean_citation_precision=0.5), _aggregate(mean_citation_precision=0.9)
    )

    precision = next(c for c in comparisons if c.name == "citation_precision")
    assert precision.status == "improved"
    assert precision.delta == pytest.approx(0.4)


def test_compare_aggregates_flags_higher_latency_as_degraded() -> None:
    comparisons = compare_aggregates(
        _aggregate(mean_latency_seconds=0.5), _aggregate(mean_latency_seconds=1.5)
    )

    latency = next(c for c in comparisons if c.name == "mean_latency_seconds")
    assert latency.status == "degraded"


def test_compare_aggregates_flags_lower_latency_as_improved() -> None:
    comparisons = compare_aggregates(
        _aggregate(mean_latency_seconds=1.5), _aggregate(mean_latency_seconds=0.5)
    )

    latency = next(c for c in comparisons if c.name == "mean_latency_seconds")
    assert latency.status == "improved"


def test_compare_aggregates_flags_more_violations_as_degraded() -> None:
    comparisons = compare_aggregates(
        _aggregate(citation_consistency_violations=0), _aggregate(citation_consistency_violations=2)
    )

    violations = next(c for c in comparisons if c.name == "citation_consistency_violations")
    assert violations.status == "degraded"


def test_compare_aggregates_flags_equal_values_as_unchanged() -> None:
    comparisons = compare_aggregates(_aggregate(), _aggregate())

    assert all(c.status == "unchanged" for c in comparisons)


def test_compare_aggregates_flags_unavailable_when_either_side_is_none() -> None:
    comparisons = compare_aggregates(
        _aggregate(mean_answer_point_coverage=None), _aggregate(mean_answer_point_coverage=0.8)
    )

    coverage = next(c for c in comparisons if c.name == "answer_point_coverage")
    assert coverage.status == "unavailable"
    assert coverage.delta is None
