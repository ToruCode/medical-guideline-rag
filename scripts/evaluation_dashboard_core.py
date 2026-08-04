"""Filtering and run-comparison logic for the evaluation dashboard
(Issue #15).

Presentation-free by design (no Streamlit import) so it can be unit
tested directly; app/ui/evaluation_dashboard.py is the only module that
renders these results. Operates entirely on
scripts.answer_quality_core's existing dataclasses (AnswerCaseResult,
AnswerAggregate, AnswerConfigurationRun) - no evaluation metric is
recomputed or redefined here, only filtered/aggregated for display.
"""

from dataclasses import dataclass

from scripts.answer_quality_core import AnswerAggregate, AnswerCaseResult, is_failure_case


def filter_case_results(
    case_results: list[AnswerCaseResult],
    *,
    failures_only: bool = False,
    category: str | None = None,
    difficulty: str | None = None,
    question_contains: str | None = None,
) -> list[AnswerCaseResult]:
    """Returns the subset of case_results matching every given filter.

    Each filter is skipped (matches everything) when its argument is
    None/False, so filter_case_results(cases) (no filters) returns
    cases unchanged. category/difficulty match case-insensitively
    against AnswerCaseResult.category/difficulty (both optional,
    dataset-defined metadata - Issue #15; a case result should be
    treated as excluded from any specific category/difficulty filter
    when it has no value for it, since None cannot equal a requested
    filter value).
    """
    results = case_results
    if failures_only:
        results = [case for case in results if is_failure_case(case)]
    if category is not None:
        results = [
            case
            for case in results
            if case.category is not None and case.category.lower() == category.lower()
        ]
    if difficulty is not None:
        results = [
            case
            for case in results
            if case.difficulty is not None and case.difficulty.lower() == difficulty.lower()
        ]
    if question_contains:
        needle = question_contains.lower()
        results = [case for case in results if needle in case.question.lower()]
    return results


def available_categories(case_results: list[AnswerCaseResult]) -> list[str]:
    """Sorted, deduplicated list of every non-None category present."""
    return sorted({case.category for case in case_results if case.category is not None})


def available_difficulties(case_results: list[AnswerCaseResult]) -> list[str]:
    """Sorted, deduplicated list of every non-None difficulty present."""
    return sorted({case.difficulty for case in case_results if case.difficulty is not None})


# (display label, AnswerAggregate attribute name, whether a higher value is better)
_METRIC_SPECS: list[tuple[str, str, bool]] = [
    ("citation_precision", "mean_citation_precision", True),
    ("citation_recall", "mean_citation_recall", True),
    ("answer_point_coverage", "mean_answer_point_coverage", True),
    ("insufficient_evidence_accuracy", "insufficient_evidence_accuracy", True),
    ("mean_latency_seconds", "mean_latency_seconds", False),
    ("citation_consistency_violations", "citation_consistency_violations", False),
]


@dataclass(frozen=True, slots=True)
class MetricComparison:
    """One aggregate metric's value in each of two compared runs.

    status is "improved"/"degraded" relative to whichever direction is
    better for this metric (e.g. lower is better for latency and
    consistency violations, higher for everything else), "unchanged"
    when the two values are within epsilon of each other, or
    "unavailable" when either run has no value for this metric (e.g. no
    case in that run had any expected_answer_points).
    """

    name: str
    value_a: float | None
    value_b: float | None
    delta: float | None
    status: str


def compare_aggregates(
    aggregate_a: AnswerAggregate, aggregate_b: AnswerAggregate, *, epsilon: float = 1e-9
) -> list[MetricComparison]:
    """Compares every metric in _METRIC_SPECS between two aggregates,
    in a's -> b's direction (delta = b - a).
    """
    comparisons = []
    for name, attr, higher_is_better in _METRIC_SPECS:
        value_a = getattr(aggregate_a, attr)
        value_b = getattr(aggregate_b, attr)
        if value_a is None or value_b is None:
            comparisons.append(MetricComparison(name, value_a, value_b, None, "unavailable"))
            continue

        delta = value_b - value_a
        if abs(delta) <= epsilon:
            status = "unchanged"
        elif (delta > 0) == higher_is_better:
            status = "improved"
        else:
            status = "degraded"
        comparisons.append(MetricComparison(name, value_a, value_b, delta, status))
    return comparisons
