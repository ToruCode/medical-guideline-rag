import pytest
from tests.support.evaluation.metrics import (
    answer_point_coverage,
    citation_precision,
    citation_recall,
    mean,
    recall_at_k,
    reciprocal_rank,
)


def test_reciprocal_rank_returns_inverse_of_first_match_rank() -> None:
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5


def test_reciprocal_rank_returns_one_for_top_ranked_match() -> None:
    assert reciprocal_rank(["a", "b"], {"a"}) == 1.0


def test_reciprocal_rank_returns_zero_when_no_match() -> None:
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0


def test_reciprocal_rank_returns_zero_for_empty_ranked_ids() -> None:
    assert reciprocal_rank([], {"a"}) == 0.0


def test_reciprocal_rank_uses_earliest_of_multiple_relevant_ids() -> None:
    assert reciprocal_rank(["a", "b", "c"], {"c", "b"}) == 0.5


def test_recall_at_k_counts_any_relevant_id_within_top_k() -> None:
    assert recall_at_k(["a", "b", "c"], {"b"}, k=2) == 1.0


def test_recall_at_k_is_zero_when_relevant_id_outside_top_k() -> None:
    assert recall_at_k(["a", "b", "c"], {"c"}, k=2) == 0.0


def test_recall_at_k_averages_over_multiple_relevant_ids() -> None:
    assert recall_at_k(["a", "b", "c"], {"a", "z"}, k=3) == 0.5


def test_recall_at_k_handles_k_larger_than_ranked_ids() -> None:
    assert recall_at_k(["a"], {"a"}, k=10) == 1.0


def test_recall_at_k_rejects_empty_relevant_ids() -> None:
    with pytest.raises(ValueError):
        recall_at_k(["a"], set(), k=1)


@pytest.mark.parametrize("k", [0, -1])
def test_recall_at_k_rejects_non_positive_k(k: int) -> None:
    with pytest.raises(ValueError):
        recall_at_k(["a"], {"a"}, k=k)


def test_mean_computes_arithmetic_average() -> None:
    assert mean([1.0, 0.5, 0.0]) == pytest.approx(0.5)


def test_mean_rejects_empty_scores() -> None:
    with pytest.raises(ValueError):
        mean([])


def test_citation_precision_is_fraction_of_citations_that_are_expected() -> None:
    assert citation_precision({1, 2}, {2, 3}) == pytest.approx(0.5)


def test_citation_precision_is_one_when_all_citations_are_expected() -> None:
    assert citation_precision({2}, {2, 3}) == 1.0


def test_citation_precision_returns_none_for_no_citations() -> None:
    assert citation_precision(set(), {1}) is None


def test_citation_recall_is_fraction_of_expected_pages_that_were_cited() -> None:
    assert citation_recall({2}, {2, 3}) == pytest.approx(0.5)


def test_citation_recall_is_one_when_all_expected_pages_were_cited() -> None:
    assert citation_recall({2, 3, 9}, {2, 3}) == 1.0


def test_citation_recall_is_zero_for_no_matching_citations() -> None:
    assert citation_recall({9}, {2, 3}) == 0.0


def test_citation_recall_rejects_empty_expected_pages() -> None:
    with pytest.raises(ValueError):
        citation_recall({1}, set())


def test_answer_point_coverage_counts_matching_substrings_case_insensitively() -> None:
    answer = "Adults should take 500 mg twice daily."
    assert answer_point_coverage(answer, ["500 MG", "twice daily", "not present"]) == pytest.approx(
        2 / 3
    )


def test_answer_point_coverage_is_one_when_all_points_are_present() -> None:
    answer = "Take 500 mg twice daily."
    assert answer_point_coverage(answer, ["500 mg", "twice daily"]) == 1.0


def test_answer_point_coverage_is_zero_when_no_points_are_present() -> None:
    assert answer_point_coverage("unrelated text", ["500 mg"]) == 0.0


def test_answer_point_coverage_returns_none_for_no_expected_points() -> None:
    assert answer_point_coverage("any answer", []) is None
