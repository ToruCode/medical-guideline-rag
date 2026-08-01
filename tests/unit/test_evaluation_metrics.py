import pytest
from tests.support.evaluation.metrics import mean, recall_at_k, reciprocal_rank


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
