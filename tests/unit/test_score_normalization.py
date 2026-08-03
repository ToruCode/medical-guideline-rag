"""Unit tests for scripts/score_normalization.py."""

import pytest
from scripts.score_normalization import min_max_normalize


def test_empty_dict_returns_empty_dict() -> None:
    assert min_max_normalize({}) == {}


def test_normalizes_to_zero_one_range() -> None:
    normalized = min_max_normalize({"a": 1.0, "b": 3.0, "c": 5.0})

    assert normalized == pytest.approx({"a": 0.0, "b": 0.5, "c": 1.0})


def test_handles_negative_values() -> None:
    normalized = min_max_normalize({"a": -1.0, "b": 0.0, "c": 1.0})

    assert normalized == pytest.approx({"a": 0.0, "b": 0.5, "c": 1.0})


def test_all_equal_scores_normalize_to_zero_without_division_by_zero() -> None:
    normalized = min_max_normalize({"a": 0.42, "b": 0.42, "c": 0.42})

    assert normalized == {"a": 0.0, "b": 0.0, "c": 0.0}


def test_single_candidate_normalizes_to_zero() -> None:
    normalized = min_max_normalize({"a": 7.5})

    assert normalized == {"a": 0.0}


def test_preserves_all_keys() -> None:
    normalized = min_max_normalize({"a": 2.0, "b": 4.0})

    assert set(normalized.keys()) == {"a", "b"}
