"""Unit tests for scripts/hybrid_scorer.py."""

import pytest
from scripts.hybrid_scorer import HybridScorer


def test_alpha_one_uses_only_dense_signal() -> None:
    scorer = HybridScorer(alpha=1.0)
    dense_scores = {"a": 0.1, "b": 0.9}
    bm25_scores = {"a": 100.0, "b": 0.0}  # BM25 disagrees; must be ignored entirely.

    fused = scorer.fuse(dense_scores, bm25_scores)

    assert fused == pytest.approx({"a": 0.0, "b": 1.0})


def test_alpha_zero_uses_only_bm25_signal() -> None:
    scorer = HybridScorer(alpha=0.0)
    dense_scores = {"a": 0.9, "b": 0.1}  # Dense disagrees; must be ignored entirely.
    bm25_scores = {"a": 1.0, "b": 5.0}

    fused = scorer.fuse(dense_scores, bm25_scores)

    assert fused == pytest.approx({"a": 0.0, "b": 1.0})


def test_alpha_half_averages_normalized_signals() -> None:
    scorer = HybridScorer(alpha=0.5)
    dense_scores = {"a": 0.0, "b": 1.0}
    bm25_scores = {"a": 1.0, "b": 0.0}

    fused = scorer.fuse(dense_scores, bm25_scores)

    assert fused == pytest.approx({"a": 0.5, "b": 0.5})


def test_fused_keys_match_input_keys() -> None:
    scorer = HybridScorer(alpha=0.7)
    fused = scorer.fuse({"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0})

    assert set(fused.keys()) == {"x", "y"}


@pytest.mark.parametrize("invalid_alpha", [-0.01, 1.01, -1.0, 2.0])
def test_rejects_alpha_outside_unit_range(invalid_alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha must be within"):
        HybridScorer(alpha=invalid_alpha)


@pytest.mark.parametrize("boundary_alpha", [0.0, 1.0])
def test_accepts_boundary_alpha_values(boundary_alpha: float) -> None:
    HybridScorer(alpha=boundary_alpha)
