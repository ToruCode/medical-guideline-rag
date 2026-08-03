"""Fuses per-signal candidate scores (Dense, BM25) into one ranking
score (Issue #24's Hybrid Search comparison).

ScoreFuser is a narrow Protocol specifically so an alternative fusion
strategy - e.g. Reciprocal Rank Fusion (RRF), which fuses by rank
rather than by normalized score - can be added later as a second
implementation without changing scripts/hybrid_retrieval_core.py's
calling code. HybridScorer is the only implementation this issue adds
(alpha-weighted linear combination, per the issue's spec).
"""

from typing import Protocol

from scripts.score_normalization import min_max_normalize


class ScoreFuser(Protocol):
    """Combines per-candidate Dense/BM25 scores into one fused score per
    candidate.

    Implementations must return a dict with exactly the same keys as
    dense_scores (which callers construct with the same keys as
    bm25_scores - see hybrid_retrieval_core.hybrid_search), with higher
    fused values meaning a better match.
    """

    def fuse(
        self, dense_scores: dict[str, float], bm25_scores: dict[str, float]
    ) -> dict[str, float]: ...


class HybridScorer:
    """alpha-weighted linear combination of min-max-normalized Dense and
    BM25 scores.

    hybrid_score = alpha * dense_score_normalized + (1 - alpha) * bm25_score_normalized

    alpha=1.0 uses only the (normalized) Dense signal; alpha=0.0 uses
    only the (normalized) BM25 signal. Normalization is candidate-set
    relative (scripts.score_normalization.min_max_normalize) - see that
    module for the zero-division (all-equal-scores) convention.
    """

    def __init__(self, alpha: float) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be within [0.0, 1.0], got {alpha}")
        self._alpha = alpha

    def fuse(
        self, dense_scores: dict[str, float], bm25_scores: dict[str, float]
    ) -> dict[str, float]:
        dense_normalized = min_max_normalize(dense_scores)
        bm25_normalized = min_max_normalize(bm25_scores)
        return {
            doc_id: self._alpha * dense_normalized[doc_id]
            + (1 - self._alpha) * bm25_normalized[doc_id]
            for doc_id in dense_scores
        }
