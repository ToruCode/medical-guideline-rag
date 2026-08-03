"""Candidate-set-relative min-max score normalization (Issue #24's
Hybrid Search comparison).

Normalizes each signal (Dense, BM25) within the candidate set being
scored for one query, not against any global/corpus-wide range - a
simple, easily explained convention (see
docs/adr/0019-hybrid-search-comparison.md).
"""


def min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Maps scores to [0.0, 1.0] via min-max scaling over scores' own
    values.

    Returns 0.0 for every id when all scores are equal (including a
    single-candidate input), instead of dividing by zero: with no
    spread across the candidate set, this signal carries no
    distinguishing information, so it should contribute nothing to a
    score fused from it.
    """
    if not scores:
        return {}

    values = scores.values()
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        return dict.fromkeys(scores, 0.0)

    span = maximum - minimum
    return {doc_id: (value - minimum) / span for doc_id, value in scores.items()}
