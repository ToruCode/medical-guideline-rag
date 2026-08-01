"""Pure retrieval-quality metrics: Recall@k and Mean Reciprocal Rank (MRR).

No framework or embedding-model dependency: these operate purely on
ranked identifier lists, so they can be unit-tested without a real
Embedder/VectorStore (tests/unit/test_evaluation_metrics.py) and are
reused by the real-embedding evaluation gate
(tests/integration/test_retrieval_evaluation.py).
"""


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    """1 / (1-based rank) of the first id in ranked_ids that is relevant.

    Returns 0.0 when no relevant id appears anywhere in ranked_ids,
    matching the standard MRR convention for a fully missed query.
    """
    for rank, candidate_id in enumerate(ranked_ids, start=1):
        if candidate_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant_ids present anywhere in the top k of ranked_ids.

    Raises ValueError for an empty relevant_ids or non-positive k, since
    both make "recall" undefined rather than a meaningful 0.0/1.0 - a
    malformed evaluation case should fail loudly, not silently score.
    """
    if not relevant_ids:
        raise ValueError("relevant_ids must not be empty")
    if k <= 0:
        raise ValueError("k must be positive")
    top_k = set(ranked_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def mean(scores: list[float]) -> float:
    """Arithmetic mean of one score per evaluation case (e.g. the output
    of reciprocal_rank() or recall_at_k() across a whole dataset).
    """
    if not scores:
        raise ValueError("scores must not be empty")
    return sum(scores) / len(scores)
