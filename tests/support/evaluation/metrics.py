"""Pure retrieval- and answer-quality metrics: Recall@k, Mean Reciprocal
Rank (MRR), citation precision/recall, and answer-point coverage.

No framework or embedding-model/LLM dependency: these operate purely on
plain identifier lists, sets, and strings, so they can be unit-tested
without a real Embedder/VectorStore/Llm
(tests/unit/test_evaluation_metrics.py) and are reused by both the
real-embedding retrieval gate (tests/integration/test_retrieval_evaluation.py)
and the answer-quality tooling (scripts/answer_quality_core.py, Issue #10).
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


def citation_precision(cited_pages: set[int], expected_pages: set[int]) -> float | None:
    """Fraction of returned citations whose page is actually expected.

    Returns None (undefined) when no citations were returned at all -
    there is nothing to measure precision over, and a 0.0 would
    misleadingly read as "every citation was wrong" rather than "no
    citations were made" (contrast with citation_recall, which is
    always well-defined as long as expected_pages is non-empty).
    """
    if not cited_pages:
        return None
    return len(cited_pages & expected_pages) / len(cited_pages)


def citation_recall(cited_pages: set[int], expected_pages: set[int]) -> float:
    """Fraction of expected pages that were actually cited.

    Raises ValueError for an empty expected_pages, since recall is
    undefined there (mirrors recall_at_k's convention above) - callers
    should skip this metric entirely for cases where no evidence is
    expected to exist, rather than call it with an empty set.
    """
    if not expected_pages:
        raise ValueError("expected_pages must not be empty")
    return len(cited_pages & expected_pages) / len(expected_pages)


def answer_point_coverage(answer: str, expected_points: list[str]) -> float | None:
    """Fraction of expected_points found as a case-insensitive substring
    of answer.

    Returns None when expected_points is empty (no answer-point ground
    truth was defined for this case, which is different from "0%
    covered"). This is a deliberately simple, deterministic and
    explainable approximation - it cannot detect a correct answer that
    paraphrases or uses a synonym instead of the exact expected
    substring; see
    docs/adr/0023-answer-quality-and-citation-consistency-evaluation.md
    for why this limitation was accepted instead of using an
    LLM-as-a-Judge.
    """
    if not expected_points:
        return None
    answer_lower = answer.lower()
    matched = sum(1 for point in expected_points if point.lower() in answer_lower)
    return matched / len(expected_points)
