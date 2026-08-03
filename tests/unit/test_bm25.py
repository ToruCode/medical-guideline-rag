"""Unit tests for scripts/bm25.py.

Uses only small, hand-constructed token lists - no real guideline
content and no PDF parsing.
"""

import pytest
from scripts.bm25 import Bm25Index


def test_score_all_ranks_document_containing_query_term_higher() -> None:
    index = Bm25Index(
        [
            ("doc_a", ["医療", "医療", "注意"]),
            ("doc_b", ["注意", "事項"]),
        ]
    )

    scores = index.score_all(["医療"])

    assert scores["doc_a"] > scores["doc_b"]
    assert scores["doc_b"] == 0.0


def test_score_all_returns_zero_for_query_term_absent_from_corpus() -> None:
    index = Bm25Index([("doc_a", ["医療", "注意"])])

    scores = index.score_all(["未知語"])

    assert scores["doc_a"] == 0.0


def test_score_raises_key_error_for_unknown_doc_id() -> None:
    index = Bm25Index([("doc_a", ["医療"])])

    with pytest.raises(KeyError):
        index.score(["医療"], "doc_missing")


def test_top_k_orders_descending_by_score() -> None:
    # doc_a/doc_b are the same length so BM25's length normalization
    # does not confound the comparison: doc_a repeats the query term
    # more often and must score strictly higher than doc_b, which must
    # in turn score higher than doc_c (which never mentions it).
    index = Bm25Index(
        [
            ("doc_a", ["糖尿病", "糖尿病", "治療", "治療"]),
            ("doc_b", ["糖尿病", "食事", "食事", "食事"]),
            ("doc_c", ["高血圧", "治療"]),
        ]
    )

    ranked = index.top_k(["糖尿病"], k=3)

    assert [doc_id for doc_id, _ in ranked] == ["doc_a", "doc_b", "doc_c"]
    assert ranked[2][1] == 0.0


def test_top_k_respects_k() -> None:
    index = Bm25Index(
        [
            ("doc_a", ["a", "b"]),
            ("doc_b", ["a"]),
            ("doc_c", ["b"]),
        ]
    )

    ranked = index.top_k(["a"], k=1)

    assert len(ranked) == 1


def test_top_k_breaks_ties_by_ascending_doc_id() -> None:
    index = Bm25Index(
        [
            ("doc_b", ["a"]),
            ("doc_a", ["a"]),
        ]
    )

    ranked = index.top_k(["a"], k=2)

    assert [doc_id for doc_id, _ in ranked] == ["doc_a", "doc_b"]


def test_top_k_rejects_non_positive_k() -> None:
    index = Bm25Index([("doc_a", ["a"])])

    with pytest.raises(ValueError, match="k must be positive"):
        index.top_k(["a"], k=0)


def test_empty_query_tokens_score_zero_for_every_document() -> None:
    index = Bm25Index([("doc_a", ["a", "b"]), ("doc_b", ["c"])])

    scores = index.score_all([])

    assert scores == {"doc_a": 0.0, "doc_b": 0.0}


def test_single_document_corpus_does_not_raise() -> None:
    index = Bm25Index([("doc_a", ["医療", "機関"])])

    scores = index.score_all(["医療"])

    assert scores["doc_a"] > 0.0
