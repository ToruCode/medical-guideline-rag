"""Unit tests for scripts/reranking_core.py.

Uses only hand-constructed IndexedCorpus fixtures and FakeReranker - no
real guideline PDF and no real Cross Encoder model is loaded or
downloaded.
"""

from app.application.services.search_chunks import SearchChunksService
from app.domain.models.chunk import Chunk
from app.domain.models.embedding import EmbeddedChunk
from app.infrastructure.vector_store.in_memory_vector_store import InMemoryVectorStore
from scripts.bm25 import Bm25Index
from scripts.hybrid_retrieval_core import IndexedCorpus, dense_search, hybrid_search
from scripts.hybrid_scorer import HybridScorer
from scripts.japanese_tokenizer import tokenize_japanese_text
from scripts.reranking_core import hybrid_rerank_search


def _chunk(text: str, page_number: int, chunk_index: int = 0) -> Chunk:
    return Chunk(
        document_id="doc",
        source_name="doc.pdf",
        source_path="/doc.pdf",
        page_number=page_number,
        chunk_index=chunk_index,
        text=text,
        title=None,
    )


def _manual_corpus() -> IndexedCorpus:
    """A 3-chunk corpus where Dense and BM25 clearly disagree (mirrors
    tests/unit/test_hybrid_retrieval_core.py's fixture): chunk_a's
    vector is closest to the test query_vector ([1.0, 0.0]); chunk_b's
    text is the only one sharing bigrams with the test query text
    ("オムツカブレ"); chunk_c matches neither signal well.
    """
    chunk_a = _chunk("一般的な血圧管理について", page_number=1)
    chunk_b = _chunk("オムツカブレ治療薬の使用法", page_number=2)
    chunk_c = _chunk("標準的な食事療法の説明", page_number=3)
    chunks = [chunk_a, chunk_b, chunk_c]

    vector_store = InMemoryVectorStore()
    vector_store.upsert(
        [
            EmbeddedChunk(chunk=chunk_a, vector=[1.0, 0.0]),
            EmbeddedChunk(chunk=chunk_b, vector=[0.0, 1.0]),
            EmbeddedChunk(chunk=chunk_c, vector=[0.5, 0.5]),
        ]
    )
    bm25_index = Bm25Index(
        [(chunk.chunk_id, tokenize_japanese_text(chunk.text)) for chunk in chunks]
    )
    return IndexedCorpus(
        vector_store=vector_store, chunks=chunks, bm25_index=bm25_index, page_count=3
    )


class _RecordingReranker:
    """Reranker test double recording every score() call's arguments."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls.append((query, texts))
        return self._scores


_QUERY_VECTOR = [1.0, 0.0]  # closest to chunk_a
_QUERY_TEXT = "オムツカブレ"  # bigram-overlaps only chunk_b


def _query_tokens() -> list[str]:
    return tokenize_japanese_text(_QUERY_TEXT)


# --- query/text pairs passed correctly ---


def test_reranker_receives_full_chunk_text_for_every_candidate() -> None:
    corpus = _manual_corpus()
    reranker = _RecordingReranker(scores=[1.0, 1.0])

    hybrid_rerank_search(
        _QUERY_VECTOR,
        _query_tokens(),
        _QUERY_TEXT,
        corpus,
        HybridScorer(alpha=0.5),
        reranker,
        dense_candidate_k=1,
        bm25_candidate_k=1,
        reranker_candidate_k=5,
        final_top_k=5,
    )

    assert len(reranker.calls) == 1
    query, texts = reranker.calls[0]
    assert query == _QUERY_TEXT
    assert set(texts) == {"一般的な血圧管理について", "オムツカブレ治療薬の使用法"}


# --- descending score order ---


def test_results_are_sorted_by_reranker_score_descending() -> None:
    corpus = _manual_corpus()
    # 3 candidates (a, b, c); reranker prefers c > a > b regardless of hybrid rank.
    reranker = _RecordingReranker(scores=[])

    def scoring(query: str, texts: list[str]) -> list[float]:
        score_by_text = {
            "一般的な血圧管理について": 0.5,
            "オムツカブレ治療薬の使用法": 0.1,
            "標準的な食事療法の説明": 0.9,
        }
        return [score_by_text[text] for text in texts]

    reranker.score = scoring  # type: ignore[method-assign]

    results, _, _ = hybrid_rerank_search(
        _QUERY_VECTOR,
        _query_tokens(),
        _QUERY_TEXT,
        corpus,
        HybridScorer(alpha=0.5),
        reranker,
        dense_candidate_k=3,
        bm25_candidate_k=3,
        reranker_candidate_k=3,
        final_top_k=3,
    )

    assert [r.page_number for r in results] == [3, 1, 2]
    assert [r.reranker_score for r in results] == [0.9, 0.5, 0.1]


# --- candidate_k / final_top_k enforcement ---


def test_reranker_candidate_k_limits_candidates_sent_to_reranker() -> None:
    corpus = _manual_corpus()
    reranker = _RecordingReranker(scores=[1.0])

    hybrid_rerank_search(
        _QUERY_VECTOR,
        _query_tokens(),
        _QUERY_TEXT,
        corpus,
        HybridScorer(alpha=0.5),
        reranker,
        dense_candidate_k=3,
        bm25_candidate_k=3,
        reranker_candidate_k=1,
        final_top_k=5,
    )

    _, texts = reranker.calls[0]
    assert len(texts) == 1


def test_final_top_k_is_respected() -> None:
    corpus = _manual_corpus()
    reranker = _RecordingReranker(scores=[0.3, 0.9, 0.1])

    results, _, _ = hybrid_rerank_search(
        _QUERY_VECTOR,
        _query_tokens(),
        _QUERY_TEXT,
        corpus,
        HybridScorer(alpha=0.5),
        reranker,
        dense_candidate_k=3,
        bm25_candidate_k=3,
        reranker_candidate_k=3,
        final_top_k=2,
    )

    assert len(results) == 2


# --- deterministic tie-break: ties keep original Hybrid rank order ---


def test_tied_reranker_scores_keep_original_hybrid_rank_order() -> None:
    corpus = _manual_corpus()
    reranker = _RecordingReranker(scores=[1.0, 1.0, 1.0])

    results, _, _ = hybrid_rerank_search(
        _QUERY_VECTOR,
        _query_tokens(),
        _QUERY_TEXT,
        corpus,
        HybridScorer(alpha=0.5),
        reranker,
        dense_candidate_k=3,
        bm25_candidate_k=3,
        reranker_candidate_k=3,
        final_top_k=3,
    )

    assert [r.rank_before_rerank for r in results] == [1, 2, 3]
    assert [r.rank_after_rerank for r in results] == [1, 2, 3]


# --- empty / fewer-than-candidate_k candidates ---


def test_empty_corpus_returns_no_results_without_calling_reranker() -> None:
    empty_corpus = IndexedCorpus(
        vector_store=InMemoryVectorStore(), chunks=[], bm25_index=Bm25Index([]), page_count=0
    )
    reranker = _RecordingReranker(scores=[])

    results, retrieval_ms, reranking_ms = hybrid_rerank_search(
        _QUERY_VECTOR,
        ["x"],
        "x",
        empty_corpus,
        HybridScorer(alpha=0.5),
        reranker,
        dense_candidate_k=5,
        bm25_candidate_k=5,
        reranker_candidate_k=5,
        final_top_k=5,
    )

    assert results == []
    assert reranking_ms == 0.0
    assert reranker.calls == []


def test_works_with_fewer_candidates_than_reranker_candidate_k() -> None:
    corpus = _manual_corpus()  # only 3 chunks total
    reranker = _RecordingReranker(scores=[1.0, 2.0, 3.0])

    results, _, _ = hybrid_rerank_search(
        _QUERY_VECTOR,
        _query_tokens(),
        _QUERY_TEXT,
        corpus,
        HybridScorer(alpha=0.5),
        reranker,
        dense_candidate_k=20,
        bm25_candidate_k=20,
        reranker_candidate_k=20,  # far more than the 3 available chunks
        final_top_k=20,
    )

    assert len(results) == 3


# --- FakeReranker-driven rank improvement / regression ---


def test_rerank_improves_rank_when_reranker_favors_a_lower_hybrid_candidate() -> None:
    corpus = _manual_corpus()
    # Hybrid (alpha=1.0 -> Dense only) ranks chunk_a (page 1) first,
    # chunk_c (page 3) second. The reranker strongly prefers chunk_c.
    reranker = _RecordingReranker(scores=[])
    reranker.score = lambda query, texts: [  # type: ignore[method-assign]
        {
            "一般的な血圧管理について": 0.1,
            "標準的な食事療法の説明": 0.9,
        }[text]
        for text in texts
    ]

    results, _, _ = hybrid_rerank_search(
        _QUERY_VECTOR,
        _query_tokens(),
        _QUERY_TEXT,
        corpus,
        HybridScorer(alpha=1.0),
        reranker,
        dense_candidate_k=2,
        bm25_candidate_k=1,
        reranker_candidate_k=2,
        final_top_k=2,
    )

    assert results[0].page_number == 3  # chunk_c moved to rank 1 after rerank
    assert results[0].rank_before_rerank == 2
    assert results[0].rank_after_rerank == 1


def test_rerank_worsens_rank_when_reranker_disfavors_top_hybrid_candidate() -> None:
    corpus = _manual_corpus()
    reranker = _RecordingReranker(scores=[])
    reranker.score = lambda query, texts: [  # type: ignore[method-assign]
        {
            "一般的な血圧管理について": 0.1,  # was Hybrid's #1, reranker likes it least
            "標準的な食事療法の説明": 0.9,
        }[text]
        for text in texts
    ]

    results, _, _ = hybrid_rerank_search(
        _QUERY_VECTOR,
        _query_tokens(),
        _QUERY_TEXT,
        corpus,
        HybridScorer(alpha=1.0),
        reranker,
        dense_candidate_k=2,
        bm25_candidate_k=1,
        reranker_candidate_k=2,
        final_top_k=2,
    )

    chunk_a_result = next(r for r in results if r.page_number == 1)
    assert chunk_a_result.rank_before_rerank == 1
    assert chunk_a_result.rank_after_rerank == 2  # dropped from 1st to last


# --- Dense/Hybrid single-strategy results are unchanged by this issue ---


def test_dense_search_still_matches_search_chunks_service() -> None:
    corpus = _manual_corpus()

    ranked = dense_search(_QUERY_VECTOR, corpus, top_k=3)
    expected = SearchChunksService(corpus.vector_store).execute(_QUERY_VECTOR, top_k=3)

    assert [(r.page_number, r.final_score) for r in ranked] == [
        (result.embedded_chunk.chunk.page_number, result.score) for result in expected
    ]


def test_hybrid_search_unaffected_by_reranking_core_additions() -> None:
    corpus = _manual_corpus()

    results = hybrid_search(
        _QUERY_VECTOR,
        _query_tokens(),
        corpus,
        HybridScorer(alpha=0.7),
        dense_candidate_k=3,
        bm25_candidate_k=3,
        final_top_k=3,
    )

    assert len(results) == 3
    assert all(r.bm25_score is not None for r in results)
