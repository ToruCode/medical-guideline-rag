"""Unit tests for scripts/hybrid_retrieval_core.py.

Uses only synthetic PDFs (tests/support/pdf_factory.py), hand-constructed
IndexedCorpus fixtures, and FakeEmbedder - no real guideline PDF is
needed or used, and no real sentence-transformers model is loaded.
"""

from pathlib import Path

import pytest
from app.application.services.search_chunks import SearchChunksService
from app.domain.models.chunk import Chunk
from app.domain.models.embedding import EmbeddedChunk
from app.infrastructure.embedding.fake_embedder import FakeEmbedder
from app.infrastructure.pdf.pymupdf_extractor import PyMuPdfExtractor
from app.infrastructure.vector_store.in_memory_vector_store import InMemoryVectorStore
from scripts.bm25 import Bm25Index
from scripts.hybrid_retrieval_core import (
    STRATEGY_DENSE,
    STRATEGY_HYBRID,
    IndexedCorpus,
    build_indexed_corpus,
    dense_search,
    evaluate_strategy,
    hybrid_search,
)
from scripts.hybrid_scorer import HybridScorer
from scripts.japanese_tokenizer import tokenize_japanese_text
from scripts.retrieval_baseline_core import DatasetCase, DatasetDocument
from tests.support.pdf_factory import build_pdf


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
    """A 3-chunk corpus where Dense and BM25 clearly disagree:

    - chunk_a's vector is closest to the test query_vector ([1.0, 0.0]).
    - chunk_b's text is the only one sharing bigrams with the test
      query text ("オムツカブレ") - the only chunk with a nonzero BM25
      score for it.
    - chunk_c matches neither signal well, and is not a candidate under
      dense_candidate_k=1/bm25_candidate_k=1 in the tests below.
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


# --- dense_search ---


def test_dense_search_matches_existing_search_chunks_service() -> None:
    corpus = _manual_corpus()
    query_vector = [1.0, 0.0]

    ranked = dense_search(query_vector, corpus, top_k=3)
    expected = SearchChunksService(corpus.vector_store).execute(query_vector, top_k=3)

    assert [(r.page_number, r.chunk_index, r.final_score) for r in ranked] == [
        (
            result.embedded_chunk.chunk.page_number,
            result.embedded_chunk.chunk.chunk_index,
            result.score,
        )
        for result in expected
    ]
    assert all(r.bm25_score is None for r in ranked)


# --- hybrid_search: candidate union ---


def test_hybrid_search_unions_dense_only_and_bm25_only_candidates() -> None:
    corpus = _manual_corpus()

    results = hybrid_search(
        query_vector=[1.0, 0.0],
        query_tokens=tokenize_japanese_text("オムツカブレ"),
        corpus=corpus,
        scorer=HybridScorer(alpha=0.5),
        dense_candidate_k=1,
        bm25_candidate_k=1,
        final_top_k=2,
    )

    retrieved_pages = {r.page_number for r in results}
    assert retrieved_pages == {1, 2}  # chunk_a (dense-only) and chunk_b (bm25-only)


def test_hybrid_search_respects_final_top_k() -> None:
    corpus = _manual_corpus()

    results = hybrid_search(
        query_vector=[1.0, 0.0],
        query_tokens=tokenize_japanese_text("オムツカブレ"),
        corpus=corpus,
        scorer=HybridScorer(alpha=0.5),
        dense_candidate_k=3,
        bm25_candidate_k=3,
        final_top_k=1,
    )

    assert len(results) == 1


def test_hybrid_search_alpha_zero_ranks_by_bm25_only() -> None:
    corpus = _manual_corpus()

    results = hybrid_search(
        query_vector=[1.0, 0.0],  # would favor chunk_a (page 1) if Dense mattered
        query_tokens=tokenize_japanese_text("オムツカブレ"),
        corpus=corpus,
        scorer=HybridScorer(alpha=0.0),
        dense_candidate_k=1,
        bm25_candidate_k=1,
        final_top_k=2,
    )

    assert results[0].page_number == 2  # chunk_b: the only BM25 match


def test_hybrid_search_alpha_one_ranks_by_dense_only() -> None:
    corpus = _manual_corpus()

    results = hybrid_search(
        query_vector=[1.0, 0.0],  # favors chunk_a (page 1)
        query_tokens=tokenize_japanese_text("オムツカブレ"),  # would favor chunk_b if BM25 mattered
        corpus=corpus,
        scorer=HybridScorer(alpha=1.0),
        dense_candidate_k=1,
        bm25_candidate_k=1,
        final_top_k=2,
    )

    assert results[0].page_number == 1  # chunk_a: the closest Dense match


def test_hybrid_search_populates_both_scores_for_every_result() -> None:
    corpus = _manual_corpus()

    results = hybrid_search(
        query_vector=[1.0, 0.0],
        query_tokens=tokenize_japanese_text("オムツカブレ"),
        corpus=corpus,
        scorer=HybridScorer(alpha=0.5),
        dense_candidate_k=1,
        bm25_candidate_k=1,
        final_top_k=2,
    )

    assert all(r.dense_score is not None and r.bm25_score is not None for r in results)


def test_hybrid_search_returns_empty_list_for_empty_corpus() -> None:
    empty_corpus = IndexedCorpus(
        vector_store=InMemoryVectorStore(), chunks=[], bm25_index=Bm25Index([]), page_count=0
    )

    results = hybrid_search(
        query_vector=[1.0, 0.0],
        query_tokens=["x"],
        corpus=empty_corpus,
        scorer=HybridScorer(alpha=0.5),
        dense_candidate_k=5,
        bm25_candidate_k=5,
        final_top_k=5,
    )

    assert results == []


# --- build_indexed_corpus + evaluate_strategy (synthetic PDF, FakeEmbedder) ---


def test_build_indexed_corpus_indexes_one_chunk_per_short_page(tmp_path: Path) -> None:
    pdf_path = build_pdf(
        tmp_path / "guideline.pdf", ["First page content.", "Second page content."]
    )
    document = DatasetDocument(source_path=pdf_path, label="Test Guideline")
    embedder = FakeEmbedder(dimension=4)

    corpus = build_indexed_corpus(
        document, embedder, chunk_size=1000, chunk_overlap=0, pdf_extractor=PyMuPdfExtractor()
    )

    assert corpus.page_count == 2
    assert len(corpus.chunks) == 2
    assert isinstance(corpus.bm25_index, Bm25Index)


def test_alpha_one_hybrid_strategy_matches_dense_strategy_exactly(tmp_path: Path) -> None:
    """Requirement: alpha=1.0 must reproduce the Dense-only strategy's
    results exactly (ranked locations and aggregate Recall@k/MRR), since
    at alpha=1.0 the BM25 signal is weighted out entirely and min-max
    normalization does not change the ranking order of the Dense
    signal, given dense_candidate_k >= top_k (satisfied here: 10 >= 3).
    """
    pdf_path = build_pdf(
        tmp_path / "guideline.pdf",
        [
            "Page one content about topic Alpha.",
            "Page two content about topic Beta and some more detail here.",
            "Page three content about topic Gamma with even more additional detail text.",
        ],
    )
    document = DatasetDocument(source_path=pdf_path, label="Test Guideline")
    cases = [
        DatasetCase(
            question="What is topic Alpha?", granularity="page", expected_locations=[(1, None)]
        ),
        DatasetCase(
            question="Tell me about Beta", granularity="page", expected_locations=[(2, None)]
        ),
        DatasetCase(
            question="Explain Gamma in detail please",
            granularity="page",
            expected_locations=[(3, None)],
        ),
    ]
    embedder = FakeEmbedder(dimension=4)

    corpus = build_indexed_corpus(
        document, embedder, chunk_size=1000, chunk_overlap=0, pdf_extractor=PyMuPdfExtractor()
    )

    dense_run = evaluate_strategy(
        STRATEGY_DENSE,
        corpus,
        cases,
        embedder,
        top_k=3,
        dense_candidate_k=10,
        bm25_candidate_k=10,
        alpha=1.0,
        chunk_size=1000,
        chunk_overlap=0,
        embedding_model_name="fake",
        scorer=HybridScorer(alpha=1.0),
    )
    hybrid_run = evaluate_strategy(
        STRATEGY_HYBRID,
        corpus,
        cases,
        embedder,
        top_k=3,
        dense_candidate_k=10,
        bm25_candidate_k=10,
        alpha=1.0,
        chunk_size=1000,
        chunk_overlap=0,
        embedding_model_name="fake",
        scorer=HybridScorer(alpha=1.0),
    )

    dense_locations = [c.ranked_locations for c in dense_run.case_results]
    hybrid_locations = [c.ranked_locations for c in hybrid_run.case_results]
    assert hybrid_locations == dense_locations
    assert hybrid_run.aggregate == dense_run.aggregate


def test_evaluate_strategy_rejects_unknown_strategy() -> None:
    corpus = _manual_corpus()
    cases = [DatasetCase(question="q", granularity="page", expected_locations=[(1, None)])]
    embedder = FakeEmbedder(dimension=2)

    with pytest.raises(ValueError, match="Unknown strategy"):
        evaluate_strategy(
            "reciprocal_rank_fusion",
            corpus,
            cases,
            embedder,
            top_k=1,
            dense_candidate_k=1,
            bm25_candidate_k=1,
            alpha=0.5,
            chunk_size=1000,
            chunk_overlap=0,
            embedding_model_name="fake",
            scorer=HybridScorer(alpha=0.5),
        )
