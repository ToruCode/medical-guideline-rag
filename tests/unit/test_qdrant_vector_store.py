"""Contract-parity tests for QdrantVectorStore, mirroring
tests/unit/test_in_memory_vector_store.py so both VectorStore
implementations are verified against the same behavioral contract
(docs/adr/0006-vector-store-strategy.md). Runs entirely against a
tmp_path-local Qdrant embedded instance - no network, no server, no
committed data.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from app.domain.exceptions.vector_store import (
    InvalidSearchQueryError,
    InvalidTopKError,
    VectorDimensionMismatchError,
)
from app.domain.models.chunk import Chunk
from app.domain.models.embedding import EmbeddedChunk
from app.infrastructure.vector_store.qdrant_vector_store import QdrantVectorStore


@pytest.fixture
def store(tmp_path: Path) -> Iterator[QdrantVectorStore]:
    vector_store = QdrantVectorStore(path=str(tmp_path / "qdrant"), collection_name="test")
    yield vector_store
    vector_store.close()


def _make_chunk(chunk_index: int, page_number: int = 1, document_id: str = "doc-1") -> Chunk:
    return Chunk(
        document_id=document_id,
        source_name="sample.pdf",
        source_path="/tmp/sample.pdf",
        page_number=page_number,
        chunk_index=chunk_index,
        text=f"text-{chunk_index}",
        title="Sample",
    )


def _make_embedded_chunk(
    chunk_index: int, vector: list[float], page_number: int = 1, document_id: str = "doc-1"
) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk=_make_chunk(chunk_index, page_number=page_number, document_id=document_id),
        vector=vector,
    )


def test_search_on_empty_store_returns_empty_list(store: QdrantVectorStore) -> None:
    assert store.search([1.0, 0.0], top_k=5) == []


def test_upsert_empty_list_is_noop(store: QdrantVectorStore) -> None:
    store.upsert([])

    assert store.search([1.0, 0.0], top_k=5) == []


def test_search_returns_stored_chunk(store: QdrantVectorStore) -> None:
    embedded = _make_embedded_chunk(0, [1.0, 0.0])
    store.upsert([embedded])

    results = store.search([1.0, 0.0], top_k=5)

    assert len(results) == 1
    assert results[0].embedded_chunk.chunk == embedded.chunk
    assert results[0].embedded_chunk.vector == [1.0, 0.0]


def test_search_orders_by_descending_similarity(store: QdrantVectorStore) -> None:
    close = _make_embedded_chunk(0, [1.0, 0.0])
    far = _make_embedded_chunk(1, [0.0, 1.0])
    store.upsert([far, close])

    results = store.search([1.0, 0.0], top_k=5)

    assert [r.embedded_chunk.chunk.chunk_id for r in results] == [
        close.chunk.chunk_id,
        far.chunk.chunk_id,
    ]
    assert results[0].score > results[1].score


def test_search_respects_top_k(store: QdrantVectorStore) -> None:
    store.upsert(
        [
            _make_embedded_chunk(0, [1.0, 0.0]),
            _make_embedded_chunk(1, [0.9, 0.1]),
            _make_embedded_chunk(2, [0.8, 0.2]),
        ]
    )

    results = store.search([1.0, 0.0], top_k=2)

    assert len(results) == 2


def test_search_returns_fewer_than_top_k_when_store_is_smaller(store: QdrantVectorStore) -> None:
    store.upsert([_make_embedded_chunk(0, [1.0, 0.0])])

    results = store.search([1.0, 0.0], top_k=10)

    assert len(results) == 1


def test_upsert_with_same_chunk_id_replaces_entry(store: QdrantVectorStore) -> None:
    original = _make_embedded_chunk(0, [1.0, 0.0])
    store.upsert([original])

    replacement = EmbeddedChunk(chunk=original.chunk, vector=[0.0, 1.0])
    store.upsert([replacement])

    results = store.search([0.0, 1.0], top_k=10)

    assert len(results) == 1
    assert results[0].embedded_chunk.vector == [0.0, 1.0]


def test_search_with_empty_query_vector_raises(store: QdrantVectorStore) -> None:
    store.upsert([_make_embedded_chunk(0, [1.0, 0.0])])

    with pytest.raises(InvalidSearchQueryError):
        store.search([], top_k=5)


@pytest.mark.parametrize("top_k", [0, -1])
def test_search_with_non_positive_top_k_raises(store: QdrantVectorStore, top_k: int) -> None:
    store.upsert([_make_embedded_chunk(0, [1.0, 0.0])])

    with pytest.raises(InvalidTopKError):
        store.search([1.0, 0.0], top_k=top_k)


def test_search_with_dimension_mismatch_raises(store: QdrantVectorStore) -> None:
    store.upsert([_make_embedded_chunk(0, [1.0, 0.0])])

    with pytest.raises(VectorDimensionMismatchError):
        store.search([1.0, 0.0, 0.0], top_k=5)


def test_upsert_with_dimension_mismatch_raises(store: QdrantVectorStore) -> None:
    store.upsert([_make_embedded_chunk(0, [1.0, 0.0])])

    with pytest.raises(VectorDimensionMismatchError):
        store.upsert([_make_embedded_chunk(1, [1.0, 0.0, 0.0])])


def test_zero_vector_scores_zero_instead_of_raising(store: QdrantVectorStore) -> None:
    store.upsert([_make_embedded_chunk(0, [0.0, 0.0])])

    results = store.search([1.0, 0.0], top_k=5)

    assert results[0].score == 0.0


def test_zero_query_vector_scores_zero_instead_of_raising(store: QdrantVectorStore) -> None:
    store.upsert([_make_embedded_chunk(0, [1.0, 0.0])])

    results = store.search([0.0, 0.0], top_k=5)

    assert results[0].score == 0.0


def test_tied_scores_are_ordered_deterministically_by_chunk_id(store: QdrantVectorStore) -> None:
    a = _make_embedded_chunk(0, [1.0, 0.0])
    b = _make_embedded_chunk(1, [1.0, 0.0])
    store.upsert([b, a])

    results = store.search([1.0, 0.0], top_k=5)

    assert [r.embedded_chunk.chunk.chunk_id for r in results] == [
        a.chunk.chunk_id,
        b.chunk.chunk_id,
    ]


def test_upsert_does_not_mutate_caller_vector(store: QdrantVectorStore) -> None:
    vector = [1.0, 0.0]
    embedded = EmbeddedChunk(chunk=_make_chunk(0), vector=vector)
    store.upsert([embedded])

    vector.append(9.0)

    results = store.search([1.0, 0.0], top_k=5)
    assert results[0].embedded_chunk.vector == [1.0, 0.0]


def test_upsert_does_not_mutate_input_list(store: QdrantVectorStore) -> None:
    chunks = [_make_embedded_chunk(0, [1.0, 0.0])]
    original_len = len(chunks)

    store.upsert(chunks)

    assert len(chunks) == original_len


def test_rebuild_discards_existing_data(store: QdrantVectorStore) -> None:
    store.upsert([_make_embedded_chunk(0, [1.0, 0.0])])

    store.rebuild()

    assert store.search([1.0, 0.0], top_k=5) == []


def test_rebuild_allows_a_new_dimension(store: QdrantVectorStore) -> None:
    store.upsert([_make_embedded_chunk(0, [1.0, 0.0])])

    store.rebuild()
    store.upsert([_make_embedded_chunk(0, [1.0, 0.0, 0.0])])

    results = store.search([1.0, 0.0, 0.0], top_k=5)
    assert len(results) == 1


def test_rebuild_on_a_store_with_no_collection_yet_is_a_noop(store: QdrantVectorStore) -> None:
    store.rebuild()

    assert store.search([1.0, 0.0], top_k=5) == []
