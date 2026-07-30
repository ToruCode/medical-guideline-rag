import pytest
from app.domain.exceptions.vector_store import (
    InvalidSearchQueryError,
    InvalidTopKError,
    VectorDimensionMismatchError,
)
from app.domain.models.chunk import Chunk
from app.domain.models.embedding import EmbeddedChunk
from tests.support.in_memory_vector_store import InMemoryVectorStore


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


def test_search_on_empty_store_returns_empty_list() -> None:
    store = InMemoryVectorStore()

    assert store.search([1.0, 0.0], top_k=5) == []


def test_upsert_empty_list_is_noop() -> None:
    store = InMemoryVectorStore()

    store.upsert([])

    assert store.search([1.0, 0.0], top_k=5) == []


def test_search_returns_stored_chunk() -> None:
    store = InMemoryVectorStore()
    embedded = _make_embedded_chunk(0, [1.0, 0.0])
    store.upsert([embedded])

    results = store.search([1.0, 0.0], top_k=5)

    assert len(results) == 1
    assert results[0].embedded_chunk.chunk == embedded.chunk


def test_search_orders_by_descending_similarity() -> None:
    store = InMemoryVectorStore()
    close = _make_embedded_chunk(0, [1.0, 0.0])
    far = _make_embedded_chunk(1, [0.0, 1.0])
    store.upsert([far, close])

    results = store.search([1.0, 0.0], top_k=5)

    assert [r.embedded_chunk.chunk.chunk_id for r in results] == [
        close.chunk.chunk_id,
        far.chunk.chunk_id,
    ]
    assert results[0].score > results[1].score


def test_search_respects_top_k() -> None:
    store = InMemoryVectorStore()
    store.upsert(
        [
            _make_embedded_chunk(0, [1.0, 0.0]),
            _make_embedded_chunk(1, [0.9, 0.1]),
            _make_embedded_chunk(2, [0.8, 0.2]),
        ]
    )

    results = store.search([1.0, 0.0], top_k=2)

    assert len(results) == 2


def test_search_returns_fewer_than_top_k_when_store_is_smaller() -> None:
    store = InMemoryVectorStore()
    store.upsert([_make_embedded_chunk(0, [1.0, 0.0])])

    results = store.search([1.0, 0.0], top_k=10)

    assert len(results) == 1


def test_upsert_with_same_chunk_id_replaces_entry() -> None:
    store = InMemoryVectorStore()
    original = _make_embedded_chunk(0, [1.0, 0.0])
    store.upsert([original])

    replacement = EmbeddedChunk(chunk=original.chunk, vector=[0.0, 1.0])
    store.upsert([replacement])

    results = store.search([0.0, 1.0], top_k=10)

    assert len(results) == 1
    assert results[0].embedded_chunk.vector == [0.0, 1.0]


def test_search_with_empty_query_vector_raises() -> None:
    store = InMemoryVectorStore()
    store.upsert([_make_embedded_chunk(0, [1.0, 0.0])])

    with pytest.raises(InvalidSearchQueryError):
        store.search([], top_k=5)


@pytest.mark.parametrize("top_k", [0, -1])
def test_search_with_non_positive_top_k_raises(top_k: int) -> None:
    store = InMemoryVectorStore()
    store.upsert([_make_embedded_chunk(0, [1.0, 0.0])])

    with pytest.raises(InvalidTopKError):
        store.search([1.0, 0.0], top_k=top_k)


def test_search_with_dimension_mismatch_raises() -> None:
    store = InMemoryVectorStore()
    store.upsert([_make_embedded_chunk(0, [1.0, 0.0])])

    with pytest.raises(VectorDimensionMismatchError):
        store.search([1.0, 0.0, 0.0], top_k=5)


def test_upsert_with_dimension_mismatch_raises() -> None:
    store = InMemoryVectorStore()
    store.upsert([_make_embedded_chunk(0, [1.0, 0.0])])

    with pytest.raises(VectorDimensionMismatchError):
        store.upsert([_make_embedded_chunk(1, [1.0, 0.0, 0.0])])


def test_zero_vector_scores_zero_instead_of_raising() -> None:
    store = InMemoryVectorStore()
    store.upsert([_make_embedded_chunk(0, [0.0, 0.0])])

    results = store.search([1.0, 0.0], top_k=5)

    assert results[0].score == 0.0


def test_zero_query_vector_scores_zero_instead_of_raising() -> None:
    store = InMemoryVectorStore()
    store.upsert([_make_embedded_chunk(0, [1.0, 0.0])])

    results = store.search([0.0, 0.0], top_k=5)

    assert results[0].score == 0.0


def test_tied_scores_are_ordered_deterministically_by_chunk_id() -> None:
    store = InMemoryVectorStore()
    a = _make_embedded_chunk(0, [1.0, 0.0])
    b = _make_embedded_chunk(1, [1.0, 0.0])
    store.upsert([b, a])

    results = store.search([1.0, 0.0], top_k=5)

    assert [r.embedded_chunk.chunk.chunk_id for r in results] == [
        a.chunk.chunk_id,
        b.chunk.chunk_id,
    ]


def test_upsert_does_not_mutate_caller_vector() -> None:
    store = InMemoryVectorStore()
    vector = [1.0, 0.0]
    embedded = EmbeddedChunk(chunk=_make_chunk(0), vector=vector)
    store.upsert([embedded])

    vector.append(9.0)

    results = store.search([1.0, 0.0], top_k=5)
    assert results[0].embedded_chunk.vector == [1.0, 0.0]


def test_upsert_does_not_mutate_input_list() -> None:
    store = InMemoryVectorStore()
    chunks = [_make_embedded_chunk(0, [1.0, 0.0])]
    original_len = len(chunks)

    store.upsert(chunks)

    assert len(chunks) == original_len
