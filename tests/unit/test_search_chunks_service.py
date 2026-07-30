import logging

import pytest
from app.application.services.search_chunks import SearchChunksService
from app.domain.exceptions.vector_store import InvalidSearchQueryError, InvalidTopKError
from app.domain.models.chunk import Chunk
from app.domain.models.embedding import EmbeddedChunk
from app.domain.models.search_result import SearchResult


class FakeVectorStore:
    def __init__(self, results: list[SearchResult] | None = None) -> None:
        self._results = results if results is not None else []
        self.received_query_vector: list[float] | None = None
        self.received_top_k: int | None = None

    def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        raise NotImplementedError

    def search(self, query_vector: list[float], top_k: int) -> list[SearchResult]:
        self.received_query_vector = query_vector
        self.received_top_k = top_k
        return self._results


def _make_search_result(text: str = "sensitive guideline text") -> SearchResult:
    chunk = Chunk(
        document_id="doc-1",
        source_name="sample.pdf",
        source_path="/tmp/sample.pdf",
        page_number=1,
        chunk_index=0,
        text=text,
        title="Guideline",
    )
    return SearchResult(embedded_chunk=EmbeddedChunk(chunk=chunk, vector=[0.1, 0.2]), score=0.9)


def test_execute_delegates_to_vector_store_search() -> None:
    expected = [_make_search_result()]
    store = FakeVectorStore(results=expected)
    service = SearchChunksService(store)

    results = service.execute([1.0, 0.0], top_k=3)

    assert results == expected
    assert store.received_query_vector == [1.0, 0.0]
    assert store.received_top_k == 3


def test_execute_with_empty_query_vector_raises_without_calling_store() -> None:
    store = FakeVectorStore()
    service = SearchChunksService(store)

    with pytest.raises(InvalidSearchQueryError):
        service.execute([], top_k=3)

    assert store.received_query_vector is None


@pytest.mark.parametrize("top_k", [0, -1])
def test_execute_with_non_positive_top_k_raises_without_calling_store(top_k: int) -> None:
    store = FakeVectorStore()
    service = SearchChunksService(store)

    with pytest.raises(InvalidTopKError):
        service.execute([1.0, 0.0], top_k=top_k)

    assert store.received_top_k is None


def test_execute_logs_counts_without_chunk_text_or_vector(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = FakeVectorStore(results=[_make_search_result(text="do-not-leak-this-text")])
    service = SearchChunksService(store)

    with caplog.at_level(logging.INFO):
        service.execute([1.0, 0.0], top_k=5)

    log_output = "\n".join(caplog.messages)
    assert "5" in log_output
    assert "do-not-leak-this-text" not in log_output
    assert "0.1" not in log_output
