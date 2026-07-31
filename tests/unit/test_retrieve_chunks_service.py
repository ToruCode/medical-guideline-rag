import logging

import pytest
from app.application.services.retrieve_chunks import RetrieveChunksService
from app.application.services.search_chunks import SearchChunksService
from app.domain.exceptions.embedding import EmbeddingCountMismatchError
from app.domain.exceptions.retrieval import EmptyQueryError
from app.domain.exceptions.vector_store import InvalidTopKError, VectorStoreError
from app.domain.models.chunk import Chunk
from app.domain.models.embedding import EmbeddedChunk
from app.domain.models.search_result import SearchResult
from app.infrastructure.embedding.fake_embedder import FakeEmbedder, FixedVectorsEmbedder


class FakeVectorStore:
    def __init__(
        self, results: list[SearchResult] | None = None, error: Exception | None = None
    ) -> None:
        self._results = results if results is not None else []
        self._error = error
        self.received_query_vector: list[float] | None = None
        self.received_top_k: int | None = None

    def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        raise NotImplementedError

    def search(self, query_vector: list[float], top_k: int) -> list[SearchResult]:
        if self._error is not None:
            raise self._error
        self.received_query_vector = query_vector
        self.received_top_k = top_k
        return self._results


def _make_search_result(text: str = "sensitive guideline text", score: float = 0.9) -> SearchResult:
    chunk = Chunk(
        document_id="doc-1",
        source_name="sample.pdf",
        source_path="/tmp/sample.pdf",
        page_number=1,
        chunk_index=0,
        text=text,
        title="Guideline",
    )
    return SearchResult(embedded_chunk=EmbeddedChunk(chunk=chunk, vector=[0.1, 0.2]), score=score)


def _build_service(
    store: FakeVectorStore, embedder: object | None = None
) -> tuple[RetrieveChunksService, object]:
    embedder = embedder if embedder is not None else FakeEmbedder()
    service = RetrieveChunksService(
        embedder=embedder,  # type: ignore[arg-type]
        search_chunks=SearchChunksService(store),
    )
    return service, embedder


def test_execute_embeds_query_and_delegates_to_search_chunks() -> None:
    expected = [_make_search_result()]
    store = FakeVectorStore(results=expected)
    query = "how to treat condition x"
    expected_vector = FakeEmbedder().embed([query])[0]
    service, embedder = _build_service(store)

    results = service.execute(query, top_k=3)

    assert results == expected
    assert embedder.received_texts == [query]  # type: ignore[union-attr]
    assert store.received_query_vector == expected_vector
    assert store.received_top_k == 3


def test_execute_returns_results_in_store_order() -> None:
    expected = [_make_search_result(score=0.9), _make_search_result(score=0.5)]
    store = FakeVectorStore(results=expected)
    service, _ = _build_service(store)

    results = service.execute("query text", top_k=5)

    assert results == expected


def test_execute_with_no_results_returns_empty_list() -> None:
    store = FakeVectorStore(results=[])
    service, _ = _build_service(store)

    results = service.execute("query text", top_k=5)

    assert results == []


@pytest.mark.parametrize("query", ["", "   ", "\t\n"])
def test_execute_with_empty_or_whitespace_query_raises_without_embedding(query: str) -> None:
    store = FakeVectorStore()
    service, embedder = _build_service(store)

    with pytest.raises(EmptyQueryError):
        service.execute(query, top_k=3)

    assert embedder.received_texts is None  # type: ignore[union-attr]
    assert store.received_query_vector is None


@pytest.mark.parametrize("top_k", [0, -1])
def test_execute_with_non_positive_top_k_raises_without_embedding(top_k: int) -> None:
    store = FakeVectorStore()
    service, embedder = _build_service(store)

    with pytest.raises(InvalidTopKError):
        service.execute("query text", top_k=top_k)

    assert embedder.received_texts is None  # type: ignore[union-attr]
    assert store.received_top_k is None


@pytest.mark.parametrize("vector_count", [0, 2])
def test_execute_with_embedder_returning_wrong_vector_count_raises(vector_count: int) -> None:
    store = FakeVectorStore()
    embedder = FixedVectorsEmbedder(vectors=[[0.1, 0.2]] * vector_count)
    service, _ = _build_service(store, embedder=embedder)

    with pytest.raises(EmbeddingCountMismatchError):
        service.execute("query text", top_k=3)

    assert store.received_query_vector is None


def test_execute_propagates_vector_store_error() -> None:
    store = FakeVectorStore(error=VectorStoreError("boom"))
    service, embedder = _build_service(store)

    with pytest.raises(VectorStoreError):
        service.execute("query text", top_k=3)

    assert embedder.received_texts == ["query text"]  # type: ignore[union-attr]


def test_execute_propagates_embedder_error() -> None:
    class RaisingEmbedder:
        def embed(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("embedder unavailable")

    store = FakeVectorStore()
    service, _ = _build_service(store, embedder=RaisingEmbedder())

    with pytest.raises(RuntimeError):
        service.execute("query text", top_k=3)

    assert store.received_query_vector is None


def test_execute_logs_top_k_and_count_without_query_or_vector(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = FakeVectorStore(results=[_make_search_result(text="do-not-leak-this-text")])
    service, _ = _build_service(store)

    with caplog.at_level(logging.INFO):
        service.execute("do-not-leak-this-query", top_k=5)

    log_output = "\n".join(caplog.messages)
    assert "5" in log_output
    assert "do-not-leak-this-query" not in log_output
    assert "do-not-leak-this-text" not in log_output
    assert "0.1" not in log_output
