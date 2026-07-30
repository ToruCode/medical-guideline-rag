import logging

import pytest
from app.application.services.index_chunks import IndexChunksService
from app.domain.models.chunk import Chunk
from app.domain.models.embedding import EmbeddedChunk
from app.domain.models.search_result import SearchResult


class FakeVectorStore:
    def __init__(self) -> None:
        self.upserted: list[EmbeddedChunk] | None = None

    def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        self.upserted = chunks

    def search(self, query_vector: list[float], top_k: int) -> list[SearchResult]:
        raise NotImplementedError


def _make_embedded_chunk(chunk_index: int, text: str = "sensitive guideline text") -> EmbeddedChunk:
    chunk = Chunk(
        document_id="doc-1",
        source_name="sample.pdf",
        source_path="/tmp/sample.pdf",
        page_number=1,
        chunk_index=chunk_index,
        text=text,
        title="Guideline",
    )
    return EmbeddedChunk(chunk=chunk, vector=[0.1, 0.2, 0.3])


def test_execute_with_empty_list_does_not_call_vector_store() -> None:
    store = FakeVectorStore()
    service = IndexChunksService(store)

    service.execute([])

    assert store.upserted is None


def test_execute_delegates_to_vector_store_upsert() -> None:
    store = FakeVectorStore()
    service = IndexChunksService(store)
    chunks = [_make_embedded_chunk(0), _make_embedded_chunk(1)]

    service.execute(chunks)

    assert store.upserted == chunks


def test_execute_logs_count_without_chunk_text_or_vector(caplog: pytest.LogCaptureFixture) -> None:
    store = FakeVectorStore()
    service = IndexChunksService(store)
    chunks = [_make_embedded_chunk(0, text="do-not-leak-this-text")]

    with caplog.at_level(logging.INFO):
        service.execute(chunks)

    log_output = "\n".join(caplog.messages)
    assert "1" in log_output
    assert "do-not-leak-this-text" not in log_output
    assert "0.1" not in log_output
