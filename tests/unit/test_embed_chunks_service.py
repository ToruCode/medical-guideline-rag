import pytest
from app.application.services.embed_chunks import EmbedChunksService
from app.domain.exceptions.embedding import (
    EmbeddingCountMismatchError,
    EmbeddingDimensionMismatchError,
)
from app.domain.models.chunk import Chunk
from tests.support.fake_embedder import FakeEmbedder, FixedVectorsEmbedder


def _make_chunk(chunk_index: int, text: str, page_number: int = 1) -> Chunk:
    return Chunk(
        document_id="doc-1",
        source_name="sample.pdf",
        source_path="/tmp/sample.pdf",
        page_number=page_number,
        chunk_index=chunk_index,
        text=text,
        title="Guideline",
    )


def test_execute_with_empty_chunks_returns_empty_list_without_calling_embedder() -> None:
    embedder = FakeEmbedder()
    service = EmbedChunksService(embedder)

    result = service.execute([])

    assert result == []
    assert embedder.received_texts is None


def test_execute_produces_embedded_chunk_per_chunk_in_order() -> None:
    chunks = [_make_chunk(0, "first"), _make_chunk(1, "second")]
    embedder = FakeEmbedder(dimension=3)
    service = EmbedChunksService(embedder)

    result = service.execute(chunks)

    assert len(result) == 2
    assert [item.chunk for item in result] == chunks
    assert all(len(item.vector) == 3 for item in result)


def test_execute_preserves_chunk_metadata() -> None:
    chunk = _make_chunk(0, "content", page_number=2)
    service = EmbedChunksService(FakeEmbedder())

    result = service.execute([chunk])

    assert result[0].chunk.document_id == "doc-1"
    assert result[0].chunk.source_name == "sample.pdf"
    assert result[0].chunk.source_path == "/tmp/sample.pdf"
    assert result[0].chunk.page_number == 2
    assert result[0].chunk.title == "Guideline"


def test_dimension_mismatch_raises_error() -> None:
    chunks = [_make_chunk(0, "a"), _make_chunk(1, "b")]
    embedder = FixedVectorsEmbedder(vectors=[[0.1, 0.2], [0.1, 0.2, 0.3]])
    service = EmbedChunksService(embedder)

    with pytest.raises(EmbeddingDimensionMismatchError):
        service.execute(chunks)


def test_count_mismatch_raises_error() -> None:
    chunks = [_make_chunk(0, "a"), _make_chunk(1, "b")]
    embedder = FixedVectorsEmbedder(vectors=[[0.1, 0.2]])
    service = EmbedChunksService(embedder)

    with pytest.raises(EmbeddingCountMismatchError):
        service.execute(chunks)
