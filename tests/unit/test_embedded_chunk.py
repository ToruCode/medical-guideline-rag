import dataclasses

import pytest
from app.domain.models.chunk import Chunk
from app.domain.models.embedding import EmbeddedChunk


def _make_chunk() -> Chunk:
    return Chunk(
        document_id="abc123",
        source_name="sample.pdf",
        source_path="/tmp/sample.pdf",
        page_number=1,
        chunk_index=0,
        text="hello",
        title="Sample",
    )


def test_embedded_chunk_holds_chunk_and_vector() -> None:
    chunk = _make_chunk()
    embedded = EmbeddedChunk(chunk=chunk, vector=[0.1, 0.2, 0.3])

    assert embedded.chunk == chunk
    assert embedded.vector == [0.1, 0.2, 0.3]


def test_embedded_chunk_is_frozen() -> None:
    embedded = EmbeddedChunk(chunk=_make_chunk(), vector=[0.1])

    with pytest.raises(dataclasses.FrozenInstanceError):
        embedded.vector = [0.9]  # type: ignore[misc]
