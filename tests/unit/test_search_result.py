import dataclasses

import pytest
from app.domain.models.chunk import Chunk
from app.domain.models.embedding import EmbeddedChunk
from app.domain.models.search_result import SearchResult


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


def _make_embedded_chunk() -> EmbeddedChunk:
    return EmbeddedChunk(chunk=_make_chunk(), vector=[0.1, 0.2, 0.3])


def test_search_result_holds_embedded_chunk_and_score() -> None:
    embedded_chunk = _make_embedded_chunk()
    result = SearchResult(embedded_chunk=embedded_chunk, score=0.75)

    assert result.embedded_chunk == embedded_chunk
    assert result.score == 0.75


def test_search_result_is_frozen() -> None:
    result = SearchResult(embedded_chunk=_make_embedded_chunk(), score=0.5)

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.score = 0.9  # type: ignore[misc]
