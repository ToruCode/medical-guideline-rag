import dataclasses

import pytest
from app.domain.models.chunk import Chunk


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


def test_chunk_is_frozen() -> None:
    chunk = _make_chunk()

    with pytest.raises(dataclasses.FrozenInstanceError):
        chunk.text = "changed"  # type: ignore[misc]


def test_chunk_allows_none_title() -> None:
    chunk = dataclasses.replace(_make_chunk(), title=None)

    assert chunk.title is None
