"""Integration test verifying QdrantVectorStore data survives a process
restart: a fresh instance reopened at the same on-disk path sees data
written by a now-closed prior instance. Uses hand-built vectors only - no
real embedding model, no network, no committed data. See
docs/adr/0026-persistent-vector-store.md.
"""

from pathlib import Path

import pytest
from app.domain.exceptions.vector_store import VectorDimensionMismatchError
from app.domain.models.chunk import Chunk
from app.domain.models.embedding import EmbeddedChunk
from app.infrastructure.vector_store.qdrant_vector_store import QdrantVectorStore


def _make_embedded_chunk(chunk_index: int, vector: list[float]) -> EmbeddedChunk:
    chunk = Chunk(
        document_id="doc-1",
        source_name="guideline.pdf",
        source_path="/tmp/guideline.pdf",
        page_number=1,
        chunk_index=chunk_index,
        text=f"guideline text {chunk_index}",
        title="Sample Guideline",
    )
    return EmbeddedChunk(chunk=chunk, vector=vector)


def test_data_survives_reopening_the_same_path(tmp_path: Path) -> None:
    path = str(tmp_path / "qdrant")

    first_chunk = _make_embedded_chunk(0, [1.0, 0.0])
    first_process = QdrantVectorStore(path=path, collection_name="guideline_chunks")
    first_process.upsert([first_chunk, _make_embedded_chunk(1, [0.0, 1.0])])
    first_process.close()

    second_process = QdrantVectorStore(path=path, collection_name="guideline_chunks")
    try:
        results = second_process.search([1.0, 0.0], top_k=5)
    finally:
        second_process.close()

    assert len(results) == 2
    assert results[0].embedded_chunk.chunk.chunk_id == first_chunk.chunk.chunk_id


def test_reindexing_the_same_chunk_after_restart_replaces_it_not_duplicates(
    tmp_path: Path,
) -> None:
    path = str(tmp_path / "qdrant")

    first_process = QdrantVectorStore(path=path, collection_name="guideline_chunks")
    first_process.upsert([_make_embedded_chunk(0, [1.0, 0.0])])
    first_process.close()

    second_process = QdrantVectorStore(path=path, collection_name="guideline_chunks")
    try:
        second_process.upsert([_make_embedded_chunk(0, [0.0, 1.0])])
        results = second_process.search([0.0, 1.0], top_k=10)
    finally:
        second_process.close()

    assert len(results) == 1
    assert results[0].embedded_chunk.vector == [0.0, 1.0]


def test_dimension_mismatch_is_detected_after_restart(tmp_path: Path) -> None:
    path = str(tmp_path / "qdrant")

    first_process = QdrantVectorStore(path=path, collection_name="guideline_chunks")
    first_process.upsert([_make_embedded_chunk(0, [1.0, 0.0])])
    first_process.close()

    second_process = QdrantVectorStore(path=path, collection_name="guideline_chunks")
    try:
        with pytest.raises(VectorDimensionMismatchError):
            second_process.upsert([_make_embedded_chunk(1, [1.0, 0.0, 0.0])])
    finally:
        second_process.close()


def test_rebuild_then_restart_starts_with_a_clean_index(tmp_path: Path) -> None:
    path = str(tmp_path / "qdrant")

    first_process = QdrantVectorStore(path=path, collection_name="guideline_chunks")
    first_process.upsert([_make_embedded_chunk(0, [1.0, 0.0])])
    first_process.rebuild()
    first_process.close()

    second_process = QdrantVectorStore(path=path, collection_name="guideline_chunks")
    try:
        assert second_process.search([1.0, 0.0], top_k=5) == []
        second_process.upsert([_make_embedded_chunk(0, [1.0, 0.0, 0.0])])
        results = second_process.search([1.0, 0.0, 0.0], top_k=5)
    finally:
        second_process.close()

    assert len(results) == 1
