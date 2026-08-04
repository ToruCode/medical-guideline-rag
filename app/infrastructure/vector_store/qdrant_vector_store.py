"""Persistent VectorStore implementation backed by Qdrant's embedded/local mode.

Runs entirely in-process against a local on-disk directory
(qdrant_client.QdrantClient(path=...)) - no Qdrant server process, no
network access. Data survives a process restart, unlike
InMemoryVectorStore. See docs/adr/0026-persistent-vector-store.md.
"""

import logging
import shutil
import uuid
from typing import Any, cast

from qdrant_client import QdrantClient, models

from app.domain.exceptions.vector_store import (
    InvalidSearchQueryError,
    InvalidTopKError,
    VectorDimensionMismatchError,
    VectorStoreUnavailableError,
)
from app.domain.models.chunk import Chunk
from app.domain.models.embedding import EmbeddedChunk
from app.domain.models.search_result import SearchResult

logger = logging.getLogger(__name__)

# Fixed, arbitrary namespace for deriving a Qdrant point ID from
# Chunk.chunk_id (uuid.uuid5): Qdrant only accepts an unsigned integer or a
# UUID as a point ID, never an arbitrary string. Any fixed namespace works
# as long as it never changes - changing it would silently orphan every
# previously stored point (a new, different UUID for the same chunk_id).
_POINT_ID_NAMESPACE = uuid.UUID("a3f1c6f0-1b8e-4e9a-9c2d-6f6f6b8b7a10")


def _point_id_for(chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, chunk_id))


class QdrantVectorStore:
    """Persists EmbeddedChunks to a local directory via Qdrant's embedded mode.

    The collection is created lazily on the first upsert(), once the
    embedding dimension is known - mirroring InMemoryVectorStore's lazy
    dimension learning. If the collection already exists on disk (created
    by a prior process), its configured dimension is loaded immediately in
    __init__ and used to validate every subsequent upsert()/search() call:
    this is how a dimension mismatch across a process restart (e.g. after
    changing Settings.embedding_provider) is detected, raising the same
    VectorDimensionMismatchError InMemoryVectorStore raises.

    Point IDs are derived from Chunk.chunk_id via uuid.uuid5, so
    re-upserting the same chunk_id always maps to the same point and
    replaces it in place, preserving the upsert-idempotency contract from
    docs/adr/0006-vector-store-strategy.md.
    """

    def __init__(self, path: str, collection_name: str) -> None:
        self._path = path
        self._collection_name = collection_name
        self._client = self._open_client()
        self._dimension = self._load_existing_dimension()

    def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        if not chunks:
            return

        for chunk in chunks:
            self._check_dimension(chunk.vector)

        if self._dimension is None:
            dimension = len(chunks[0].vector)
            self._create_collection(dimension)
            self._dimension = dimension

        points = [
            models.PointStruct(
                id=_point_id_for(chunk.chunk.chunk_id),
                vector=list(chunk.vector),
                payload=_payload_for(chunk.chunk),
            )
            for chunk in chunks
        ]
        self._client.upsert(collection_name=self._collection_name, points=points)
        logger.info("qdrant_vector_store upsert count=%d", len(chunks))

    def search(self, query_vector: list[float], top_k: int) -> list[SearchResult]:
        if top_k <= 0:
            raise InvalidTopKError(f"top_k must be positive, got {top_k}")
        if not query_vector:
            raise InvalidSearchQueryError("query_vector must not be empty")
        if self._dimension is None:
            return []

        self._check_dimension(query_vector)

        response = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=True,
        )
        results = [
            SearchResult(embedded_chunk=self._embedded_chunk_from_point(point), score=point.score)
            for point in response.points
        ]
        results.sort(key=lambda result: (-result.score, result.embedded_chunk.chunk.chunk_id))
        return results[:top_k]

    def rebuild(self) -> None:
        """Discards every stored chunk, so the next upsert() recreates the
        collection from scratch (possibly with a different dimension, e.g.
        after switching embedding_provider). Intended for explicit,
        operator-initiated use (scripts/index_documents.py --rebuild) -
        never called automatically by API dependency wiring.

        Deletes and recreates the entire vector_store_path directory rather
        than just calling the client's delete_collection(): Qdrant's
        embedded/local mode does not reliably release a deleted
        collection's on-disk storage within the same path (verified
        empirically - a collection recreated with a different vector
        dimension right after delete_collection() raises an internal
        numpy shape error from stale on-disk state). vector_store_path is
        assumed to be dedicated to this one collection; do not point two
        different purposes at the same path.
        """
        self._client.close()
        shutil.rmtree(self._path, ignore_errors=True)
        self._client = self._open_client()
        self._dimension = None
        logger.info("qdrant_vector_store rebuild collection=%s", self._collection_name)

    def close(self) -> None:
        """Releases the on-disk lock so another QdrantVectorStore (in this
        process or another) can subsequently open the same path.
        """
        self._client.close()

    def _open_client(self) -> QdrantClient:
        try:
            return QdrantClient(path=self._path)
        except Exception as exc:
            raise VectorStoreUnavailableError(
                f"Could not open the vector store at {self._path!r}: {exc}"
            ) from exc

    def _load_existing_dimension(self) -> int | None:
        if not self._client.collection_exists(self._collection_name):
            return None
        info = self._client.get_collection(self._collection_name)
        vectors_config = info.config.params.vectors
        if not isinstance(vectors_config, models.VectorParams):
            raise VectorStoreUnavailableError(
                f"Collection {self._collection_name!r} has an unsupported vector "
                "configuration (expected a single unnamed vector)."
            )
        return vectors_config.size

    def _create_collection(self, dimension: int) -> None:
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
        )

    def _check_dimension(self, vector: list[float]) -> None:
        if self._dimension is not None and len(vector) != self._dimension:
            raise VectorDimensionMismatchError(
                f"Expected vector of dimension {self._dimension}, got {len(vector)}"
            )

    def _embedded_chunk_from_point(self, point: Any) -> EmbeddedChunk:
        payload = point.payload
        if not isinstance(payload, dict):
            raise VectorStoreUnavailableError(
                f"Point {point.id!r} in collection {self._collection_name!r} has no payload."
            )
        vector = point.vector
        if not isinstance(vector, list):
            raise VectorStoreUnavailableError(
                f"Point {point.id!r} in collection {self._collection_name!r} has an "
                "unsupported vector shape (expected a single unnamed vector)."
            )
        chunk = Chunk(
            document_id=payload["document_id"],
            source_name=payload["source_name"],
            source_path=payload["source_path"],
            page_number=payload["page_number"],
            chunk_index=payload["chunk_index"],
            text=payload["text"],
            title=payload["title"],
        )
        return EmbeddedChunk(chunk=chunk, vector=cast(list[float], vector))


def _payload_for(chunk: Chunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "source_name": chunk.source_name,
        "source_path": chunk.source_path,
        "page_number": chunk.page_number,
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
        "title": chunk.title,
    }
