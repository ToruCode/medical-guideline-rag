"""Abstract interface for storing and searching EmbeddedChunks."""

from typing import Protocol

from app.domain.models.embedding import EmbeddedChunk
from app.domain.models.search_result import SearchResult


class VectorStore(Protocol):
    """Stores EmbeddedChunks and finds the most similar ones to a query vector.

    Implementations must:
    - Treat upsert as idempotent by chunk_id: re-upserting an
      EmbeddedChunk with a chunk_id that already exists replaces the
      stored entry rather than duplicating it.
    - Raise app.domain.exceptions.vector_store.InvalidTopKError for
      top_k <= 0, and InvalidSearchQueryError for an empty query_vector.
    - Raise VectorDimensionMismatchError when a vector's dimension does
      not match the store's established dimension.
    - Return SearchResults ordered by descending score, with a
      deterministic tie-break (by chunk_id) for equal scores.
    - Return fewer than top_k results (never an error) when fewer than
      top_k entries are stored.
    """

    def upsert(self, chunks: list[EmbeddedChunk]) -> None: ...

    def search(self, query_vector: list[float], top_k: int) -> list[SearchResult]: ...
