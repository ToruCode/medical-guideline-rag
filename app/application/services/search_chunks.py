"""Use case for searching a VectorStore for chunks similar to a query vector."""

import logging

from app.domain.exceptions.vector_store import InvalidSearchQueryError, InvalidTopKError
from app.domain.models.search_result import SearchResult
from app.domain.ports.vector_store import VectorStore

logger = logging.getLogger(__name__)


class SearchChunksService:
    """Searches an injected VectorStore implementation for similar chunks.

    Validates query_vector/top_k itself, in addition to the same
    validation VectorStore implementations are required to perform, so
    that callers see consistent errors regardless of which VectorStore
    implementation is injected.
    """

    def __init__(self, vector_store: VectorStore) -> None:
        self._vector_store = vector_store

    def execute(self, query_vector: list[float], top_k: int) -> list[SearchResult]:
        if not query_vector:
            raise InvalidSearchQueryError("query_vector must not be empty")
        if top_k <= 0:
            raise InvalidTopKError(f"top_k must be positive, got {top_k}")

        results = self._vector_store.search(query_vector, top_k)
        logger.info("search_chunks top_k=%d returned=%d", top_k, len(results))
        return results
