"""Use case for retrieving chunks most similar to a natural-language query."""

import logging

from app.application.services.search_chunks import SearchChunksService
from app.domain.exceptions.embedding import EmbeddingCountMismatchError
from app.domain.exceptions.retrieval import EmptyQueryError
from app.domain.exceptions.vector_store import InvalidTopKError
from app.domain.models.search_result import SearchResult
from app.domain.ports.embedder import Embedder

logger = logging.getLogger(__name__)


class RetrieveChunksService:
    """Embeds a natural-language query and searches for similar chunks.

    Composes an injected Embedder with an injected SearchChunksService
    (rather than a VectorStore directly), so query_vector/top_k
    validation and the search step's own logging stay defined in
    exactly one place (SearchChunksService), matching how
    IndexDocumentService composes existing Application services.

    top_k is validated here too, before the Embedder is called, even
    though SearchChunksService validates it again: this avoids paying
    for an Embedder call (e.g. a real model or API request) on input
    that is already known to be invalid. This mirrors SearchChunksService
    re-validating what its injected VectorStore is also required to
    validate.
    """

    def __init__(self, embedder: Embedder, search_chunks: SearchChunksService) -> None:
        self._embedder = embedder
        self._search_chunks = search_chunks

    def execute(self, query: str, top_k: int) -> list[SearchResult]:
        if top_k <= 0:
            raise InvalidTopKError(f"top_k must be positive, got {top_k}")
        if not query.strip():
            raise EmptyQueryError("query must not be empty or whitespace-only")

        vectors = self._embedder.embed([query])
        if len(vectors) != 1:
            raise EmbeddingCountMismatchError(f"Expected 1 vector for 1 query, got {len(vectors)}")

        results = self._search_chunks.execute(vectors[0], top_k)
        logger.info("retrieve_chunks top_k=%d returned=%d", top_k, len(results))
        return results
