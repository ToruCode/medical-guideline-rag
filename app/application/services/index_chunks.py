"""Use case for storing embedded chunks into a VectorStore."""

import logging

from app.domain.models.embedding import EmbeddedChunk
from app.domain.ports.vector_store import VectorStore

logger = logging.getLogger(__name__)


class IndexChunksService:
    """Stores EmbeddedChunks via an injected VectorStore implementation."""

    def __init__(self, vector_store: VectorStore) -> None:
        self._vector_store = vector_store

    def execute(self, chunks: list[EmbeddedChunk]) -> None:
        if not chunks:
            logger.info("index_chunks skipped: empty input")
            return

        self._vector_store.upsert(chunks)
        logger.info("index_chunks stored count=%d", len(chunks))
