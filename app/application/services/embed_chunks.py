"""Use case for embedding chunks into vectors."""

from app.domain.exceptions.embedding import (
    EmbeddingCountMismatchError,
    EmbeddingDimensionMismatchError,
)
from app.domain.models.chunk import Chunk
from app.domain.models.embedding import EmbeddedChunk
from app.domain.ports.embedder import Embedder


class EmbedChunksService:
    """Embeds chunks via an injected Embedder implementation."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    def execute(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        if not chunks:
            return []

        texts = [chunk.text for chunk in chunks]
        vectors = self._embedder.embed(texts)
        self._validate(texts, vectors)

        return [
            EmbeddedChunk(chunk=chunk, vector=vector)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

    def _validate(self, texts: list[str], vectors: list[list[float]]) -> None:
        if len(vectors) != len(texts):
            raise EmbeddingCountMismatchError(f"Expected {len(texts)} vectors, got {len(vectors)}")

        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) > 1:
            raise EmbeddingDimensionMismatchError(
                f"Embedder returned inconsistent vector dimensions: {sorted(dimensions)}"
            )
