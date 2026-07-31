"""Deterministic, dependency-free VectorStore implementation.

Never connects to a real vector database. Storage is a plain
in-process dict, so data does not survive a process restart and is
not shared across processes. Used both by tests and by the FastAPI
app's default dependency wiring (app/api/dependencies.py) as a
stand-in until a real vector database adapter (Qdrant, pgvector) is
implemented; see docs/adr/0006-vector-store-strategy.md.
"""

import math

from app.domain.exceptions.vector_store import (
    InvalidSearchQueryError,
    InvalidTopKError,
    VectorDimensionMismatchError,
)
from app.domain.models.embedding import EmbeddedChunk
from app.domain.models.search_result import SearchResult


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._entries: dict[str, EmbeddedChunk] = {}
        self._dimension: int | None = None

    def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        if not chunks:
            return

        for chunk in chunks:
            self._check_dimension(chunk.vector)

        for chunk in chunks:
            self._entries[chunk.chunk.chunk_id] = EmbeddedChunk(
                chunk=chunk.chunk, vector=list(chunk.vector)
            )
            if self._dimension is None:
                self._dimension = len(chunk.vector)

    def search(self, query_vector: list[float], top_k: int) -> list[SearchResult]:
        if top_k <= 0:
            raise InvalidTopKError(f"top_k must be positive, got {top_k}")
        if not query_vector:
            raise InvalidSearchQueryError("query_vector must not be empty")
        if not self._entries:
            return []

        self._check_dimension(query_vector)

        results = [
            SearchResult(
                embedded_chunk=EmbeddedChunk(chunk=entry.chunk, vector=list(entry.vector)),
                score=self._cosine_similarity(query_vector, entry.vector),
            )
            for entry in self._entries.values()
        ]
        results.sort(key=lambda result: (-result.score, result.embedded_chunk.chunk.chunk_id))
        return results[:top_k]

    def _check_dimension(self, vector: list[float]) -> None:
        if self._dimension is not None and len(vector) != self._dimension:
            raise VectorDimensionMismatchError(
                f"Expected vector of dimension {self._dimension}, got {len(vector)}"
            )

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            raise VectorDimensionMismatchError(
                f"Expected vector of dimension {len(a)}, got {len(b)}"
            )

        dot_product = sum(x * y for x, y in zip(a, b, strict=True))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot_product / (norm_a * norm_b)
