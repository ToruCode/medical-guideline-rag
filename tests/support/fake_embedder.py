"""Deterministic test-only Embedder implementation.

Never loads a real model or performs network access; used to test
Embedder-dependent code without downloading anything.
"""


class FakeEmbedder:
    def __init__(self, dimension: int = 4) -> None:
        self._dimension = dimension
        self.received_texts: list[str] | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.received_texts = texts
        return [self._vector_for(text) for text in texts]

    def _vector_for(self, text: str) -> list[float]:
        return [float((len(text) + i) % 7) for i in range(self._dimension)]


class FixedVectorsEmbedder:
    """Returns a pre-configured list of vectors regardless of input, for
    exercising count/dimension mismatch handling in EmbedChunksService.
    """

    def __init__(self, vectors: list[list[float]]) -> None:
        self._vectors = vectors

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._vectors
