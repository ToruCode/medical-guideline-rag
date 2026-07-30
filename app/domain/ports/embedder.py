"""Abstract interface for embedding raw text into vectors."""

from typing import Protocol


class Embedder(Protocol):
    """Converts a batch of texts into a batch of embedding vectors."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...
