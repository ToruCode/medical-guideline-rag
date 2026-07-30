"""Exceptions raised while embedding chunks into vectors."""


class EmbeddingError(Exception):
    """Base class for embedding failures."""


class EmbeddingCountMismatchError(EmbeddingError):
    """Raised when the embedder returns a different number of vectors than input texts."""


class EmbeddingDimensionMismatchError(EmbeddingError):
    """Raised when returned vectors do not share a consistent dimension."""
