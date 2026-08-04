"""Exceptions raised while storing or searching EmbeddedChunks."""


class VectorStoreError(Exception):
    """Base class for vector store failures."""


class InvalidSearchQueryError(VectorStoreError):
    """Raised when a search query is structurally invalid (e.g. an empty query_vector)."""


class InvalidTopKError(VectorStoreError):
    """Raised when top_k is not a positive integer."""


class VectorDimensionMismatchError(VectorStoreError):
    """Raised when a vector's dimension does not match the store's established dimension."""


class VectorStoreUnavailableError(VectorStoreError):
    """Raised when a persistent vector store cannot be opened (e.g. a locked or
    corrupted on-disk index), as distinct from a structurally invalid request.
    """
