"""Exceptions raised while retrieving chunks for a natural-language query."""


class RetrievalError(Exception):
    """Base class for retrieval failures."""


class EmptyQueryError(RetrievalError):
    """Raised when a query is empty or contains only whitespace."""
