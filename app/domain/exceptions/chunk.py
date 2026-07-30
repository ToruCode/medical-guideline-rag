"""Exceptions raised while configuring or performing text chunking."""


class InvalidChunkConfigError(Exception):
    """Raised when chunk_size/chunk_overlap values are invalid."""
