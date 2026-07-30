"""Embedding-related domain models."""

from dataclasses import dataclass

from app.domain.models.chunk import Chunk


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    """A Chunk together with its embedding vector.

    vector is a list[float] as required by the embedding contract;
    note that a frozen dataclass prevents reassigning this attribute
    but does not make the list's contents immutable - callers must not
    mutate it in place.
    """

    chunk: Chunk
    vector: list[float]
