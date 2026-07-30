"""Chunk-related domain models."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Chunk:
    """One piece of text split from a source document page.

    chunk_index is a 0-based position of this chunk within its
    originating page, used for internal ordering (unlike page_number,
    which is 1-based for citation display).
    """

    document_id: str
    source_name: str
    source_path: str
    page_number: int
    chunk_index: int
    text: str
    title: str | None
