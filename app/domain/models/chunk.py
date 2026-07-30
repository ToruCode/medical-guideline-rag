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

    @property
    def chunk_id(self) -> str:
        """A stable identifier for deduplication in a VectorStore.

        Derived from document_id, page_number, and chunk_index rather
        than stored as a field, so it always stays consistent with
        those fields and requires no extra construction argument.
        """
        return f"{self.document_id}:{self.page_number}:{self.chunk_index}"
