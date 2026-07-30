"""Document-related domain models."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DocumentPage:
    """One page of text extracted from a source document.

    page_number is 1-based, matching how a human reader refers to a
    page in citations.
    """

    document_id: str
    source_name: str
    source_path: str
    page_number: int
    text: str
    title: str | None
