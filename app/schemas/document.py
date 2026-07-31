"""Response schemas for the document indexing endpoint."""

from pydantic import BaseModel


class IndexDocumentResponse(BaseModel):
    """Response body for POST /api/v1/documents/index.

    source_name is a sanitized version of the originally uploaded file
    name (path separators and other unsafe characters stripped), the
    same value stored as every indexed Chunk's source_name and later
    surfaced in citations from POST /api/v1/questions/ask.
    """

    document_id: str | None
    source_name: str
    page_count: int
    chunk_count: int
    indexed_count: int
