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


class UploadDocumentResponse(BaseModel):
    """Response body for POST /api/v1/documents/upload.

    key identifies the uploaded object in the configured S3 bucket
    (Settings.s3_bucket_name); pass it to POST
    /api/v1/documents/index-from-s3 to index it.
    """

    key: str


class IndexFromS3Request(BaseModel):
    """Request body for POST /api/v1/documents/index-from-s3.

    key must already exist in the configured S3 bucket - see POST
    /api/v1/documents/upload.
    """

    key: str
