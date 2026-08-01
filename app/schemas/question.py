"""Request/response schemas for the question answering endpoint."""

from pydantic import BaseModel, Field


class AskQuestionRequest(BaseModel):
    """Request body for POST /api/v1/questions/ask.

    min_length=1 only rejects the empty string; a whitespace-only
    question is rejected by GenerateAnswerService's own validation
    (EmptyQueryError), which the endpoint maps to HTTP 400.
    """

    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, gt=0)


class CitationSchema(BaseModel):
    """One cited passage, projected from SearchResult for API responses.

    Deliberately excludes the embedding vector: callers of this API
    never need it, and it must not be exposed over HTTP.

    chunk_index and text_preview exist to let callers debug retrieval
    quality (e.g. confirm which chunk of a page actually matched, and
    a preview of what it contains) without querying the VectorStore
    directly. text_preview is truncated
    (CITATION_TEXT_PREVIEW_MAX_CHARS in app.core.constants) rather than
    the full chunk text, so this debug field cannot become a channel
    for leaking long passages of guideline text over HTTP.
    """

    document_id: str
    source_name: str
    title: str | None
    page_number: int
    chunk_index: int
    score: float
    text_preview: str


class AskQuestionResponse(BaseModel):
    """Response body for POST /api/v1/questions/ask."""

    answer: str
    citations: list[CitationSchema]
    is_insufficient_evidence: bool
