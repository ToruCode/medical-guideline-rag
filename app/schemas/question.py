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
    """

    document_id: str
    source_name: str
    title: str | None
    page_number: int
    chunk_index: int
    score: float


class AskQuestionResponse(BaseModel):
    """Response body for POST /api/v1/questions/ask."""

    answer: str
    citations: list[CitationSchema]
    is_insufficient_evidence: bool
