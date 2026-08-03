"""Question answering endpoint."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_ask_question_service
from app.application.services.ask_question import AskQuestionService
from app.core.constants import CITATION_TEXT_PREVIEW_MAX_CHARS
from app.domain.exceptions.embedding import EmbeddingError
from app.domain.exceptions.llm import LlmGenerationError
from app.domain.exceptions.retrieval import EmptyQueryError
from app.domain.exceptions.vector_store import InvalidTopKError, VectorStoreError
from app.domain.models.search_result import SearchResult
from app.schemas.question import AskQuestionRequest, AskQuestionResponse, CitationSchema

logger = logging.getLogger(__name__)

router = APIRouter()


def _truncate_text(text: str, max_chars: int) -> str:
    """Truncates text to at most max_chars, appending "..." when cut.

    The returned string's own length can exceed max_chars by the
    length of the "..." marker; callers relying on a hard byte/char
    cap should account for that.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def _to_citation_schema(result: SearchResult) -> CitationSchema:
    chunk = result.embedded_chunk.chunk
    return CitationSchema(
        document_id=chunk.document_id,
        source_name=chunk.source_name,
        title=chunk.title,
        page_number=chunk.page_number,
        chunk_index=chunk.chunk_index,
        score=result.score,
        text_preview=_truncate_text(chunk.text, CITATION_TEXT_PREVIEW_MAX_CHARS),
    )


@router.post("/questions/ask", response_model=AskQuestionResponse)
def ask_question(
    request: AskQuestionRequest,
    ask_question_service: Annotated[AskQuestionService, Depends(get_ask_question_service)],
) -> AskQuestionResponse:
    """Answer a question, grounded only in indexed guideline passages.

    Returns an explicit insufficient-evidence result (never an error)
    when no indexed passages match the question.
    """
    try:
        result = ask_question_service.execute(request.question, request.top_k)
    except EmptyQueryError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="question must not be empty or whitespace-only.",
        ) from exc
    except InvalidTopKError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="top_k must be a positive integer.",
        ) from exc
    except LlmGenerationError as exc:
        logger.exception("ask_question failed: llm generation error")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to generate an answer from the language model.",
        ) from exc
    except (EmbeddingError, VectorStoreError) as exc:
        logger.exception("ask_question failed unexpectedly")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error while answering the question.",
        ) from exc

    logger.info(
        "ask_question request completed",
        extra={
            "top_k": request.top_k,
            "citation_count": len(result.citations),
            "is_insufficient_evidence": result.is_insufficient_evidence,
        },
    )
    return AskQuestionResponse(
        answer=result.answer,
        citations=[_to_citation_schema(citation) for citation in result.citations],
        is_insufficient_evidence=result.is_insufficient_evidence,
    )
