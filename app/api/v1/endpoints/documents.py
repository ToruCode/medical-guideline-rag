"""Document indexing endpoint."""

import logging
import re
import shutil
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.dependencies import get_index_document_service
from app.application.services.index_document import IndexDocumentService
from app.domain.exceptions.chunk import InvalidChunkConfigError
from app.domain.exceptions.document import (
    DocumentLoadError,
    EncryptedDocumentError,
    InvalidPdfError,
    UnsupportedDocumentTypeError,
)
from app.domain.exceptions.embedding import EmbeddingError
from app.domain.exceptions.vector_store import VectorStoreError
from app.schemas.document import IndexDocumentResponse

logger = logging.getLogger(__name__)

router = APIRouter()

_UNSAFE_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _validate_pdf_filename(filename: str | None) -> str:
    if not filename or not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF files are supported.",
        )
    return filename


def _sanitize_filename(filename: str) -> str:
    """Strips path separators and other unsafe characters, keeping the
    file's display name recognizable (it becomes Chunk.source_name, and
    is shown in citations) while making it safe to use as a file's
    basename. Never lets a caller-supplied filename influence which
    directory the file is written into; see _save_temp_pdf.
    """
    name = Path(filename).name
    name = _UNSAFE_FILENAME_CHARS.sub("_", name).strip()
    return name or "upload.pdf"


def _save_temp_pdf(content: bytes, filename: str) -> Path:
    """Writes content to `filename` inside a freshly-created, randomly
    named temporary directory. `filename` must already be sanitized by
    _sanitize_filename; unpredictability and collision-avoidance come
    from the random directory, not from the filename itself.
    """
    temp_dir = Path(tempfile.mkdtemp())
    temp_path = temp_dir / filename
    temp_path.write_bytes(content)
    return temp_path


@router.post(
    "/documents/index",
    response_model=IndexDocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def index_document(
    file: Annotated[UploadFile, File(...)],
    index_document_service: Annotated[IndexDocumentService, Depends(get_index_document_service)],
) -> IndexDocumentResponse:
    """Index a PDF: extract text, chunk, embed, and store it for later search.

    The uploaded file is saved under a sanitized version of its own name,
    inside a freshly-created, randomly-named temporary directory (so the
    uploaded name never determines *where* the file is written), and the
    whole directory is always removed afterward, whether indexing
    succeeds or fails.
    """
    original_filename = _validate_pdf_filename(file.filename)
    sanitized_filename = _sanitize_filename(original_filename)

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty."
        )

    temp_path = _save_temp_pdf(content, sanitized_filename)
    try:
        result = index_document_service.execute(temp_path)
    except EncryptedDocumentError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"'{sanitized_filename}' is an encrypted PDF and cannot be processed.",
        ) from exc
    except (UnsupportedDocumentTypeError, InvalidPdfError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"'{sanitized_filename}' could not be parsed as a valid PDF.",
        ) from exc
    except DocumentLoadError as exc:
        logger.exception(
            "index_document failed unexpectedly", extra={"source_name": sanitized_filename}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error while processing the uploaded file.",
        ) from exc
    except (InvalidChunkConfigError, EmbeddingError, VectorStoreError) as exc:
        logger.exception(
            "index_document failed unexpectedly", extra={"source_name": sanitized_filename}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error while indexing the document.",
        ) from exc
    finally:
        shutil.rmtree(temp_path.parent, ignore_errors=True)

    logger.info(
        "index_document request completed",
        extra={
            "source_name": result.source_name,
            "document_id": result.document_id,
            "page_count": result.page_count,
            "chunk_count": result.chunk_count,
            "indexed_count": result.indexed_count,
        },
    )
    return IndexDocumentResponse(
        document_id=result.document_id,
        source_name=result.source_name,
        page_count=result.page_count,
        chunk_count=result.chunk_count,
        indexed_count=result.indexed_count,
    )
