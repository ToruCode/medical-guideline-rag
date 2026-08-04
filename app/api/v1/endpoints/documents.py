"""Document indexing endpoint."""

import logging
import re
import shutil
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import get_index_document_service, get_object_storage
from app.application.services.index_document import IndexDocumentService
from app.domain.exceptions.chunk import InvalidChunkConfigError
from app.domain.exceptions.document import (
    DocumentLoadError,
    EncryptedDocumentError,
    InvalidPdfError,
    UnsupportedDocumentTypeError,
)
from app.domain.exceptions.embedding import EmbeddingError
from app.domain.exceptions.storage import ObjectNotFoundError, ObjectStorageError
from app.domain.exceptions.vector_store import VectorStoreError
from app.domain.ports.object_storage import ObjectStorage
from app.schemas.document import (
    IndexDocumentResponse,
    IndexFromS3Request,
    UploadDocumentResponse,
)

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


def _download_temp_pdf(object_storage: ObjectStorage, key: str) -> Path:
    """Downloads `key` from ObjectStorage and writes it into a freshly
    created, randomly named temporary directory, mirroring
    _save_temp_pdf's temp-file lifecycle for the S3-backed path.
    """
    content = object_storage.download(key)
    return _save_temp_pdf(content, _sanitize_filename(Path(key).name))


async def _index_temp_pdf(
    index_document_service: IndexDocumentService,
    temp_path: Path,
    sanitized_filename: str,
) -> IndexDocumentResponse:
    """Runs IndexDocumentService against an already-local PDF path and
    maps its exceptions to HTTPException, regardless of whether that
    path came from a direct upload (POST /documents/index) or an S3
    download (POST /documents/index-from-s3). Always removes the
    temp_path's parent directory afterward, success or failure.

    IndexDocumentService.execute is synchronous and, under a real
    embedding model, can take a non-trivial amount of time; it runs in a
    worker thread (run_in_threadpool) so it never blocks the event loop.
    """
    try:
        result = await run_in_threadpool(index_document_service.execute, temp_path)
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
    return await _index_temp_pdf(index_document_service, temp_path, sanitized_filename)


@router.post(
    "/documents/upload",
    response_model=UploadDocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: Annotated[UploadFile, File(...)],
    object_storage: Annotated[ObjectStorage, Depends(get_object_storage)],
) -> UploadDocumentResponse:
    """Uploads a PDF to the configured S3 bucket without indexing it.

    Requires MEDICAL_RAG_STORAGE_PROVIDER=s3. The returned key can be
    passed to POST /documents/index-from-s3 to index it afterward.
    Uploading the same (sanitized) filename twice overwrites the
    previous object under that key - relies on the S3 bucket's own
    versioning to avoid data loss, rather than generating a unique key
    here.
    """
    original_filename = _validate_pdf_filename(file.filename)
    sanitized_filename = _sanitize_filename(original_filename)

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty."
        )

    try:
        await run_in_threadpool(object_storage.upload, sanitized_filename, content)
    except ObjectStorageError as exc:
        logger.exception(
            "upload_document failed unexpectedly", extra={"source_name": sanitized_filename}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error while uploading the file.",
        ) from exc

    return UploadDocumentResponse(key=sanitized_filename)


@router.post(
    "/documents/index-from-s3",
    response_model=IndexDocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def index_document_from_s3(
    request: IndexFromS3Request,
    object_storage: Annotated[ObjectStorage, Depends(get_object_storage)],
    index_document_service: Annotated[IndexDocumentService, Depends(get_index_document_service)],
) -> IndexDocumentResponse:
    """Downloads a PDF already stored in S3 (see POST /documents/upload)
    and indexes it - the S3 counterpart of POST /documents/index,
    sharing the same underlying IndexDocumentService call and exception
    mapping (_index_temp_pdf). Requires MEDICAL_RAG_STORAGE_PROVIDER=s3.
    """
    try:
        temp_path = await run_in_threadpool(_download_temp_pdf, object_storage, request.key)
    except ObjectNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Object not found: {request.key!r}",
        ) from exc
    except ObjectStorageError as exc:
        logger.exception("index_document_from_s3 failed to download", extra={"key": request.key})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error while downloading the file.",
        ) from exc

    sanitized_filename = temp_path.name
    return await _index_temp_pdf(index_document_service, temp_path, sanitized_filename)
