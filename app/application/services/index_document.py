"""Use case for indexing a single PDF document end to end."""

import logging
from dataclasses import dataclass
from pathlib import Path

from app.application.services.chunk_document import ChunkDocumentService
from app.application.services.embed_chunks import EmbedChunksService
from app.application.services.index_chunks import IndexChunksService
from app.application.services.load_document import LoadDocumentService

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IndexDocumentResult:
    """Summary of one IndexDocumentService.execute() run.

    document_id is None when the source PDF has zero pages, since no
    DocumentPage (and therefore no document_id) is produced in that
    case.
    """

    document_id: str | None
    source_name: str
    page_count: int
    chunk_count: int
    indexed_count: int


class IndexDocumentService:
    """Loads, chunks, embeds, and stores a PDF by composing the existing
    per-step Application services (LoadDocumentService, ChunkDocumentService,
    EmbedChunksService, IndexChunksService).

    Does not catch exceptions raised by any step; they propagate to the
    caller unchanged (fail-fast), so a failure's original error type
    (e.g. DocumentLoadError, InvalidChunkConfigError, EmbeddingError,
    VectorStoreError) is preserved.
    """

    def __init__(
        self,
        load_document: LoadDocumentService,
        chunk_document: ChunkDocumentService,
        embed_chunks: EmbedChunksService,
        index_chunks: IndexChunksService,
    ) -> None:
        self._load_document = load_document
        self._chunk_document = chunk_document
        self._embed_chunks = embed_chunks
        self._index_chunks = index_chunks

    def execute(self, file_path: Path) -> IndexDocumentResult:
        logger.info("index_document started", extra={"source_name": file_path.name})

        pages = self._load_document.execute(file_path)
        logger.info("index_document loaded", extra={"page_count": len(pages)})

        chunks = self._chunk_document.execute(pages)
        logger.info("index_document chunked", extra={"chunk_count": len(chunks)})

        embedded_chunks = self._embed_chunks.execute(chunks)
        logger.info("index_document embedded", extra={"embedded_count": len(embedded_chunks)})

        self._index_chunks.execute(embedded_chunks)

        result = IndexDocumentResult(
            document_id=pages[0].document_id if pages else None,
            source_name=file_path.name,
            page_count=len(pages),
            chunk_count=len(chunks),
            indexed_count=len(embedded_chunks),
        )
        logger.info(
            "index_document completed",
            extra={"document_id": result.document_id, "source_name": result.source_name},
        )
        return result
