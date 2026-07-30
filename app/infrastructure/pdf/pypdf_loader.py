"""pypdf-based implementation of the PdfLoader port."""

import hashlib
import io
import logging
from pathlib import Path

from pypdf import PdfReader

from app.domain.exceptions.document import (
    DocumentLoadError,
    DocumentNotFoundError,
    EncryptedDocumentError,
    InvalidPdfError,
    UnsupportedDocumentTypeError,
)
from app.domain.models.document import DocumentPage

logger = logging.getLogger(__name__)


class PypdfLoader:
    """Loads text-layer PDFs into DocumentPage models using pypdf."""

    def load(self, file_path: Path) -> list[DocumentPage]:
        self._validate_path(file_path)
        content = file_path.read_bytes()
        document_id = hashlib.sha256(content).hexdigest()

        logger.info("PDF load started", extra={"source_name": file_path.name})
        pages = self._parse(content, document_id, file_path)
        logger.info(
            "PDF load completed",
            extra={"source_name": file_path.name, "page_count": len(pages)},
        )
        return pages

    def _validate_path(self, file_path: Path) -> None:
        if not file_path.exists():
            raise DocumentNotFoundError(f"File not found: {file_path.name}")
        if not file_path.is_file():
            raise DocumentNotFoundError(f"Not a regular file: {file_path.name}")
        if file_path.suffix.lower() != ".pdf":
            raise UnsupportedDocumentTypeError(f"Unsupported file type: {file_path.name}")

    def _parse(self, content: bytes, document_id: str, file_path: Path) -> list[DocumentPage]:
        # Every pypdf call below (construction, metadata, page access,
        # text extraction) can raise low-level exceptions on malformed
        # input. They must never reach the application layer, so the
        # whole parse is wrapped and re-raised as InvalidPdfError -
        # except for our own DocumentLoadError subclasses (e.g. the
        # EncryptedDocumentError raised inside this block), which are
        # already the intended domain error and must pass through
        # unchanged.
        try:
            reader = PdfReader(io.BytesIO(content))

            if reader.is_encrypted:
                raise EncryptedDocumentError(f"PDF is encrypted: {file_path.name}")

            title = reader.metadata.title if reader.metadata is not None else None

            return [
                DocumentPage(
                    document_id=document_id,
                    source_name=file_path.name,
                    source_path=str(file_path),
                    page_number=index,
                    text=self._normalize_text(page.extract_text() or ""),
                    title=title,
                )
                for index, page in enumerate(reader.pages, start=1)
            ]
        except DocumentLoadError:
            raise
        except Exception as exc:
            raise InvalidPdfError(f"Failed to parse PDF: {file_path.name}") from exc

    def _normalize_text(self, text: str) -> str:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        return normalized.strip()
