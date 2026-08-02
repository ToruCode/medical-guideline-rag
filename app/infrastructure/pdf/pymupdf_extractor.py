"""PyMuPDF-based implementation of the PdfExtractor protocol.

Added for Issue #22 to compare Japanese PDF text extraction quality
against the production pypdf path (PypdfExtractor/PypdfLoader), not to
replace it. See docs/adr/0017-pdf-extraction-comparison-tooling.md.
"""

import hashlib
import logging
from pathlib import Path

import pymupdf

from app.domain.exceptions.document import (
    DocumentLoadError,
    DocumentNotFoundError,
    EncryptedDocumentError,
    InvalidPdfError,
    UnsupportedDocumentTypeError,
)
from app.domain.models.document import DocumentPage

logger = logging.getLogger(__name__)


class PyMuPdfExtractor:
    """Extracts text-layer PDFs into DocumentPage models using PyMuPDF."""

    def extract(self, file_path: Path) -> list[DocumentPage]:
        self._validate_path(file_path)
        content = file_path.read_bytes()
        document_id = hashlib.sha256(content).hexdigest()

        logger.info("PDF extract started", extra={"source_name": file_path.name})
        pages = self._parse(content, document_id, file_path)
        logger.info(
            "PDF extract completed",
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
        # Every pymupdf call below can raise low-level exceptions on
        # malformed input. They must never reach the application layer,
        # so the whole parse is wrapped and re-raised as InvalidPdfError -
        # except for our own DocumentLoadError subclasses (e.g. the
        # EncryptedDocumentError raised inside this block), which are
        # already the intended domain error and must pass through
        # unchanged.
        try:
            with pymupdf.open(stream=content, filetype="pdf") as document:
                if document.is_encrypted:
                    raise EncryptedDocumentError(f"PDF is encrypted: {file_path.name}")

                title = document.metadata.get("title") or None

                return [
                    DocumentPage(
                        document_id=document_id,
                        source_name=file_path.name,
                        source_path=str(file_path),
                        page_number=index,
                        text=self._normalize_text(page.get_text()),
                        title=title,
                    )
                    for index, page in enumerate(document, start=1)
                ]
        except DocumentLoadError:
            raise
        except Exception as exc:
            raise InvalidPdfError(f"Failed to parse PDF: {file_path.name}") from exc

    def _normalize_text(self, text: str) -> str:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        return normalized.strip()
