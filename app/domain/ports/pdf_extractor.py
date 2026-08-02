"""Abstract interface for text extraction strategies used by comparison
tooling (scripts/pdf_extraction_comparison_core.py).

Deliberately separate from PdfLoader (app/domain/ports/pdf_loader.py),
which remains the production extraction interface used by
LoadDocumentService/app.api.dependencies and stays pypdf-only for this
issue. PdfExtractor exists so several extraction strategies - pypdf,
PyMuPDF, and future candidates such as Docling or an OCR pipeline - can
be compared side by side without touching production wiring. Adding a
new strategy only requires a new class implementing extract() below.
"""

from pathlib import Path
from typing import Protocol

from app.domain.models.document import DocumentPage


class PdfExtractor(Protocol):
    """Extracts a PDF file into a list of per-page domain models."""

    def extract(self, file_path: Path) -> list[DocumentPage]: ...
