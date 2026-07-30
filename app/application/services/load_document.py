"""Use case for loading a source document into domain page models."""

from pathlib import Path

from app.domain.models.document import DocumentPage
from app.domain.ports.pdf_loader import PdfLoader


class LoadDocumentService:
    """Loads a document via an injected PdfLoader implementation."""

    def __init__(self, pdf_loader: PdfLoader) -> None:
        self._pdf_loader = pdf_loader

    def execute(self, file_path: Path) -> list[DocumentPage]:
        return self._pdf_loader.load(file_path)
