"""Abstract interface for loading PDF documents into domain models."""

from pathlib import Path
from typing import Protocol

from app.domain.models.document import DocumentPage


class PdfLoader(Protocol):
    """Loads a PDF file into a list of per-page domain models."""

    def load(self, file_path: Path) -> list[DocumentPage]: ...
