"""PdfExtractor adapter around the existing pypdf-based PdfLoader.

A thin wrapper rather than a reimplementation: the production pypdf
extraction logic already lives in PypdfLoader and must not be
duplicated or diverge from it. This class only adapts that logic to
the PdfExtractor protocol's extract() method name so it can sit
alongside PyMuPdfExtractor in the comparison tooling's extractor
registry.
"""

from pathlib import Path

from app.domain.models.document import DocumentPage
from app.infrastructure.pdf.pypdf_loader import PypdfLoader


class PypdfExtractor:
    """Adapts PypdfLoader to the PdfExtractor protocol."""

    def __init__(self) -> None:
        self._loader = PypdfLoader()

    def extract(self, file_path: Path) -> list[DocumentPage]:
        return self._loader.load(file_path)
