"""PdfLoader adapter around PyMuPdfExtractor.

A thin wrapper rather than a reimplementation, mirroring
PypdfExtractor's relationship to PypdfLoader (pypdf_extractor.py) but
inverted: the tested PyMuPDF extraction logic already lives in
PyMuPdfExtractor (added for Issue #22's comparison tooling), so this
class only adapts it to the PdfLoader protocol's load() method name,
letting it be wired into production (app/api/dependencies.py) without
duplicating or diverging from that logic. See
docs/adr/0018-adopt-pymupdf-for-production-pdf-extraction.md.
"""

from pathlib import Path

from app.domain.models.document import DocumentPage
from app.infrastructure.pdf.pymupdf_extractor import PyMuPdfExtractor


class PyMuPdfLoader:
    """Adapts PyMuPdfExtractor to the PdfLoader protocol."""

    def __init__(self) -> None:
        self._extractor = PyMuPdfExtractor()

    def load(self, file_path: Path) -> list[DocumentPage]:
        return self._extractor.extract(file_path)
