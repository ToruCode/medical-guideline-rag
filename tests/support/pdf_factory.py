"""Helpers for generating synthetic PDFs used only in tests.

Never place real medical guideline content or patient information
here; all text must be self-authored placeholder content.
"""

from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

_FONT_NAME = "Helvetica"
_FONT_SIZE = 12
_LEFT_MARGIN = 72
_MIN_PAGE_WIDTH = 612  # reportlab's default Letter width
_PAGE_HEIGHT = 792  # reportlab's default Letter height


def build_pdf(path: Path, pages: list[str], title: str | None = None) -> Path:
    """Create a PDF with one page per string in `pages` (empty string -> blank page).

    Page width grows to fit each string's drawn text on one line
    (instead of always using the default Letter width): layout-aware
    extractors such as PyMuPDF only return text within a page's
    mediabox, unlike pypdf, so a too-narrow page would silently
    truncate extracted text for long single-line strings.
    """
    max_text_width = max(
        (stringWidth(text, _FONT_NAME, _FONT_SIZE) for text in pages if text), default=0
    )
    page_width = max(_MIN_PAGE_WIDTH, _LEFT_MARGIN * 2 + int(max_text_width) + 1)

    pdf_canvas = canvas.Canvas(str(path), pagesize=(page_width, _PAGE_HEIGHT))
    if title is not None:
        pdf_canvas.setTitle(title)

    for page_text in pages:
        if page_text:
            pdf_canvas.drawString(_LEFT_MARGIN, 720, page_text)
        pdf_canvas.showPage()

    pdf_canvas.save()
    return path


def build_non_pdf_file(path: Path) -> Path:
    """Create a plain text file with a non-.pdf extension."""
    path.write_text("this is not a pdf")
    return path


def build_corrupted_pdf(path: Path) -> Path:
    """Create a .pdf file whose content is not valid PDF data."""
    path.write_bytes(b"not a valid pdf file")
    return path


def build_encrypted_pdf(path: Path, password: str) -> Path:
    """Create a password-encrypted PDF."""
    source_path = path.with_suffix(".source.pdf")
    build_pdf(source_path, ["encrypted content"])

    reader = PdfReader(str(source_path))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.encrypt(user_password=password)

    with path.open("wb") as output_file:
        writer.write(output_file)

    source_path.unlink()
    return path
