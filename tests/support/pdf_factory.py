"""Helpers for generating synthetic PDFs used only in tests.

Never place real medical guideline content or patient information
here; all text must be self-authored placeholder content.
"""

from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas


def build_pdf(path: Path, pages: list[str], title: str | None = None) -> Path:
    """Create a PDF with one page per string in `pages` (empty string -> blank page)."""
    pdf_canvas = canvas.Canvas(str(path))
    if title is not None:
        pdf_canvas.setTitle(title)

    for page_text in pages:
        if page_text:
            pdf_canvas.drawString(72, 720, page_text)
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
