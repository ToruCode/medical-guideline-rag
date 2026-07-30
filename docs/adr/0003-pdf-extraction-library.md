# 0003. Use pypdf for PDF text extraction, decoupled from LangChain

## Status

Accepted

## Context

The project needs to read text-layer medical guideline PDFs page by
page and convert them into an internal, library-independent
representation that later chunking, embedding, and retrieval steps can
consume. This is the first infrastructure integration in the project,
so the boundary between the domain model and any specific PDF or LLM
framework library needs to be established now.

## Decision

- Use `pypdf` in the infrastructure layer (`app/infrastructure/pdf`)
  to open PDFs, read metadata, and extract per-page text. It is a pure
  Python library with no native dependencies, is actively maintained,
  and provides the encryption detection and metadata access this
  project needs without pulling in a larger toolkit.
- Do not use LangChain's `Document` class anywhere. The domain layer
  defines its own `DocumentPage` dataclass
  (`app/domain/models/document.py`) and a `PdfLoader` Protocol
  (`app/domain/ports/pdf_loader.py`) that infrastructure implementations
  satisfy. This keeps business rules (citation metadata, page
  numbering, insufficient-evidence handling) independent of any
  specific framework's document model, matching the layered
  architecture in `docs/architecture.md` and `CLAUDE.md`'s rule that
  the domain layer must not depend on LangChain.
- Scanned PDFs and OCR are explicitly out of scope. This issue only
  extracts text that already exists in the PDF's text layer. Adding
  OCR would introduce a materially different processing pipeline
  (image preprocessing, OCR engine dependency, accuracy trade-offs)
  that deserves its own design and is not needed for the guideline
  documents this system currently targets.
- Complex tables, multi-column layouts, and mathematical notation are
  not structurally preserved. `pypdf`'s `extract_text()` returns text
  in a heuristic reading order; multi-column pages or tables may have
  their content reordered or interleaved. This issue only guarantees
  raw per-page text extraction, not layout-aware reconstruction.

## Consequences

- Because `PdfLoader` is a `Protocol` consumed only through
  `LoadDocumentService`, a future PyMuPDF-based loader, an
  OCR-augmented loader, or any other extraction strategy can be added
  as a new infrastructure class implementing the same `load(file_path)
  -> list[DocumentPage]` contract, without changes to the domain or
  application layers.
- Guideline PDFs with meaningful tabular data or multi-column layouts
  may need a different extraction strategy later; this is a known
  limitation to revisit once real guideline documents are evaluated.
- Low-level `pypdf` exceptions are translated into the domain
  exceptions defined in `app/domain/exceptions/document.py`
  (`DocumentNotFoundError`, `UnsupportedDocumentTypeError`,
  `EncryptedDocumentError`, `InvalidPdfError`) so infrastructure
  details never leak into the application layer.
