# Architecture

This project follows a layered architecture with dependencies pointing
inward.

## Layers

- **API** (`app/api`, `app/schemas`): receives HTTP requests, validates
  input, calls application use cases, converts results into HTTP
  responses. Must not call Qdrant, PostgreSQL, LangChain, or an LLM
  directly.
- **Application** (`app/application`): executes use cases and
  coordinates retrieval, reranking, answerability judgment, and
  generation.
- **Domain** (`app/domain`): defines entities, value objects, and
  interfaces independent of frameworks. Must not depend on API,
  Application, or Infrastructure, nor on any framework or SDK.
  - `app/domain/models/document.py` defines `DocumentPage`, an
    immutable (`frozen=True, slots=True`) dataclass representing one
    page of extracted text, with a 1-based `page_number`.
  - `app/domain/ports/pdf_loader.py` defines the `PdfLoader` Protocol
    (`load(file_path: Path) -> list[DocumentPage]`) that infrastructure
    PDF loaders implement. The domain layer never imports pypdf.
  - `app/domain/exceptions/document.py` defines
    `DocumentLoadError` and its subtypes
    (`DocumentNotFoundError`, `UnsupportedDocumentTypeError`,
    `EncryptedDocumentError`, `InvalidPdfError`) used to translate
    low-level parsing failures into domain-level errors.
- **Infrastructure** (`app/infrastructure`): implements domain
  interfaces using external libraries (PDF loading, embeddings, LLMs,
  Qdrant, PostgreSQL, S3).
  - `app/infrastructure/pdf/pypdf_loader.py` implements `PdfLoader`
    using `pypdf`. It validates the input path, computes
    `document_id` as the SHA-256 hex digest of the raw file bytes,
    reads PDF title metadata, extracts and normalizes per-page text
    (CRLF/CR -> LF, then strip), and converts pypdf's low-level
    exceptions into the domain exceptions above. See
    `docs/adr/0003-pdf-extraction-library.md` for the library choice
    and its constraints (text-layer PDFs only, no OCR, no table/layout
    reconstruction).
- **Core** (`app/core`): application settings, logging, common
  exceptions, security, shared constants.
  - `app/core/config.py` defines a `Settings` class (pydantic-settings)
    loaded from environment variables (`MEDICAL_RAG_` prefix) and an
    optional `.env` file, exposed via a cached `get_settings()`
    accessor usable as a FastAPI dependency.
  - `app/core/logging.py` defines `setup_logging(settings)`, which
    configures the standard library `logging` module via
    `logging.config.dictConfig`. Modules obtain loggers with
    `logging.getLogger(__name__)`.

## Dependency Rule

```
API -> Application -> Domain
Infrastructure -> Domain (implements interfaces)
Core is referenced by all layers
```

Domain must never depend on API, Application, or Infrastructure.

## Status

Minimal FastAPI setup with a health check endpoint, environment-based
settings, standard logging, and a PDF loading foundation (Issue #4).
There is no upload API yet; retrieval, generation, and other external
service integrations land in subsequent issues.

See `docs/adr/0001-project-architecture.md`,
`docs/adr/0002-configuration-and-logging.md`, and
`docs/adr/0003-pdf-extraction-library.md` for the architecture
decision records.
