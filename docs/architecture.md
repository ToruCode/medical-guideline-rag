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
  - `app/application/services/load_document.py` defines
    `LoadDocumentService`, which delegates to an injected `PdfLoader`.
  - `app/application/services/chunk_document.py` defines
    `ChunkDocumentService`, which delegates text splitting to an
    injected `TextSplitter` and carries `document_id`, `page_number`,
    `source_name`, `source_path`, and `title` from each `DocumentPage`
    into the resulting `Chunk`s.
  - `app/application/services/embed_chunks.py` defines
    `EmbedChunksService`, which delegates vectorization to an injected
    `Embedder`, returns an empty list for empty input without calling
    the embedder, and validates that the returned vector count and
    dimensions are consistent before building `EmbeddedChunk`s.
- **Domain** (`app/domain`): defines entities, value objects, and
  interfaces independent of frameworks. Must not depend on API,
  Application, or Infrastructure, nor on any framework or SDK.
  - `app/domain/models/document.py` defines `DocumentPage`, an
    immutable (`frozen=True, slots=True`) dataclass representing one
    page of extracted text, with a 1-based `page_number`.
  - `app/domain/models/chunk.py` defines `Chunk`, an immutable
    (`frozen=True, slots=True`) dataclass representing one piece of
    text split from a page, with a 0-based `chunk_index` for ordering
    within the page.
  - `app/domain/models/embedding.py` defines `EmbeddedChunk`, an
    immutable (`frozen=True, slots=True`) dataclass composing a
    `Chunk` with its `vector: list[float]`.
  - `app/domain/ports/pdf_loader.py` defines the `PdfLoader` Protocol
    (`load(file_path: Path) -> list[DocumentPage]`) that infrastructure
    PDF loaders implement. The domain layer never imports pypdf.
  - `app/domain/ports/text_splitter.py` defines the `TextSplitter`
    Protocol (`split(text: str) -> list[str]`) that infrastructure
    chunkers implement. It operates on raw text only and knows nothing
    about `DocumentPage` metadata.
  - `app/domain/ports/embedder.py` defines the `Embedder` Protocol
    (`embed(texts: list[str]) -> list[list[float]]`) that infrastructure
    embedding adapters implement. The domain layer never imports
    sentence-transformers, an LLM SDK, or LangChain.
  - `app/domain/exceptions/document.py` defines
    `DocumentLoadError` and its subtypes
    (`DocumentNotFoundError`, `UnsupportedDocumentTypeError`,
    `EncryptedDocumentError`, `InvalidPdfError`) used to translate
    low-level parsing failures into domain-level errors.
  - `app/domain/exceptions/chunk.py` defines
    `InvalidChunkConfigError`, raised when `chunk_size`/`chunk_overlap`
    values are invalid.
  - `app/domain/exceptions/embedding.py` defines `EmbeddingError` and
    its subtypes (`EmbeddingCountMismatchError`,
    `EmbeddingDimensionMismatchError`) used to translate invalid
    embedder output into domain-level errors.
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
  - `app/infrastructure/chunking/fixed_size_text_splitter.py`
    implements `TextSplitter` with a dependency-free, character-count
    based sliding window algorithm (no LangChain). It validates
    `chunk_size`/`chunk_overlap` in its constructor and raises
    `InvalidChunkConfigError` on invalid values. See
    `docs/adr/0004-text-chunking-strategy.md` for the chunking
    strategy and its constraints.
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
  - `Settings.chunk_size` (default 1000) and `Settings.chunk_overlap`
    (default 200) configure `FixedSizeTextSplitter`, in characters.
  - `Settings.embedding_provider` (default `"fake"`) and
    `Settings.embedding_model_name` (default
    `"intfloat/multilingual-e5-large"`) are provisional placeholders;
    no infrastructure implementation reads them yet. See
    `docs/adr/0005-embedding-strategy.md`.

## Dependency Rule

```
API -> Application -> Domain
Infrastructure -> Domain (implements interfaces)
Core is referenced by all layers
```

Domain must never depend on API, Application, or Infrastructure.

## Status

Minimal FastAPI setup with a health check endpoint, environment-based
settings, standard logging, a PDF loading foundation, a text chunking
foundation, and an embedding abstraction foundation (Issue #6). There
is no upload API, concrete embedding model adapter, VectorDB storage,
retrieval, or generation yet; these land in subsequent issues.

See `docs/adr/0001-project-architecture.md`,
`docs/adr/0002-configuration-and-logging.md`,
`docs/adr/0003-pdf-extraction-library.md`,
`docs/adr/0004-text-chunking-strategy.md`, and
`docs/adr/0005-embedding-strategy.md` for the architecture decision
records.
