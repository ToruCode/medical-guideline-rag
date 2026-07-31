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
  - `app/application/services/index_chunks.py` defines
    `IndexChunksService`, which stores `EmbeddedChunk`s via an injected
    `VectorStore`, skipping the call entirely for empty input, and logs
    the stored count (never chunk text or vectors).
  - `app/application/services/search_chunks.py` defines
    `SearchChunksService`, which validates `query_vector`/`top_k` and
    delegates similarity search to an injected `VectorStore`, logging
    `top_k` and the returned count (never chunk text or vectors).
  - `app/application/services/index_document.py` defines
    `IndexDocumentService`, which composes `LoadDocumentService`,
    `ChunkDocumentService`, `EmbedChunksService`, and
    `IndexChunksService` to index one PDF end to end, returning an
    `IndexDocumentResult` (`document_id`, `source_name`, `page_count`,
    `chunk_count`, `indexed_count`). It does not catch exceptions raised
    by any step (fail-fast) and logs only counts and identifiers, never
    chunk text or vectors. See
    `docs/adr/0007-indexing-pipeline.md`.
  - `app/application/services/retrieve_chunks.py` defines
    `RetrieveChunksService`, which composes an injected `Embedder` with
    an injected `SearchChunksService` to turn a natural-language query
    into similar chunks: it embeds the query, validates that exactly
    one vector comes back, and delegates the search itself to
    `SearchChunksService`. It validates `top_k` and rejects an
    empty/whitespace-only query before calling the `Embedder`, and does
    not catch any exception raised by the `Embedder` or
    `SearchChunksService` (fail-fast). It logs only `top_k` and the
    returned result count, never query text or vectors. Returns
    `list[SearchResult]` directly; no separate retrieval result model
    is introduced. See `docs/adr/0008-retrieval-strategy.md`.
  - `app/application/services/generate_answer.py` defines
    `GenerateAnswerService`, which generates a citation-grounded answer
    from a `question: str` and an already-retrieved
    `list[SearchResult]` by delegating to an injected `Llm`. It rejects
    an empty/whitespace-only `question` (`EmptyQueryError`) before
    calling the `Llm`, and never calls the `Llm` at all when
    `search_results` is empty, returning a fixed
    `INSUFFICIENT_EVIDENCE_ANSWER` with `is_insufficient_evidence=True`
    instead. The module-level `build_context` function formats
    `SearchResult`s into a numbered, citable context block. Returns
    `GenerationResult` (`answer`, `citations: list[SearchResult]`,
    `is_insufficient_evidence`), kept in the Application layer since no
    Domain Protocol returns it. Does not catch any exception raised by
    the `Llm` (fail-fast), and logs only `citation_count` and the
    insufficient-evidence outcome, never question/context/answer text.
    No use case yet composes this with `RetrieveChunksService`; see
    `docs/adr/0009-generation-strategy.md`.
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
  - `app/domain/models/chunk.py`'s `Chunk` also exposes a computed
    `chunk_id` property (`document_id:page_number:chunk_index`), used
    as a stable identifier for `VectorStore` upsert/deduplication.
  - `app/domain/models/search_result.py` defines `SearchResult`, an
    immutable (`frozen=True, slots=True`) dataclass composing an
    `EmbeddedChunk` with a `score: float`, where a higher score always
    means a closer match regardless of which `VectorStore`
    implementation produced it.
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
  - `app/domain/ports/vector_store.py` defines the `VectorStore`
    Protocol (`upsert(chunks: list[EmbeddedChunk]) -> None`,
    `search(query_vector: list[float], top_k: int) -> list[SearchResult]`)
    that infrastructure vector database adapters implement. The domain
    layer never imports Qdrant, Chroma, FAISS, PostgreSQL, or pgvector.
  - `app/domain/ports/llm.py` defines the `Llm` Protocol
    (`generate(prompt: str) -> str`) that infrastructure LLM adapters
    implement. The domain layer never imports an LLM SDK; the caller
    is responsible for composing the full prompt string.
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
  - `app/domain/exceptions/vector_store.py` defines `VectorStoreError`
    and its subtypes (`InvalidSearchQueryError`, `InvalidTopKError`,
    `VectorDimensionMismatchError`) used to translate invalid
    storage/search input into domain-level errors.
  - `app/domain/exceptions/retrieval.py` defines `RetrievalError` and
    its subtype `EmptyQueryError`, raised when a natural-language query
    is empty or whitespace-only, distinct from
    `InvalidSearchQueryError` (which concerns an already-embedded
    `query_vector`, not the raw query string).
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
foundation, an embedding abstraction foundation (Issue #6), a vector
store abstraction foundation (Issue #7), an indexing pipeline
(Issue #8) that composes all of the above end to end, a retrieval use
case (Issue #9) that embeds a natural-language query and returns
similar chunks, and a generation use case (Issue #10) that turns
retrieved chunks into a citation-grounded answer, or an explicit
insufficient-evidence result when nothing was retrieved. There is no
upload API, search API, concrete embedding model adapter, concrete
vector database adapter, concrete LLM adapter, or a use case composing
retrieval and generation together yet; these land in subsequent
issues.

See `docs/adr/0001-project-architecture.md`,
`docs/adr/0002-configuration-and-logging.md`,
`docs/adr/0003-pdf-extraction-library.md`,
`docs/adr/0004-text-chunking-strategy.md`,
`docs/adr/0005-embedding-strategy.md`,
`docs/adr/0006-vector-store-strategy.md`,
`docs/adr/0007-indexing-pipeline.md`,
`docs/adr/0008-retrieval-strategy.md`, and
`docs/adr/0009-generation-strategy.md` for the architecture decision
records.
