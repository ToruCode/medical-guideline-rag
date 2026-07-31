# Architecture

This project follows a layered architecture with dependencies pointing
inward.

## Layers

- **API** (`app/api`, `app/schemas`): receives HTTP requests, validates
  input, calls application use cases, converts results into HTTP
  responses. Must not call Qdrant, PostgreSQL, LangChain, or an LLM
  directly.
  - `app/api/dependencies.py` is the only module the API layer uses to
    construct Infrastructure implementations and compose them into
    Application services; endpoint modules depend only on its provider
    functions via `Depends(...)`. `get_passage_embedder`/
    `get_query_embedder`/`get_vector_store`/`get_llm` are `@lru_cache`
    singletons shared across all requests for the life of the process
    (the same pattern `app/core/config.py::get_settings` already uses),
    so a document indexed via `POST /documents/index` is searchable via
    `POST /questions/ask`. Each reads `Settings.embedding_provider`/
    `llm_provider` to choose between the Fake implementations and the
    real `sentence-transformers`/OpenAI adapters (see Infrastructure,
    below); `get_passage_embedder`/`get_query_embedder` share one
    loaded `SentenceTransformer` model, applying a `"passage: "`/
    `"query: "` prefix respectively. See
    `docs/adr/0010-fastapi-rag-api.md` and
    `docs/adr/0011-real-embedding-and-llm-adapters.md`.
  - `app/api/v1/endpoints/documents.py` defines
    `POST /documents/index`: validates the uploaded file's name ends in
    `.pdf`, rejects an empty file, sanitizes the file name
    (`_sanitize_filename`, strips path separators/unsafe characters),
    writes it under that sanitized name inside a freshly-created,
    randomly-named temp directory (`_save_temp_pdf`), and always
    removes that directory afterward (`shutil.rmtree` in a `finally`
    block), whether `IndexDocumentService.execute` succeeds or fails.
    `IndexDocumentService.execute` runs via
    `starlette.concurrency.run_in_threadpool`, so a real (potentially
    slow) embedding model call never blocks the event loop. Domain
    exceptions are mapped to HTTP status codes in the endpoint itself
    (415/400/422/500 as appropriate). Logs only the sanitized file name
    and result counts, never document text.
  - `app/api/v1/endpoints/questions.py` defines `POST /questions/ask`:
    delegates to `AskQuestionService`, maps `EmptyQueryError`/
    `InvalidTopKError` to 400 and `EmbeddingError`/`VectorStoreError` to
    500, and converts each `SearchResult` citation into a
    `CitationSchema` that excludes the embedding vector. Logs only
    `top_k`, `citation_count`, and `is_insufficient_evidence`, never
    question/passage/answer text.
  - `app/schemas/document.py` defines `IndexDocumentResponse`;
    `app/schemas/question.py` defines `AskQuestionRequest`,
    `CitationSchema`, and `AskQuestionResponse`.
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
    See `docs/adr/0009-generation-strategy.md`.
  - `app/application/services/ask_question.py` defines
    `AskQuestionService`, which composes `RetrieveChunksService` and
    `GenerateAnswerService` to answer a question end to end, returning
    `GenerationResult` unchanged. Does not catch any exception raised
    by either step (fail-fast). See
    `docs/adr/0010-fastapi-rag-api.md`.
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
  - `app/infrastructure/embedding/fake_embedder.py` (`FakeEmbedder`,
    `FixedVectorsEmbedder`), `app/infrastructure/llm/fake_llm.py`
    (`FakeLlm`, `RaisingLlm`), and
    `app/infrastructure/vector_store/in_memory_vector_store.py`
    (`InMemoryVectorStore`) are deterministic, dependency-free
    implementations of `Embedder`/`Llm`/`VectorStore` with no network
    access. They are used both by tests and as the FastAPI app's
    default (`"fake"` provider) dependency wiring
    (`app/api/dependencies.py`). Moved here from `tests/support/` in
    Issue #11; see `docs/adr/0010-fastapi-rag-api.md` for why (they
    must be importable from application code shipped in the built
    package, which `tests/support/` is not).
  - `app/infrastructure/embedding/sentence_transformer_embedder.py`
    defines `SentenceTransformerEmbedder`, implementing `Embedder` over
    a local `sentence-transformers` model, prepending a configurable
    `prefix` (`"query: "`/`"passage: "`) to every input text for
    asymmetric retrieval models. `load_sentence_transformer_model`
    imports `sentence_transformers` lazily (not at module import time),
    so merely referencing this module never triggers the real,
    expensive import unless a model is actually loaded. Selected via
    `Settings.embedding_provider = "sentence_transformers"`. See
    `docs/adr/0011-real-embedding-and-llm-adapters.md`.
  - `app/infrastructure/llm/openai_llm.py` defines `OpenAiLlm`,
    implementing `Llm` over OpenAI's Chat Completions API
    (`openai.OpenAI(...).chat.completions.create(...)`), sending the
    single prompt string as one user message. Imports `openai` lazily
    inside `__init__` for the same reason as
    `SentenceTransformerEmbedder`. Selected via
    `Settings.llm_provider = "openai"`. See
    `docs/adr/0011-real-embedding-and-llm-adapters.md`.
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
  - `Settings.embedding_provider` (`Literal["fake", "sentence_transformers"]`,
    default `"fake"`) and `Settings.embedding_model_name` (default
    `"intfloat/multilingual-e5-base"`) select and configure the
    `Embedder` implementation `app/api/dependencies.py` constructs. See
    `docs/adr/0005-embedding-strategy.md` and
    `docs/adr/0011-real-embedding-and-llm-adapters.md`.
  - `Settings.llm_provider` (`Literal["fake", "openai"]`, default
    `"fake"`), `Settings.llm_model_name` (default `"gpt-4o-mini"`),
    `Settings.llm_api_key` (`SecretStr | None`, never logged even if the
    whole `Settings` object is), and `Settings.llm_timeout_seconds`
    (default `30.0`) select and configure the `Llm` implementation. See
    `docs/adr/0011-real-embedding-and-llm-adapters.md`.

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
similar chunks, a generation use case (Issue #10) that turns retrieved
chunks into a citation-grounded answer, a FastAPI RAG API (Issue #11)
exposing document indexing and question answering end to end, real
`Embedder`/`Llm` adapters (Issue #12: a local `sentence-transformers`
model and OpenAI's Chat Completions API), selectable via
`Settings.embedding_provider`/`llm_provider` alongside the still-default
Fake implementations, and an opt-in live end-to-end test
(Issue #13, `tests/integration/test_live_rag_e2e.py`) verifying the
full index-then-ask flow with both real adapters together. There is no
concrete vector database adapter (Qdrant/pgvector) yet; that lands in a
subsequent issue.

See `docs/adr/0001-project-architecture.md`,
`docs/adr/0002-configuration-and-logging.md`,
`docs/adr/0003-pdf-extraction-library.md`,
`docs/adr/0004-text-chunking-strategy.md`,
`docs/adr/0005-embedding-strategy.md`,
`docs/adr/0006-vector-store-strategy.md`,
`docs/adr/0007-indexing-pipeline.md`,
`docs/adr/0008-retrieval-strategy.md`,
`docs/adr/0009-generation-strategy.md`,
`docs/adr/0010-fastapi-rag-api.md`,
`docs/adr/0011-real-embedding-and-llm-adapters.md`, and
`docs/adr/0012-live-e2e-verification.md` for the architecture decision
records.
