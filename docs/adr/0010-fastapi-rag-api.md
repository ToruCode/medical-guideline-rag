# 0010. Expose indexing, retrieval, and generation over a FastAPI RAG API

## Status

Accepted

## Context

Issues #8-#10 produced three Application services (`IndexDocumentService`,
`RetrieveChunksService`, `GenerateAnswerService`) with no HTTP access.
Issue #11 exposes them as two endpoints, `POST /api/v1/documents/index`
and `POST /api/v1/questions/ask`, using only the existing Fake/in-memory
implementations (no real embedding model, LLM, or Qdrant).

## Decision

- **`AskQuestionService`** (`app/application/services/ask_question.py`)
  composes `RetrieveChunksService` and `GenerateAnswerService`,
  returning `GenerationResult` unchanged. This is added in this issue
  (not deferred further, as `docs/adr/0009-generation-strategy.md`
  originally anticipated), because CLAUDE.md explicitly forbids
  business logic in FastAPI endpoint functions, and the retrieve-then-
  generate sequencing is exactly that: it belongs in the Application
  layer, so `POST /questions/ask` can depend on a single service.
- **Fake implementations moved from `tests/support/` to
  `app/infrastructure/`** (`app/infrastructure/embedding/fake_embedder.py`,
  `app/infrastructure/llm/fake_llm.py`,
  `app/infrastructure/vector_store/in_memory_vector_store.py`), with no
  behavior change. `tests/support/in_memory_vector_store.py` was
  previously documented as "not production code... so no application
  code can accidentally depend on it", and `pyproject.toml`'s
  `[tool.hatch.build.targets.wheel] packages = ["app"]` excludes
  `tests/` from any built wheel/Docker image. Importing from
  `tests.support` inside `app/api/dependencies.py` would both violate
  that documented intent and silently work in local dev while breaking
  with an `ImportError` the moment the app is packaged/deployed. Since
  these Fakes are legitimate (if simplistic) `Embedder`/`Llm`/
  `VectorStore` Protocol implementations - deterministic, no network
  access - `app/infrastructure` is their correct home regardless of
  whether they're "real" backends. All existing tests that imported
  them from `tests.support` were updated to import from
  `app.infrastructure...` instead; no test behavior changed.
- **Dependency injection lives in `app/api/dependencies.py`**, using
  FastAPI's own `Depends` chaining rather than a separate `container.py`
  object graph. `get_embedder`, `get_vector_store`, and `get_llm` are
  `@lru_cache` singletons - the same pattern `app/core/config.py`
  already established for `get_settings()` - so a document indexed via
  `POST /documents/index` is searchable via `POST /questions/ask`
  within the same process. Endpoint modules only ever import from
  `app.api.dependencies`, never from `app.infrastructure` directly.
  Tests clear these caches between test functions (`tests/conftest.py`)
  to avoid state leaking across tests, mirroring the existing
  `_clear_settings_cache` fixture.
- **Temp file handling for `POST /documents/index`**: the uploaded
  file's name is validated (must end in `.pdf`, case-insensitive) and
  then sanitized (`_sanitize_filename` in
  `app/api/v1/endpoints/documents.py`: strips path separators and other
  unsafe characters via `Path(...).name` plus a character allowlist).
  The sanitized name is used only as the file's basename, written
  inside a freshly-created, randomly-named directory
  (`tempfile.mkdtemp()`); the random directory - not the filename - is
  what makes the path unpredictable and collision-free, so the
  uploaded name never determines *where* the file is written. Using the
  sanitized name as the basename (rather than a fully random name) is
  deliberate: `PypdfLoader` sets `DocumentPage.source_name` (and
  therefore every `Chunk.source_name`, permanently stored in the
  `VectorStore`) from the file's actual name, so a fully-random temp
  name would have leaked into every future citation returned by
  `POST /questions/ask`. `IndexDocumentResponse.source_name` is
  `IndexDocumentResult.source_name` unchanged - no separate
  substitution is needed, since it already equals the sanitized name.
  The whole temp directory is removed in a `finally` block
  (`shutil.rmtree(..., ignore_errors=True)`), on both success and
  failure.
- **Exception-to-HTTP mapping happens per-endpoint via `try`/`except`**,
  not a global `@app.exception_handler`. `POST /documents/index` needs
  the sanitized original filename in its error messages instead of the
  domain exception's own message (which embeds the temp file's name,
  e.g. `EncryptedDocumentError`'s `f"PDF is encrypted: {file_path.name}"`)
  - context a global handler cannot see. `CLAUDE.md` assigns "select
  appropriate HTTP status codes" to the API layer, so this mapping is
  within the endpoint's stated responsibility, not business logic.
  Uncaught/unexpected exception types are not special-cased; FastAPI's
  default handling returns a generic 500.
- **Response schemas project Domain objects rather than serializing them
  directly**: `CitationSchema` (`app/schemas/question.py`) excludes the
  embedding vector carried by `SearchResult`/`EmbeddedChunk`, exposing
  only `document_id`, `source_name`, `title`, `page_number`,
  `chunk_index`, `score`. This was already anticipated in
  `docs/adr/0009-generation-strategy.md` ("a future API layer can
  project just the citation-relevant fields").
- **`top_k` validation is layered**: `AskQuestionRequest.top_k` uses
  Pydantic's `Field(gt=0)`, so a non-positive `top_k` is rejected by
  FastAPI's request validation (`422`) before the endpoint function
  even runs. `InvalidTopKError` is still caught and mapped to `400` as
  a backstop, matching the project's existing double-validation pattern
  (`SearchChunksService` re-validates what its injected `VectorStore` is
  also required to validate).
- **New dependency: `python-multipart`**, required by Starlette/FastAPI
  to parse `multipart/form-data` file uploads (`UploadFile`/`File(...)`).
  There is no way to accept a file upload without it.
- Logging stays minimal: `POST /documents/index` logs the sanitized
  file name (not guideline text) and result counts;
  `POST /questions/ask` logs `top_k`, `citation_count`, and
  `is_insufficient_evidence` only. Neither endpoint logs question text,
  guideline passage text, or generated answer text, per the project's
  medical-safety and logging rules.

## Consequences

- A future real `Embedder`/`Llm`/`VectorStore` adapter replaces the
  `@lru_cache` singleton functions in `app/api/dependencies.py` (e.g.
  reading `Settings.embedding_provider` to select an implementation);
  endpoint code does not need to change, since it only depends on the
  provider functions and the Application services they build.
- `InMemoryVectorStore` remains process-local and non-persistent:
  restarting the server loses all indexed data. This is expected and
  unchanged from `docs/adr/0006-vector-store-strategy.md`.
- Neither endpoint enforces an upload size limit, rate limit, or
  authentication/authorization. Acceptable for this technical
  demonstration; must be addressed before any real deployment.
- `index_document` is `async def` (needed for `await file.read()`) but
  calls the synchronous `IndexDocumentService` inline, blocking the
  event loop for the duration of indexing. Negligible with the current
  Fake implementations; a future real backend may need
  `starlette.concurrency.run_in_threadpool` or an async service
  interface.
- `CitationSchema` only carries `title`/`source_name`/`page_number`
  (not `edition`/`chapter`/`section`), the same limitation already
  noted in `docs/adr/0009-generation-strategy.md`, since `Chunk` does
  not yet model those fields.
