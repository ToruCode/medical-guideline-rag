# 0026. Persistent vector store (Qdrant embedded/local mode)

## Status

Accepted

## Context

`docs/adr/0006-vector-store-strategy.md` defined the `VectorStore`
Protocol and `InMemoryVectorStore`, explicitly deferring a concrete,
persistent adapter to a follow-up issue. Since then, `InMemoryVectorStore`
has been the only implementation wired into
`app/api/dependencies.py::get_vector_store()`: every indexed chunk is
lost when the FastAPI process restarts. Issue #17 adds a persistent
option while keeping `InMemoryVectorStore` available (it remains the
default, and is still what most unit/integration tests use).

`CLAUDE.md`'s Technology Stack already names Qdrant (not ChromaDB or
another vector database), and `compose.yaml`, `.gitignore`
(`qdrant_storage/`), and `docs/architecture.md`/`docs/adr/0006`'s own
docstrings already anticipated a `Qdrant`/pgvector adapter - this issue
follows that existing direction rather than introducing a new one.

## Decision

- **Qdrant's embedded/local mode** (`qdrant_client.QdrantClient(path=...)`)
  is the persistent backend, not a Qdrant server. It runs entirely
  in-process against a local on-disk directory - no Docker, no network,
  no server process to run or configure - which satisfies the issue's
  "lightweight embedded option" requirement while staying on the
  project's already-declared vector database (Qdrant). A ChromaDB or
  pgvector/PostgreSQL-server adapter was considered and rejected for
  this issue: ChromaDB is not in `CLAUDE.md`'s stack and would be an
  unexplained new dependency; pgvector requires a running server,
  which conflicts with "embedded" and is better suited to a later,
  production-deployment issue.
- **`app/infrastructure/vector_store/qdrant_vector_store.py`** defines
  `QdrantVectorStore`, a second `VectorStore` implementation alongside
  `InMemoryVectorStore`. Both are selected via
  `Settings.vector_store_provider: Literal["memory", "qdrant"] = "memory"`
  in `app/api/dependencies.py::get_vector_store()`, following the exact
  branching pattern `get_pdf_loader`/`get_llm` already use (`ValueError`
  for an unrecognized value). The default stays `"memory"`, so existing
  behavior and every existing test are unaffected unless
  `MEDICAL_RAG_VECTOR_STORE_PROVIDER=qdrant` is set explicitly.
  `Settings.vector_store_path` (default `./qdrant_storage`, reusing the
  directory name already present in `.gitignore`) and
  `Settings.vector_store_collection_name` (default `guideline_chunks`)
  configure it.
- **Point IDs are derived from `Chunk.chunk_id` via `uuid.uuid5`** with a
  fixed, arbitrary namespace UUID. Qdrant only accepts an unsigned
  integer or a UUID as a point ID, never an arbitrary string, so
  `chunk_id` (`document_id:page_number:chunk_index`) cannot be used
  directly. `uuid.uuid5` is deterministic: the same `chunk_id` always
  maps to the same point, so re-upserting it replaces the existing
  point rather than duplicating it - preserving the upsert-idempotency
  contract from `docs/adr/0006-vector-store-strategy.md`. The namespace
  constant must never change once chosen; changing it would silently
  orphan every previously stored point.
- **Distance metric is `Distance.COSINE`**, matching
  `InMemoryVectorStore`'s cosine similarity computation exactly. Qdrant
  already reports Cosine-configured scores as "higher is more similar",
  so no sign inversion is needed to satisfy `SearchResult.score`'s
  fixed convention (`docs/adr/0006`) - unlike a distance metric such as
  Euclidean, where a future adapter would need to invert the sign.
- **The collection is created lazily on the first `upsert()`**, once the
  embedding dimension is known from the first `EmbeddedChunk.vector`,
  mirroring `InMemoryVectorStore`'s lazy `_dimension` learning exactly.
  If the collection already exists on disk (created by a prior process),
  `__init__` loads its configured dimension immediately and validates
  every subsequent `upsert()`/`search()` call against it, raising the
  existing `VectorDimensionMismatchError` - this is how a dimension
  change across a restart (e.g. switching
  `Settings.embedding_provider`/`embedding_model_name`) is caught,
  exactly as it would be caught by `InMemoryVectorStore` within a single
  process.
- **A locked or corrupted on-disk store raises the new
  `VectorStoreUnavailableError`** (`app/domain/exceptions/vector_store.py`,
  a `VectorStoreError` subtype), wrapping whatever `QdrantClient(path=...)`
  itself raised (e.g. `RuntimeError` when the path is already open by
  another process/instance). Because `app/api/v1/endpoints/documents.py`
  and `questions.py` already catch `VectorStoreError` broadly and return
  500, **no API-layer code changes were needed** to handle this new
  exception.
- **Explicit rebuild is a CLI flag, not a `Settings`/env-var flag.**
  `scripts/index_documents.py --rebuild` calls the new
  `QdrantVectorStore.rebuild()` method before indexing. This was a
  deliberate choice made with the user during planning: every other
  provider/behavior toggle in this project is a `MEDICAL_RAG_*`
  environment variable, but an *operator action* ("wipe and rebuild
  the index now") is qualitatively different from a *configuration
  value* ("which backend to use") - an env var would risk an
  accidental, silent wipe on a routine restart if left set, whereas a
  CLI flag only ever fires when explicitly invoked. `get_vector_store()`
  (the API's own dependency wiring) therefore never rebuilds anything on
  its own.
- **`QdrantVectorStore.rebuild()` deletes and recreates the entire
  `vector_store_path` directory**, not just the one collection, and not
  via the client's own `delete_collection()`. This is a deliberate
  workaround for a verified limitation of `qdrant-client`'s
  embedded/local mode: calling `delete_collection()` followed by
  `create_collection()` with a *different* vector dimension, within the
  same process/session and even across a client close+reopen at the
  same path, still raises an internal `ValueError` ("could not broadcast
  input array...") from stale on-disk collection state -
  `delete_collection()` does not actually remove the collection's
  on-disk storage file in this mode. Recreating the whole client after
  `shutil.rmtree(path)` was confirmed (by direct experimentation while
  implementing this issue) to avoid the bug reliably. Consequence:
  `vector_store_path` must be dedicated to one purpose (one collection)
  - never point two different collections at the same path, since
  `rebuild()` on one would silently destroy the other.
- **`scripts/index_documents.py`** is a new CLI, composing
  `IndexDocumentService` the same way `scripts/retrieval_baseline_core.py`
  already does (building `PdfLoader`/`Embedder`/`TextSplitter`
  implementations directly from `Settings`, not by importing
  `app/api/dependencies.py` - consistent with how existing `scripts/`
  tooling is its own composition root). It indexes every `*.pdf` under
  `--input-dir` (default `data/raw/`, already gitignored) or explicit
  positional file paths, and refuses to run at all when
  `Settings.vector_store_provider != "qdrant"`, since indexing into an
  in-memory store from a one-shot CLI process would be discarded the
  instant the process exits.
- **Payload stores the full `Chunk`** (`document_id`, `source_name`,
  `source_path`, `page_number`, `chunk_index`, `text`, `title`, plus
  `chunk_id` for debugging) so `search()` can fully reconstruct an
  `EmbeddedChunk`/`SearchResult` without any additional lookup - the
  same information `InMemoryVectorStore` already keeps in-process.
  `with_vectors=True` is passed to `query_points()` so
  `EmbeddedChunk.vector` round-trips too, matching the existing
  `InMemoryVectorStore` contract that tests already assert on.
- **Known, accepted behavioral limitation**: Qdrant does not guarantee a
  deterministic tie-break for equal scores natively.
  `QdrantVectorStore.search()` re-sorts whatever Qdrant returns by
  `(-score, chunk_id)` client-side to match `InMemoryVectorStore`'s
  documented tie-break, but if the number of tied entries at the
  `top_k` cutoff boundary exceeds what Qdrant chose to return, the
  exact candidate *set* is not guaranteed to match
  `InMemoryVectorStore`'s. Tests avoid constructing that boundary
  condition rather than asserting a guarantee the implementation cannot
  make.
- **Tests**: `tests/unit/test_qdrant_vector_store.py` mirrors every
  `tests/unit/test_in_memory_vector_store.py` case (contract parity)
  against a `tmp_path`-local embedded instance - no network, no server,
  no committed data - plus `QdrantVectorStore`-specific `rebuild()`
  cases. `tests/integration/test_persistent_vector_store_restart.py`
  simulates a process restart by closing one `QdrantVectorStore`
  instance and opening a second one at the same path, verifying data
  survival, chunk_id-based upsert idempotency across the restart,
  cross-restart dimension-mismatch detection, and `rebuild()` producing
  a clean index on the next open. All of these use hand-built vectors
  (no real embedding model), consistent with `CLAUDE.md`'s rule against
  committing real guideline content.

## Consequences

- Operators who want persistence set `MEDICAL_RAG_VECTOR_STORE_PROVIDER=qdrant`
  and run `uv run python -m scripts.index_documents` (optionally
  `--rebuild`) to populate it; `POST /documents/index` continues to work
  unchanged for incremental, one-file-at-a-time indexing against
  whichever provider is configured.
- `vector_store_path` (default `./qdrant_storage`) must never be shared
  between two different purposes, both because `rebuild()` wipes the
  entire directory and because Qdrant's embedded mode allows only one
  process to hold a given path open at a time (a second `QdrantClient`
  pointed at an already-open path raises `RuntimeError`, mapped here to
  `VectorStoreUnavailableError`). Running the FastAPI app and
  `scripts/index_documents.py` against the same path at the same time
  will fail for this reason; index first, then start (or restart) the
  API server.
- `InMemoryVectorStore` remains the default and the implementation most
  tests exercise; switching the default itself was explicitly out of
  scope for this issue.
- A future cloud-deployment issue that needs concurrent multi-process
  access would need a real Qdrant server (or another server-backed
  store) instead of local mode - this issue only addresses single-process
  local persistence, per the issue's stated scope.
