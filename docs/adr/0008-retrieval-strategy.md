# 0008. Compose Embedder and SearchChunksService into a query-facing retrieval use case

## Status

Accepted

## Context

Issues #6-#8 produced `Embedder` (embeds text into vectors),
`VectorStore` (stores/searches `EmbeddedChunk`s), and
`SearchChunksService` (validates `query_vector`/`top_k` and delegates
to an injected `VectorStore`). None of these accept a natural-language
query string directly: `SearchChunksService.execute` requires an
already-embedded `query_vector`. Issue #9 needs a use case that takes a
user's search query and returns the most similar chunks, without any
caller having to embed the query itself.

## Decision

- `RetrieveChunksService`
  (`app/application/services/retrieve_chunks.py`) composes an injected
  `Embedder` with an injected `SearchChunksService` — not a
  `VectorStore` directly. This mirrors `IndexDocumentService`
  (`docs/adr/0007-indexing-pipeline.md`) composing existing
  Application services rather than re-implementing what they already
  validate and log. `query_vector`/`top_k` validation against the
  store and the search step's own logging stay defined in exactly one
  place: `SearchChunksService`.
- `EmbedChunksService` is not reused here, because its contract is
  `list[Chunk] -> list[EmbeddedChunk]`: it is shaped around indexing
  document chunks, not around embedding a single query string.
  `RetrieveChunksService` calls `Embedder.embed([query])` directly and
  validates that exactly one vector comes back, raising the existing
  `EmbeddingCountMismatchError` (`app/domain/exceptions/embedding.py`)
  on mismatch rather than introducing a new exception type for what is
  the same underlying failure mode `EmbedChunksService` already names.
- `top_k` is validated in `RetrieveChunksService` itself, raising the
  existing `InvalidTopKError`
  (`app/domain/exceptions/vector_store.py`), **before** the `Embedder`
  is called, even though `SearchChunksService` validates it again
  further down the call chain. This is a deliberate, precedented
  duplication (`SearchChunksService` itself re-validates what its
  injected `VectorStore` is also required to validate) that avoids
  spending an `Embedder` call — a real model or API request in
  production — on a `top_k` that is already known to be invalid.
- A new `app/domain/exceptions/retrieval.py` defines `RetrievalError`
  and `EmptyQueryError`, raised when a query is empty or
  whitespace-only. This is not the same failure as
  `InvalidSearchQueryError` (`app/domain/exceptions/vector_store.py`),
  which is about a structurally invalid `query_vector`, not the
  natural-language query string a caller supplies before embedding;
  reusing it would conflate two different validation concerns under
  one exception type.
- No new result model is introduced. `RetrieveChunksService.execute`
  returns `list[SearchResult]` — the same type `SearchResult`
  (`app/domain/models/search_result.py`) already returned by
  `VectorStore.search`/`SearchChunksService.execute` — since nothing
  about retrieval-by-text changes what a result contains.
- No exceptions are caught. `EmptyQueryError`, `InvalidTopKError`,
  `EmbeddingCountMismatchError`, any other `EmbeddingError` raised by
  the `Embedder`, and any `VectorStoreError` raised through
  `SearchChunksService` all propagate to the caller unchanged, matching
  the fail-fast approach already established for
  `IndexDocumentService`.
- Logging includes only `top_k` and the returned result count, at INFO
  level, matching `SearchChunksService`'s own logging. Query text and
  vector values are never logged, per the project's medical-safety
  logging rules.
- **No API endpoint is added in this issue.** `RetrieveChunksService`
  is Application-layer only, wired directly in tests; a future issue
  adds the API layer that calls it.

## Consequences

- A future change to `SearchChunksService`'s validation, logging, or
  delegation behavior automatically applies to `RetrieveChunksService`
  without touching it.
- Because `top_k` is validated in two places by design, a future change
  to the validation rule (e.g. an upper bound on `top_k`) must be
  applied in both `RetrieveChunksService` and `SearchChunksService` to
  stay consistent.
- A future API layer must still translate `EmptyQueryError`,
  `InvalidTopKError`, `EmbeddingError`, and `VectorStoreError` into HTTP
  responses itself; this issue does not provide a unified error type
  for that translation.
- `RetrieveChunksService`'s constructor takes an `Embedder` Protocol and
  a concrete `SearchChunksService`, not two Protocols. This deliberately
  mixes a Protocol dependency (an infrastructure boundary) with a
  concrete Application service dependency (an already-tested use case
  being composed), the same mixed pattern `IndexDocumentService`
  established for composing existing services.
