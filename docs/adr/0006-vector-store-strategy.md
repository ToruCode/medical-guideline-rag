# 0006. VectorStore abstraction, deferred to a concrete adapter

## Status

Accepted

## Context

`EmbeddedChunk`s produced by the embedding foundation (Issue #6,
`docs/adr/0005-embedding-strategy.md`) need to be stored and searched
by similarity before they can support retrieval. No concrete vector
database has been evaluated yet, and this project's architecture rules
(`CLAUDE.md`) require that the Domain and Application layers not import
a specific vector database library (Qdrant, Chroma, FAISS, PostgreSQL,
pgvector).

## Decision

- Define a `VectorStore` Protocol (`app/domain/ports/vector_store.py`)
  with two methods: `upsert(chunks: list[EmbeddedChunk]) -> None` and
  `search(query_vector: list[float], top_k: int) -> list[SearchResult]`.
  Domain and Application code depend only on this Protocol, never on a
  concrete vector database library.
- **`upsert`, not `add`.** Re-storing an `EmbeddedChunk` with a
  `chunk_id` that already exists must replace the existing entry, not
  duplicate it (e.g. re-indexing a document after a chunking change).
  `add` would leave that behavior ambiguous. Qdrant and pgvector
  (`ON CONFLICT`) both expose idempotent upsert-shaped writes as their
  primary write API, so this also matches how a future concrete adapter
  will naturally be implemented.
- **`Chunk` gains a computed `chunk_id` property**
  (`app/domain/models/chunk.py`), returning
  `f"{document_id}:{page_number}:{chunk_index}"`. It is a property, not
  a stored field, so the existing `frozen=True, slots=True` dataclass
  shape and all existing call sites are unaffected. `EmbeddedChunk`
  reuses `chunk.chunk_id` rather than duplicating identity logic.
  Defining this once in the Domain layer prevents every future
  `VectorStore` implementation (Qdrant point ID, pgvector primary key,
  Chroma document ID) from reinventing — and potentially
  inconsistently reinventing — the same composition.
- Represent a match as `SearchResult` (`app/domain/models/search_result.py`),
  composing an `EmbeddedChunk` with a `score: float`.
  **`score` follows one fixed convention: a higher value always means a
  closer match.** It is intentionally not clamped to `[0.0, 1.0]`,
  since cosine similarity ranges over `[-1.0, 1.0]` and other metrics
  (e.g. dot product) are unbounded; forcing a `[0.0, 1.0]` range would
  either misrepresent the underlying metric or require an arbitrary
  rescaling. **Translating a concrete backend's native distance metric
  into this "higher is more similar" convention (including sign
  inversion, if the backend reports a raw distance where lower is
  closer) is the responsibility of the Infrastructure-layer
  `VectorStore` implementation**, so Application code and any future
  ranking/reranking logic never needs to know which backend produced a
  score.
- `VectorStore` implementations (both `InMemoryVectorStore` here and
  future concrete adapters) must raise
  `InvalidTopKError` for `top_k <= 0`, `InvalidSearchQueryError` for an
  empty `query_vector`, and `VectorDimensionMismatchError` for a vector
  whose dimension does not match the store's established dimension
  (`app/domain/exceptions/vector_store.py`, all subtypes of
  `VectorStoreError`). These are distinct from
  `app/domain/exceptions/embedding.py`'s `EmbeddingError` subtypes,
  which validate an `Embedder`'s output at embedding time; the new
  exceptions validate storage/search-time input and are raised by
  `VectorStore` implementations, not `Embedder` implementations.
- `SearchChunksService` (`app/application/services/search_chunks.py`)
  performs the same `query_vector`/`top_k` validation itself, before
  delegating to the injected `VectorStore`. This is intentional
  duplication: it keeps validation behavior consistent for callers
  regardless of which `VectorStore` implementation is wired in, while
  each `VectorStore` implementation still independently protects itself
  as a Protocol that may be called directly.
- `IndexChunksService.execute([])` returns without calling the
  injected `VectorStore`, mirroring `EmbedChunksService`'s empty-batch
  short circuit (`docs/adr/0005-embedding-strategy.md`).
- **`InMemoryVectorStore` lives only under `tests/support/`, not
  `app/infrastructure/`.** It is a deterministic test double (cosine
  similarity computed with the standard library `math` module only, no
  NumPy or a real vector database), not a production storage backend,
  so it must not be reachable from application code.
  `InMemoryVectorStore` computes cosine similarity for a zero vector
  (query or stored) as `score = 0.0` instead of raising, since a
  degenerate embedding is not a structurally invalid input (unlike a
  dimension mismatch) and cosine similarity is mathematically undefined
  at zero rather than incorrect. Ties are broken deterministically by
  `chunk_id` so search results never depend on dict iteration order.
  `InMemoryVectorStore` defensively copies each vector on `upsert` so
  neither caller mutation of the original list nor a later mutation of
  a returned `SearchResult`'s vector can corrupt its internal state.
- **This issue implements only the abstraction.** No concrete vector
  database adapter (Qdrant, pgvector, Chroma) is implemented, and no
  new dependency is added. No `Settings` fields
  (`vector_store_provider`, `vector_store_collection_name`,
  `vector_dimension`) are added either, unlike the provisional
  placeholders added in `docs/adr/0005-embedding-strategy.md`: none of
  them would be read by any code in this issue, and `CLAUDE.md`
  explicitly discourages adding settings that stay unused. They are
  deferred to the issue that adds a concrete adapter and a
  provider-selection factory, at which point their names and defaults
  can be chosen against an actual implementation instead of guessed in
  advance.
- No logging existed in `LoadDocumentService`, `ChunkDocumentService`,
  or `EmbedChunksService` before this issue. `IndexChunksService` and
  `SearchChunksService` introduce structured logging
  (`logging.getLogger(__name__)`) as a new but deliberate pattern:
  stored/returned counts, `top_k`, and success are logged, but chunk
  text and vector values are never logged, consistent with
  `CLAUDE.md`'s logging rules. This pattern is not retrofitted onto the
  three earlier services in this issue.

## Consequences

- Downstream retrieval code can be designed against `VectorStore` and
  `SearchResult` without waiting for a vector database decision.
- A follow-up issue must add a concrete adapter (e.g.
  `app/infrastructure/vector_store/qdrant_vector_store.py`), the
  corresponding dependency, the `Settings` fields listed above, and a
  provider-selection factory.
- `SearchResult.score`'s meaning depends entirely on documentation
  discipline (the Protocol docstring and this ADR) rather than a type-
  level guarantee, since `float` cannot itself encode "higher is
  better." A future adapter that gets the sign wrong would not be
  caught by the type checker; adapter-level tests must verify the
  convention explicitly.
- `Chunk.chunk_id`'s format (`document_id:page_number:chunk_index`) is
  now a de facto contract for any `VectorStore` implementation that
  uses it as a storage key. If a future adapter needs a different key
  shape (e.g. a UUID), `chunk_id` can still be reused as an input to
  derive one, without changing `Chunk`'s stored fields.
- `SearchChunksService`'s input validation is duplicated in every
  `VectorStore` implementation. This is accepted as boundary defense
  rather than treated as duplication to eliminate; if it becomes
  burdensome, validation could move to a shared base implementation in
  a future revision.
