# 0007. Compose existing services into a fail-fast indexing pipeline

## Status

Accepted

## Context

Issues #4-#7 produced four independent, already-tested Application
services — `LoadDocumentService`, `ChunkDocumentService`,
`EmbedChunksService`, `IndexChunksService` — each wrapping exactly one
Domain Protocol (`PdfLoader`, `TextSplitter`, `Embedder`, `VectorStore`).
Indexing one PDF end to end requires running all four in sequence, but
no orchestrating use case existed yet.

## Decision

- `IndexDocumentService` (`app/application/services/index_document.py`)
  composes the four **existing Application services**, injected via its
  constructor, rather than depending on the four Domain Protocols
  directly. Each of the existing services already owns its own
  empty-input short circuit, output validation
  (`EmbedChunksService`), and logging (`IndexChunksService`); depending
  on the Protocols directly would require re-implementing that
  behavior inside the pipeline and risking drift between the two call
  paths (using a service directly vs. through the pipeline).
- The pipeline is a straight-line, four-step call chain with **no
  extra branching for empty input**. An empty `pages` list already
  produces an empty `chunks` list through `ChunkDocumentService`, and
  `EmbedChunksService`/`IndexChunksService` already short-circuit on
  empty input, so a zero-page or all-blank-page PDF flows through
  unchanged.
- **No new dedup logic.** Re-running the pipeline on the same PDF
  produces the same `chunk_id`s (`document_id:page_number:chunk_index`,
  `docs/adr/0006-vector-store-strategy.md`), so `VectorStore.upsert`'s
  existing idempotency guarantee prevents duplicate entries without
  the pipeline needing to know about identity at all.
- **No exceptions are caught.** `DocumentLoadError` (from
  `LoadDocumentService`), `InvalidChunkConfigError` (from
  `ChunkDocumentService`), `EmbeddingError` (from `EmbedChunksService`),
  and `VectorStoreError` (from `IndexChunksService`) all propagate to
  the caller unchanged. No new pipeline-level exception type wraps
  them. This is fail-fast by design: it matches how each underlying
  service already behaves, keeps the original error's type and message
  intact for the caller, and avoids introducing an error-translation
  layer before any caller (e.g. a future API endpoint) exists to define
  what it would need. Progress up to the point of failure is visible
  through the pipeline's own step-level INFO logs
  (`page_count`/`chunk_count`/`embedded_count`), not through catching
  and re-logging the exception.
- `IndexDocumentResult` (same module as `IndexDocumentService`, not a
  Domain model) summarizes one run: `document_id: str | None`,
  `source_name`, `page_count`, `chunk_count`, `indexed_count`. It is
  kept in the Application layer, not `app/domain/models`, because
  nothing in the Domain layer returns it — unlike `SearchResult`, which
  is returned by the `VectorStore` Protocol itself, `IndexDocumentResult`
  only exists to summarize this one orchestrating use case's outcome.
- `document_id` is `str | None`, `None` only when the PDF has zero
  pages. `PdfLoader` already treats a zero-page PDF as valid (not an
  error), and `document_id` is only ever known via a produced
  `DocumentPage`, so a page-less PDF has no `document_id` to report.
  Introducing a new exception for this case would make the pipeline
  reject input that `PdfLoader` itself accepts.
- **No API endpoint is added in this issue.** `IndexDocumentService`
  is Application-layer only, wired directly in tests; a future issue
  adds the API layer (`app/api`, `app/schemas`) that calls it, along
  with a decision on file upload handling and storage location for
  uploaded PDFs.

## Consequences

- A future change to any of the four underlying services'
  short-circuit or validation behavior automatically applies to the
  pipeline without touching `IndexDocumentService`.
- Because errors are not wrapped, a future API layer must translate
  four different domain exception hierarchies
  (`DocumentLoadError`/`InvalidChunkConfigError`/`EmbeddingError`/
  `VectorStoreError`) into HTTP responses itself; this issue does not
  provide a unified error type for that translation.
- Changing chunking configuration (`chunk_size`/`chunk_overlap`)
  between indexing runs of the same PDF can leave stale chunks in the
  `VectorStore` under `chunk_id`s the new run no longer produces, since
  neither `upsert` nor this pipeline deletes anything. Cleanup (e.g. a
  delete-by-`document_id` operation) is out of scope here and deferred
  to a future issue.
- `IndexDocumentService`'s constructor takes four concrete service
  classes rather than four Protocols. This is a deliberate deviation
  from the Protocol-per-dependency pattern used inside each of those
  four services, justified because they are already-tested
  Application-layer use cases being composed, not infrastructure
  boundaries being abstracted.
