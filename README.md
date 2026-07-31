# Medical Guideline RAG

A citation-grounded RAG system for searching medical guidelines. It
helps healthcare professionals locate relevant passages from medical
guideline documents.

This system is a technical demonstration. It must not be used to
provide medical diagnoses, individual treatment decisions, or
patient-specific recommendations. Always confirm the original guideline
and current clinical information.

## Status

Minimal FastAPI setup with a health check endpoint, environment-based
settings, standard logging, a PDF loading foundation, a text chunking
foundation, an embedding foundation (Issue #6), a vector store
abstraction foundation (Issue #7), an indexing pipeline that composes
all of the above end to end (Issue #8), and a retrieval use case
(Issue #9) that embeds a natural-language query and returns similar
chunks. No concrete embedding model, concrete vector database, upload
API, search API, or generation logic is implemented yet.

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync
cp .env.example .env
```

Edit `.env` as needed. All settings are environment variables prefixed
with `MEDICAL_RAG_` (see `app/core/config.py` and `.env.example` for
the full list). `.env` is not committed to Git.

Future plan: settings will be split per environment (Local / Test /
Staging / Production) instead of a single `Settings` class.

## Development commands

```bash
make dev        # run the API with uvicorn --reload
make lint       # ruff check
make format     # ruff format --check
make typecheck  # mypy
make test       # pytest
make check      # lint + typecheck + test
```

Once the server is running, open `http://127.0.0.1:8000/docs` for the
Swagger UI, or call `GET /api/v1/health` directly.

## PDF loading

Text-layer PDFs can be loaded page by page via
`app.infrastructure.pdf.pypdf_loader.PypdfLoader`, which implements
the `app.domain.ports.pdf_loader.PdfLoader` interface and returns
immutable `DocumentPage` values. Scanned PDFs (image-only, no text
layer) and encrypted PDFs are not supported; see
`docs/adr/0003-pdf-extraction-library.md` for the reasoning and
constraints.

There is no upload API yet. Only self-authored, license-free sample
PDFs belong under `data/sample/` (see `data/sample/README.md`); real
or copyrighted guideline PDFs must never be committed, and
`data/raw/` is git-ignored for that reason.

## Text chunking

`DocumentPage`s produced by the PDF loader can be split into
`Chunk`s for downstream embedding and search via
`app.infrastructure.chunking.fixed_size_text_splitter.FixedSizeTextSplitter`
(implements `app.domain.ports.text_splitter.TextSplitter`) together
with `app.application.services.chunk_document.ChunkDocumentService`,
which carries `document_id`, `page_number`, `source_name`,
`source_path`, and `title` from each page into its chunks.

Chunking is character-count based (not LangChain, not token-based) and
configured via `MEDICAL_RAG_CHUNK_SIZE` (default 1000) and
`MEDICAL_RAG_CHUNK_OVERLAP` (default 200). See
`docs/adr/0004-text-chunking-strategy.md` for the reasoning and
constraints.

## Embedding

`Chunk`s can be converted into `EmbeddedChunk`s (a `Chunk` plus a
`vector: list[float]`) via
`app.application.services.embed_chunks.EmbedChunksService`, which
delegates the actual vectorization to an injected
`app.domain.ports.embedder.Embedder` implementation
(`embed(texts: list[str]) -> list[list[float]]`). Empty input returns
an empty list without calling the embedder, and mismatched vector
counts or dimensions raise a domain-level `EmbeddingError` instead of
producing corrupted data.

This issue only builds the abstraction; no concrete model adapter
(e.g. sentence-transformers) is implemented yet, and there is no
VectorDB storage or API endpoint. `Settings.embedding_provider` and
`Settings.embedding_model_name` are provisional placeholders not yet
read by any implementation. See
`docs/adr/0005-embedding-strategy.md` for the reasoning and candidate
models under consideration for a follow-up issue.

## Vector store

`EmbeddedChunk`s can be stored and searched by similarity through the
`app.domain.ports.vector_store.VectorStore` Protocol
(`upsert(chunks: list[EmbeddedChunk]) -> None`,
`search(query_vector: list[float], top_k: int) -> list[SearchResult]`),
via `app.application.services.index_chunks.IndexChunksService` and
`app.application.services.search_chunks.SearchChunksService`. A
`SearchResult` pairs an `EmbeddedChunk` with a `score: float`, where a
higher score always means a closer match regardless of which
`VectorStore` implementation produced it; converting a concrete
backend's native distance metric into that convention is the
implementation's responsibility. `Chunk` exposes a computed `chunk_id`
property (`document_id:page_number:chunk_index`) used as a stable
identifier for upsert/deduplication.

This issue only builds the abstraction; no concrete vector database
adapter (Qdrant, pgvector, Chroma) is implemented yet, and no new
dependency is added. A deterministic, dependency-free test double,
`tests/support/in_memory_vector_store.py::InMemoryVectorStore`, is
available for tests but is not production code. See
`docs/adr/0006-vector-store-strategy.md` for the reasoning, including
why `upsert` (not `add`) was chosen and the `score` convention.

## Indexing pipeline

`app.application.services.index_document.IndexDocumentService` indexes
one PDF end to end by composing `LoadDocumentService`,
`ChunkDocumentService`, `EmbedChunksService`, and `IndexChunksService`
in sequence, returning an `IndexDocumentResult` (`document_id`,
`source_name`, `page_count`, `chunk_count`, `indexed_count`).
`document_id` is `None` for a zero-page PDF, since no `DocumentPage` is
produced in that case. Re-indexing the same PDF does not create
duplicate `VectorStore` entries, since it relies on the existing
`chunk_id`-based upsert idempotency
(`docs/adr/0006-vector-store-strategy.md`). Exceptions raised by any
step (document loading, chunking, embedding, storage) are not caught
and propagate to the caller unchanged. See
`docs/adr/0007-indexing-pipeline.md` for the reasoning.

This issue only builds the Application-layer pipeline and its tests;
there is no upload API yet, and no concrete embedding model or vector
database adapter.

## Retrieval

`app.application.services.retrieve_chunks.RetrieveChunksService` takes
a natural-language query string and `top_k`, embeds the query via an
injected `Embedder`, and delegates the similarity search to an
injected `app.application.services.search_chunks.SearchChunksService`.
`top_k` and an empty/whitespace-only query are rejected before the
`Embedder` is called; a mismatch between the number of vectors the
`Embedder` returns and the single query it was given raises the
existing `EmbeddingCountMismatchError`. No new result model is
introduced: the return value is `list[SearchResult]`, the same type
`VectorStore.search`/`SearchChunksService.execute` already return.
Exceptions from the `Embedder` or `SearchChunksService` are not caught
and propagate to the caller unchanged. Logging includes only `top_k`
and the returned result count, never query text or vector values. See
`docs/adr/0008-retrieval-strategy.md` for the reasoning.

This issue only builds the Application-layer use case and its tests;
there is no search API endpoint yet, and no concrete embedding model or
vector database adapter.

## Project layout

See `docs/architecture.md` for the layered architecture and
`CLAUDE.md` for the full set of project rules.

```
app/            # application source (api, application, domain, infrastructure, core, schemas)
tests/          # unit, integration, api tests
docs/           # requirements, architecture, ADRs
scripts/        # development and operational scripts
data/           # raw, processed, and sample guideline data (see data/sample/README.md)
```
