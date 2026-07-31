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
all of the above end to end (Issue #8), a retrieval use case
(Issue #9) that embeds a natural-language query and returns similar
chunks, a generation use case (Issue #10) that turns retrieved chunks
into a citation-grounded answer, a FastAPI RAG API (Issue #11) exposing
document indexing and question answering end to end, and real
`Embedder`/`Llm` adapters (Issue #12: a local `sentence-transformers`
model and OpenAI's Chat Completions API), selectable alongside the
still-default Fake implementations via `Settings`. No concrete vector
database adapter is implemented yet.

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

`uv sync` installs `sentence-transformers` (and its transitive
`torch`/`transformers` dependencies) and `openai` regardless of which
provider is configured; this is a sizeable download (hundreds of MB,
CPU-only). Both remain unused unless you opt in via
`MEDICAL_RAG_EMBEDDING_PROVIDER=sentence_transformers` and/or
`MEDICAL_RAG_LLM_PROVIDER=openai` - see [Embedding](#embedding) and
[Generation](#generation) below.

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

PDFs can be uploaded via `POST /api/v1/documents/index` (see
[API](#api) below). Only self-authored, license-free sample PDFs belong
under `data/sample/` (see `data/sample/README.md`); real or
copyrighted guideline PDFs must never be committed, and `data/raw/` is
git-ignored for that reason.

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

By default the FastAPI app uses
`app.infrastructure.embedding.fake_embedder.FakeEmbedder`, a
deterministic, dependency-free stand-in with no network access. Setting
`MEDICAL_RAG_EMBEDDING_PROVIDER=sentence_transformers` switches to
`app.infrastructure.embedding.sentence_transformer_embedder.SentenceTransformerEmbedder`,
which runs a local, multilingual `sentence-transformers` model
(`MEDICAL_RAG_EMBEDDING_MODEL_NAME`, default
`intfloat/multilingual-e5-base`, downloaded on first use). That model
family is asymmetric - it needs a `"query: "` prefix for search queries
and a `"passage: "` prefix for indexed text - so the API layer actually
constructs two `Embedder`s sharing one loaded model
(`get_passage_embedder`/`get_query_embedder` in
`app/api/dependencies.py`), not one. See
`docs/adr/0005-embedding-strategy.md` for the original abstraction
decision and `docs/adr/0011-real-embedding-and-llm-adapters.md` for the
concrete adapter and provider-switching design.

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

No concrete vector database adapter (Qdrant, pgvector, Chroma) is
implemented yet. A deterministic, dependency-free implementation,
`app.infrastructure.vector_store.in_memory_vector_store.InMemoryVectorStore`,
is used both by tests and as the FastAPI app's default dependency
wiring (see [API](#api) below); its data does not survive a process
restart. See `docs/adr/0006-vector-store-strategy.md` for the
reasoning, including why `upsert` (not `add`) was chosen and the
`score` convention.

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
there is no concrete embedding model or vector database adapter yet
(the API layer wires it to Fake implementations - see [API](#api)
below).

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
there is no concrete embedding model or vector database adapter yet
(see [API](#api) below for how it is exposed over HTTP today).

## Generation

`app.application.services.generate_answer.GenerateAnswerService` takes
a question and an already-retrieved `list[SearchResult]` (from
`RetrieveChunksService`) and generates a citation-grounded answer by
delegating to an injected `app.domain.ports.llm.Llm`
(`generate(prompt: str) -> str`). An empty/whitespace-only question is
rejected before the `Llm` is called. When `search_results` is empty,
the `Llm` is never called at all: a fixed insufficient-evidence answer
is returned instead, so the system can never invent an answer when no
guideline evidence was retrieved. `GenerationResult` (`answer`,
`citations: list[SearchResult]`, `is_insufficient_evidence`) reuses
`SearchResult` rather than introducing a separate citation model.
Exceptions from the `Llm` are not caught and propagate to the caller
unchanged. Logging includes only the citation count and the
insufficient-evidence outcome, never question text, guideline passage
text, or the generated answer text. See
`docs/adr/0009-generation-strategy.md` for the reasoning, including the
current limitation that citations only carry title/source name and
page number (not edition/chapter/section, which `Chunk` does not yet
model).

By default the FastAPI app uses `app.infrastructure.llm.fake_llm.FakeLlm`,
a deterministic, dependency-free stand-in with no network access.
Setting `MEDICAL_RAG_LLM_PROVIDER=openai` (plus a real
`MEDICAL_RAG_LLM_API_KEY`) switches to
`app.infrastructure.llm.openai_llm.OpenAiLlm`, which sends the composed
prompt as a single user message to OpenAI's Chat Completions API
(`MEDICAL_RAG_LLM_MODEL_NAME`, default `gpt-4o-mini`). `llm_api_key` is
a `pydantic.SecretStr`, so it never appears in logs even if the whole
`Settings` object is accidentally printed. See
`docs/adr/0011-real-embedding-and-llm-adapters.md` for the adapter and
provider-switching design.

## API

Two endpoints expose the services above over HTTP, prefixed with
`Settings.api_v1_prefix` (default `/api/v1`):

- **`POST /documents/index`** - `multipart/form-data` upload of a
  single PDF (`file`). Rejects a non-`.pdf` file name (415) and an
  empty file (400). The upload is saved under a sanitized version of
  its own name inside a freshly-created, randomly-named temp
  directory - never under the raw uploaded name directly, and never in
  a predictable location - and the whole directory is always removed
  afterward, whether indexing succeeds or fails. Returns 201 with an
  `IndexDocumentResponse` (`document_id`, `source_name`, `page_count`,
  `chunk_count`, `indexed_count`) on success; an encrypted or
  unparseable PDF returns 422.
- **`POST /questions/ask`** - JSON body (`question`, `top_k`, default
  5). Composes `RetrieveChunksService` and `GenerateAnswerService` via
  the new `app.application.services.ask_question.AskQuestionService`.
  Returns 200 with an `AskQuestionResponse` (`answer`,
  `citations: list[CitationSchema]`, `is_insufficient_evidence`) -
  including when no evidence was retrieved, which is a normal result,
  not an error. An empty/whitespace-only question or non-positive
  `top_k` returns 400/422.

`app/api/dependencies.py` is the only module that constructs
Infrastructure implementations (Fake or real, depending on
`Settings.embedding_provider`/`llm_provider` - see
[Embedding](#embedding) and [Generation](#generation) above) and wires
them into Application services; endpoint modules never import
Infrastructure directly. `get_passage_embedder`/`get_query_embedder`/
`get_vector_store`/`get_llm` are process-wide singletons, so a document
indexed via `POST /documents/index` is searchable via
`POST /questions/ask` for the life of the running process (data is
lost on restart). Neither endpoint logs question text, guideline
passage text, generated answer text, or embedding vectors - only
counts, file names, and outcome flags.

There is no concrete vector database adapter yet (`InMemoryVectorStore`
remains the only `VectorStore`). Swagger UI at `/docs` can exercise
both endpoints directly, including file upload, regardless of which
providers are configured.

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
