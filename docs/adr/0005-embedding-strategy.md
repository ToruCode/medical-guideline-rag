# 0005. Embedder abstraction, deferred to a concrete model adapter

## Status

Accepted

## Context

`Chunk`s produced by the text chunking foundation (Issue #5,
`docs/adr/0004-text-chunking-strategy.md`) need to be converted into
embedding vectors before they can be indexed and searched. The
guideline documents this system targets are expected to include
Japanese content alongside English, so any concrete embedding model
must support multilingual text.

## Decision

- Define an `Embedder` Protocol
  (`app/domain/ports/embedder.py`, `embed(texts: list[str]) ->
  list[list[float]]`) with the same shape as the existing
  `TextSplitter` Protocol: it operates on raw strings only and knows
  nothing about `Chunk` metadata. Carrying `Chunk` metadata into the
  embedding result is the responsibility of
  `app/application/services/embed_chunks.py::EmbedChunksService`, not
  the `Embedder` implementation.
- Represent the result as `EmbeddedChunk`
  (`app/domain/models/embedding.py`), composing an existing `Chunk`
  with a `vector: list[float]`, rather than duplicating `Chunk`'s
  fields a third time (as `Chunk` itself did when built from
  `DocumentPage`). `EmbeddedChunk`'s shape is identical to `Chunk`'s
  plus exactly one new field, which is the natural case for
  composition.
- `EmbedChunksService.execute([])` returns `[]` immediately without
  calling the injected `Embedder`, so an empty batch never triggers a
  model call (and, once a real model-backed adapter exists, never
  triggers a model load).
- `EmbedChunksService` validates the embedder's output before building
  `EmbeddedChunk`s: a returned vector count that does not match the
  input text count raises `EmbeddingCountMismatchError`, and vectors
  with inconsistent lengths within the same batch raise
  `EmbeddingDimensionMismatchError`. Both are subtypes of
  `EmbeddingError` (`app/domain/exceptions/embedding.py`), so no
  low-level indexing bug in an `Embedder` implementation can silently
  produce misaligned `EmbeddedChunk`s.
- **This issue implements only the abstraction** (`Embedder` Protocol,
  `EmbeddedChunk`, `EmbedChunksService`, and a test-only
  `tests/support/fake_embedder.py`). No concrete model adapter is
  implemented yet, and no new dependency is added. `sentence-transformers`
  and a specific multilingual model were deliberately not adopted in
  this issue, to avoid committing to a model before it can be evaluated
  and to keep this issue's dependency footprint at zero.
- `Settings.embedding_provider` (default `"fake"`) and
  `Settings.embedding_model_name` (default
  `"intfloat/multilingual-e5-large"`) are added as provisional
  placeholders only; no infrastructure implementation reads them yet.

## Candidate models for a follow-up issue

| Candidate | Notes |
|---|---|
| `intfloat/multilingual-e5-large` | Designed for retrieval/RAG (asymmetric `query:`/`passage:` prefixes), strong multilingual coverage, ~2GB. Current leading candidate. |
| `intfloat/multilingual-e5-base` | Same retrieval-oriented design, smaller footprint; a candidate if image size/inference speed outweighs the accuracy gain of the `large` variant. |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | General-purpose multilingual sentence similarity model, well established, but not retrieval-optimized. |
| OpenAI Embeddings API | No heavy local dependency, but sends guideline text to a third-party API, adds a network dependency and per-token cost, and is inconsistent with keeping document content local. |

A local `sentence-transformers`-based adapter is the current leaning,
since it keeps guideline text from being sent to a third-party API,
but the final model and the decision to adopt `sentence-transformers`
as a production dependency (and where to place it in
`pyproject.toml`) are deferred to the issue that implements the
concrete adapter.

## Consequences

- Downstream code (retrieval, VectorDB storage) can be designed against
  `Embedder` and `EmbeddedChunk` without waiting for a model decision.
- A follow-up issue must add the concrete adapter (e.g.
  `app/infrastructure/embedding/sentence_transformer_embedder.py`),
  the corresponding dependency, and wire `Settings.embedding_provider`
  /`embedding_model_name` into an actual provider selection; until
  then these settings are inert.
- `vector: list[float]` is mutable even though `EmbeddedChunk` is a
  frozen dataclass (only attribute reassignment is prevented, not
  in-place mutation of the list). This is accepted because `list[float]`
  is the required representation; callers must not mutate a vector in
  place.
- Dimension-mismatch detection only checks consistency within a single
  `execute()` batch, not across separate calls or model swaps. If that
  becomes necessary, `Embedder` can gain a `dimension` property in a
  future revision without breaking this design.
