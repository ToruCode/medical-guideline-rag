# 0011. Add sentence-transformers and OpenAI adapters, selected via Settings

## Status

Accepted

## Context

Issue #5/`docs/adr/0005-embedding-strategy.md` and Issue #9/
`docs/adr/0009-generation-strategy.md` deliberately deferred adopting
any concrete `Embedder`/`Llm` implementation, keeping `Settings.
embedding_provider`/`embedding_model_name` as inert placeholders. Issue
#11 wired the existing `FakeEmbedder`/`FakeLlm`/`InMemoryVectorStore`
into the FastAPI app via `app/api/dependencies.py`. This issue adds real
implementations - a local `sentence-transformers` model for embedding
and OpenAI's Chat Completions API for generation - selectable via
`Settings`, without removing the Fake path (still the default).

## Decision

- **`intfloat/multilingual-e5-base`** is the new default
  `embedding_model_name` (previously `-large`, never actually read by
  any implementation until now). `-base` is chosen over `-large` for a
  smaller download/memory footprint appropriate for a technical
  demonstration; either remains a plain `Settings` string, changeable
  via `MEDICAL_RAG_EMBEDDING_MODEL_NAME` with no code change.
- **The `Embedder` Protocol is unchanged**
  (`embed(texts: list[str]) -> list[list[float]]`), but the DI layer
  now provides **two** `Embedder`s instead of one:
  `get_passage_embedder()` (used by `EmbedChunksService`, i.e.
  `POST /documents/index`) and `get_query_embedder()` (used by
  `RetrieveChunksService`, i.e. `POST /questions/ask`). This is
  necessary because `intfloat/multilingual-e5-*` models are
  *asymmetric*: they require a `"query: "` prefix on search queries and
  a `"passage: "` prefix on indexed text to produce good retrieval
  quality, and the previous single `get_embedder()` had no way to know
  which prefix applied to a given call. `SentenceTransformerEmbedder`
  (`app/infrastructure/embedding/sentence_transformer_embedder.py`)
  takes a `prefix: str` constructor argument (default `""`) and
  prepends it to every input text; the `"fake"` provider is unaffected
  (`FakeEmbedder` ignores prefixes entirely; two separate `FakeEmbedder`
  instances are still constructed, which is inconsequential since it
  holds no meaningful state or expensive resources).
- **The underlying model is loaded once and shared.** `_get_
  sentence_transformer_model()` (`app/api/dependencies.py`) is a
  separate `@lru_cache`'d function that loads the `SentenceTransformer`
  exactly once; `get_passage_embedder()`/`get_query_embedder()` each
  wrap that same shared model with a different `prefix`, rather than
  loading the (multi-hundred-MB) model twice for no benefit.
- **`app/infrastructure/llm/openai_llm.py::OpenAiLlm`** implements
  `Llm.generate(prompt: str) -> str` via `openai.OpenAI(...)
  .chat.completions.create(model=..., messages=[{"role": "user",
  "content": prompt}])`, returning `response.choices[0].message.
  content` (or `""` if `None`). This matches the `Llm` Protocol's
  existing single-prompt-string shape (`docs/adr/0009-generation-strategy.md`)
  without any change to the Protocol itself. No exception is caught;
  `GenerateAnswerService` already propagates any `Llm` exception
  unchanged. No custom retry logic is added; the OpenAI SDK's own
  built-in retry behavior (default `max_retries=2`) is relied on
  instead, consistent with keeping this adapter minimal.
- **Provider selection** happens inside `get_passage_embedder`/
  `get_query_embedder`/`get_llm` themselves, reading
  `Settings.embedding_provider`
  (`Literal["fake", "sentence_transformers"]`, changed from a bare
  `str`) and `Settings.llm_provider` (new,
  `Literal["fake", "openai"]`). Both default to `"fake"`, so existing
  behavior is unchanged unless `.env` opts in. Reading `Literal` instead
  of `str` gives fail-fast `ValidationError`s on a typo'd provider name
  at `Settings` construction time, matching the existing
  `environment`/`log_level` fields' style. Each provider-dispatch
  function still has an `else: raise ValueError(...)` branch as a
  defensive fallback (currently unreachable through `Settings` itself
  given the `Literal` typing, but guards against a future refactor that
  loosens it).
- **Why `get_settings()` is called directly instead of declared as a
  `Depends(...)` parameter**: `get_passage_embedder`/`get_query_embedder`/
  `get_llm` are `@lru_cache`'d for process-wide singleton behavior
  (`docs/adr/0010-fastapi-rag-api.md`). `lru_cache` requires hashable
  arguments; `Settings` (`pydantic_settings.BaseSettings`, not
  `frozen=True`) is not hashable, so adding it as a parameter would
  raise `TypeError` on the first call. Calling `get_settings()` directly
  inside the function body keeps these providers zero-argument (safe
  for `lru_cache`) while still reading whatever `Settings` was current
  the first time each was invoked - `get_settings()` is itself cached,
  so this is consistent with how the rest of the app already treats
  `Settings` as fixed for a process's lifetime.
- **`Settings.llm_api_key` is `pydantic.SecretStr | None`**, not `str`.
  A `SecretStr`'s `repr()`/`str()` always renders as `**********`, so
  accidentally logging or printing the whole `Settings` object (e.g. a
  future debug statement) cannot leak the key - a stronger guarantee
  than only remembering not to log the field explicitly. `OpenAiLlm`
  receives `settings.llm_api_key.get_secret_value()` (the raw string) at
  construction time, never logged.
- **`get_llm()` raises `ValueError` at construction time** (not on the
  first request) when `llm_provider == "openai"` but `llm_api_key` is
  unset, so a misconfiguration produces a clear, immediate error instead
  of an opaque `AuthenticationError` from inside the OpenAI SDK on the
  first real question.
- **Heavy imports (`sentence_transformers`, `openai`) are deferred to
  each adapter's `__init__`/factory function**, not placed at module
  top level. `app/api/dependencies.py` unconditionally imports both
  adapter *classes* at its own top level (cheap - no heavy transitive
  import happens merely by referencing a class), and is itself imported
  by `app.main`/`app.api.v1.router`, which every test that touches the
  FastAPI app (nearly the whole suite) imports. Since `sentence_
  transformers` transitively pulls in `torch`/`transformers` (import
  alone costs real seconds and hundreds of MB), placing that import at
  a module's top level would force every test run - even ones that
  never touch embedding, using the default `"fake"` provider - to pay
  that cost. Deferring the import means it only runs when a
  `sentence_transformers`/`openai` provider is actually selected and
  constructed.
- **`POST /documents/index` now runs `IndexDocumentService.execute` via
  `starlette.concurrency.run_in_threadpool`** instead of calling it
  directly inside the `async def` endpoint. This was flagged as a
  theoretical risk in `docs/adr/0010-fastapi-rag-api.md` when only
  `FakeEmbedder` existed (near-instant); now that a real, potentially
  slow local model inference call can happen in the same code path,
  blocking the event loop is a real, not theoretical, cost. `POST
  /questions/ask`'s endpoint function is already a plain `def`, which
  FastAPI already runs in a worker thread automatically, so no
  equivalent change was needed there.
- **Testing**: unit tests for `SentenceTransformerEmbedder`/`OpenAiLlm`/
  provider dispatch (`tests/unit/test_sentence_transformer_embedder.py`,
  `test_openai_llm.py`, `test_dependencies.py`) monkeypatch
  `sentence_transformers.SentenceTransformer`/`openai.OpenAI` and never
  perform real model loading or network access. Two additional,
  **opt-in-only** live tests
  (`tests/integration/test_live_sentence_transformer_embedder.py`,
  gated on `RUN_SLOW_TESTS`; `tests/integration/test_live_openai_llm.py`,
  gated on a real `MEDICAL_RAG_LLM_API_KEY` being set) exercise the real
  model/API and are skipped by default, so the default `pytest` run
  never downloads a model, never makes a network call, and never
  requires a secret.

## Consequences

- Installing this project now pulls in `torch`/`transformers`/
  `huggingface-hub` (via `sentence-transformers`) and `openai`,
  meaningfully increasing install size (hundreds of MB, CPU-only; no
  CUDA/GPU setup is attempted). This will affect a future Docker image's
  size - out of scope here, deferred to the Docker/deployment issue.
- Switching `embedding_provider` (or `embedding_model_name`) after
  documents have already been indexed leaves the `VectorStore`'s
  existing vectors in the old model's space; a query embedded with a
  new model/dimension against old vectors either raises
  `VectorDimensionMismatchError` (if dimensions differ) or silently
  returns poor-quality matches (if dimensions happen to match). No
  automatic re-indexing or migration is provided; re-indexing affected
  documents is a manual operational step.
- No prompt/context length truncation is implemented. A very large
  number of retrieved passages (large `top_k`) could exceed the
  configured OpenAI model's context window, and `OpenAiLlm.generate`
  does not guard against this - the request would simply fail with the
  SDK's own error, which propagates unchanged.
- `llm_timeout_seconds` (default 30) is the only new tunable for
  request resilience; `temperature`, `max_tokens`, and similar
  generation parameters are not exposed as `Settings` and use the
  OpenAI API's own defaults.
- The live tests are real integration tests: `test_live_openai_llm.py`
  makes a real, billable API call when run with a real key, and
  `test_live_sentence_transformer_embedder.py` downloads a real model on
  first run. Neither runs in a default `pytest`/CI invocation.
