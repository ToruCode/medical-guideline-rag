# 0030. Isolate pytest's provider settings from a real local .env

## Status

Accepted

## Context

`docs/adr/0012-live-e2e-verification.md` added `load_dotenv()` to
`tests/conftest.py` so that filling in `MEDICAL_RAG_LLM_API_KEY` and
`RUN_SLOW_TESTS` in `.env` is enough to opt into the live tests
(`tests/integration/test_live_*.py`), without also exporting them into
the shell. That ADR stated: "No test currently depends on `os.environ`
*not* reflecting `.env`, so this is not expected to change any other
test's behavior."

That assumption no longer holds. A developer's real `.env` (used for
local `make dev`/`make ui` against real providers) typically also sets
`MEDICAL_RAG_EMBEDDING_PROVIDER=sentence_transformers`,
`MEDICAL_RAG_LLM_PROVIDER=openai`, and a real `MEDICAL_RAG_LLM_API_KEY`
- values meant for the running app, not for pytest. Because
`load_dotenv()` populates `os.environ` for the entire test session and
nothing ever undoes it, every ordinary test's `Settings()`/
`get_settings()` picked up these real values instead of `Settings`'
fake/no-key defaults. Five tests failed as a result, all of them tests
that assert default (no-monkeypatch) provider behavior:

- `test_get_passage_embedder_returns_fake_embedder_by_default`
- `test_get_query_embedder_returns_fake_embedder_by_default`
- `test_get_llm_returns_fake_llm_by_default`
- `test_get_llm_with_openai_provider_and_no_api_key_raises` (a real key
  from `.env` was present, so the expected `ValueError` never fired)
- `test_index_document_returns_created_with_counts` (the real
  `SentenceTransformerEmbedder` loads PyTorch, which writes a
  `torchinductor` cache directory under the same temp directory the
  test asserts is left empty)

CI never has a `.env` file (it is gitignored), so `load_dotenv()` is a
no-op there and this never reproduces in CI - only locally, for any
developer whose `.env` is configured for real local use rather than
left at its fake defaults.

## Decision

- **`tests/conftest.py` gains an autouse, function-scoped fixture,
  `_isolate_provider_settings_from_dotenv`**, that runs
  `monkeypatch.delenv(...)` on `MEDICAL_RAG_EMBEDDING_PROVIDER`,
  `MEDICAL_RAG_LLM_PROVIDER`, and `MEDICAL_RAG_LLM_API_KEY` before every
  test, then clears `get_settings` and the `app/api/dependencies.py`
  provider `lru_cache`s. `monkeypatch.delenv` restores each variable to
  its prior (real `.env`) value once the test ends, and the caches are
  cleared again on teardown so the next test starts from a clean slate.
- **Ordinary tests never need to change.** Any test that already calls
  `monkeypatch.setenv("MEDICAL_RAG_EMBEDDING_PROVIDER", ...)` etc. (most
  of `tests/unit/test_dependencies.py`, `tests/unit/test_config.py`)
  keeps working unmodified: that `setenv` call happens inside the test
  body, after this fixture's setup, so it simply overrides the
  now-absent variable for that one test.
- **Live tests are unaffected in gating, with one small opt-in change.**
  `tests/integration/test_live_*.py` decide whether to skip by reading
  `os.environ` directly at *collection* time (module import), which
  happens before any function-scoped fixture - including this new one -
  ever runs. `test_live_openai_llm.py` and
  `test_live_sentence_transformer_embedder.py` need no change: neither
  goes through `Settings()`/`get_settings()` for the real provider, so
  this fixture's per-test `delenv` never applies to them.
  `test_live_rag_e2e.py` does use `get_llm()` (via the running app), so
  it now captures `MEDICAL_RAG_LLM_API_KEY` into a module-level `_API_KEY`
  at collection time - the same pattern `test_live_openai_llm.py`
  already used - and each test restores it with
  `monkeypatch.setenv("MEDICAL_RAG_LLM_API_KEY", _API_KEY)` alongside its
  existing provider `monkeypatch.setenv` calls.
- **A regression test**
  (`tests/unit/test_dependencies.py::test_providers_stay_fake_even_when_dotenv_style_env_vars_are_simulated`)
  asserts the three variables are absent from `os.environ` and that
  `Settings`/the provider functions resolve to their fake/no-key
  defaults, so a future change that accidentally removes or narrows this
  fixture is caught locally, not just noticed as a CI/local behavior
  mismatch.
- `.env` and `.env.example` are unchanged - this is a test-harness-only
  fix.

## Consequences

- Running the full `pytest` suite locally now succeeds with a real
  `.env` present, matching CI, regardless of what
  `MEDICAL_RAG_EMBEDDING_PROVIDER`/`MEDICAL_RAG_LLM_PROVIDER`/
  `MEDICAL_RAG_LLM_API_KEY` are set to for local `make dev` use.
- `os.environ` still reflects the rest of `.env` (e.g.
  `MEDICAL_RAG_APP_NAME`, `MEDICAL_RAG_CHUNK_SIZE`) for the whole
  session, as `docs/adr/0012` originally decided. If a future test
  starts asserting a default value for one of those other
  `MEDICAL_RAG_*` settings and a developer's `.env` happens to override
  it, the same class of failure could reappear for that variable; only
  the three provider-selection variables actually observed to cause
  failures are covered here, not `.env` wholesale.
- `RUN_SLOW_TESTS`/`MEDICAL_RAG_LLM_API_KEY`-gated opt-in live tests
  keep working exactly as `docs/adr/0012` intended.
