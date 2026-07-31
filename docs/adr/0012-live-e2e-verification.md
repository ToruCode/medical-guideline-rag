# 0012. Verify the full RAG flow with real Embedding and LLM adapters

## Status

Accepted

## Context

Issue #12 (`docs/adr/0011-real-embedding-and-llm-adapters.md`) added
`SentenceTransformerEmbedder` and `OpenAiLlm`, selectable via
`Settings.embedding_provider`/`llm_provider`, plus two isolated live
tests: `tests/integration/test_live_sentence_transformer_embedder.py`
(real model, no API key needed) and
`tests/integration/test_live_openai_llm.py` (real API, one bare
`generate()` call). Neither exercises the full stack: a PDF indexed via
`POST /documents/index` with the real embedder, then a question
answered via `POST /questions/ask` with the real LLM, against the
shared `InMemoryVectorStore`. This issue closes that gap with a
verification pass; no new Domain/Application/Infrastructure code is
needed; everything required already exists.

## Decision

- **`tests/integration/test_live_rag_e2e.py`** drives the actual
  FastAPI app (`app.main.app`, via `TestClient`) through
  `POST /documents/index` then `POST /questions/ask`, with
  `MEDICAL_RAG_EMBEDDING_PROVIDER=sentence_transformers` and
  `MEDICAL_RAG_LLM_PROVIDER=openai` set via `monkeypatch.setenv` inside
  the test. This is the first test that exercises `RetrieveChunksService`'s
  `"query: "`-prefixed embedding, `EmbedChunksService`'s
  `"passage: "`-prefixed embedding, `InMemoryVectorStore`'s real (768-d,
  for `intfloat/multilingual-e5-base`) vectors, and `OpenAiLlm`'s real
  completion, all in one path - functionally confirming the query/
  passage prefix split and dimension consistency end to end, rather
  than re-testing them as isolated units (already covered by
  `tests/unit/test_sentence_transformer_embedder.py`,
  `test_openai_llm.py`, `test_dependencies.py`).
- **Gating reuses the existing two conventions, combined**: skipped
  unless both `RUN_SLOW_TESTS` (the model-download cost, already used
  by `test_live_sentence_transformer_embedder.py`) and
  `MEDICAL_RAG_LLM_API_KEY` (the billable-API-call cost, already used
  by `test_live_openai_llm.py`) are set. No new environment variable is
  introduced; running the full stack requires opting into both existing
  costs, which is exactly what it does.
- **The sample PDF is generated at test-run time**
  (`tests/support/pdf_factory.build_pdf`, writing into `tmp_path`), not
  committed to the repository. Its content is a single, obviously
  fictional sentence ("Adults should take 500 mg of Medicamentum X
  twice daily with food.") - not a real drug, guideline, or patient
  fact - consistent with `data/sample/README.md`'s and `CLAUDE.md`'s
  data/copyright rules, and avoiding any ambiguity about whether the
  content could be mistaken for real medical guidance.
- **`tests/conftest.py` now calls `load_dotenv()`** (via `python-dotenv`,
  newly declared as an explicit dev dependency though it was already
  installed transitively through `pydantic-settings`) at import time,
  before any test module is collected. Without this, a developer who
  fills in `MEDICAL_RAG_LLM_API_KEY`/`RUN_SLOW_TESTS` only in `.env`
  would see every live test skip anyway: `Settings()` reads `.env`
  correctly, but each live test's `skipif`/module-level `os.environ.get(...)`
  check runs at collection time, before any `Settings()` is
  constructed, and reads the real process environment directly - so
  without `load_dotenv()`, only variables actually exported into the
  shell would be seen. `load_dotenv()` never overrides a variable
  already present in the real environment, so CI/shell-exported values
  still take precedence. This applies retroactively to the two
  existing live tests too, unifying the developer experience with how
  `uvicorn`/the running app already reads `.env` through
  `pydantic-settings`.
- **Assertions check only counts, status codes, and boolean flags** -
  `chunk_count`, `citation_count` (via `len(citations)`), `page_number`,
  `is_insufficient_evidence`, and that `answer` is a non-empty string -
  never the API key or the generated answer's exact wording, so a
  failing assertion's message cannot leak either.
- **No manual CLI script** (e.g. `scripts/live_e2e_check.py`) is added.
  Manual verification is documented in `README.md` as a Swagger UI
  walkthrough instead, avoiding duplicating the same
  index-then-ask logic in two places for one project.

## Consequences

- Running `test_live_rag_e2e.py` costs real time (model load/inference)
  and real money (one OpenAI Chat Completions call); it must never run
  in default `pytest`/CI invocations, and does not.
- `load_dotenv()` in `tests/conftest.py` means `.env`'s contents are now
  visible to `os.environ` for the entire test session, not just to
  `Settings()` instances. No test currently depends on `os.environ`
  *not* reflecting `.env`, so this is not expected to change any other
  test's behavior.
- If `intfloat/multilingual-e5-base` or the configured OpenAI model is
  ever swapped, this test's fixed sample sentence/question pair should
  still work (it does not depend on exact embedding values), but is not
  a substitute for re-running the opt-in live tests after such a
  change.
