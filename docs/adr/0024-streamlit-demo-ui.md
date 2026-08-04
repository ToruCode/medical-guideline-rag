# 0024. Streamlit demo UI

## Status

Accepted

## Context

Issues #6-#12 built the full question-answering flow (retrieval,
generation, citations, real/Fake providers) behind two FastAPI
endpoints (`docs/adr/0010-fastapi-rag-api.md`), exercised so far only
through Swagger UI or direct HTTP calls. Issue #13 adds a simple
Streamlit UI so a user can type a question and see the generated
answer and its citations without using Swagger UI directly. The UI
must reuse the existing question-answering flow rather than
duplicating retrieval/generation logic, must not call OpenAI directly,
and is explicitly not responsible for indexing documents (no file
upload) or any other functionality beyond asking a question.

## Decision

- **The Streamlit UI is an HTTP client of the existing FastAPI API,
  not a second process that constructs its own `AskQuestionService`
  in-process.** `app/ui/api_client.py`'s `QuestionApiClient` calls
  `POST {api_v1_prefix}/questions/ask` on an already-running API
  server via `httpx`. This is the only design that works correctly
  given two existing, unchanged constraints: `InMemoryVectorStore` is
  a process-wide singleton
  (`docs/adr/0006-vector-store-strategy.md`) with no persistence
  across processes or restarts, and this issue explicitly keeps
  document indexing (`POST /documents/index`) out of the UI's scope.
  If the UI built its own service graph in-process, it would only
  ever see an empty vector store - the documents indexed via Swagger
  UI/`curl` into the separately-running API process would be
  invisible to it. Calling the same running API process the documents
  were indexed into is the only way for the UI to see that data.
  This also means the UI never touches `Settings.llm_provider` (or any
  other provider setting) directly: whichever provider the API server
  is configured with is transparently what the UI uses, satisfying the
  "single configuration system" requirement without the UI needing its
  own copy of that logic.
- **`Settings.ui_api_base_url`** (new, `MEDICAL_RAG_UI_API_BASE_URL`,
  default `http://127.0.0.1:8000`) is the one new setting this issue
  adds, following the existing single-`Settings`-class pattern rather
  than introducing a second, UI-only configuration system.
  `api_v1_prefix` (already existing) is reused as-is for the path.
- **Three-module split inside `app/ui/`**, mirroring how
  `app/application/services/generate_answer.py` already separates pure
  functions (`build_context`, `select_chunks_within_budget`) from the
  stateful service class, so the UI's logic is unit-testable without a
  Streamlit runtime or a network call:
  - `api_client.py` - `QuestionApiClient`, translating `httpx`
    connection failures and HTTP 4xx/5xx responses into two narrow
    exception types (`ApiConnectionError`, `ApiRequestError`), mirroring
    how `OpenAiLlm.generate()` already translates the `openai` SDK's
    exceptions into `LlmGenerationError`
    (`docs/adr/0022-context-length-control-and-llm-error-handling.md`)
    rather than letting a third-party library's exception type leak to
    callers.
  - `presentation.py` - pure functions with no Streamlit import:
    `validate_question` (empty/whitespace-only check, mirroring
    `GenerateAnswerService`'s own validation), `citation_label` (formats
    one citation's already-safe metadata - source, page, chunk index,
    score - into a display line; deliberately never touches
    `text_preview`, which the caller displays separately, so this
    function's output can never grow with guideline text), and
    `describe_error` (maps an `ApiClientError` to one of a small, fixed
    set of Japanese messages by error category - connection failure,
    400, 502, other 5xx, unexpected - never the underlying exception's
    own message or any response body detail, so an API key, prompt, or
    internal path can never reach the UI regardless of what the
    underlying error happens to contain).
  - `streamlit_app.py` - rendering and control flow only: text input,
    submit button, spinner while the request is in flight, and
    dispatching to the two modules above. Contains no validation,
    formatting, or error-mapping logic of its own.
- **Medical disclaimer is always visible** (`MEDICAL_DISCLAIMER` in
  `presentation.py`), independent of any question being asked, per
  CLAUDE.md's Medical Safety Rules.
- **Insufficient evidence is shown as a fixed Japanese notice**
  (`INSUFFICIENT_EVIDENCE_NOTICE`), not by displaying
  `GenerateAnswerService.INSUFFICIENT_EVIDENCE_ANSWER` (the English
  fallback string returned by the API in this case, per
  `docs/adr/0009-generation-strategy.md`) - showing the raw English
  fallback to a Japanese user was judged less useful than a dedicated
  Japanese UI-level message.

## Consequences

- Running the demo end to end requires two separate processes: the
  API server (`make dev`) with a document already indexed via
  `POST /documents/index`, and the UI (`make ui`) pointed at that same
  server via `MEDICAL_RAG_UI_API_BASE_URL`. This is an accepted
  two-step setup, not a regression: no single-process alternative
  exists without either adding file upload to the UI (explicitly out
  of scope for this issue) or a persistent, shared vector store
  (no concrete adapter exists yet).
  Because the UI never builds its own DI graph, all of the existing
  provider-switching machinery (`Settings.llm_provider`,
  `app/api/dependencies.py`) is reused as-is; if a future issue changes
  the API's error responses or adds a third provider, only
  `api_client.py`/`presentation.py` need updating, not
  `streamlit_app.py`.
- `describe_error`'s fixed, category-based messages mean a caller
  cannot distinguish, from the UI alone, *why* a given 5xx occurred
  beyond "server error" - matching `POST /questions/ask`'s own existing
  behavior of returning a generic detail message and logging the real
  exception server-side only.
