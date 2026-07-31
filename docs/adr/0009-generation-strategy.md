# 0009. Add an Llm abstraction and a citation-grounded GenerateAnswerService

## Status

Accepted

## Context

Issue #9 produced `RetrieveChunksService`, which turns a natural-language
query into `list[SearchResult]`. Nothing yet turns those retrieved
passages into an answer. The project's medical-safety rules require
that any generated answer be grounded only in retrieved passages, carry
citation metadata, and fall back to an explicit insufficient-evidence
result when no evidence was retrieved — with no possibility of an LLM
inventing an answer in that case.

## Decision

- `app/domain/ports/llm.py` defines the `Llm` Protocol with a single
  method, `generate(prompt: str) -> str`. Like `Embedder`
  (`embed(texts: list[str]) -> list[list[float]]`), it is a minimal
  text-in/text-out abstraction: the caller supplies one fully-composed
  prompt string, and the Protocol says nothing about chat message
  roles, system prompts, or any specific LLM SDK's request shape. This
  keeps all prompt-engineering logic in exactly one place
  (`GenerateAnswerService`), testable without any concrete LLM adapter.
  A two-argument alternative (`generate(question, context)`) was
  considered, which would let a future concrete adapter own prompt
  templating (e.g. to build native chat messages) — rejected for now
  to keep the Domain Protocol as simple as `Embedder` and avoid
  splitting prompt-construction responsibility across layers before a
  concrete adapter exists to justify it.
- `app/application/services/generate_answer.py` defines
  `GenerateAnswerService`, which takes a `question: str` and an
  already-retrieved `list[SearchResult]` (not a `top_k`; validating
  `top_k` remains `RetrieveChunksService`/`SearchChunksService`'s
  responsibility for whatever produced those `SearchResult`s, per
  `docs/adr/0008-retrieval-strategy.md`). It rejects an
  empty/whitespace-only `question` by raising the existing
  `EmptyQueryError` (`app/domain/exceptions/retrieval.py`), reused
  as-is rather than duplicated under a generation-specific exception,
  since it names the same failure (an empty user-supplied text input)
  regardless of whether that text feeds embedding+search or an LLM
  prompt.
- When `search_results` is empty, the `Llm` is **never called**.
  `GenerateAnswerService` returns a fixed `INSUFFICIENT_EVIDENCE_ANSWER`
  string with `is_insufficient_evidence=True` and `citations=[]`
  instead. This mirrors `EmbedChunksService`/`IndexChunksService`
  short-circuiting on empty input, and makes it structurally impossible
  for the `Llm` to invent an answer when there is no retrieved
  evidence to ground it in — a stronger guarantee than asking the
  `Llm` to say "insufficient evidence" itself.
- `is_insufficient_evidence` reflects **only** the structural zero-results
  case, not any semantic judgment of the `Llm`'s own answer text (e.g. the
  `Llm` saying "I cannot answer this" despite having context). Judging
  answerability from the `Llm`'s output is explicitly out of scope: CLAUDE.md
  lists "answerability judgment" as a distinct future responsibility of the
  Application layer, separate from generation itself.
- `build_context(search_results: list[SearchResult]) -> str`
  (same module) formats retrieved chunks into a numbered block
  (`[1] {title or source_name}, page {page_number}\n{text}`, ...), in
  `search_results`' existing order (descending similarity). It is a
  plain module-level function, not a separate class or Domain module,
  since `GenerateAnswerService` is its only caller today and it has no
  state or injected dependency of its own.
  - **Known limitation**: `Chunk` currently has no `edition`, `chapter`,
    or `section` fields, so the context (and therefore any citation
    the `Llm` produces) can only reference title/source name and page
    number, not the fuller citation (`document title, edition,
    chapter, section, and page`) named in `docs/requirements.md`.
    Extending `Chunk` with those fields is out of scope here — it
    would ripple through `Chunk`, `ChunkDocumentService`,
    `EmbedChunksService`, `IndexChunksService`, and every existing test
    built against the current `Chunk` shape (Issues #4-#9) — and is
    deferred to a future issue, likely alongside richer PDF structure
    extraction.
- `GenerationResult` (same module, not `app/domain/models`) is a frozen
  dataclass: `answer: str`, `citations: list[SearchResult]`,
  `is_insufficient_evidence: bool`. It is kept in the Application layer
  for the same reason `IndexDocumentResult` is
  (`docs/adr/0007-indexing-pipeline.md`): no Domain Protocol returns
  it — `Llm.generate` returns a plain `str` — so it only exists to
  summarize this one use case's outcome. `citations` reuses
  `SearchResult` (the same type `SearchChunksService`/
  `RetrieveChunksService` already return) rather than introducing a
  separate `Citation` model; a future API layer can project just the
  citation-relevant fields (title, page, source name) into its own
  response schema without a new Domain/Application type being needed
  now.
- No exception is caught. `EmptyQueryError` and any exception raised by
  the injected `Llm` (a future concrete adapter's network/API errors)
  propagate to the caller unchanged, matching the fail-fast approach
  established for `IndexDocumentService` and `RetrieveChunksService`.
- Logging includes only `citation_count` and the insufficient-evidence
  outcome, at INFO level. Question text, context text, chunk text, and
  the `Llm`'s answer text are never logged, per the project's
  medical-safety and logging rules. No API key handling exists yet in
  this layer, since no concrete `Llm` adapter is implemented in this
  issue; a future Infrastructure-layer adapter (e.g. calling an LLM
  API) must follow the same "never log credentials" rule.
- **No orchestrating use case is added in this issue.** A future issue
  is expected to compose `RetrieveChunksService` and
  `GenerateAnswerService` into a single "ask a question end to end"
  use case, the same way `IndexDocumentService` (Issue #8) composed
  four existing services only after all four existed (Issues #4-#7).
  Composing retrieval and generation now, before reranking and
  answerability judgment (both named as future Application-layer
  responsibilities in CLAUDE.md) exist, risks an orchestrator that
  needs restructuring as soon as those steps land. The connection
  between the two services is demonstrated only in
  `tests/integration/test_generation_pipeline.py`, which calls
  `RetrieveChunksService.execute` and feeds its result directly into
  `GenerateAnswerService.execute`.
- **No API endpoint and no concrete `Llm` adapter (e.g. an OpenAI
  client) are added in this issue.** `GenerateAnswerService` is
  Application-layer only, wired directly in tests with `FakeLlm`
  (`tests/support/fake_llm.py`), which never performs network access.

## Consequences

- A future orchestrating use case (composing `RetrieveChunksService` +
  `GenerateAnswerService`, and eventually reranking/answerability
  judgment) still needs to be designed and built; this issue does not
  provide it.
- A future API layer must translate `EmptyQueryError` and whatever
  exception a concrete `Llm` adapter raises into HTTP responses itself;
  this issue does not provide a unified error type for that
  translation.
- Citations are limited to title/source name and page number until
  `Chunk` gains edition/chapter/section fields in a future issue.
- Because `Llm.generate` takes one plain string, a future concrete
  adapter for a chat-style API (e.g. OpenAI's Chat Completions) must
  translate that single string into whatever message structure its SDK
  requires itself; the Domain Protocol does not help with that
  translation.
