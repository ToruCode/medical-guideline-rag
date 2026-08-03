# ADR 0023: Answer Quality and Citation Consistency Evaluation

## Status

Accepted

## Context

Issue #8 added LLM-based answer generation: `GenerateAnswerService`
retrieves guideline chunks, builds a bounded context
(`select_chunks_within_budget()`), generates a Japanese answer, returns
citations, and reports insufficient evidence when nothing was
retrieved. Issue #10 asks whether that generated answer is actually
grounded in the retrieved context and whether the returned citations
are consistent with the evidence - a reproducible evaluation, not a
one-off manual check.

Retrieval quality is already evaluated separately
(`scripts/retrieval_baseline_core.py`,
`docs/adr/0013-retrieval-evaluation.md`), per
`docs/requirements.md`'s "separate retrieval quality from generation
quality". This ADR only covers the additional, generation-side
evaluation: citation consistency and answer quality.

## Decision

### Deterministic metrics only - no LLM-as-a-Judge

Every metric this evaluation computes is plain set arithmetic or
substring matching over structured data (citations' page numbers,
`GenerationResult.is_insufficient_evidence`, and the answer text) -
never a second LLM call scoring the first LLM's answer. This keeps the
evaluation:

- **Reproducible**: the same inputs always produce the same score, so
  results are comparable across runs and machines.
- **Explainable**: every score traces to a concrete, printable reason
  (which page was expected but not cited, which substring was not
  found), rather than an opaque judge-model rating.
- **Free of a third-party judge dependency**: no second model's
  quality/cost/availability enters the loop.

The accepted cost is that these metrics cannot detect a *correct but
differently worded* answer (see "Answer-point coverage is lexical, not
semantic" below). A future LLM-as-a-Judge pass remains possible, but is
out of scope for this issue and would be additive, not a replacement -
see "Future extensibility" below.

### Citations are correct by construction, not by runtime validation

`GenerateAnswerService.execute()` (Issue #8) already guarantees, by
construction, that `GenerationResult.citations` is exactly the subset
of `search_results` that was selected for the LLM's context
(`select_chunks_within_budget()`) - never a chunk the LLM was not
shown. Consequently:

- "Every citation was actually supplied to the LLM" is not something
  this evaluation can meaningfully fail to observe; it is a structural
  property of the code being evaluated.
- What this evaluation adds is `citations_are_subset_of_retrieved()`
  (`scripts/answer_quality_core.py`): an explicit regression check that
  every citation's `chunk_id` appears among the chunks that were
  actually retrieved for that question. It is expected to always
  report `True` today. Its purpose is to catch a *future* change that
  breaks this invariant (e.g. a refactor that lets
  `GenerateAnswerService` fabricate or mix in citations from another
  question) - a safety net, not a currently-observed failure mode.
- Reporting still counts and surfaces any violation
  (`AnswerAggregate.citation_consistency_violations`,
  `print_failure_analysis()`) rather than asserting on it silently,
  so a regression is visible in both the CLI report and the committed
  unit tests (`tests/unit/test_answer_quality_core.py`), not just
  discovered by chance.

### "Insufficient evidence" only ever means "nothing was retrieved"

`GenerateAnswerService.execute()` returns
`is_insufficient_evidence=True` only when `search_results` is empty -
there is no similarity-score threshold anywhere in
`RetrieveChunksService`/`SearchChunksService`/`VectorStore`. A
semantically off-topic question against a non-empty index still
retrieves its top-`k` nearest chunks and is answered from them; it does
not become "insufficient evidence". This is a pre-existing, unchanged
scope boundary of the retrieval/generation design, not something this
issue introduces or fixes.

Practically, this means `expected_insufficient_evidence: true` is only
a meaningful ground-truth label for a case run against an **empty**
index (see `tests/unit/test_answer_quality_core.py`'s
`test_evaluate_answer_case_skips_citation_metrics_when_insufficient_evidence_expected`).
`tests/support/evaluation/qa_dataset.py::ANSWER_EVALUATION_CASES`
(all 8 cases, real questions against the populated `SAMPLE_PAGES`
index) therefore has no such case; `insufficient_evidence_accuracy`
is still computed and reported for every dataset, in case a future
dataset does include one.

### Answer-point coverage is lexical, not semantic

`answer_point_coverage()` (`tests/support/evaluation/metrics.py`)
checks whether each `expected_answer_points` string appears as a
case-insensitive substring of the generated answer. It has a known,
accepted limitation: it cannot detect a correct answer that paraphrases
or uses a synonym instead of the exact expected substring (e.g. "check
for blood-thinner interactions" would not match an expected point of
"anticoagulant"). This is the same trade-off retrieval evaluation
already makes with Recall/MRR against a curated page list rather than a
semantic judgment of relevance - accepted for the same reason
(reproducibility over completeness).

`FakeLlm`-based evaluation (`tests/unit/test_answer_quality_core.py`)
sidesteps this limitation entirely by scripting the fake answer to
literally contain the expected points - it verifies the **evaluation
harness's own logic** (aggregation, dataset parsing, consistency
checking), not real answer quality. Only the opt-in, real-OpenAI test
(`tests/integration/test_live_answer_quality_evaluation.py`) measures
coverage against a genuinely model-generated answer, and even there,
coverage is only printed, not asserted on (see next section) - the
real-LLM test's hard assertions are limited to the deterministic,
retrieval-driven metrics (citation recall, citation consistency).

### Two-tier testing: committed Fake-LLM gate + opt-in live measurement

| | `tests/unit/test_answer_quality_core.py` | `tests/integration/test_live_answer_quality_evaluation.py` |
|---|---|---|
| Runs by default (CI) | Yes | No (skipped) |
| Requires network / API key | No | Yes (`RUN_SLOW_TESTS=1` + `MEDICAL_RAG_LLM_API_KEY`) |
| LLM | `FakeLlm` (scripted, deterministic) | Real `OpenaiLlm` |
| Embeddings | Fake stand-in model (like `test_retrieval_baseline_core.py`) | Real `sentence-transformers` |
| What it actually verifies | The evaluation harness's own logic is correct | Citation recall/consistency against a real model's behavior; answer-point coverage is printed, not gated |
| Cost | Free | Billable |

This mirrors the existing precedent for retrieval evaluation
(`tests/integration/test_retrieval_evaluation.py`, gated on
`RUN_SLOW_TESTS` only, and `tests/integration/test_live_openai_llm.py`,
gated on a real API key) and directly satisfies Issue #10's
requirement that a default evaluation run needs no external service.

### Shared library structure (`scripts/evaluation_common.py`)

`scripts/retrieval_baseline_core.py`'s dataset-loading primitives
(`DatasetDocument`, `DatasetCase`, `location_key()`, `truncate_text()`,
`load_dataset()`, `resolve_report_path()`) were evaluation-type-agnostic
already; they have been extracted into `scripts/evaluation_common.py`
so `scripts/answer_quality_core.py` (this issue) and any future
evaluation tool (e.g. an LLM-as-a-Judge pass) can reuse them without
depending on retrieval-specific code (`evaluate_configuration()`,
`CaseResult`, `Aggregate`, Recall/MRR reporting), which stays in
`retrieval_baseline_core.py`.

`DatasetCase` gained two new, optional fields for this issue:
`expected_answer_points: list[str]` (defaults to `[]`) and
`expected_insufficient_evidence: bool` (defaults to `False`). Every
existing retrieval-only dataset file (with no such keys) parses
unchanged.

`retrieval_baseline_core.py` re-exports every name it previously
defined itself from `scripts.evaluation_common`, so none of its 10
existing call sites (`scripts/compare_chunk_sizes.py`,
`scripts/compare_pdf_extractors.py`,
`scripts/compare_reranking_strategies.py`,
`scripts/compare_retrieval_strategies.py`,
`scripts/compare_chunking_strategies.py`,
`scripts/chunking_comparison_core.py`,
`scripts/hybrid_retrieval_core.py`,
`scripts/pdf_extraction_comparison_core.py`,
`scripts/reranking_core.py`,
`scripts/evaluate_retrieval_baseline.py`, plus their unit tests) needed
to change. This was verified by running the existing test suite
unchanged before and after the extraction.

### Why `evaluate_answer_case()` composes RetrieveChunksService and GenerateAnswerService directly instead of calling AskQuestionService

`scripts/answer_quality_core.py::evaluate_answer_case()` calls
`RetrieveChunksService.execute()` then
`GenerateAnswerService.execute()` itself, rather than calling the
existing `AskQuestionService.execute()` (which composes exactly those
same two services). This is not a reimplementation of retrieval or
generation logic - both calls are the same two existing Application
services, in the same order, that `AskQuestionService` itself calls -
it is done only so this module can also see the intermediate
`search_results` (needed for `citations_are_subset_of_retrieved()`),
which `AskQuestionService.execute()` does not expose in its return
value (`GenerationResult` only, by design - see
`docs/adr/0009-generation-strategy.md`).

## Consequences

- No production code (`app/domain`, `app/application`,
  `app/infrastructure`, `app/api`) changed; this issue is evaluation
  tooling only, matching the pattern of Issues #24/#25/#29/#30.
- `data/eval/guideline_qa.json`'s existing, real, gitignored cases
  continue to parse unchanged; `expected_answer_points` and
  `expected_insufficient_evidence` can be added to them incrementally
  and locally, never committed (see `docs/evaluation-dataset-format.md`).
- A future LLM-as-a-Judge evaluation (explicitly out of scope for
  Issue #10) could be added as a third `scripts/*_core.py` module
  reusing `scripts/evaluation_common.py`'s dataset loading, without
  touching this issue's deterministic metrics or their tests.
