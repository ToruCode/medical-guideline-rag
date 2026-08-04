# 0025. Evaluation dashboard

## Status

Accepted

## Context

`docs/adr/0023-answer-quality-and-citation-consistency-evaluation.md`
added `scripts/evaluate_answer_quality.py`, which measures citation
precision/recall, answer-point coverage, insufficient-evidence
accuracy, and latency against a local dataset and can save a detailed
per-question JSON report (`--save-report`, written by
`scripts/answer_quality_core.py::write_local_report`). Inspecting that
report has so far meant reading raw JSON or the tool's own CLI
printout. Issue #15 adds a Streamlit dashboard to browse one or more
saved reports interactively, without re-running any evaluation or
duplicating its metrics.

## Decision

- **Presentation/logic split mirrors `app/ui/streamlit_app.py`**
  (Issue #13, `docs/adr/0024-streamlit-demo-ui.md`): the only
  Streamlit file is `app/ui/evaluation_dashboard.py` (rendering and
  control flow only), backed by two new, Streamlit-free modules under
  `scripts/` rather than `app/ui/`, since this is developer-only
  tooling (report inspection), not part of the production
  API/Application/Domain/Infrastructure stack that `app/` otherwise
  contains - the same reasoning that already put
  `scripts/answer_quality_core.py` outside `app/`.
  - **`scripts/evaluation_report_loader.py`** parses a report JSON file
    back into the *existing* `AnswerRunConfig`/`AnswerAggregate`/
    `AnswerCaseResult`/`AnswerConfigurationRun` dataclasses from
    `scripts/answer_quality_core.py`, rather than defining a second,
    parallel schema - "reuse the existing evaluation report format",
    literally.
  - **`scripts/evaluation_dashboard_core.py`** provides filtering
    (`filter_case_results`, by failure/category/difficulty/question
    substring) and two-report metric comparison
    (`compare_aggregates`), operating only on those same dataclasses.
- **`data/eval/results/` is shared by several unrelated tools**
  (`scripts/retrieval_baseline_core.py`'s Recall/MRR reports, and the
  chunk-size/PDF-extraction/reranking comparison tools), all of which
  happen to serialize the same outer
  `{"document_label", "runs": [{"config", "aggregate", "cases"}]}`
  envelope as `scripts/answer_quality_core.py`, but with completely
  different `cases[]`/`config`/`aggregate` fields (no LLM is involved,
  so no `citation_precision`, `answer_point_coverage`, etc.).
  `evaluation_report_loader.load_report()` checks that every case
  contains the answer-quality-specific keys
  (`citation_precision`, `answer_point_coverage`,
  `insufficient_evidence_correct`, `citations_consistent`) and raises
  `ReportFormatError` otherwise;
  `load_answer_quality_reports()` uses this to silently skip
  non-matching files when scanning a directory, so the dashboard can
  point at `data/eval/results/` as-is without the user first sorting
  reports by which tool produced them.
- **`category`/`difficulty` are new, optional, per-case dataset fields**
  (`docs/evaluation-dataset-format.md`), added the same way Issue #10
  added `expected_answer_points`/`expected_insufficient_evidence`:
  default to `None`/absent, ignored by every metric and by every
  existing dataset file. They exist solely so
  `filter_case_results()` has something free-form to filter on,
  satisfying Issue #15's "filter by category/question type/difficulty"
  requirement without inventing a fixed taxonomy or a new metric.
  "Question type" was not added as a third field distinct from
  `category` - a dataset author can already express that distinction
  using `category` itself (e.g. `"category": "dosage-lookup"` vs.
  `"category": "contraindication-lookup"`), and adding a fourth
  free-form metadata field for what is conceptually the same kind of
  grouping was judged unnecessary duplication.
- **Failure criteria are extracted, not reimplemented.**
  `scripts/answer_quality_core.py::print_failure_analysis` previously
  computed its four failure conditions (citation-consistency
  violation, insufficient-evidence mismatch, `citation_recall < 1.0`,
  `answer_point_coverage < 1.0`) inline. This issue extracts that into
  `failure_reasons()`/`is_failure_case()` (same module,
  behavior-preserving - `print_failure_analysis`'s own output and
  existing tests are unchanged), so the dashboard's "failures only"
  filter and failure-analysis view use the exact same definition of
  failure as the CLI tool, per Issue #15's "no evaluation logic should
  be duplicated".
- **Comparison is aggregate-metric-only, not per-question.** For each
  of the six aggregate metrics (citation precision/recall,
  answer-point coverage, insufficient-evidence accuracy, mean latency,
  citation-consistency violations), `compare_aggregates()` classifies
  run B relative to run A as `"improved"`/`"degraded"`/`"unchanged"`
  (within a `1e-9` epsilon, to absorb floating-point noise, not to
  hide a real small change) or `"unavailable"` when either run has no
  value for a mean-of-optional metric (e.g. no case defined
  `expected_answer_points`). Direction-awareness
  (`_METRIC_SPECS`'s `higher_is_better` flag) means latency and
  consistency-violation *decreases* are correctly reported as
  "improved", not "degraded". A per-question diff (matching cases by
  question text across two reports) was considered and left out of
  this issue's scope - the aggregate view already satisfies "highlight
  improved/degraded/unchanged metrics", and a good per-question diff
  UI (handling added/removed/reworded questions between two dataset
  versions) is a large enough feature to design separately if a real
  need for it appears.
- **`answer_preview` (already truncated to 300 characters by
  `scripts/evaluation_common.py::truncate_text`, not the full generated
  answer or any extracted PDF text) is shown only inside a per-case
  expander in the failure-analysis view**, labeled as local-only/never
  to be committed - consistent with the CLI tool already printing the
  same field locally (`print_case_report`). Full extracted PDF content,
  prompts, and API keys are never read or displayed by this dashboard
  at all - it has no access to them (it only ever reads a saved JSON
  report, never a PDF, a prompt, or `Settings.llm_api_key`).

## Consequences

- The dashboard is read-only: it cannot regenerate, edit, or delete a
  report, matching Issue #15's "do not regenerate evaluation results"
  and "do not modify evaluation results" constraints by construction,
  not by a runtime check.
- Because `category`/`difficulty` are optional and unused by any
  existing dataset, every dataset file written before this issue keeps
  working unchanged, and its cases simply have nothing for those two
  filters to narrow down (`available_categories`/`available_difficulties`
  return an empty list).
- Filtering and comparison are pure functions over already-loaded,
  in-memory dataclasses - no new dependency on `pandas`/a real
  database was needed (`streamlit`'s own transitive `pandas` dependency,
  already present since Issue #13, is used only for `st.dataframe`'s
  table rendering, not by any of this issue's own code).
