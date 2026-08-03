# 0021. Compare fixed-size vs table-aware chunking via a
comparison-only pipeline

## Status

Accepted

## Context

Issue #29's 30-question failure analysis (Dense/Hybrid/Hybrid+Rerank)
found no evaluation-label errors, but did find
`duplicate_or_near_duplicate_content` as the most common retrieval-
pipeline-side cause among the priority cases reviewed (Q11, Q29, Q30):
the same substance name, standard value, or term appears in several
distinct tables/sections of this guideline (e.g. a chemical-limits
table and a toxicity table both listing "アルミニウム" as their first
entry), and the existing fixed-size chunker
(`FixedSizeTextSplitter`) has no awareness of table structure - it can
split a table's title/column headers away from its data rows, or
produce a chunk containing little more than a few numbers with no
surrounding context to disambiguate which table they came from. Issue
#30 tests whether a rule-based "table-aware" chunker that keeps that
context attached improves Recall@1/MRR@5 over the existing chunker.

## Decision

- **Everything new lives under `scripts/`**, following the pattern of
  Issues #24-#29: `app/domain`, `app/application`, `app/infrastructure`,
  and `app/api/dependencies.py` remain untouched, and production
  chunking (`FixedSizeTextSplitter`) is not changed or replaced. A new
  CLI, `scripts/compare_chunking_strategies.py`, compares "fixed"
  (the existing chunker, called unmodified) against "table_aware"
  (`scripts/table_aware_chunking.py`, new in this issue) under the
  exact `hybrid_rerank` configuration Issue #29 recommended
  (`alpha=0.7`, `reranker_candidate_k=5`).

### Rule-based, not layout-based (requirement 4)

Table detection operates purely on the line-structured text PyMuPDF
already extracts (`app/infrastructure/pdf/pymupdf_extractor.py`) - it
does not read PDF coordinates, fonts, or cell/column geometry. This
keeps the comparison self-contained (no new PDF-parsing dependency) and
directly testable against synthetic text fixtures
(`tests/unit/test_table_aware_chunking.py`), at the cost of being tuned
to one specific, empirically observed extraction pattern in this
document: PyMuPDF tends to extract each table cell as its own line (a
table-title line, several short column-header lines, then many short
data-cell lines, in sequence). **This is a heuristic for that pattern,
not a general table recognizer** - it will both miss real tables that
extract differently and occasionally misclassify unusually terse prose
as a table.

### Detection heuristics (requirement 5)

`scripts/table_aware_chunking.py`'s `TableBlockDetector` combines:

- **Table title lines**: a line starting with 表/別表/補足表/付記表
  followed by a number (all four are literal patterns confirmed present
  in this guideline).
- **A run of table-row-like lines**: a line is "row-like" if it is
  short (≤20 characters - matching the one-cell-per-line extraction
  pattern) or has a high digit/symbol ratio. A run of at least 3
  consecutive row-like lines is treated as a table even without an
  explicit title line (untitled tables/continuations).
- **A preceding heading-like line**: the nearest short, non-`。`-
  terminated line immediately before the detected table becomes its
  `heading_context`.
- **Trailing annotation lines**: lines starting with 注/※/備考/＊
  immediately after the last row-like line are attached to the table
  block as `annotation_lines` (a small lookahead, stopping at the first
  line that isn't blank or annotation-shaped).
- **Header vs. data rows within a table**: the leading run of a table's
  rows that do *not* look numeric/symbol-heavy is treated as
  column-header lines (e.g. "グループ", "最大濃度（mg/L）"); the first
  row that does look numeric/symbol-heavy starts the actual data. This
  is also a heuristic and is documented as such in
  `table_aware_chunking.py`'s docstrings - it will misclassify a
  non-numeric data label (e.g. a row whose "value" is itself a chemical
  name) as a header row in some real tables.

### `table_max_chars` takes priority over `table_row_group_size`

Per explicit clarification during planning: a table's data rows are
grouped into chunks by adding one row at a time and cutting *before*
either limit would be exceeded - `table_max_chars` first, then
`table_row_group_size`. `table_row_group_size=20` is this comparison's
**initial value only** (not a value tuned to be optimal for this or any
PDF); the comparison report exposes enough detail to tell which limit
actually triggered each split (`split_by_max_chars_count` vs.
`split_by_row_group_size_count`), the per-chunk row-count distribution,
and how often header/title duplication pushed a chunk over
`table_max_chars` anyway (`exceeded_max_chars_after_header_count`), so
these defaults can be re-tuned from real measurements later.

### Component design (requirement 6)

`TextBlock` (plain prose lines) and `TableBlock` (`heading_context`,
`title`, `rows`, `annotation_lines`) are simple dataclasses;
`TableBlockDetector.detect()` turns one page's lines into an ordered
list of both. `TableAwareTextSplitter.split_page()` delegates every
`TextBlock` to the existing, unmodified `FixedSizeTextSplitter` (so
"fixed" and "table_aware" behave identically outside detected tables),
and builds one or more `TableAwareChunk`s per `TableBlock` (duplicating
heading/title/column-headers into every split part, attaching
annotation to the last part only). A separate `HeadingContext` class
was considered and rejected as unnecessary abstraction - a single
`heading_context: str | None` field is sufficient.

`TableAwareChunk` (text, `is_table_chunk`, `heading_context`,
`table_title`, `row_count`, `split_trigger`, `has_header_lines`,
`is_header_duplicate`, `exceeded_max_chars_after_header`) is a
comparison-only type, distinct from the production `Chunk` model, which
must not and does not gain these fields.
`scripts/chunking_comparison_core.py`'s `build_chunks_table_aware()`
still produces real `Chunk` instances for the retrieval pipeline to
consume unchanged, alongside a `chunk_id -> TableAwareChunk` side table
used only for statistics and `--verbose`/`--save-report` output.

### Scope limitation: still page-scoped (like the existing chunker)

Neither chunker merges content across a page boundary - table-aware
chunking here only reorganizes content *within* a page differently. A
table or answer that continues onto the next page (as Q28's real answer
does, spanning pages 26-27) is not bridged by this issue; `cross_page_
chunk_count` is reported as `0` for both strategies for this reason,
documented rather than silently omitted.

### Comparison methodology (requirement 9-11)

Retrieval config is fixed at Issue #29's recommendation
(`hybrid_rerank`, PyMuPDF, `chunk_size=1000`/`chunk_overlap=200` for
prose, `dense_candidate_k=20`, `bm25_candidate_k=20`, `alpha=0.7`,
`reranker_candidate_k=5`, `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`,
`device=cpu`, `batch_size=8`); only chunk production varies between
"fixed" and "table_aware". Chunk-production is fully decoupled from
corpus building
(`build_chunks_fixed()`/`build_chunks_table_aware()` vs.
`build_corpus_from_chunks()`), so both strategies share identical
embedding/indexing/Dense/BM25/Hybrid/reranking logic - only the input
chunk list differs.

### Per-question comparison and failure-cause tagging (requirements 13, 15)

`compare_fixed_to_table_aware()` classifies each question into exactly
one of `fixed_only_success` / `table_aware_only_success` /
`both_success` (further split into `rank_improved`/`rank_worsened`/
`rank_unchanged`) / `both_fail`.

`auto_tag_failure_causes()` is an explicitly **best-effort, mechanically
derived** tagger, not a semantic classifier: it only uses candidate-pool
membership and rerank rank movement (both fully recoverable from
`RerankedChunkResult.rank_before_rerank`/`rank_after_rerank`, because
Issue #30 fixes `reranker_candidate_k == final_top_k == 5` - nothing is
cut between reranking and the final result, so the final ranked list
*is* the reranker's full candidate pool) and, for `table_aware`,
whether the expected page's winning chunk is missing a table
title/heading/column-header. Semantic causes from Issue #29's taxonomy
(`duplicate_or_near_duplicate_content`, `query_document_vocabulary_gap`,
`row_split_across_chunks`, `unit_detached_from_value`,
`note_detached_from_table`, `ambiguous_question`) are **not**
auto-detected - reaching those conclusions requires the same kind of
manual, per-question deep-dive Issue #29's analysis did. Every tag this
function produces is a hypothesis to manually confirm, not a finding.

### CLI and reporting (requirements 8, 11, 12, 16)

`scripts/compare_chunking_strategies.py`: `--strategies fixed,table_aware`
(default both), `--dataset --chunk-size --chunk-overlap --table-max-chars
--table-row-group-size --top-k --dense-candidate-k --bm25-candidate-k
--reranker-candidate-k --alpha --embedding-model-name
--reranker-model-name --device --batch-size --verbose --save-report`.
Output: a comparison table (`strategy | chunks | avg_chars |
short_chunks | table_chunks | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
avg_latency_ms`) and a matching Markdown table for
`docs/table-aware-chunking-comparison-results.md`, plus the fixed-vs-
table_aware per-question breakdown when both strategies are evaluated.
`--verbose` additionally prints, per strategy and per question: rank,
Dense/BM25/Hybrid/reranker score, `chunk_index`, `is_table_chunk`,
`heading_context`, `table_title`, and `text_preview` (local use only,
never committed). `--save-report` writes chunk statistics, per-question
ranks, the fixed/table_aware diff, table-block detection statistics,
latency, and the best-effort failure tags to `data/eval/results/`
(already gitignored in full).

- **Not a pytest test**, for the same reason as every prior comparison
  script in this series: the real PDF and dataset exist only on the
  operator's machine and are never committed.
- **New unit tests use only self-authored, fictional synthetic text**
  (no real guideline PDF, no PyMuPDF extraction needed):
  `tests/unit/test_table_aware_chunking.py` (title/row-run detection,
  prose not misclassified, heading/annotation carried into table
  chunks, column-header duplication across split parts, `table_max_chars`
  enforcement, empty-page handling), `tests/unit/test_chunking_comparison_core.py`
  ("fixed" strategy byte-for-byte matches calling
  `FixedSizeTextSplitter`/`ChunkDocumentService` directly, page_number/
  chunk_index correctness, chunk-stats and per-question comparison
  aggregation), and `tests/unit/test_compare_chunking_strategies.py`
  (CLI arg parsing, Markdown table formatting).

## Consequences

- Production chunking is unchanged by this issue; adopting table-aware
  chunking in production would need its own follow-up issue, informed
  by what this comparison measures on the real guideline.
- `docs/table-aware-chunking-comparison-results.md` starts empty,
  exactly like every prior comparison results doc in this series:
  recording a real comparison requires a human, with the real guideline
  document and dataset, to run `compare_chunking_strategies.py` locally
  and choose to paste its (reviewed) output.
- The detection heuristics (line-length thresholds, digit/symbol
  ratios, the header/data row split, `table_row_group_size=20`) are
  tuned to this one guideline's PyMuPDF extraction pattern, not
  validated against other documents or extraction styles; treat this
  issue's results as a signal for *this* corpus, not a general claim
  about table-aware chunking.
- The auto-tagged failure causes in `--save-report` output cover only
  the mechanically-detectable subset of Issue #29's taxonomy; a full
  root-cause analysis (as Issue #29 did for Q11/Q29/Q30) still requires
  manual, per-question review before acting on any single tag.
- Cross-page tables/answers (e.g. Q28) remain out of scope; a future
  issue could explore merging chunks across a page boundary when a
  table or heading context is detected as continuing onto the next
  page.

## Outcome (2026-08-04)

Real-PDF comparison (`fixed` vs. `table_aware`) under the fixed
`hybrid_rerank` configuration (`chunk_size=1000`, `chunk_overlap=200`,
`table_max_chars=1000`, `table_row_group_size=20`, `top_k=5`,
`dense_candidate_k=20`, `bm25_candidate_k=20`, `alpha=0.7`,
`reranker_candidate_k=5`, `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`,
`device=cpu`, `batch_size=8`):

| strategy | Recall@1 | Recall@3 | Recall@5 | MRR@5 | avg_latency_ms |
|---|---:|---:|---:|---:|---:|
| fixed | 0.750 | 0.917 | 0.917 | 0.839 | 701.1 |
| table_aware | 0.733 | 0.867 | 0.900 | 0.801 | 665.4 |

`table_aware` underperformed `fixed` on every accuracy metric
(Recall@1 -0.017, Recall@3 -0.050, Recall@5 -0.017, MRR@5 -0.038),
while only modestly reducing latency (-35.7ms, ~5%) - not enough to
offset the accuracy regression.

**Decision: not adopted.** Production chunking remains the existing
`FixedSizeTextSplitter`. This issue is closed as a comparison result,
not a production change. See
`docs/table-aware-chunking-comparison-results.md` for the recorded
aggregate numbers (document title anonymized).
