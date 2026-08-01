# 0015. Compare chunk_size candidates against the same real dataset

## Status

Accepted

## Context

Issue #18 (`docs/adr/0014-real-data-retrieval-baseline.md`) measured
Recall@1/3/5/MRR for whatever `chunk_size`/`chunk_overlap` the current
`.env` happened to configure. Issue #19 asks a comparative question:
across `chunk_size` in `{300, 500, 700, 1000, 1500}` (with
`chunk_overlap=200`, `top_k=5`, and
`intfloat/multilingual-e5-base` all held fixed), which `chunk_size`
retrieves best against the same real, local dataset? This issue is
measurement/comparison only - no retrieval or chunking code changes,
no default `chunk_size` change.

## Decision

- **`scripts/retrieval_baseline_core.py`** is extracted from Issue
  #18's `evaluate_retrieval_baseline.py`, which previously inlined
  dataset loading, indexing, per-question evaluation, aggregation, and
  report formatting directly in `main()`. The extraction is a
  behavior-preserving refactor (same CLI, same output) that exists
  specifically to let Issue #19 reuse this logic without duplicating
  it: `evaluate_configuration(document, cases, model, *, chunk_size,
  chunk_overlap, top_k, embedding_model_name)` is the one function
  both scripts call to run a single configuration end to end (index →
  retrieve → score every case → aggregate). `chunk_size`/
  `chunk_overlap`/`top_k`/`embedding_model_name` are now explicit
  parameters rather than being read from `Settings` inside the
  function - `evaluate_retrieval_baseline.py`'s `main()` still sources
  them from `get_settings()` (unchanged behavior: "measure the
  current `.env` configuration"), while `compare_chunk_sizes.py`'s
  `main()` sources them from its own CLI arguments, independent of
  `.env`. This directly answers "how is `chunk_size` externally
  specified" - by making it a parameter of the shared core instead of
  an implicit `Settings` read.
- **The sentence-transformers model is loaded exactly once per
  `compare_chunk_sizes.py` run**, before the `chunk_size` loop, and
  passed into every `evaluate_configuration()` call - loading it is
  the expensive part (seconds, hundreds of MB), while chunking/
  embedding/indexing one already-small guideline PDF five times is
  comparatively cheap. Each call still builds a **fresh
  `InMemoryVectorStore`** internally, so one candidate's chunks can
  never leak into another's search results (verified in
  `tests/unit/test_retrieval_baseline_core.py`).
- **`compare_chunk_sizes.py` never reads `Settings`/`.env`** for
  `chunk_size`/`chunk_overlap`/`top_k`/`embedding_model_name` (unlike
  `evaluate_retrieval_baseline.py`, which deliberately does). The
  whole point of a comparison is a controlled, explicit sweep,
  independent of whatever a developer's local `.env` currently has
  configured; `--chunk-sizes` defaults to `300,500,700,1000,1500` and
  `--chunk-overlap`/`--top-k`/`--embedding-model-name` all default to
  this issue's fixed values but remain CLI-overridable for future
  reuse (e.g. comparing a different overlap later) without code
  changes.
- **Default output is the comparison table only; per-question detail
  requires `--verbose`.** Printing every question's rank for all five
  `chunk_size` candidates by default would be `5 × case_count` lines,
  drowning out the one thing this issue actually wants to compare.
  `--verbose` prints each candidate's full per-question breakdown
  (via the same `print_case_report()` Issue #18 already has, called
  once per candidate) for when that detail is needed.
- **Two-tier output, same review discipline as Issue #18**: a console
  comparison table (`chunk_size | chunks | Recall@1 | Recall@3 |
  Recall@5 | MRR@k`) plus a ready-to-review Markdown *table* (not a
  single-entry snippet like Issue #18's) for
  `docs/chunk-size-comparison.md` - aggregate numbers, run
  configuration, and the dataset's anonymized `document.label` only,
  printed for a human to check before pasting, never auto-written.
  `--save-report` writes every candidate's full per-question results
  into one local JSON file under `data/eval/results/` (gitignored),
  reusing `write_local_report()`'s `runs: list[ConfigurationRun]`
  shape (a one-element list for Issue #18's script, one element per
  `chunk_size` here).
- **A separate results doc (`docs/chunk-size-comparison.md`), not
  reusing `docs/baseline-retrieval-evaluation.md`.** A comparison
  table across five configurations is a structurally different
  artifact from a single-point-in-time baseline entry; keeping them in
  separate files avoids awkwardly overloading one Markdown format for
  two different reporting purposes. Both docs cross-link to each
  other.
- **New unit tests, now possible because the core logic is a plain,
  side-effect-free module**: `tests/unit/test_retrieval_baseline_core.py`
  covers dataset parsing/validation, `evaluate_case()`'s page- vs.
  chunk-granularity matching, `summarize()`, and
  `evaluate_configuration()`'s wiring (config fields, indexed page/
  chunk counts, per-call vector-store isolation) - all using a
  minimal fake `SentenceTransformer.encode()` stand-in (deterministic,
  text-length-based, the same spirit as `FakeEmbedder`) so no real
  model download or real guideline PDF is needed.
  `tests/unit/test_compare_chunk_sizes.py` covers `--chunk-sizes`
  parsing and the Markdown comparison table formatter. As with Issue
  #18, no automated test touches a real PDF or real dataset; this
  remains a manual, local-only measurement tool, not a CI gate.

## Consequences

- `evaluate_retrieval_baseline.py`'s public CLI is unchanged by this
  issue (same flags, same output); only its internals now delegate to
  `retrieval_baseline_core.py`. Existing usage instructions in
  `README.md`/`docs/baseline-retrieval-evaluation.md` remain valid
  as-is.
- `docs/chunk-size-comparison.md` starts empty, exactly like
  `docs/baseline-retrieval-evaluation.md` did after Issue #18:
  recording a real comparison requires a human, with a real guideline
  document and dataset, to run `compare_chunk_sizes.py` locally and
  choose to paste its output.
- Because ground truth can be page-level (`docs/evaluation-dataset-format.md`),
  a "hit" at a given `chunk_size` does not always mean the exact
  relevant sentence was retrieved, only that a chunk from the right
  page was - the same caveat already noted in
  `docs/adr/0014-real-data-retrieval-baseline.md`, worth re-reading
  when interpreting a comparison across `chunk_size` values (a larger
  `chunk_size` can look artificially better at page-level recall
  simply by packing more of a page into fewer, broader chunks).
  `granularity: "chunk"` cases are more sensitive to this and should
  be re-verified if `chunk_size`/`chunk_overlap` changes shift chunk
  boundaries.
- If a future issue acts on a comparison's findings (e.g. changing the
  default `chunk_size` in `Settings`), that is explicitly out of scope
  here - this issue only measures and reports.
