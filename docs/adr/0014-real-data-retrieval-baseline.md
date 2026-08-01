# 0014. Measure a retrieval baseline against a real guideline, without committing it

## Status

Accepted

## Context

Issue #17 (`docs/adr/0013-retrieval-evaluation.md`) built Recall@k/MRR
metrics and a synthetic, committed, CI-reproducible evaluation gate.
Issue #18 asks a different question: what is Recall@1/Recall@3/
Recall@5/MRR of the *current* retrieval configuration against a real
Japanese medical guideline? This is a one-off (or occasionally
repeated) baseline *measurement*, not a regression gate, and it
necessarily involves a real guideline document - which `CLAUDE.md`'s
data/copyright rules forbid committing, in either PDF or extracted-text
form. This issue is measurement only; no retrieval improvements are
made here.

## Decision

- **The real PDF, the real dataset (questions + expected pages/
  chunks), and any per-question results are never committed.** A new
  gitignored directory, `data/eval/` (alongside the existing
  `data/raw/`), holds the local dataset file(s) and
  `scripts/evaluate_retrieval_baseline.py --save-report` output. Only
  three things are committed: the measurement tool itself
  (`scripts/evaluate_retrieval_baseline.py`), documentation of the
  dataset format (`docs/evaluation-dataset-format.md`, fictional
  examples only), and a place to record **aggregate** results
  (`docs/baseline-retrieval-evaluation.md`).
- **Even bibliographic details (title, publisher, edition) are kept
  out of committed docs by default**, per explicit user instruction
  when this issue's plan was approved. The dataset's
  `document.label` field (e.g. `"Guideline A"`) is what the script
  prints in its Markdown snippet; it is the operator's responsibility
  to keep that label anonymized if they intend to paste the snippet
  into `docs/baseline-retrieval-evaluation.md`. `CLAUDE.md` itself
  does not forbid citing a title, but this project chooses the more
  conservative default anyway.
- **A standalone script (`scripts/`), not a pytest test** - a
  deliberate departure from Issue #17's precedent
  (`docs/adr/0012-live-e2e-verification.md`'s and
  `docs/adr/0013-retrieval-evaluation.md`'s "no manual CLI script"
  decisions). Those precedents apply to gates with a fixed,
  committed, universally-reproducible dataset and a pass/fail
  threshold; neither holds here. A real dataset exists only on one
  operator's machine, cannot be reproduced by another developer or
  CI, and this issue explicitly wants a *reported number*, not a
  pass/fail signal. Modeling that as a pytest test would misuse the
  framework's pass/fail semantics for what is actually a reporting
  tool - exactly what `scripts/` (already documented in `README.md`
  as "development and operational scripts") exists for.
- **Ground truth is page-number-based by default, chunk-index-based
  per case when needed** (`granularity: "page" | "chunk"` in
  `docs/evaluation-dataset-format.md`). Real Japanese guideline pages
  are likely to exceed `chunk_size` (1000 chars) and split into
  multiple chunks, unlike Issue #17's short synthetic sentences: a
  purely page-level match (as Issue #17 uses throughout) stays robust
  to exactly where those chunk boundaries fall, while an optional
  chunk-level override lets an evaluator pin down a specific chunk on
  a multi-chunk page when that precision matters. Both granularities
  reduce to a single string identifier
  (`"{page_number}"` or `"{page_number}:{chunk_index}"`), so
  `tests/support/evaluation/metrics.py`'s `recall_at_k`/
  `reciprocal_rank`/`mean` (Issue #17) are reused completely
  unchanged.
- **`scripts/evaluate_retrieval_baseline.py` imports
  `tests.support.evaluation.metrics`.** Both `scripts/` and `tests/`
  are excluded from the packaged wheel (`pyproject.toml`'s
  `packages = ["app"]`), so this is not the same concern
  `docs/adr/0010-fastapi-rag-api.md` raised about `app/` depending on
  `tests.support` (production code accidentally depending on
  test-only code that could vanish from a built artifact); it is one
  dev-only tool reusing pure functions from another. Duplicating
  `recall_at_k`/`reciprocal_rank`/`mean` instead was rejected as
  needless drift between two copies of the same three functions.
- **Configuration is captured from the running `Settings`, not
  hardcoded**: `chunk_size`, `chunk_overlap`, and
  `embedding_model_name` come from `get_settings()` at run time (so
  the report always reflects whatever `.env` currently configures,
  matching the issue's "現状構成" framing), while the query/passage
  prefixes (`"query: "`/`"passage: "`) are recorded as constants
  matching what `app/api/dependencies.py` actually uses for the
  `sentence_transformers` provider. The script refuses to run unless
  `MEDICAL_RAG_EMBEDDING_PROVIDER=sentence_transformers`, since
  `FakeEmbedder`'s vectors (derived from text length only) would make
  any measured Recall/MRR meaningless.
- **`--top-k` must be `>= 5`**: Recall@1/3/5 are all computed from one
  retrieval call per question (results sliced to k=1/3/5), not three
  separate calls, so the single retrieval depth must cover the widest
  metric. MRR is therefore effectively "MRR@top_k" (bounded by however
  deep the retrieval went), not a true infinite-depth MRR; the printed
  report and the Markdown snippet both label it `MRR@{top_k}` to make
  that explicit.
- **Three-tier output**: (1) a per-question console report (question
  text, expected vs. retrieved locations, per-case scores) - local use
  only, never meant to be pasted anywhere; (2) an aggregate console
  summary; (3) a ready-to-review Markdown snippet (aggregate numbers
  and run configuration only, using the dataset's anonymized
  `document.label`) for manually pasting into
  `docs/baseline-retrieval-evaluation.md` after a human checks it for
  anything identifying. Nothing is auto-written to any committed file;
  a human review step is mandatory between "the script ran" and
  "something is committed". `--save-report` additionally writes tiers
  (1)+(2)+config as JSON under `data/eval/results/` (gitignored) for
  the operator's own record-keeping across repeated local runs.

## Consequences

- `docs/baseline-retrieval-evaluation.md` starts empty (no baseline
  recorded yet): recording an actual number requires a human, with a
  real guideline document and a real dataset file, to run the script
  locally and choose to paste its output. This ADR and the rest of
  this issue's deliverable are the tooling and process only.
- Because ground truth can be page-level, a "hit" does not always mean
  the *exact* relevant sentence was retrieved - only that a chunk from
  the right page was. This is an intentional simplification for a
  first baseline (`granularity: "chunk"` is available per-case when
  finer precision is needed) and should be kept in mind when
  interpreting Recall@k numbers here versus a stricter, chunk-exact
  evaluation.
- If a future issue adds a real vector database adapter (Qdrant,
  pgvector), `scripts/evaluate_retrieval_baseline.py` only needs its
  `InMemoryVectorStore()` construction swapped for that adapter;
  nothing else in this design depends on which `VectorStore`
  implementation is used, mirroring the same consequence already noted
  in `docs/adr/0013-retrieval-evaluation.md`.
- If `EVALUATION_CASES`-style dataset growth or a chunking config
  change ever invalidates `granularity: "chunk"` entries, there is no
  automated detection (unlike Issue #17's `chunk_count ==
  len(SAMPLE_PAGES)` sanity assertion, which only applies to that
  issue's own synthetic dataset) - re-verifying chunk-level entries
  after a config change is a manual responsibility documented in
  `docs/evaluation-dataset-format.md`.
