# 0013. Evaluate retrieval quality with Recall@k and MRR against a fixed dataset

## Status

Accepted

## Context

Issues #1-#16 built and exposed the retrieval pipeline
(`RetrieveChunksService`, `SearchChunksService`, `InMemoryVectorStore`)
and the generation pipeline (`GenerateAnswerService`), but nothing
measures whether retrieval actually finds the right passages.
`docs/requirements.md`'s non-functional requirements explicitly call
for separating retrieval quality from generation quality; this issue
adds a first, minimal measurement of retrieval quality only - no LLM
involved.

## Decision

- **Two standard, well-understood metrics**: Recall@k (does a relevant
  chunk appear anywhere in the top k results?) and Mean Reciprocal
  Rank (how high does the first relevant chunk rank, averaged over all
  questions?). Both are implemented as pure functions in
  `tests/support/evaluation/metrics.py`
  (`reciprocal_rank`, `recall_at_k`, `mean`), with no framework or
  embedding-model dependency and no new third-party dependency (plain
  `sum`/`len`, not numpy) - consistent with CLAUDE.md's "do not add a
  dependency without explaining its purpose".
- **A fixed, self-authored evaluation dataset**
  (`tests/support/evaluation/qa_dataset.py`): `SAMPLE_PAGES`, eight
  short fictional sentences about a made-up drug ("Medicamentum X"),
  and `EVALUATION_CASES`, one `EvaluationCase(question,
  expected_page_numbers)` per sentence. This mirrors
  `tests/integration/test_live_rag_e2e.py`'s existing fictional-content
  pattern and CLAUDE.md's data/copyright rules: no real guideline or
  patient content, nothing committed as a file (the PDF is built at
  test-run time via `tests/support/pdf_factory.build_pdf`).
- **Page number as the ground-truth chunk identifier**, not a full
  `chunk_id`. Every `SAMPLE_PAGES` entry is far shorter than the
  default `chunk_size` (1000 chars), so each page becomes exactly one
  chunk (`chunk_index` 0); `Chunk.page_number` is therefore a
  sufficient, stable identifier for this dataset, and keeps
  `EvaluationCase` simple. `test_retrieval_evaluation.py` asserts
  `chunk_count == len(SAMPLE_PAGES)` right after indexing, so a future
  edit that pushes a sample sentence past `chunk_size` fails loudly
  instead of silently invalidating this assumption.
- **Retrieval only, no API layer, no LLM**: the evaluation test
  composes `IndexDocumentService`/`RetrieveChunksService` directly
  (the same Application-layer-only pattern as
  `tests/integration/test_retrieval_pipeline.py`), never going through
  `POST /documents/index`/`POST /questions/ask` or any `Llm`. This
  keeps the gate specific to retrieval quality, per
  `docs/requirements.md`.
- **Gated behind `RUN_SLOW_TESTS=1` only**, the same single condition
  as `tests/integration/test_live_sentence_transformer_embedder.py`
  (real model download/load, no billable API call needed since no
  `Llm` is involved). Never runs in default `pytest`/CI, matching
  `docs/adr/0012-live-e2e-verification.md`'s existing convention.
- **`MIN_RECALL_AT_3 = 0.8` and `MIN_MRR = 0.7` are provisional
  thresholds**, calibrated against the current 8-case
  `EVALUATION_CASES` only - not a fixed production SLA. Both are named
  constants at the top of `test_retrieval_evaluation.py` with a
  comment stating this explicitly, and the test prints a full
  per-case report (`recall@k`, reciprocal rank, expected vs. actual
  page numbers) on both pass and failure, so a human recalibrating
  them after growing the dataset has the data to do so without
  re-instrumenting anything.
- **No manual CLI script** (e.g. `scripts/evaluate_retrieval.py`) is
  added, for the same reason `docs/adr/0012-live-e2e-verification.md`
  rejected one for full-stack e2e verification: it would duplicate the
  same index-then-retrieve logic a pytest test already expresses, for
  one project. `pytest tests/integration/test_retrieval_evaluation.py
  -v -s` (with `RUN_SLOW_TESTS=1`) is the way to run it manually and
  see the printed report.

## Consequences

- This gate only measures retrieval quality under the real
  `sentence_transformers` embedder; it says nothing about `FakeEmbedder`
  (not semantically meaningful - vectors are derived from text length)
  or about generation/answer quality (out of scope by design).
- Growing `EVALUATION_CASES` (more questions, more pages, harder
  distractor content) is expected over time; `MIN_RECALL_AT_3`/`MIN_MRR`
  must be revisited whenever it does, by rerunning with `-s` and
  reading the new baseline from the printed report - they are not
  meant to stay fixed forever.
- If `MEDICAL_RAG_EMBEDDING_MODEL_NAME` is ever changed from
  `intfloat/multilingual-e5-base`, this gate should be rerun to confirm
  the new model still clears the thresholds (or to recalibrate them),
  the same caution `docs/adr/0012-live-e2e-verification.md` already
  notes for the full-stack live test.
- Extending this to a real vector database adapter (Qdrant, pgvector)
  once one exists only requires swapping `InMemoryVectorStore` for that
  adapter in `test_retrieval_evaluation.py`; nothing else in this
  design depends on which `VectorStore` implementation is used.
