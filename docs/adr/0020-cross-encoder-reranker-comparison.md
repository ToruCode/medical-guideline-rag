# 0020. Compare Hybrid Search with and without Cross Encoder reranking
via a comparison-only pipeline

## Status

Accepted

## Context

Issue #24 (`docs/adr/0019-hybrid-search-comparison.md`) compared Dense-
only vs Hybrid (Dense+BM25) retrieval and found Hybrid (alpha=0.7)
improved Recall@3/MRR@5 over Dense with zero regressions on a real
guideline dataset. Issue #25 tests a further improvement candidate:
reranking Hybrid's retrieved candidates with a Cross Encoder - a model
that scores a (query, chunk_text) pair jointly (unlike Dense/BM25,
which score query and chunk independently and only compare the results
afterward) - to see whether it improves Recall@1/MRR@5 specifically.

## Decision

- **Everything new lives under `scripts/`**, following Issue #24's
  precedent: `app/domain`, `app/application`, `app/infrastructure`, and
  `app/api/dependencies.py` remain untouched, and production retrieval
  is not changed. `scripts/compare_retrieval_strategies.py` (Issue #24)
  is also untouched; a new CLI,
  `scripts/compare_reranking_strategies.py`, adds "hybrid_rerank"
  alongside "dense"/"hybrid" (the latter two reusing
  `hybrid_retrieval_core.evaluate_strategy()` exactly as Issue #24 left
  it).

### The one exception: extracting `rank_hybrid_candidates()`

Requirement: rerank *before* Hybrid's final `top_k` cut (passing only 5
already-ranked candidates to a reranker leaves it little room to
change the outcome). `hybrid_search()` (Issue #24) computed the
Dense/BM25 union and fused score, then cut to `final_top_k`, all in one
function - there was no way to get the pre-cut candidate set out.

`scripts/hybrid_retrieval_core.py` gains one new function,
`rank_hybrid_candidates()`, extracted from `hybrid_search()`'s body
with **no change to its math or output**: it returns the *entire*
Dense/BM25 union, scored and fused, descending by fused score (still
tie-broken by ascending `chunk_id`). `hybrid_search()` becomes a thin
wrapper: `rank_hybrid_candidates(...)[:final_top_k]`, mapped to
`RankedChunkResult`. This is verified behavior-preserving by Issue #24's
full existing test suite continuing to pass unchanged (including the
`alpha=1.0` exact-Dense-match test), plus new regression tests in
`tests/unit/test_reranking_core.py`
(`test_dense_search_still_matches_search_chunks_service`,
`test_hybrid_search_unaffected_by_reranking_core_additions`).

A new `HybridCandidate` dataclass (`chunk: Chunk`, `dense_score`,
`bm25_score`, `hybrid_score`) is returned instead of `RankedChunkResult`:
`RankedChunkResult.text_preview` is already truncated (see
`retrieval_baseline_core.truncate_text`), but the reranker needs the
**full, untruncated chunk text** (requirement 9) - `HybridCandidate`
carries the whole `Chunk`, so `scripts/reranking_core.py` reads
`candidate.chunk.text` directly rather than a truncated preview.
`hybrid_search()` still only ever produces `RankedChunkResult` - its
public contract is unchanged.

### Cross Encoder model candidates

| | cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 | BAAI/bge-reranker-v2-m3 | hotchpotch/japanese-reranker-cross-encoder-small-v1 |
|---|---|---|---|
| Language | Multilingual (mMARCO, ~100 languages incl. Japanese) | Multilingual (100+ languages incl. Japanese) | Japanese-specific |
| Size | Small (MiniLMv2, 12 layers / 384 hidden - lightweight transformer, larger multilingual vocab/embedding table) | Large (XLM-RoBERTa-large based, several hundred million parameters) | Small (BERT-small class) |
| CPU feasibility | Good, practical speed | Feasible but noticeably heavier | Good, fast |
| Inference speed | Fastest of this comparison | Moderate-to-slow on CPU | Fast |
| License | Apache-2.0 | MIT | Per model card (verify at adoption time) |
| sentence-transformers integration | Native (published under the `cross-encoder` org for `CrossEncoder` use) | Works via `CrossEncoder` (standard HF sequence-classification model) | Works via `CrossEncoder` |
| Windows reproducibility | Good (no native extension) | Good, but a larger download | Good |

**Adopted: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`**, as the
default `--reranker-model-name` (freely overridable per run). It best
matches this issue's stated priorities: multilingual (Japanese-capable),
CPU-friendly, a clear permissive license, and native
`sentence-transformers` `CrossEncoder` support. `bge-reranker-v2-m3` is
a plausible higher-quality alternative but is heavier on CPU and a
larger download; it remains available by passing
`--reranker-model-name BAAI/bge-reranker-v2-m3`.

### Reranking design

- **Candidate/rerank separation** (requirement 8): candidates for
  reranking are Hybrid's full Dense/BM25 union
  (`rank_hybrid_candidates()`), cut to `--reranker-candidate-k` (default
  20) - *not* Hybrid's `final_top_k` (5). The Cross Encoder then
  rescoring those 20 candidates and the final cut to `--top-k` (5)
  happens only after reranking.
- **Reranker input** (requirement 9): `(query, chunk.text)` pairs only -
  no page number, chunk index, or other metadata is passed to the
  model.
- **No score blending** (requirement 10): the Cross Encoder's raw output
  score is used directly to rank candidates. It is not combined with
  Dense/BM25/Hybrid scores - this issue is deliberately a simple
  "retrieve, then separately rerank" comparison, not a second fusion
  step. (Should reranker+Hybrid score blending prove interesting later,
  it is a natural follow-up, but out of scope here.)
- **Deterministic ties**: Python's `sorted()` is stable, and candidates
  are handed to the reranker already in Hybrid rank order, so tied
  reranker scores keep each candidate's original Hybrid rank without
  any extra tie-break key.
- **Model loaded once** (requirement 21): `CrossEncoderReranker` wraps
  an already-loaded model (`scripts/reranker.load_cross_encoder()`,
  called once by the CLI); `evaluate_hybrid_rerank()` reuses the same
  instance for every question.
- **`Reranker` Protocol** (`scripts/reranker.py`, requirement 22):
  `score(self, query: str, texts: list[str]) -> list[float]`.
  `CrossEncoderReranker` is the only production-shaped implementation
  this issue adds; `FakeReranker` (test-only, dependency-free) lets
  every reranking-logic test run without a real model. The Protocol
  boundary means a future API-based reranker (Cohere Rerank, a hosted
  BGE endpoint, etc.) is a second `Reranker` implementation away, with
  no change to `scripts/reranking_core.py`.
- **`--device`** defaults to `cpu` (Windows+CPU reproducibility is this
  issue's stated priority); pass `cuda` explicitly to use a GPU, or
  `auto` to use CUDA only if `torch.cuda.is_available()`.
  **`--batch-size`** defaults to 8, a size unlikely to exhaust memory on
  a typical CPU-only machine for `reranker_candidate_k=20`-sized
  batches.

### Latency measurement (requirement 15)

`hybrid_rerank_search()` measures **retrieval latency** (time inside
`rank_hybrid_candidates()`) and **reranking latency** (time inside
`reranker.score()`) separately per question, summed to
`total_latency_ms`; the comparison table's `avg_latency_ms` column
reports each strategy's average total latency per question. For
"dense"/"hybrid" (which have no reranking step and where
`hybrid_retrieval_core.evaluate_strategy()` is intentionally left
unmodified), the CLI instead times the whole `evaluate_strategy()` call
once and divides by the case count - a simpler, whole-run measurement,
per requirement 15's explicit allowance to report total latency only
where a finer breakdown would add complexity for no real benefit (there
is no separate "reranking" phase for these two strategies).

### Per-question categorization (requirement 18)

`scripts/reranking_core.compare_hybrid_to_rerank()` classifies every
question into exactly one of: `hybrid_correct_rerank_worse` (was found,
reranking pushed it out of the top-k entirely),
`hybrid_incorrect_rerank_better` (was missing, reranking found it),
`rank_improved` / `rank_worsened` (found by both, rank changed), or
`unchanged_correct` / `unchanged_missing` (no change either way).
`summarize_rerank_comparison()` counts these across the whole dataset;
`print_rerank_comparison_summary()` prints the counts (never the
underlying question text).

### `dense_rerank`: design note only, not implemented (requirement 6)

A `dense_rerank` strategy (rerank Dense's own top
`reranker_candidate_k` candidates, instead of Hybrid's) would need only
a `dense_candidate_k`-sized, pre-cut candidate list analogous to
`rank_hybrid_candidates()` - e.g. `corpus.vector_store.search(query_vector,
top_k=dense_candidate_k)` directly, since Dense's candidates already
carry the full `Chunk` via `SearchResult.embedded_chunk.chunk`, unlike
`RankedChunkResult`. The reranking step itself
(`hybrid_rerank_search()`'s reranker-scoring/sort/cut logic) would be
identical. This issue does not implement or test it - only "dense",
"hybrid", and "hybrid_rerank" are required (requirement 6) - but the
existing design leaves it a small, low-risk addition for a future
issue.

### CLI (`scripts/compare_reranking_strategies.py`)

- `--strategies dense,hybrid,hybrid_rerank` (default: all three).
  `--dataset --chunk-size --chunk-overlap --top-k --dense-candidate-k
  --bm25-candidate-k --reranker-candidate-k --alpha
  --embedding-model-name --reranker-model-name --device --batch-size
  --verbose --save-report`, matching requirement 11/12/13.
- Output: a comparison table (`strategy | alpha | reranker | Recall@1 |
  Recall@3 | Recall@5 | MRR@5 | avg_latency_ms`, requirement 14) and a
  matching Markdown table for
  `docs/cross-encoder-reranker-comparison-results.md`. When both
  "hybrid" and "hybrid_rerank" are evaluated, the per-question
  categorization breakdown (above) is also printed. `--verbose`
  additionally prints, per strategy and per question: expected/
  retrieved pages, and for `hybrid_rerank` specifically,
  rank-before/rank-after-rerank plus dense/bm25/hybrid/reranker score
  and `text_preview` (local use only, never committed; requirement 16).
  `--save-report` writes full per-strategy, per-question detail (plus
  the categorization counts) to `data/eval/results/` (already
  gitignored in full).
- **Not a pytest test**, for the same reason as
  `compare_retrieval_strategies.py`: the real PDF and dataset exist only
  on the operator's machine and are never committed.
- **New unit tests use only hand-constructed `IndexedCorpus` fixtures
  and `FakeReranker`/test-double rerankers** - no real guideline PDF and
  no real Cross Encoder model:
  `tests/unit/test_reranker.py` (`CrossEncoderReranker`'s
  pairing/batching against a stub model object, `FakeReranker`,
  `resolve_device()`), `tests/unit/test_reranking_core.py` (candidate/
  final_top_k enforcement, descending sort, deterministic ties, empty/
  fewer-than-candidate_k candidates, rank-improvement/regression via
  `FakeReranker`, and the Dense/Hybrid-unchanged regression checks), and
  `tests/unit/test_compare_reranking_strategies.py` (CLI arg parsing,
  Markdown table formatting).
- **A real-model smoke test**
  (`tests/integration/test_live_cross_encoder_reranker.py`) follows this
  project's existing convention for expensive, network-dependent tests
  (see `tests/integration/test_live_sentence_transformer_embedder.py`):
  `pytestmark = pytest.mark.skipif(not os.environ.get("RUN_SLOW_TESTS"),
  ...)`, so a normal `pytest` run skips it automatically (requirement
  20). This project has no custom pytest marker configuration - every
  existing real-model test uses this same `RUN_SLOW_TESTS` `skipif`
  pattern, so this issue follows it rather than introducing a new
  mechanism.

## Consequences

- Production retrieval is unchanged by this issue; adopting reranking
  in production would need its own follow-up issue, informed by what
  this comparison measures on a real guideline.
- `docs/cross-encoder-reranker-comparison-results.md` starts empty,
  exactly like Issue #24's results doc did: recording a real comparison
  requires a human, with a real guideline document and dataset, to run
  `compare_reranking_strategies.py` locally and choose to paste its
  (reviewed) output.
- Reranking adds real inference cost (a full forward pass per candidate
  pair, `reranker_candidate_k` times per question) - the latency columns
  make that cost visible alongside any Recall/MRR gain, rather than only
  reporting accuracy.
- Adding `dense_rerank`, an API-based reranker, or blending the reranker
  score with Hybrid's score are all natural follow-ups the current
  design does not block, but none are implemented here.
