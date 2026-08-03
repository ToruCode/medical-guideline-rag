# 0019. Compare Dense-only vs Hybrid (Dense+BM25) retrieval via a
comparison-only pipeline

## Status

Accepted

## Context

Issue #23 (`docs/adr/0018-adopt-pymupdf-for-production-pdf-extraction.md`)
adopted PyMuPDF as the production PDF extractor, substantially
improving Japanese retrieval quality (Recall@1: 0.583, Recall@3:
0.833, Recall@5: 0.917, MRR@5: 0.697). Production retrieval is
currently Dense-only (cosine similarity over sentence-transformers
embeddings). Medical guidelines contain drug names, abbreviations,
device names, numeric values, and chemical names where exact lexical
match matters and Dense embeddings can under-perform. Issue #24 adds a
comparison-only pipeline measuring whether blending a lexical signal
(BM25) with the existing Dense signal improves Recall@1/3/5/MRR@5 on
the same PDF and evaluation dataset, without replacing production
Dense search.

## Decision

- **Everything new lives under `scripts/`**, mirroring
  `scripts/retrieval_baseline_core.py` (Issue #18/#19) and
  `scripts/pdf_extraction_comparison_core.py` (Issue #22): `app/domain`,
  `app/application`, `app/infrastructure`, and `app/api/dependencies.py`
  are completely untouched. `InMemoryVectorStore`/`SearchChunksService`/
  `RetrieveChunksService` are called directly and unmodified;
  `scripts/hybrid_retrieval_core.py`'s `dense_search()` calls the exact
  same `SearchChunksService.execute()` production code calls, and
  `evaluate_strategy(STRATEGY_HYBRID, ...)` composes a new BM25 index
  alongside it, never replacing it.

### BM25: self-implemented (`scripts/bm25.py`), not `rank-bm25`

| | `rank-bm25` | Self-implemented |
|---|---|---|
| Implementation size | ~0 (library call) | ~60 lines, standard Okapi BM25 |
| Testability | Black-box internals | Every step traceable; known-input/known-order tests are easy to write |
| Japanese fit | Tokenization is still ours either way | Same |
| License | Apache-2.0 (fine) | N/A (no new dependency) |
| Maintainability | New dependency, new CVE surface | Matches this project's existing convention: `InMemoryVectorStore`'s cosine similarity and `tests/support/evaluation/metrics.py`'s Recall/MRR are also self-implemented rather than pulled from a library |

Chosen: self-implemented. Tokenization has to be supplied by us
regardless of which BM25 implementation is used, so `rank-bm25`'s only
benefit (not writing ~60 lines of well-understood math) is outweighed
by adding a dependency this project's existing pattern avoids for
comparably small, explainable algorithms. `k1=1.5`, `b=0.75` (the
conventional Okapi defaults, also `rank-bm25`'s defaults); idf uses the
always-non-negative "+1" variant
(`log((N - df + 0.5) / (df + 0.5) + 1)`).

### Japanese tokenization: ASCII-token + Japanese-character-bigram hybrid (`scripts/japanese_tokenizer.py`), not a morphological analyzer

| | Character bigram (chosen) | SudachiPy | Janome | fugashi/MeCab |
|---|---|---|---|---|
| Dependency | None | Dictionary package (tens of MB) | Pure Python, bundled dict (MIT) | C extension + dictionary |
| Reproducibility | Deterministic, no download | Dictionary download/version drift | Good (pure Python) | Native build/dictionary friction, especially on Windows |
| Fit for this issue | Directly captures the "exact term/substring match" cases this issue is motivated by (drug names, abbreviations, numbers) | Best linguistic accuracy, but is analytical overkill for a comparison-only tool | Reasonable middle ground | Highest accuracy, but heaviest setup |

Chosen: a hybrid scheme, not a morphological analyzer. ASCII
alphanumeric runs (drug names in Roman letters, device model numbers,
abbreviations, numeric values) are kept as single lowercased tokens;
any other letter/digit run (hiragana/katakana/kanji) is split into
overlapping character bigrams (a single-character run becomes one
unigram). No dictionary, no native extension, fully deterministic
across machines - matching this issue's explicit preference for a
reproducible, easily-tested method over deeper linguistic accuracy.
**Recorded here as a future option, not implemented**: if a later issue
pursues production adoption, Janome (pure Python, bundled dictionary,
MIT-licensed) is the most likely upgrade path, since it needs no native
build step.

### Score normalization and fusion (`scripts/score_normalization.py`, `scripts/hybrid_scorer.py`)

- `SearchResult.score`'s existing convention ("higher is always
  better" - see `app/domain/models/search_result.py`) means Dense
  scores need no inversion before blending.
- Both signals are normalized via candidate-set-relative min-max
  scaling (`min_max_normalize()`): scaled against the min/max of the
  *specific candidate set being scored for one query*, not any
  corpus-wide range - simple and easy to explain. When every value in
  the set is equal (including a single-candidate set), normalization
  returns 0.0 for all of them rather than dividing by zero: with no
  spread, that signal carries no distinguishing information for this
  query, so it should contribute nothing to the fused score.
- **`ScoreFuser` (`scripts/hybrid_scorer.py`) is a narrow Protocol**
  (`fuse(dense_scores, bm25_scores) -> dict[str, float]`), and
  `HybridScorer` is its only implementation this issue adds:
  `hybrid_score = alpha * dense_score_normalized + (1 - alpha) *
  bm25_score_normalized`. This separation exists specifically so an
  alternative fusion strategy - most plausibly Reciprocal Rank Fusion
  (RRF), which fuses by each candidate's *rank* in each signal rather
  than by its normalized score - can be added later as a second
  `ScoreFuser` implementation without changing
  `hybrid_retrieval_core.py`'s calling code (`hybrid_search()` only
  depends on the `ScoreFuser` protocol, never on `HybridScorer`
  directly). `alpha` is validated to `[0.0, 1.0]` at construction.

### Candidate generation and re-ranking (`scripts/hybrid_retrieval_core.py`)

- `hybrid_search()` takes the union of the top `dense_candidate_k` (by
  raw Dense score) and top `bm25_candidate_k` (by raw BM25 score)
  candidates, then **re-scores every candidate in that union against
  the *full* corpus for both signals** (not only within each signal's
  own top-k slice). This avoids needing to invent a placeholder score
  for a candidate that one signal's top-k missed but the other's did
  not - both a Dense re-score (`InMemoryVectorStore.search()` with
  `top_k = len(corpus.chunks)`) and a BM25 re-score
  (`Bm25Index.score_all()`) are cheap in-memory operations at this
  project's scale. The fused top `final_top_k` (ties broken by
  ascending `chunk_id`, matching `InMemoryVectorStore`'s tie-break
  convention) is returned.
- `dense_candidate_k`/`bm25_candidate_k`/`final_top_k` (`--top-k`) are
  all CLI arguments (defaults: 20/20/5).
- **Consequence used as a correctness guarantee**: at `alpha=1.0`, and
  whenever `dense_candidate_k >= final_top_k`, `hybrid_search()`
  produces the *exact same ranking* as `dense_search()` (which calls
  the unmodified `SearchChunksService` directly). Min-max normalization
  is a monotonic transform of the raw Dense score, and the true top
  `final_top_k` Dense results are always inside the
  `dense_candidate_k` slice by construction. This is asserted directly
  in `tests/unit/test_hybrid_retrieval_core.py`
  (`test_alpha_one_hybrid_strategy_matches_dense_strategy_exactly`).

### CLI (`scripts/compare_retrieval_strategies.py`)

- One shared `IndexedCorpus` (PyMuPDF extraction, fixed
  `chunk_size`/`chunk_overlap`) is built once and evaluated under both
  `--strategies dense,hybrid` (the default) so the two strategies are
  compared on identical indexed content. `--alpha` is a free CLI float
  in `[0.0, 1.0]` (default 0.7), rejected outside that range;
  `--dense-candidate-k`/`--bm25-candidate-k` must each be `>=
  --top-k`, guaranteeing the alpha=1.0 exact-match property above holds
  for whatever depths an operator chooses.
- Output mirrors `compare_pdf_extractors.py`: a comparison table
  (`strategy | alpha | Recall@1 | Recall@3 | Recall@5 | MRR@5`, `alpha`
  blank for `dense`) and a matching Markdown table for
  `docs/hybrid-search-comparison-results.md`. `--verbose` additionally
  prints, per strategy and per question: expected page(s), retrieved
  pages, and each retrieved chunk's rank/final score/dense score/bm25
  score/text_preview (local use only, never committed). `--save-report`
  writes full per-strategy, per-question detail to
  `data/eval/results/` (already gitignored in full, per
  `.gitignore`'s `data/eval/` line).
- **Question-type analysis (numeric/abbreviation/plain-Japanese/proper-noun)
  is a design note only, not implemented this issue**: a regex-based
  heuristic tagger (in the spirit of
  `pdf_extraction_comparison_core.py`'s `is_suspicious_page()`) could
  tag each question by whether it contains digits or a 3+ character
  ASCII run, printed alongside `--verbose` output. Reliable automatic
  detection of "chemical name or proper noun" specifically was judged
  too unstable to commit to as a required feature; a future issue can
  revisit this with real dataset examples.
- **Not a pytest test**, for the same reason as
  `evaluate_retrieval_baseline.py`/`compare_pdf_extractors.py`: the
  real PDF and dataset exist only on the operator's machine and are
  never committed; this is a one-off/occasional measurement, not a CI
  gate.
- **New unit tests use only synthetic PDFs
  (`tests/support/pdf_factory.py`), hand-constructed `IndexedCorpus`
  fixtures, and `FakeEmbedder`** - no real guideline PDF and no real
  sentence-transformers model: `tests/unit/test_japanese_tokenizer.py`,
  `tests/unit/test_bm25.py`, `tests/unit/test_score_normalization.py`,
  `tests/unit/test_hybrid_scorer.py`,
  `tests/unit/test_hybrid_retrieval_core.py` (including the alpha=1.0
  exact-match guarantee above), and
  `tests/unit/test_compare_retrieval_strategies.py` (CLI arg parsing,
  Markdown table formatting).

## Consequences

- Production retrieval is unchanged by this issue; adopting Hybrid
  search in production would need its own follow-up issue, informed by
  what this comparison measures on a real guideline.
- `docs/hybrid-search-comparison-results.md` starts empty, exactly like
  `docs/baseline-retrieval-evaluation.md`/
  `docs/pdf-extraction-comparison-results.md` did: recording a real
  comparison requires a human, with a real guideline document and
  dataset, to run `compare_retrieval_strategies.py` locally and choose
  to paste its (reviewed) output.
- Adding Reciprocal Rank Fusion (or any other fusion strategy) later is
  a new `ScoreFuser` implementation plus a CLI flag to select it - no
  changes to `hybrid_search()`'s candidate-union logic are needed for
  that alone.
- The character-bigram tokenizer's fit for real guideline text is not
  validated against real data in this issue; treat its BM25 scores as
  a relative signal for this comparison, not a tuned production-grade
  lexical ranker.
