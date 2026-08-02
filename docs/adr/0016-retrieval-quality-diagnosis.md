# 0016. Diagnosis: suspect pypdf's Japanese extraction quality, not retrieval configuration

## Status

Accepted

## Context

Issue #21 investigated why retrieval quality against a real Japanese
medical guideline document was lower than expected, using the
measurement tooling already built by Issues #18/#19
(`scripts/evaluate_retrieval_baseline.py`,
`scripts/compare_chunk_sizes.py`). This issue was diagnosis only: no
retrieval, chunking, or extraction code was changed, and no commit was
made on its branch (`feature/issue-21-retrieval-diagnosis` is
identical to `main`).

Per `CLAUDE.md`'s data/copyright rules, the real guideline PDF, the
real evaluation dataset (questions and expected pages), and any
per-question results are never committed. This ADR records only the
diagnosis's conclusion and its rationale, not any of that underlying
material.

## Decision

Retrieval quality was measured while varying the parameters retrieval
tuning would normally target - embedding model choice, `chunk_size`,
and `top_k` - and none of those variations explained the shortfall.
What stood out instead, on manual inspection of the actual text being
indexed, was that a meaningful share of the pages extracted via
`pypdf` (`app/infrastructure/pdf/pypdf_loader.py`, per
`docs/adr/0003-pdf-extraction-library.md`) contained garbled or
malformed Japanese text - not merely imperfect layout/reading-order
(already a known, accepted limitation per ADR 0003), but text that did
not look like valid Japanese content at all.

**Conclusion: pypdf's Japanese text-layer extraction quality, not
embedding/chunk_size/top_k, is suspected to be the primary driver of
the retrieval-quality shortfall.** This is a hypothesis based on
manual, local inspection of one real document - not a quantified,
reproducible measurement - which is exactly what Issue #22
(`docs/adr/0017-pdf-extraction-comparison-tooling.md`) sets out to
test: compare `pypdf` against an alternative extraction library
(PyMuPDF) on the same document, with the same fixed retrieval
configuration, and measure whether extraction-quality statistics and
Recall@k/MRR actually differ.

## Consequences

- No code changes result from this issue. Production PDF extraction
  remains `pypdf`-only (`app/api/dependencies.py`) until Issue #22's
  comparison produces evidence one way or the other.
- Issue #22 is scoped as a comparison/measurement issue, not a
  migration - the same discipline Issues #18/#19 already established
  for this kind of real-document, non-reproducible-by-CI measurement
  work.
- If Issue #22 confirms the hypothesis, a future issue would be needed
  to actually change production extraction (with its own migration,
  testing, and rollout considerations) - out of scope for both Issue
  #21 and Issue #22.
