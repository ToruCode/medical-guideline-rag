# Hybrid search comparison results

Aggregate Recall@1/3/5/MRR@5 measurements from
`scripts/compare_retrieval_strategies.py`, run manually against real
guideline documents kept entirely locally, comparing Dense-only search
against Hybrid (Dense+BM25) search under a fixed retrieval
configuration (`chunk_size=1000`, `chunk_overlap=200`, `top_k=5`,
`dense_candidate_k=20`, `bm25_candidate_k=20`, `alpha=0.7`,
`intfloat/multilingual-e5-base`, PDF extractor: PyMuPDF). See
`docs/evaluation-dataset-format.md` for the dataset format and
`docs/adr/0019-hybrid-search-comparison.md` for the full design
reasoning (BM25 implementation choice, Japanese tokenization, score
normalization/fusion). For a single Dense-only baseline, a `chunk_size`
sweep, or a PDF extractor comparison instead, see
`docs/baseline-retrieval-evaluation.md`/`docs/chunk-size-comparison.md`/
`docs/pdf-extraction-comparison-results.md`.

This is a comparison/measurement result only - it does not reflect a
production adoption decision. Production retrieval remains Dense-only
until a separate issue changes it.

Only aggregate numbers and measurement configuration are recorded
here. Document titles/publishers are anonymized (e.g. `"Guideline
A"`); the underlying PDF, dataset, questions, expected pages/chunks,
and any extracted text (including `text_preview`) are never committed -
see `CLAUDE.md`'s data/copyright rules.

To add an entry: run `scripts/compare_retrieval_strategies.py`, review
the Markdown table it prints for anything identifying, then paste it
below (most recent first).

---

_(No comparison recorded yet.)_
