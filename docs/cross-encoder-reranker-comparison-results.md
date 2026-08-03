# Cross Encoder reranking comparison results

Aggregate Recall@1/3/5/MRR@5/latency measurements from
`scripts/compare_reranking_strategies.py`, run manually against real
guideline documents kept entirely locally, comparing Dense-only,
Hybrid (Dense+BM25), and Hybrid+Cross-Encoder-Rerank search under a
fixed retrieval configuration (`chunk_size=1000`, `chunk_overlap=200`,
`top_k=5`, `dense_candidate_k=20`, `bm25_candidate_k=20`,
`reranker_candidate_k=20`, `alpha=0.7`,
`intfloat/multilingual-e5-base`, PDF extractor: PyMuPDF). See
`docs/evaluation-dataset-format.md` for the dataset format and
`docs/adr/0020-cross-encoder-reranker-comparison.md` for the full
design reasoning (candidate/rerank separation, Cross Encoder model
choice, latency measurement, per-question categorization). For
Dense-vs-Hybrid only, see
`docs/hybrid-search-comparison-results.md`/
`scripts/compare_retrieval_strategies.py`.

This is a comparison/measurement result only - it does not reflect a
production adoption decision. Production retrieval remains Dense-only
until a separate issue changes it.

Only aggregate numbers and measurement configuration are recorded
here. Document titles/publishers are anonymized (e.g. `"Guideline
A"`); the underlying PDF, dataset, questions, expected pages/chunks,
and any extracted text (including `text_preview`) are never committed -
see `CLAUDE.md`'s data/copyright rules.

To add an entry: run `scripts/compare_reranking_strategies.py`, review
the Markdown table it prints for anything identifying, then paste it
below (most recent first).

---

_(No comparison recorded yet.)_
