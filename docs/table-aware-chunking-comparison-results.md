# Table-aware chunking comparison results

Aggregate Recall@1/3/5/MRR@5, chunk statistics, and latency measurements
from `scripts/compare_chunking_strategies.py`, run manually against real
guideline documents kept entirely locally, comparing the existing
fixed-size chunker against a rule-based table-aware chunker under a
fixed `hybrid_rerank` retrieval configuration (`chunk_size=1000`,
`chunk_overlap=200`, `top_k=5`, `dense_candidate_k=20`,
`bm25_candidate_k=20`, `alpha=0.7`, `reranker_candidate_k=5`,
`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, `device=cpu`,
`batch_size=8`, PDF extractor: PyMuPDF). See
`docs/evaluation-dataset-format.md` for the dataset format and
`docs/adr/0021-table-aware-chunking-comparison.md` for the full design
reasoning (detection heuristics and their limitations, `table_max_chars`
vs. `table_row_group_size` priority, comparison methodology). For
Dense/Hybrid/Hybrid+Rerank comparisons instead, see
`docs/hybrid-search-comparison-results.md`/
`docs/cross-encoder-reranker-comparison-results.md`.

This is a comparison/measurement result only - it does not reflect a
production adoption decision. Production chunking remains the existing
fixed-size chunker until a separate issue changes it.

Only aggregate numbers and measurement configuration are recorded
here. Document titles/publishers are anonymized (e.g. `"Guideline
A"`); the underlying PDF, dataset, questions, expected pages/chunks,
and any extracted text (including `text_preview`) are never committed -
see `CLAUDE.md`'s data/copyright rules.

To add an entry: run `scripts/compare_chunking_strategies.py`, review
the Markdown table it prints for anything identifying, then paste it
below (most recent first).

---

## Table-aware chunking comparison (2026-08-04)

- Document: Guideline A
- Configuration: `hybrid_rerank`, `chunk_size=1000`, `chunk_overlap=200`,
  `table_max_chars=1000`, `table_row_group_size=20`, `top_k=5`,
  `dense_candidate_k=20`, `bm25_candidate_k=20`, `alpha=0.7`,
  `reranker_candidate_k=5`,
  `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, `device=cpu`,
  `batch_size=8`

| strategy | Recall@1 | Recall@3 | Recall@5 | MRR@5 | avg_latency_ms |
|---|---:|---:|---:|---:|---:|
| fixed | 0.750 | 0.917 | 0.917 | 0.839 | 701.1 |
| table_aware | 0.733 | 0.867 | 0.900 | 0.801 | 665.4 |

**Conclusion: table-aware chunking is not adopted.** It underperformed
the existing fixed-size chunker on every accuracy metric (Recall@1/3/5,
MRR@5), and its latency improvement (~5%) was not large enough to
offset that regression. See
`docs/adr/0021-table-aware-chunking-comparison.md` for the full design
and rationale.
