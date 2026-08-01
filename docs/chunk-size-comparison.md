# Chunk size comparison results

Aggregate Recall@1/3/5 and MRR measurements from
`scripts/compare_chunk_sizes.py`, run manually against real guideline
documents kept entirely locally, across several `chunk_size` values
(`chunk_overlap`, `top_k`, and the embedding model held fixed). See
`docs/evaluation-dataset-format.md` for the dataset format and
`docs/adr/0015-chunk-size-comparison.md` for the full design
reasoning. For a single-configuration baseline (not a comparison), see
`docs/baseline-retrieval-evaluation.md`.

Only aggregate numbers and measurement configuration are recorded
here. Document titles/publishers are anonymized (e.g. `"Guideline
A"`); the underlying PDF, dataset, questions, and expected
pages/chunks are never committed - see `CLAUDE.md`'s data/copyright
rules.

To add an entry: run `scripts/compare_chunk_sizes.py`, review the
Markdown table it prints for anything identifying, then paste it below
(most recent first).

---

_(No comparison recorded yet.)_
