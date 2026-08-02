# PDF extraction comparison results

Aggregate extraction-quality and Recall@1/3/5/MRR measurements from
`scripts/compare_pdf_extractors.py`, run manually against real
guideline documents kept entirely locally, comparing `pypdf` against
PyMuPDF under a fixed retrieval configuration (`chunk_size=1000`,
`chunk_overlap=200`, `top_k=5`,
`intfloat/multilingual-e5-base`). See
`docs/evaluation-dataset-format.md` for the dataset format and
`docs/adr/0016-retrieval-quality-diagnosis.md`/
`docs/adr/0017-pdf-extraction-comparison-tooling.md` for the full
design reasoning. For a single-extractor baseline or a `chunk_size`
sweep instead, see `docs/baseline-retrieval-evaluation.md`/
`docs/chunk-size-comparison.md`.

Only aggregate numbers and measurement configuration are recorded
here. Document titles/publishers are anonymized (e.g. `"Guideline
A"`); the underlying PDF, dataset, questions, expected pages/chunks,
and any extracted text (including `representative_text_preview`) are
never committed - see `CLAUDE.md`'s data/copyright rules.

Suspicious-page counts/ratios come from a comparison-only heuristic
(Unicode replacement/control/private-use characters, abnormal ASCII
symbol runs, an abnormally low Japanese character ratio) - not a
quality guarantee. See
`docs/adr/0017-pdf-extraction-comparison-tooling.md` for what it does
and does not detect.

To add an entry: run `scripts/compare_pdf_extractors.py`, review the
Markdown table it prints for anything identifying, then paste it below
(most recent first).

---

_(No comparison recorded yet.)_
