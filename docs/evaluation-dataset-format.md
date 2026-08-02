# Retrieval evaluation dataset format (real documents)

This describes the JSON format used to measure retrieval quality
(Recall@1/3/5, MRR) against a real guideline document, shared by
`scripts/evaluate_retrieval_baseline.py` (a single configuration),
`scripts/compare_chunk_sizes.py` (several `chunk_size` values), and
`scripts/compare_pdf_extractors.py` (several PDF text-extraction
strategies) against the same dataset. See
`docs/adr/0014-real-data-retrieval-baseline.md`,
`docs/adr/0015-chunk-size-comparison.md`, and
`docs/adr/0017-pdf-extraction-comparison-tooling.md` for the reasoning
behind this design.

**Never commit an actual dataset file.** `data/eval/` is gitignored for
exactly this reason: a real dataset's questions and expected
page/chunk numbers are derived from a specific real (often
copyrighted) guideline document, and both the real PDF itself
(`data/raw/`, also gitignored) and anything derived from its content
must stay local. See `CLAUDE.md`'s data/copyright rules.

## Location

Place your dataset anywhere under `data/eval/`, e.g.
`data/eval/my_guideline_qa.json`.

## Schema

```json
{
  "document": {
    "source_path": "data/raw/example_guideline.pdf",
    "label": "Guideline A"
  },
  "cases": [
    { "question": "...", "granularity": "page", "expected": [12] },
    { "question": "...", "granularity": "chunk", "expected": [[12, 0]] }
  ]
}
```

- **`document.source_path`**: path to the real PDF (relative to the
  repo root or absolute). Must already exist under `data/raw/` (or
  elsewhere outside version control).
- **`document.label`**: a short, anonymized label (e.g. `"Guideline
  A"`), used only in the script's own output - including the Markdown
  snippet it prints for `docs/baseline-retrieval-evaluation.md`. Do
  not put the real title, publisher, or edition here unless you have
  specifically confirmed that's acceptable to record; this project's
  convention is to keep even bibliographic detail out of committed
  docs by default (`docs/adr/0014-real-data-retrieval-baseline.md`).
- **`cases[].question`**: a natural-language question, evaluated
  exactly as `POST /questions/ask` would receive it.
- **`cases[].granularity`**: `"page"` or `"chunk"`.
  - `"page"` (recommended default): `expected` is a list of 1-based
    page numbers (`Chunk.page_number`). A hit counts if *any* chunk
    from one of these pages appears in the top-k results - robust to
    exactly how a page happens to split into chunks under the current
    `chunk_size`/`chunk_overlap`.
  - `"chunk"`: `expected` is a list of `[page_number, chunk_index]`
    pairs (`Chunk.page_number`, `Chunk.chunk_index`), for when only
    one specific chunk on a page is the correct passage. This is
    coupled to the current `chunk_size`/`chunk_overlap`: if either
    changes, chunk boundaries shift and these indices may no longer
    point at the intended text - re-verify after any chunking config
    change.

## Example (fictional content only)

For a PDF whose pages happen to be, e.g.:

1. "Adults should take 500 mg of Medicamentum X twice daily with food."
2. "Pediatric dosing of Medicamentum X is weight-based; consult a specialist."
3. "Common side effects of Medicamentum X include mild nausea and headache."

a dataset might look like:

```json
{
  "document": { "source_path": "data/raw/example_guideline.pdf", "label": "Guideline A" },
  "cases": [
    { "question": "What is the adult dosage of Medicamentum X?", "granularity": "page", "expected": [1] },
    { "question": "How should Medicamentum X be dosed in children?", "granularity": "page", "expected": [2] },
    { "question": "What are the common side effects of Medicamentum X?", "granularity": "chunk", "expected": [[3, 0]] }
  ]
}
```

## Running

```bash
# .env must have MEDICAL_RAG_EMBEDDING_PROVIDER=sentence_transformers
uv run python -m scripts.evaluate_retrieval_baseline \
  --dataset data/eval/my_guideline_qa.json --save-report
```

`--save-report` writes a detailed per-question JSON report under
`data/eval/results/` (also gitignored) - useful for tracking your own
progress across chunking/embedding experiments locally. See
`scripts/evaluate_retrieval_baseline.py --help` for all options, and
`docs/baseline-retrieval-evaluation.md` for where aggregate results
are recorded.
