# 0017. Compare PDF extraction strategies via a PdfExtractor protocol

## Status

Accepted

## Context

Issue #21 (`docs/adr/0016-retrieval-quality-diagnosis.md`) concluded
that pypdf's Japanese text extraction quality is suspected - but not
confirmed - to be the primary driver of a retrieval-quality shortfall.
Issue #22 tests that hypothesis: extract the same real guideline PDF
with `pypdf` and PyMuPDF, measure extraction-quality statistics for
each, and run the identical retrieval evaluation
(`scripts/retrieval_baseline_core.py`'s `evaluate_configuration()`,
Issue #18/#19) against each extractor's output under a fixed
configuration (`chunk_size=1000`, `chunk_overlap=200`, `top_k=5`,
`intfloat/multilingual-e5-base`). This issue is comparison/measurement
only - production PDF extraction is not replaced.

## Decision

- **A new `PdfExtractor` protocol
  (`app/domain/ports/pdf_extractor.py`), separate from the existing
  production `PdfLoader` protocol
  (`app/domain/ports/pdf_loader.py`).** `PdfLoader`/`PypdfLoader`/
  `LoadDocumentService`/`app/api/dependencies.py`'s `get_pdf_loader()`
  are completely untouched by this issue - production still wires up
  `pypdf` only. `PdfExtractor.extract(file_path) -> list[DocumentPage]`
  exists specifically so several extraction strategies can be compared
  side by side and so future strategies (e.g. Docling, an OCR
  pipeline) can be added later without touching production DI: each
  new strategy only needs a class implementing `extract()`, registered
  in `scripts/pdf_extraction_comparison_core.py`'s
  `EXTRACTOR_FACTORIES` dict.
- **`PypdfExtractor` (`app/infrastructure/pdf/pypdf_extractor.py`) is a
  thin adapter delegating to the existing `PypdfLoader`**, not a
  reimplementation - the production pypdf extraction logic must not
  duplicate or diverge from itself. **`PyMuPdfExtractor`
  (`app/infrastructure/pdf/pymupdf_extractor.py`) is a new
  implementation** using the `pymupdf` library (added as a main
  dependency, `pymupdf>=1.28.0`), mirroring `PypdfLoader`'s structure:
  same path validation, same SHA-256 `document_id`, same
  `DocumentLoadError` subclass translation
  (`DocumentNotFoundError`/`UnsupportedDocumentTypeError`/
  `EncryptedDocumentError`/`InvalidPdfError`).
  - **License note**: PyMuPDF is dual-licensed AGPL-3.0/commercial,
    unlike pypdf's more permissive license. Acceptable for this
    comparison-only, non-shipped use; a future production adoption
    would need its own license review.
- **A garbled-text heuristic
  (`scripts/pdf_extraction_comparison_core.py`'s
  `is_suspicious_page()` and helpers) is explicitly a comparison-only
  signal, not a quality guarantee.** It flags pages containing the
  Unicode replacement character, unnatural control characters,
  Private-Use-Area code points or `(cid:` markers (patterns associated
  with broken CID-keyed font mappings), abnormal runs of ASCII
  symbols, or an abnormally low Japanese-character ratio for text long
  enough to expect Japanese content. Each signal can both miss real
  corruption and flag unusual-but-legitimate text; the module docstring
  and every heuristic function's docstring say so explicitly, and the
  Japanese-ratio check in particular assumes the source document is a
  Japanese-language guideline (this tool's intended use case).
- **`evaluate_configuration()`
  (`scripts/retrieval_baseline_core.py`) gains one optional parameter,
  `pdf_loader: PdfLoader | None = None`** (defaulting to
  `PypdfLoader()` when omitted), so a `PdfExtractor`'s already-extracted
  pages can be injected for retrieval evaluation without re-parsing
  the PDF. `scripts/evaluate_retrieval_baseline.py` and
  `scripts/compare_chunk_sizes.py`'s existing calls are unaffected
  (they never pass this argument). A local `_StaticPageLoader` adapter
  in `pdf_extraction_comparison_core.py` wraps the extractor's
  already-loaded `list[DocumentPage]` as a `PdfLoader`, so each
  extractor's PDF parse happens exactly once per run - reused for both
  the extraction-quality statistics and the retrieval evaluation,
  guaranteeing both are measured against identical extracted content.
- **`ExtractionStats`** (pages, empty pages, total/average chars per
  page, suspicious page count/ratio, a representative page's truncated
  text preview) is computed per extractor from that one extraction
  pass. **`average_chars_per_chunk`** is computed separately by
  chunking those same pages with the fixed `chunk_size`/
  `chunk_overlap` via the existing `ChunkDocumentService`/
  `FixedSizeTextSplitter` - giving an exact figure (not an
  overlap-distorted approximation from raw page character counts).
- **`scripts/compare_pdf_extractors.py`** (new CLI, mirroring
  `compare_chunk_sizes.py`'s structure) defaults to comparing every
  registered extractor (`--extractors` to select a subset) under the
  fixed configuration above, printing a comparison table (`extractor |
  pages | empty_pages | total_chars | average_chars_per_page |
  suspicious_pages | suspicious_ratio | average_chars_per_chunk |
  Recall@1 | Recall@3 | Recall@5 | MRR@5`) and a ready-to-review
  Markdown table for `docs/pdf-extraction-comparison-results.md`.
  `--verbose` additionally prints, per extractor, each question's
  rank/page/chunk/score/text_preview (local use only, never
  committed); `--save-report` writes full per-extractor,
  per-question detail to `data/eval/results/` (already gitignored).
- **Not a pytest test**, for the same reason as
  `evaluate_retrieval_baseline.py`/`compare_chunk_sizes.py`: the real
  PDF and dataset exist only on the operator's machine and are never
  committed; this is a one-off/occasional measurement, not a CI gate.
- **New unit tests use only synthetic PDFs
  (`tests/support/pdf_factory.py`) and inline strings** - no real
  guideline PDF is needed or used:
  `tests/unit/test_pypdf_extractor.py` and
  `tests/unit/test_pymupdf_extractor.py` (extractor mechanics, mirroring
  `test_pypdf_loader.py`'s coverage), and
  `tests/unit/test_pdf_extraction_comparison_core.py` (extractor
  selection, page-number preservation and empty-page handling across
  both extractors, the garbled-text heuristic on known inputs, and
  `ExtractionStats`/`average_chars_per_chunk` aggregation).
  `tests/unit/test_compare_pdf_extractors.py` covers `--extractors`
  CLI parsing and the Markdown table formatter, mirroring
  `test_compare_chunk_sizes.py`.

## Consequences

- Production PDF extraction is unchanged by this issue; a future issue
  would be needed to actually switch production to a different
  extractor, informed by what this comparison measures.
- `docs/pdf-extraction-comparison-results.md` starts empty, exactly
  like `docs/baseline-retrieval-evaluation.md`/
  `docs/chunk-size-comparison.md` did: recording a real comparison
  requires a human, with a real guideline document and dataset, to run
  `compare_pdf_extractors.py` locally and choose to paste its
  (reviewed) output.
- Adding a further extraction strategy later (Docling, OCR, etc.) is a
  new `PdfExtractor` implementation plus one `EXTRACTOR_FACTORIES`
  entry - no changes to `evaluate_extractor()`, the comparison table,
  or report format are needed for that alone.
- The garbled-text heuristic's thresholds (symbol-run length, Japanese
  character ratio) are not tuned against real guideline data in this
  issue; treat its suspicious-page counts as a relative signal between
  extractors on the same document, not an absolute quality score.
