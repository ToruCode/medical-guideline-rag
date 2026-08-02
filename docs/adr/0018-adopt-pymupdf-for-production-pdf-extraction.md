# 0018. Adopt PyMuPDF as the default production PDF extractor

## Status

Accepted

## Context

Issue #22 (`docs/adr/0017-pdf-extraction-comparison-tooling.md`)
compared `pypdf` against PyMuPDF on one Japanese medical guideline PDF,
under one fixed retrieval configuration (`chunk_size=1000`,
`chunk_overlap=200`, `top_k=5`, `intfloat/multilingual-e5-base`), using
the same evaluation dataset for both. PyMuPDF's comparison-only
garbled-text heuristic (`suspicious_ratio`) was near zero for that
document, versus a heuristic-flagged majority of pages for `pypdf`, and
PyMuPDF's Recall@1/3/5 and MRR@5 were substantially higher than
`pypdf`'s under the identical configuration. Only aggregate,
non-identifying figures are referenced here; see
`docs/pdf-extraction-comparison-results.md` for where a reviewed,
anonymized aggregate table belongs. This result is from a single
document and evaluation set - it indicates PyMuPDF is the better
choice for the Japanese guideline PDFs this project currently targets,
**not** that PyMuPDF is guaranteed superior for all PDF formats,
layouts, or languages.

## Decision

- **`app/core/config.py`'s `Settings` gains
  `pdf_extractor: Literal["pymupdf", "pypdf"] = "pymupdf"`**
  (env var `MEDICAL_RAG_PDF_EXTRACTOR`), following the same
  provider-switch pattern already used for `embedding_provider`/
  `llm_provider`. An unrecognized value raises a pydantic
  `ValidationError` at `Settings()` construction (startup/first
  `get_settings()` call).
- **A new `PyMuPdfLoader`
  (`app/infrastructure/pdf/pymupdf_loader.py`) implements the
  production `PdfLoader` protocol
  (`app/domain/ports/pdf_loader.py`)** as a thin adapter delegating to
  the already-tested `PyMuPdfExtractor`
  (`app/infrastructure/pdf/pymupdf_extractor.py`, added in Issue #22),
  mirroring how `PypdfExtractor` adapts `PypdfLoader` for the
  comparison-only `PdfExtractor` protocol - just inverted, since the
  tested core logic already lived on the `PdfExtractor` side for
  PyMuPDF.
- **`app/api/dependencies.py`'s `get_pdf_loader()` now takes `Settings`
  via `Depends(get_settings)`** (matching `get_text_splitter`'s
  pattern) and returns `PyMuPdfLoader()` or `PypdfLoader()` based on
  `settings.pdf_extractor`, raising `ValueError` for any other value as
  defense-in-depth (mirroring `get_llm()`'s fail-fast behavior) for
  callers that pass a `Settings`-like object directly rather than
  through env-var-validated construction.
- **`PypdfLoader`/`PypdfExtractor` are not removed.** They remain the
  rollback path (`MEDICAL_RAG_PDF_EXTRACTOR=pypdf`) and stay available
  to `scripts/compare_pdf_extractors.py` for future comparisons.
- **Both extraction strategies stay reachable only through the
  `PdfLoader`/`PdfExtractor` protocols.** No application or domain code
  depends on `PyMuPdfLoader`/`PyMuPdfExtractor` directly; a future
  extraction strategy (Docling, an OCR pipeline, etc.) can be adopted
  the same way - a new class implementing the relevant protocol plus a
  `Settings`/DI wiring change, without touching `LoadDocumentService`
  or any layer above it.

## License follow-up (not resolved by this issue)

PyMuPDF is dual-licensed AGPL-3.0/commercial (unlike `pypdf`'s more
permissive license), as already flagged in
`docs/adr/0017-pdf-extraction-comparison-tooling.md` when it was added
for comparison-only, non-shipped use. This issue makes PyMuPDF the
**default, invoked-at-runtime** production extractor, which is a
materially different situation: AGPL-3.0's network-use provisions can
require offering source code to users interacting with the service
over a network, which is directly relevant to this project's goal of
deploying to AWS.

- This adoption is accepted at the current technical-demonstration
  stage on the basis that PyMuPDF's extraction/retrieval quality
  advantage justified the switch, and that `pypdf` remains available as
  an immediate, config-only rollback.
- **Before any commercial use or public/production deployment, the
  PyMuPDF license position must be explicitly reconfirmed** (either
  compliance with AGPL-3.0 obligations, a commercial license from
  Artifex, or switching the default back to `pypdf` or another
  extractor). This is an open item, not resolved by this issue.

## Consequences

- Default PDF extraction quality and retrieval accuracy improve for
  Japanese guideline PDFs matching the profile evaluated in Issue #22,
  without changing anything above the `PdfLoader` protocol boundary.
- Existing tests using `FakePdfLoader`
  (`tests/unit/test_load_document_service.py`,
  `tests/unit/test_index_document_service.py`,
  `tests/integration/test_index_document_pipeline.py`, etc.) are
  unaffected: `LoadDocumentService` still accepts any `PdfLoader`
  implementation via constructor injection.
- The PyMuPDF license follow-up above is a known, tracked risk, not a
  blocker for this issue's technical-demonstration scope - but it must
  be resolved before this system is deployed for real, public, or
  commercial use.
- The single-document comparison in Issue #22 is not proof this result
  generalizes to every PDF layout (multi-column pages, heavy tables,
  scanned/image-only pages remain out of scope for both extractors, as
  already noted in `docs/adr/0003-pdf-extraction-library.md`).
