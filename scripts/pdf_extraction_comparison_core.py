"""Shared library for comparing PDF text-extraction strategies against a
real, local retrieval evaluation dataset (Issue #22).

Not a CLI entry point - imported by
scripts/compare_pdf_extractors.py. Builds on
scripts/retrieval_baseline_core.py (Issue #18/#19): retrieval
Recall@k/MRR is measured via evaluate_configuration() exactly as
before, with a different PdfLoader injected per extractor under
comparison. This module additionally measures per-extractor,
extraction-quality statistics (page/char counts, a garbled-text
heuristic) that retrieval_baseline_core.py has no reason to know
about.

Extractors under comparison implement PdfExtractor
(app/domain/ports/pdf_extractor.py), not the production PdfLoader
port: this keeps app/api/dependencies.py's production wiring (pypdf
only) completely untouched. Adding a future extraction strategy (e.g.
Docling, an OCR pipeline) only requires a new PdfExtractor
implementation registered in EXTRACTOR_FACTORIES below - no changes to
this module's comparison logic.

Garbled-text detection (is_suspicious_page() and friends) is a rough,
comparison-only heuristic, not a quality guarantee. It can both miss
real corruption and flag unusual-but-legitimate text; it exists only
to give a relative signal when comparing extractors against each
other on the same document, per docs/adr/0017-pdf-extraction-comparison-tooling.md.

Never commit a dataset file, the PDF it points to, or this script's
--save-report output; data/eval/ is gitignored for exactly this
reason. See docs/evaluation-dataset-format.md.
"""

import json
import re
import unicodedata
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from app.application.services.chunk_document import ChunkDocumentService
from app.domain.models.chunk import Chunk
from app.domain.models.document import DocumentPage
from app.domain.ports.pdf_extractor import PdfExtractor
from app.infrastructure.chunking.fixed_size_text_splitter import FixedSizeTextSplitter
from app.infrastructure.pdf.pymupdf_extractor import PyMuPdfExtractor
from app.infrastructure.pdf.pypdf_extractor import PypdfExtractor
from scripts.retrieval_baseline_core import (
    ConfigurationRun,
    DatasetCase,
    DatasetDocument,
    evaluate_configuration,
    location_key,
    print_case_report,
    print_verbose_case_results,
    truncate_text,
)

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

EXTRACTOR_FACTORIES: dict[str, Callable[[], PdfExtractor]] = {
    "pypdf": PypdfExtractor,
    "pymupdf": PyMuPdfExtractor,
}

# --- Garbled-text heuristic (comparison-only, not a quality guarantee) ---

_REPLACEMENT_CHAR = "�"
_CID_MARKER = "(cid:"
_PRIVATE_USE_RANGES = [(0xE000, 0xF8FF), (0xF0000, 0xFFFFD), (0x100000, 0x10FFFD)]
_JAPANESE_RANGES = [
    (0x3040, 0x309F),  # Hiragana
    (0x30A0, 0x30FF),  # Katakana
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
]
_SYMBOL_RUN_MIN_LENGTH = 8
_SYMBOL_RUN_PATTERN = re.compile(r"[!-/:-@\[-`{-~]{" + str(_SYMBOL_RUN_MIN_LENGTH) + r",}")
_JAPANESE_RATIO_MIN_LENGTH = 40
_JAPANESE_RATIO_THRESHOLD = 0.05


def _in_ranges(code_point: int, ranges: list[tuple[int, int]]) -> bool:
    return any(low <= code_point <= high for low, high in ranges)


def _is_private_use_char(char: str) -> bool:
    return _in_ranges(ord(char), _PRIVATE_USE_RANGES)


def _is_japanese_char(char: str) -> bool:
    return _in_ranges(ord(char), _JAPANESE_RANGES)


def count_suspicious_chars(text: str) -> int:
    """Counts characters that heuristically suggest garbled extraction:
    the Unicode replacement character, unnatural control characters,
    and Private Use Area code points (a pattern sometimes produced by
    CID-keyed fonts with a broken/missing ToUnicode mapping). Also
    counts literal "(cid:" markers some extractors emit for
    unmappable glyphs.

    Heuristic only - see module docstring.
    """
    count = 0
    for char in text:
        if char == _REPLACEMENT_CHAR:
            count += 1
        elif char not in ("\n", "\t") and unicodedata.category(char) == "Cc":
            count += 1
        elif _is_private_use_char(char):
            count += 1
    count += text.count(_CID_MARKER)
    return count


def has_abnormal_symbol_run(text: str) -> bool:
    """True if text contains a run of at least _SYMBOL_RUN_MIN_LENGTH
    consecutive ASCII punctuation/symbol characters - a pattern
    sometimes produced by misdecoded fonts. Heuristic only.
    """
    return _SYMBOL_RUN_PATTERN.search(text) is not None


def has_abnormal_japanese_char_ratio(text: str) -> bool:
    """True if text is long enough to expect Japanese content but
    contains almost no hiragana/katakana/kanji characters, suggesting
    the extracted text is not readable Japanese.

    Heuristic only: assumes the source document is a Japanese-language
    guideline (this tool's intended use case) and says nothing about
    legitimately non-Japanese documents.
    """
    stripped = "".join(char for char in text if not char.isspace())
    if len(stripped) < _JAPANESE_RATIO_MIN_LENGTH:
        return False
    japanese_count = sum(1 for char in stripped if _is_japanese_char(char))
    return (japanese_count / len(stripped)) < _JAPANESE_RATIO_THRESHOLD


def is_suspicious_page(text: str) -> bool:
    """Heuristic-only judgement of whether a page's extracted text
    looks garbled, combining several signals. Not a quality guarantee -
    see module docstring.
    """
    if not text.strip():
        return False
    return (
        count_suspicious_chars(text) > 0
        or has_abnormal_symbol_run(text)
        or has_abnormal_japanese_char_ratio(text)
    )


# --- Extractor registry ---


def available_extractor_names() -> list[str]:
    return list(EXTRACTOR_FACTORIES)


def get_extractor(name: str) -> PdfExtractor:
    try:
        factory = EXTRACTOR_FACTORIES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown extractor: {name!r} (available: {available_extractor_names()})"
        ) from exc
    return factory()


class _StaticPageLoader:
    """Adapts an already-extracted list[DocumentPage] to the PdfLoader
    protocol expected by evaluate_configuration(), so the same
    extraction pass used for extraction-quality stats is reused for
    retrieval evaluation instead of parsing the PDF a second time.
    """

    def __init__(self, pages: list[DocumentPage]) -> None:
        self._pages = pages

    def load(self, file_path: Path) -> list[DocumentPage]:
        return self._pages


# --- Extraction-quality statistics ---


@dataclass(frozen=True, slots=True)
class ExtractionStats:
    extractor_name: str
    pages: int
    empty_pages: int
    total_chars: int
    suspicious_pages: int
    suspicious_ratio: float
    average_chars_per_page: float
    representative_text_preview: str


def _representative_text_preview(pages: list[DocumentPage]) -> str:
    """The longest non-empty page's (truncated) text - a representative
    sample of real extracted content, for --verbose/--save-report use
    only.
    """
    non_empty = [page for page in pages if page.text.strip()]
    if not non_empty:
        return ""
    longest = max(non_empty, key=lambda page: len(page.text))
    return truncate_text(longest.text)


def compute_extraction_stats(extractor_name: str, pages: list[DocumentPage]) -> ExtractionStats:
    page_count = len(pages)
    empty_pages = sum(1 for page in pages if not page.text.strip())
    total_chars = sum(len(page.text) for page in pages)
    suspicious_pages = sum(1 for page in pages if is_suspicious_page(page.text))

    return ExtractionStats(
        extractor_name=extractor_name,
        pages=page_count,
        empty_pages=empty_pages,
        total_chars=total_chars,
        suspicious_pages=suspicious_pages,
        suspicious_ratio=(suspicious_pages / page_count) if page_count else 0.0,
        average_chars_per_page=(total_chars / page_count) if page_count else 0.0,
        representative_text_preview=_representative_text_preview(pages),
    )


def compute_average_chars_per_chunk(chunks: list[Chunk]) -> float:
    if not chunks:
        return 0.0
    return sum(len(chunk.text) for chunk in chunks) / len(chunks)


# --- Per-extractor evaluation (extraction stats + retrieval quality) ---


@dataclass(frozen=True, slots=True)
class ExtractorComparisonResult:
    extractor_name: str
    extraction_stats: ExtractionStats
    average_chars_per_chunk: float
    retrieval_run: ConfigurationRun


def evaluate_extractor(
    document: DatasetDocument,
    cases: list[DatasetCase],
    model: "SentenceTransformer",
    extractor_name: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    embedding_model_name: str,
) -> ExtractorComparisonResult:
    """Extracts document.source_path once with the named extractor,
    computes its extraction-quality stats, chunks it to measure
    average_chars_per_chunk, then reuses the already-extracted pages
    (via _StaticPageLoader) to run the same retrieval evaluation as
    scripts/evaluate_retrieval_baseline.py / compare_chunk_sizes.py.
    """
    extractor = get_extractor(extractor_name)
    pages = extractor.extract(document.source_path)
    extraction_stats = compute_extraction_stats(extractor_name, pages)

    chunks = ChunkDocumentService(
        FixedSizeTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    ).execute(pages)
    average_chars_per_chunk = compute_average_chars_per_chunk(chunks)

    retrieval_run = evaluate_configuration(
        document,
        cases,
        model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        embedding_model_name=embedding_model_name,
        pdf_loader=_StaticPageLoader(pages),
    )

    return ExtractorComparisonResult(
        extractor_name=extractor_name,
        extraction_stats=extraction_stats,
        average_chars_per_chunk=average_chars_per_chunk,
        retrieval_run=retrieval_run,
    )


# --- Reporting ---

_TABLE_HEADER = (
    f"{'extractor':>10} | {'pages':>5} | {'empty':>5} | {'total_chars':>11} | "
    f"{'chars/pg':>8} | {'suspic.':>7} | {'susp%':>6} | {'chars/chunk':>11} | "
    f"{'Recall@1':>8} | {'Recall@3':>8} | {'Recall@5':>8} | {'MRR@k':>6}"
)


def print_comparison_table(results: list[ExtractorComparisonResult]) -> None:
    print(f"\n{_TABLE_HEADER}")
    print("-" * len(_TABLE_HEADER))
    for result in results:
        stats = result.extraction_stats
        aggregate = result.retrieval_run.aggregate
        print(
            f"{result.extractor_name:>10} | {stats.pages:>5} | {stats.empty_pages:>5} | "
            f"{stats.total_chars:>11} | {stats.average_chars_per_page:>8.1f} | "
            f"{stats.suspicious_pages:>7} | {stats.suspicious_ratio * 100:>5.1f}% | "
            f"{result.average_chars_per_chunk:>11.1f} | "
            f"{aggregate.recall_at_1:>8.3f} | {aggregate.recall_at_3:>8.3f} | "
            f"{aggregate.recall_at_5:>8.3f} | {aggregate.mrr:>6.3f}"
        )


def markdown_comparison_table(label: str, results: list[ExtractorComparisonResult]) -> str:
    first_config = results[0].retrieval_run.config
    header = (
        f"## PDF extraction comparison ({first_config.measured_at})\n\n"
        f"- Document: {label}\n"
        f"- Cases: {first_config.case_count}\n"
        f"- chunk_size={first_config.chunk_size}, chunk_overlap={first_config.chunk_overlap}, "
        f"top_k={first_config.top_k}\n"
        f"- Embedding: sentence_transformers / {first_config.embedding_model_name} "
        f'(query prefix: "{first_config.query_prefix}", '
        f'passage prefix: "{first_config.passage_prefix}")\n\n'
        "Suspicious-page counts/ratios come from a comparison-only "
        "heuristic (Unicode replacement/control/private-use characters, "
        "abnormal ASCII symbol runs, abnormally low Japanese character "
        "ratio) - not a quality guarantee.\n\n"
        "| extractor | pages | empty_pages | total_chars | "
        "average_chars_per_page | suspicious_pages | suspicious_ratio | "
        "average_chars_per_chunk | Recall@1 | Recall@3 | Recall@5 | "
        f"MRR@{first_config.top_k} |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = "".join(
        f"| {result.extractor_name} | {result.extraction_stats.pages} | "
        f"{result.extraction_stats.empty_pages} | {result.extraction_stats.total_chars} | "
        f"{result.extraction_stats.average_chars_per_page:.1f} | "
        f"{result.extraction_stats.suspicious_pages} | "
        f"{result.extraction_stats.suspicious_ratio:.3f} | "
        f"{result.average_chars_per_chunk:.1f} | "
        f"{result.retrieval_run.aggregate.recall_at_1:.2f} | "
        f"{result.retrieval_run.aggregate.recall_at_3:.2f} | "
        f"{result.retrieval_run.aggregate.recall_at_5:.2f} | "
        f"{result.retrieval_run.aggregate.mrr:.2f} |\n"
        for result in results
    )
    return header + rows


def print_markdown_comparison_table(label: str, results: list[ExtractorComparisonResult]) -> None:
    print("\n" + "=" * 70)
    print("Markdown snippet for docs/pdf-extraction-comparison-results.md")
    print("(review before pasting - confirm nothing identifying leaks):")
    print("=" * 70)
    print(markdown_comparison_table(label, results))


def print_verbose_extractor_results(results: list[ExtractorComparisonResult]) -> None:
    """Per-question breakdown and retrieved-chunk detail for each
    extractor (local use only - never paste anywhere committed).
    """
    for result in results:
        print(f"\n{'=' * 70}\nextractor: {result.extractor_name}\n{'=' * 70}")
        print_case_report(result.retrieval_run.case_results)
        print_verbose_case_results(result.retrieval_run.case_results)


def resolve_comparison_report_path(save_report_arg: str, dataset_path: Path) -> Path:
    if save_report_arg != "__default__":
        return Path(save_report_arg)
    today = date.today().isoformat()
    return Path("data/eval/results") / f"{dataset_path.stem}_pdf_extraction_comparison_{today}.json"


def write_comparison_report(
    path: Path, document: DatasetDocument, results: list[ExtractorComparisonResult]
) -> None:
    payload = {
        "document_label": document.label,
        "extractors": [
            {
                "extractor_name": result.extractor_name,
                "extraction_stats": asdict(result.extraction_stats),
                "average_chars_per_chunk": result.average_chars_per_chunk,
                "config": asdict(result.retrieval_run.config),
                "aggregate": asdict(result.retrieval_run.aggregate),
                "cases": [
                    {
                        "question": case_result.question,
                        "expected": sorted(
                            location_key(page, idx) for page, idx in case_result.expected
                        ),
                        "ranked_locations": case_result.ranked_locations,
                        "recall_at_1": case_result.recall_at_1,
                        "recall_at_3": case_result.recall_at_3,
                        "recall_at_5": case_result.recall_at_5,
                        "reciprocal_rank": case_result.reciprocal_rank,
                        "ranked_results": [asdict(ranked) for ranked in case_result.ranked_results],
                    }
                    for case_result in result.retrieval_run.case_results
                ],
            }
            for result in results
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved detailed local report to {path}")
    print("Reminder: this file may contain guideline-derived content - never commit it.")
