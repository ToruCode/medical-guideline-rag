"""Shared library for real-data answer-quality-and-citation-consistency
evaluation tooling (Issue #10).

Not a CLI entry point - imported by scripts/evaluate_answer_quality.py.
Builds on scripts.evaluation_common's dataset loading (shared with
scripts/retrieval_baseline_core.py, the retrieval-only evaluator, per
docs/requirements.md's "separate retrieval quality from generation
quality") but additionally composes GenerateAnswerService (retrieve
THEN generate, like AskQuestionService - reimplemented inline here
rather than calling AskQuestionService directly, only so this module
can see the intermediate search_results and check citation consistency
against them) to measure whether the generated answer's citations are
consistent with what was actually retrieved, and (best-effort, via
keyword/substring matching) whether the answer covers the dataset's
expected_answer_points.

All metrics are deterministic and explainable (set arithmetic and
substring matching) - no LLM-as-a-Judge is used. See
docs/adr/0023-answer-quality-and-citation-consistency-evaluation.md for
the reasoning, including why citation-consistency is expected to always
pass (a structural guarantee of GenerateAnswerService, not something
this module enforces) and why answer-point coverage is a lexical
approximation, not a semantic one.

Never commit a dataset file, the PDF it points to, or any
--save-report output (which may contain guideline-derived content and
generated answer text); data/eval/ is gitignored for exactly this
reason. See docs/evaluation-dataset-format.md.
"""

import json
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from app.application.services.chunk_document import ChunkDocumentService
from app.application.services.embed_chunks import EmbedChunksService
from app.application.services.generate_answer import GenerateAnswerService
from app.application.services.index_chunks import IndexChunksService
from app.application.services.index_document import IndexDocumentService
from app.application.services.load_document import LoadDocumentService
from app.application.services.retrieve_chunks import RetrieveChunksService
from app.application.services.search_chunks import SearchChunksService
from app.domain.models.search_result import SearchResult
from app.domain.ports.llm import Llm
from app.domain.ports.pdf_loader import PdfLoader
from app.infrastructure.chunking.fixed_size_text_splitter import FixedSizeTextSplitter
from app.infrastructure.embedding.sentence_transformer_embedder import SentenceTransformerEmbedder
from app.infrastructure.pdf.pypdf_loader import PypdfLoader
from app.infrastructure.vector_store.in_memory_vector_store import InMemoryVectorStore
from scripts.evaluation_common import (
    DatasetCase,
    DatasetDocument,
    truncate_text,
)
from tests.support.evaluation.metrics import (
    answer_point_coverage,
    citation_precision,
    citation_recall,
    mean,
)

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


def citations_are_subset_of_retrieved(
    citations: list[SearchResult], retrieved: list[SearchResult]
) -> bool:
    """True if every citation corresponds to a chunk that was actually
    retrieved for this question.

    Expected to always be True: GenerateAnswerService.execute()
    constructs citations as a filtered subset of the search_results it
    is given (app/application/services/generate_answer.py's
    select_chunks_within_budget()), so a citation naming a chunk that
    was never retrieved should be structurally impossible. This check
    is a regression safety net for that invariant, not a condition
    expected to ever fail today - see
    docs/adr/0023-answer-quality-and-citation-consistency-evaluation.md.
    """
    retrieved_ids = {result.embedded_chunk.chunk.chunk_id for result in retrieved}
    return all(citation.embedded_chunk.chunk.chunk_id in retrieved_ids for citation in citations)


def _mean_of_defined(values: list[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    if not defined:
        return None
    return mean(defined)


@dataclass(frozen=True, slots=True)
class AnswerCaseResult:
    question: str
    expected_pages: list[int]
    expected_answer_points: list[str]
    expected_insufficient_evidence: bool
    is_insufficient_evidence: bool
    insufficient_evidence_correct: bool
    cited_pages: list[int]
    citation_precision: float | None
    citation_recall: float | None
    answer_point_coverage: float | None
    citations_consistent: bool
    latency_seconds: float
    answer_preview: str


@dataclass(frozen=True, slots=True)
class AnswerAggregate:
    insufficient_evidence_accuracy: float
    mean_citation_precision: float | None
    mean_citation_recall: float | None
    mean_answer_point_coverage: float | None
    mean_latency_seconds: float
    citation_consistency_violations: int


@dataclass(frozen=True, slots=True)
class AnswerRunConfig:
    llm_provider: str  # "fake" or "openai"
    llm_model_name: str | None
    chunk_size: int
    chunk_overlap: int
    top_k: int
    context_max_chars: int
    embedding_model_name: str
    case_count: int
    indexed_page_count: int
    indexed_chunk_count: int
    measured_at: str


@dataclass(frozen=True, slots=True)
class AnswerConfigurationRun:
    """The full result of evaluate_answer_configuration() for one
    llm_provider/chunk_size/top_k/context_max_chars combination.
    """

    config: AnswerRunConfig
    case_results: list[AnswerCaseResult]
    aggregate: AnswerAggregate


def evaluate_answer_case(
    case: DatasetCase,
    retrieve_chunks: RetrieveChunksService,
    generate_answer: GenerateAnswerService,
    top_k: int,
) -> AnswerCaseResult:
    expected_pages = {page for page, _ in case.expected_locations}

    started_at = time.perf_counter()
    search_results = retrieve_chunks.execute(case.question, top_k=top_k)
    result = generate_answer.execute(case.question, search_results)
    latency_seconds = time.perf_counter() - started_at

    cited_pages = {citation.embedded_chunk.chunk.page_number for citation in result.citations}
    insufficient_evidence_correct = (
        result.is_insufficient_evidence == case.expected_insufficient_evidence
    )

    case_citation_precision: float | None = None
    case_citation_recall: float | None = None
    case_answer_point_coverage: float | None = None
    if not case.expected_insufficient_evidence:
        case_citation_precision = citation_precision(cited_pages, expected_pages)
        case_citation_recall = citation_recall(cited_pages, expected_pages)
        case_answer_point_coverage = answer_point_coverage(
            result.answer, case.expected_answer_points
        )

    return AnswerCaseResult(
        question=case.question,
        expected_pages=sorted(expected_pages),
        expected_answer_points=case.expected_answer_points,
        expected_insufficient_evidence=case.expected_insufficient_evidence,
        is_insufficient_evidence=result.is_insufficient_evidence,
        insufficient_evidence_correct=insufficient_evidence_correct,
        cited_pages=sorted(cited_pages),
        citation_precision=case_citation_precision,
        citation_recall=case_citation_recall,
        answer_point_coverage=case_answer_point_coverage,
        citations_consistent=citations_are_subset_of_retrieved(result.citations, search_results),
        latency_seconds=latency_seconds,
        answer_preview=truncate_text(result.answer),
    )


def summarize_answers(case_results: list[AnswerCaseResult]) -> AnswerAggregate:
    return AnswerAggregate(
        insufficient_evidence_accuracy=mean(
            [1.0 if c.insufficient_evidence_correct else 0.0 for c in case_results]
        ),
        mean_citation_precision=_mean_of_defined([c.citation_precision for c in case_results]),
        mean_citation_recall=_mean_of_defined([c.citation_recall for c in case_results]),
        mean_answer_point_coverage=_mean_of_defined(
            [c.answer_point_coverage for c in case_results]
        ),
        mean_latency_seconds=mean([c.latency_seconds for c in case_results]),
        citation_consistency_violations=sum(
            0 if c.citations_consistent else 1 for c in case_results
        ),
    )


def evaluate_answer_configuration(
    document: DatasetDocument,
    cases: list[DatasetCase],
    model: "SentenceTransformer",
    llm: Llm,
    *,
    llm_provider: str,
    llm_model_name: str | None,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    context_max_chars: int,
    embedding_model_name: str,
    pdf_loader: PdfLoader | None = None,
) -> AnswerConfigurationRun:
    """Indexes document.source_path, then runs every case in cases
    through RetrieveChunksService + GenerateAnswerService(llm) at the
    given top_k/context_max_chars.

    model is a pre-loaded sentence-transformers model (loading it is
    the expensive part; callers comparing several configurations should
    load it once and reuse it, matching
    scripts/retrieval_baseline_core.py::evaluate_configuration()'s
    convention). llm may be a FakeLlm (default; no network/API key) or
    a real OpenAiLlm (opt-in; see scripts/evaluate_answer_quality.py
    --llm openai).
    """
    passage_embedder = SentenceTransformerEmbedder(model, prefix=PASSAGE_PREFIX)
    query_embedder = SentenceTransformerEmbedder(model, prefix=QUERY_PREFIX)
    vector_store = InMemoryVectorStore()

    index_document = IndexDocumentService(
        load_document=LoadDocumentService(pdf_loader if pdf_loader is not None else PypdfLoader()),
        chunk_document=ChunkDocumentService(
            FixedSizeTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        ),
        embed_chunks=EmbedChunksService(passage_embedder),
        index_chunks=IndexChunksService(vector_store),
    )
    retrieve_chunks = RetrieveChunksService(
        embedder=query_embedder, search_chunks=SearchChunksService(vector_store)
    )
    generate_answer = GenerateAnswerService(llm, context_max_chars=context_max_chars)

    index_result = index_document.execute(document.source_path)

    case_results = [
        evaluate_answer_case(case, retrieve_chunks, generate_answer, top_k) for case in cases
    ]
    config = AnswerRunConfig(
        llm_provider=llm_provider,
        llm_model_name=llm_model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        context_max_chars=context_max_chars,
        embedding_model_name=embedding_model_name,
        case_count=len(cases),
        indexed_page_count=index_result.page_count,
        indexed_chunk_count=index_result.chunk_count,
        measured_at=date.today().isoformat(),
    )
    return AnswerConfigurationRun(
        config=config, case_results=case_results, aggregate=summarize_answers(case_results)
    )


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def print_case_report(case_results: list[AnswerCaseResult]) -> None:
    print("\nPer-question results (local only - never paste this section anywhere committed):")
    for index, case_result in enumerate(case_results, start=1):
        consistency_flag = "OK" if case_result.citations_consistent else "VIOLATION"
        evidence_flag = "OK" if case_result.insufficient_evidence_correct else "MISMATCH"
        print(
            f"  [{index:>2}] precision={_format_optional(case_result.citation_precision)} "
            f"recall={_format_optional(case_result.citation_recall)} "
            f"coverage={_format_optional(case_result.answer_point_coverage)} "
            f"consistency={consistency_flag} insufficient_evidence={evidence_flag} "
            f"latency={case_result.latency_seconds:.2f}s"
        )
        print(
            f"        Q: {case_result.question}\n"
            f"        expected_pages={case_result.expected_pages} "
            f"cited_pages={case_result.cited_pages}"
        )


def print_aggregate_report(aggregate: AnswerAggregate, config: AnswerRunConfig) -> None:
    llm_label = config.llm_provider
    if config.llm_model_name:
        llm_label += f" / {config.llm_model_name}"
    print("\nAggregate:")
    print(
        f"  Indexed {config.indexed_page_count} pages into "
        f"{config.indexed_chunk_count} chunks "
        f"(chunk_size={config.chunk_size}, chunk_overlap={config.chunk_overlap})."
    )
    print(f"  Llm: {llm_label}")
    print(f"  Citation precision (mean) = {_format_optional(aggregate.mean_citation_precision)}")
    print(f"  Citation recall (mean)    = {_format_optional(aggregate.mean_citation_recall)}")
    print(
        "  Answer-point coverage (mean, lexical match only) = "
        f"{_format_optional(aggregate.mean_answer_point_coverage)}"
    )
    print(f"  Insufficient-evidence accuracy = {aggregate.insufficient_evidence_accuracy:.3f}")
    print(
        f"  Citation consistency violations = {aggregate.citation_consistency_violations} "
        "(expected: 0 - see docs/adr/0023-answer-quality-and-citation-consistency-evaluation.md)"
    )
    print(f"  Mean latency = {aggregate.mean_latency_seconds:.3f}s")
    print(f"  (over {config.case_count} questions)")


def print_failure_analysis(case_results: list[AnswerCaseResult]) -> None:
    """Prints only the cases with a citation-consistency violation, an
    insufficient-evidence mismatch, citation_recall < 1.0, or
    answer_point_coverage < 1.0 (Issue #10's "failure analysis"
    requirement), instead of repeating every passing case.
    """
    failures = [
        case_result
        for case_result in case_results
        if not case_result.citations_consistent
        or not case_result.insufficient_evidence_correct
        or (case_result.citation_recall is not None and case_result.citation_recall < 1.0)
        or (
            case_result.answer_point_coverage is not None
            and case_result.answer_point_coverage < 1.0
        )
    ]
    if not failures:
        print("\nNo failures detected.")
        return

    print(f"\nFailure analysis ({len(failures)} of {len(case_results)} cases):")
    for case_result in failures:
        reasons = []
        if not case_result.citations_consistent:
            reasons.append("citation_consistency_violation")
        if not case_result.insufficient_evidence_correct:
            reasons.append("insufficient_evidence_mismatch")
        if case_result.citation_recall is not None and case_result.citation_recall < 1.0:
            reasons.append("citation_recall<1.0")
        if (
            case_result.answer_point_coverage is not None
            and case_result.answer_point_coverage < 1.0
        ):
            reasons.append("answer_point_coverage<1.0")
        print(f"  - {case_result.question}: {', '.join(reasons)}")


def write_local_report(
    path: Path, document: DatasetDocument, runs: list[AnswerConfigurationRun]
) -> None:
    """Writes one or more AnswerConfigurationRuns to a single local JSON
    report. May contain guideline-derived content and generated answer
    text - never commit it.
    """
    payload = {
        "document_label": document.label,
        "runs": [
            {
                "config": asdict(run.config),
                "aggregate": asdict(run.aggregate),
                "cases": [asdict(case_result) for case_result in run.case_results],
            }
            for run in runs
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved detailed local report to {path}")
    print(
        "Reminder: this file may contain guideline-derived content and "
        "generated answer text - never commit it."
    )
