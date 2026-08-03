from pathlib import Path

import pytest
from app.application.services.generate_answer import GenerateAnswerService
from app.application.services.retrieve_chunks import RetrieveChunksService
from app.application.services.search_chunks import SearchChunksService
from app.domain.models.chunk import Chunk
from app.domain.models.embedding import EmbeddedChunk
from app.domain.models.search_result import SearchResult
from app.infrastructure.llm.fake_llm import FakeLlm
from app.infrastructure.vector_store.in_memory_vector_store import InMemoryVectorStore
from scripts.answer_quality_core import (
    citations_are_subset_of_retrieved,
    evaluate_answer_case,
    evaluate_answer_configuration,
    print_failure_analysis,
    summarize_answers,
    write_local_report,
)
from scripts.evaluation_common import DatasetCase, DatasetDocument
from tests.support.pdf_factory import build_pdf


class _FakeVectors(list):
    def tolist(self) -> list[list[float]]:
        return list(self)


class _FakeSentenceTransformerModel:
    """Minimal stand-in for sentence_transformers.SentenceTransformer's
    encode() interface, so evaluate_answer_configuration() can be
    unit-tested without downloading a real model. Deterministic,
    text-length-based (like FakeEmbedder) - only used to verify this
    module's own wiring/aggregation logic, matching
    tests/unit/test_retrieval_baseline_core.py's identical stand-in.
    """

    def encode(self, texts: list[str], convert_to_numpy: bool = True) -> _FakeVectors:
        del convert_to_numpy
        return _FakeVectors([[float(len(text) % 5), float((len(text) + 1) % 5)] for text in texts])


class _QueryAwareEmbedder:
    def __init__(self, vectors_by_text: dict[str, list[float]]) -> None:
        self._vectors_by_text = vectors_by_text

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors_by_text[text] for text in texts]


def _make_result(
    text: str = "chunk text", page_number: int = 1, chunk_index: int = 0, score: float = 0.9
) -> SearchResult:
    chunk = Chunk(
        document_id="doc-1",
        source_name="sample.pdf",
        source_path="/tmp/sample.pdf",
        page_number=page_number,
        chunk_index=chunk_index,
        text=text,
        title="Sample Guideline",
    )
    return SearchResult(embedded_chunk=EmbeddedChunk(chunk=chunk, vector=[0.1, 0.2]), score=score)


# --- citations_are_subset_of_retrieved ---


def test_citations_are_subset_of_retrieved_true_when_every_citation_was_retrieved() -> None:
    retrieved = [_make_result(page_number=1), _make_result(page_number=2)]

    assert citations_are_subset_of_retrieved(retrieved, retrieved) is True


def test_citations_are_subset_of_retrieved_false_when_a_citation_was_never_retrieved() -> None:
    retrieved = [_make_result(page_number=1)]
    fabricated_citation = [_make_result(page_number=9)]

    assert citations_are_subset_of_retrieved(fabricated_citation, retrieved) is False


def test_citations_are_subset_of_retrieved_true_for_no_citations() -> None:
    assert citations_are_subset_of_retrieved([], [_make_result()]) is True


# --- evaluate_answer_case ---


def test_evaluate_answer_case_computes_citation_and_coverage_metrics() -> None:
    relevant = _make_result(text="500 mg twice daily", page_number=1)
    embedder = _QueryAwareEmbedder(
        {"dosage question": [1.0, 0.0], "500 mg twice daily": [1.0, 0.0]}
    )
    store = InMemoryVectorStore()
    store.upsert([relevant.embedded_chunk])
    retrieve_chunks = RetrieveChunksService(
        embedder=embedder, search_chunks=SearchChunksService(store)
    )
    llm = FakeLlm(answer="Take 500 mg twice daily as directed.")
    generate_answer = GenerateAnswerService(llm)
    case = DatasetCase(
        question="dosage question",
        granularity="page",
        expected_locations=[(1, None)],
        expected_answer_points=["500 mg", "twice daily", "not present"],
    )

    result = evaluate_answer_case(case, retrieve_chunks, generate_answer, top_k=5)

    assert result.expected_pages == [1]
    assert result.cited_pages == [1]
    assert result.citation_precision == 1.0
    assert result.citation_recall == 1.0
    assert result.answer_point_coverage == pytest.approx(2 / 3)
    assert result.citations_consistent is True
    assert result.insufficient_evidence_correct is True
    assert result.latency_seconds >= 0.0


def test_evaluate_answer_case_skips_citation_metrics_when_insufficient_evidence_expected() -> None:
    embedder = _QueryAwareEmbedder({"unanswerable question": [1.0, 0.0]})
    store = InMemoryVectorStore()  # nothing indexed
    retrieve_chunks = RetrieveChunksService(
        embedder=embedder, search_chunks=SearchChunksService(store)
    )
    generate_answer = GenerateAnswerService(FakeLlm())
    case = DatasetCase(
        question="unanswerable question",
        granularity="page",
        expected_locations=[],
        expected_insufficient_evidence=True,
    )

    result = evaluate_answer_case(case, retrieve_chunks, generate_answer, top_k=5)

    assert result.is_insufficient_evidence is True
    assert result.insufficient_evidence_correct is True
    assert result.citation_precision is None
    assert result.citation_recall is None
    assert result.answer_point_coverage is None


def test_evaluate_answer_case_flags_insufficient_evidence_mismatch() -> None:
    embedder = _QueryAwareEmbedder({"unanswerable question": [1.0, 0.0]})
    store = InMemoryVectorStore()
    retrieve_chunks = RetrieveChunksService(
        embedder=embedder, search_chunks=SearchChunksService(store)
    )
    generate_answer = GenerateAnswerService(FakeLlm())
    case = DatasetCase(
        question="unanswerable question",
        granularity="page",
        expected_locations=[(3, None)],
        expected_insufficient_evidence=False,
    )

    result = evaluate_answer_case(case, retrieve_chunks, generate_answer, top_k=5)

    assert result.is_insufficient_evidence is True
    assert result.insufficient_evidence_correct is False


# --- summarize_answers ---


def test_summarize_answers_averages_defined_metrics_and_counts_violations() -> None:
    good = evaluate_answer_case(
        DatasetCase(question="q", granularity="page", expected_locations=[(1, None)]),
        RetrieveChunksService(
            embedder=_QueryAwareEmbedder({"q": [1.0, 0.0], "matched": [1.0, 0.0]}),
            search_chunks=SearchChunksService(
                _store_with([_make_result(text="matched", page_number=1)])
            ),
        ),
        GenerateAnswerService(FakeLlm(answer="matched")),
        top_k=5,
    )
    insufficient = evaluate_answer_case(
        DatasetCase(
            question="unanswerable",
            granularity="page",
            expected_locations=[],
            expected_insufficient_evidence=True,
        ),
        RetrieveChunksService(
            embedder=_QueryAwareEmbedder({"unanswerable": [1.0, 0.0]}),
            search_chunks=SearchChunksService(InMemoryVectorStore()),
        ),
        GenerateAnswerService(FakeLlm()),
        top_k=5,
    )

    aggregate = summarize_answers([good, insufficient])

    assert aggregate.insufficient_evidence_accuracy == 1.0
    assert aggregate.mean_citation_precision == 1.0
    assert aggregate.mean_citation_recall == 1.0
    assert aggregate.citation_consistency_violations == 0


def _store_with(results: list[SearchResult]) -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    store.upsert([result.embedded_chunk for result in results])
    return store


# --- evaluate_answer_configuration ---


def test_evaluate_answer_configuration_populates_config_and_indexes_document(
    tmp_path: Path,
) -> None:
    pdf_path = build_pdf(tmp_path / "guideline.pdf", ["short page one"], title="Sample")
    document = DatasetDocument(source_path=pdf_path, label="Sample Guideline")
    cases = [
        DatasetCase(question="short page one", granularity="page", expected_locations=[(1, None)])
    ]

    run = evaluate_answer_configuration(
        document,
        cases,
        _FakeSentenceTransformerModel(),
        FakeLlm(answer="short page one"),
        llm_provider="fake",
        llm_model_name=None,
        chunk_size=1000,
        chunk_overlap=200,
        top_k=5,
        context_max_chars=6000,
        embedding_model_name="fake-model",
    )

    assert run.config.case_count == 1
    assert run.config.indexed_page_count == 1
    assert run.config.llm_provider == "fake"
    assert len(run.case_results) == 1


# --- reporting ---


def test_print_failure_analysis_reports_no_failures_for_all_passing_cases(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = evaluate_answer_case(
        DatasetCase(question="q", granularity="page", expected_locations=[(1, None)]),
        RetrieveChunksService(
            embedder=_QueryAwareEmbedder({"q": [1.0, 0.0], "matched": [1.0, 0.0]}),
            search_chunks=SearchChunksService(
                _store_with([_make_result(text="matched", page_number=1)])
            ),
        ),
        GenerateAnswerService(FakeLlm(answer="matched")),
        top_k=5,
    )

    print_failure_analysis([result])

    assert "No failures detected." in capsys.readouterr().out


def test_print_failure_analysis_reports_insufficient_evidence_mismatch(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = evaluate_answer_case(
        DatasetCase(
            question="do-not-leak-this-question",
            granularity="page",
            expected_locations=[(3, None)],
        ),
        RetrieveChunksService(
            embedder=_QueryAwareEmbedder({"do-not-leak-this-question": [1.0, 0.0]}),
            search_chunks=SearchChunksService(InMemoryVectorStore()),
        ),
        GenerateAnswerService(FakeLlm()),
        top_k=5,
    )

    print_failure_analysis([result])

    out = capsys.readouterr().out
    assert "insufficient_evidence_mismatch" in out
    assert "do-not-leak-this-question" in out


def test_write_local_report_writes_document_label_and_case_results(tmp_path: Path) -> None:
    pdf_path = build_pdf(tmp_path / "guideline.pdf", ["short page one"], title="Sample")
    document = DatasetDocument(source_path=pdf_path, label="Sample Guideline")
    cases = [
        DatasetCase(question="short page one", granularity="page", expected_locations=[(1, None)])
    ]
    run = evaluate_answer_configuration(
        document,
        cases,
        _FakeSentenceTransformerModel(),
        FakeLlm(answer="short page one"),
        llm_provider="fake",
        llm_model_name=None,
        chunk_size=1000,
        chunk_overlap=200,
        top_k=5,
        context_max_chars=6000,
        embedding_model_name="fake-model",
    )
    report_path = tmp_path / "report.json"

    write_local_report(report_path, document, [run])

    import json

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["document_label"] == "Sample Guideline"
    assert len(payload["runs"][0]["cases"]) == 1
