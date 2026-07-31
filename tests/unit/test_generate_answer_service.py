import dataclasses
import logging

import pytest
from app.application.services.generate_answer import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    GenerateAnswerService,
    GenerationResult,
    build_context,
)
from app.domain.exceptions.retrieval import EmptyQueryError
from app.domain.models.chunk import Chunk
from app.domain.models.embedding import EmbeddedChunk
from app.domain.models.search_result import SearchResult
from app.infrastructure.llm.fake_llm import FakeLlm, RaisingLlm


def _make_search_result(
    text: str = "do-not-leak-this-chunk-text",
    title: str | None = "Sample Guideline",
    page_number: int = 1,
    chunk_index: int = 0,
    score: float = 0.9,
) -> SearchResult:
    chunk = Chunk(
        document_id="doc-1",
        source_name="sample.pdf",
        source_path="/tmp/sample.pdf",
        page_number=page_number,
        chunk_index=chunk_index,
        text=text,
        title=title,
    )
    return SearchResult(embedded_chunk=EmbeddedChunk(chunk=chunk, vector=[0.1, 0.2]), score=score)


def test_execute_passes_context_and_question_to_llm_and_returns_citations() -> None:
    results = [
        _make_search_result(text="first passage"),
        _make_search_result(text="second passage"),
    ]
    llm = FakeLlm(answer="the answer")
    service = GenerateAnswerService(llm)

    result = service.execute("do-not-leak-this-question", search_results=results)

    assert result == GenerationResult(
        answer="the answer", citations=results, is_insufficient_evidence=False
    )
    assert llm.received_prompt is not None
    assert "do-not-leak-this-question" in llm.received_prompt
    assert "first passage" in llm.received_prompt
    assert "second passage" in llm.received_prompt
    assert "[1]" in llm.received_prompt
    assert "[2]" in llm.received_prompt


def test_execute_with_no_search_results_returns_insufficient_evidence_without_calling_llm() -> None:
    llm = FakeLlm()
    service = GenerateAnswerService(llm)

    result = service.execute("a question", search_results=[])

    assert result == GenerationResult(
        answer=INSUFFICIENT_EVIDENCE_ANSWER, citations=[], is_insufficient_evidence=True
    )
    assert llm.received_prompt is None


@pytest.mark.parametrize("question", ["", "   ", "\t\n"])
def test_execute_with_empty_or_whitespace_question_raises_without_calling_llm(
    question: str,
) -> None:
    llm = FakeLlm()
    service = GenerateAnswerService(llm)

    with pytest.raises(EmptyQueryError):
        service.execute(question, search_results=[_make_search_result()])

    assert llm.received_prompt is None


def test_execute_propagates_llm_error() -> None:
    llm = RaisingLlm(RuntimeError("llm unavailable"))
    service = GenerateAnswerService(llm)

    with pytest.raises(RuntimeError):
        service.execute("a question", search_results=[_make_search_result()])


def test_execute_logs_counts_without_question_context_or_answer_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = FakeLlm(answer="do-not-leak-this-answer")
    service = GenerateAnswerService(llm)
    results = [_make_search_result(text="do-not-leak-this-chunk-text")]

    with caplog.at_level(logging.INFO):
        service.execute("do-not-leak-this-question", search_results=results)

    log_output = "\n".join(caplog.messages)
    assert "1" in log_output
    assert "do-not-leak-this-question" not in log_output
    assert "do-not-leak-this-chunk-text" not in log_output
    assert "do-not-leak-this-answer" not in log_output


def test_execute_insufficient_evidence_logs_without_question_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = FakeLlm()
    service = GenerateAnswerService(llm)

    with caplog.at_level(logging.INFO):
        service.execute("do-not-leak-this-question", search_results=[])

    log_output = "\n".join(caplog.messages)
    assert "do-not-leak-this-question" not in log_output


def test_build_context_numbers_passages_in_order_with_title_and_page() -> None:
    results = [
        _make_search_result(text="first passage", title="Guideline A", page_number=3),
        _make_search_result(text="second passage", title="Guideline B", page_number=7),
    ]

    context = build_context(results)

    assert context.index("[1] Guideline A, page 3") < context.index("first passage")
    assert context.index("[2] Guideline B, page 7") < context.index("second passage")


def test_build_context_falls_back_to_source_name_when_title_is_none() -> None:
    result = _make_search_result(text="passage text", title=None)

    context = build_context([result])

    assert "[1] sample.pdf, page 1" in context


def test_generation_result_is_frozen() -> None:
    result = GenerationResult(answer="a", citations=[], is_insufficient_evidence=False)

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.answer = "b"  # type: ignore[misc]
