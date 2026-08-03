import dataclasses
import logging

import pytest
from app.application.services.generate_answer import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    GenerateAnswerService,
    GenerationResult,
    build_context,
    select_chunks_within_budget,
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
        _make_search_result(text="first passage", page_number=1),
        _make_search_result(text="second passage", page_number=2),
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


# --- select_chunks_within_budget ---


def test_select_chunks_within_budget_keeps_all_when_under_budget() -> None:
    results = [
        _make_search_result(text="short one", page_number=1),
        _make_search_result(text="short two", page_number=2),
    ]

    selected = select_chunks_within_budget(results, max_chars=10_000)

    assert selected == results


def test_select_chunks_within_budget_drops_chunks_exceeding_the_limit() -> None:
    results = [
        _make_search_result(text="a" * 50, page_number=1, chunk_index=0),
        _make_search_result(text="b" * 50, page_number=2, chunk_index=0),
        _make_search_result(text="c" * 50, page_number=3, chunk_index=0),
    ]
    # Budget for exactly the first result's formatted block, nothing more.
    budget = len(build_context([results[0]]))

    selected = select_chunks_within_budget(results, max_chars=budget)

    assert selected == [results[0]]


def test_select_chunks_within_budget_preserves_page_number_and_chunk_index() -> None:
    results = [_make_search_result(text="x" * 50, page_number=7, chunk_index=2)]

    selected = select_chunks_within_budget(results, max_chars=len(build_context(results)))

    assert selected[0].embedded_chunk.chunk.page_number == 7
    assert selected[0].embedded_chunk.chunk.chunk_index == 2


def test_select_chunks_within_budget_always_keeps_first_result_even_if_oversized() -> None:
    oversized = _make_search_result(text="z" * 10_000)

    selected = select_chunks_within_budget([oversized], max_chars=10)

    assert selected == [oversized]


def test_select_chunks_within_budget_deduplicates_by_chunk_id() -> None:
    original = _make_search_result(text="same chunk", page_number=1, chunk_index=0)
    duplicate = _make_search_result(text="same chunk", page_number=1, chunk_index=0)

    selected = select_chunks_within_budget([original, duplicate], max_chars=10_000)

    assert selected == [original]


def test_select_chunks_within_budget_empty_input_returns_empty() -> None:
    assert select_chunks_within_budget([], max_chars=1000) == []


# --- GenerateAnswerService: context budget integration ---


def test_execute_citations_reflect_only_chunks_that_fit_the_budget() -> None:
    fitting = _make_search_result(text="fits", page_number=1, chunk_index=0)
    dropped = _make_search_result(text="y" * 10_000, page_number=2, chunk_index=0)
    llm = FakeLlm(answer="the answer")
    service = GenerateAnswerService(llm, context_max_chars=len(build_context([fitting])))

    result = service.execute("a question", search_results=[fitting, dropped])

    assert result.citations == [fitting]
    assert dropped.embedded_chunk.chunk.text not in (llm.received_prompt or "")


def test_execute_default_context_budget_keeps_small_fixtures_unchanged() -> None:
    results = [
        _make_search_result(text="first", page_number=1),
        _make_search_result(text="second", page_number=2),
    ]
    llm = FakeLlm(answer="the answer")
    service = GenerateAnswerService(llm)

    result = service.execute("a question", search_results=results)

    assert result.citations == results


# --- Prompt safety strengthening ---


def test_prompt_instructs_answering_in_japanese() -> None:
    llm = FakeLlm(answer="the answer")
    service = GenerateAnswerService(llm)

    service.execute("a question", search_results=[_make_search_result()])

    assert "Japanese" in (llm.received_prompt or "")


def test_prompt_treats_context_as_reference_not_instructions() -> None:
    llm = FakeLlm(answer="the answer")
    service = GenerateAnswerService(llm)

    service.execute("a question", search_results=[_make_search_result()])

    prompt = llm.received_prompt or ""
    assert "reference material" in prompt
    assert "not instructions" in prompt


def test_prompt_forbids_diagnosis_and_treatment_decisions() -> None:
    llm = FakeLlm(answer="the answer")
    service = GenerateAnswerService(llm)

    service.execute("a question", search_results=[_make_search_result()])

    prompt = llm.received_prompt or ""
    assert "diagnosis" in prompt
    assert "treatment decision" in prompt
