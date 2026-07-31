import pytest
from app.application.services.ask_question import AskQuestionService
from app.application.services.generate_answer import GenerationResult
from app.domain.exceptions.retrieval import EmptyQueryError
from app.domain.models.chunk import Chunk
from app.domain.models.embedding import EmbeddedChunk
from app.domain.models.search_result import SearchResult


class FakeRetrieveChunksService:
    def __init__(
        self, results: list[SearchResult] | None = None, error: Exception | None = None
    ) -> None:
        self._results = results if results is not None else []
        self._error = error
        self.received_question: str | None = None
        self.received_top_k: int | None = None

    def execute(self, question: str, top_k: int) -> list[SearchResult]:
        if self._error is not None:
            raise self._error
        self.received_question = question
        self.received_top_k = top_k
        return self._results


class FakeGenerateAnswerService:
    def __init__(
        self, result: GenerationResult | None = None, error: Exception | None = None
    ) -> None:
        self._result = result
        self._error = error
        self.received_question: str | None = None
        self.received_search_results: list[SearchResult] | None = None

    def execute(self, question: str, search_results: list[SearchResult]) -> GenerationResult:
        if self._error is not None:
            raise self._error
        self.received_question = question
        self.received_search_results = search_results
        assert self._result is not None
        return self._result


def _make_search_result() -> SearchResult:
    chunk = Chunk(
        document_id="doc-1",
        source_name="sample.pdf",
        source_path="/tmp/sample.pdf",
        page_number=1,
        chunk_index=0,
        text="passage text",
        title="Guideline",
    )
    return SearchResult(embedded_chunk=EmbeddedChunk(chunk=chunk, vector=[0.1, 0.2]), score=0.9)


def test_execute_passes_retrieved_results_from_retrieve_to_generate() -> None:
    search_results = [_make_search_result()]
    expected = GenerationResult(
        answer="the answer", citations=search_results, is_insufficient_evidence=False
    )
    retrieve_chunks = FakeRetrieveChunksService(results=search_results)
    generate_answer = FakeGenerateAnswerService(result=expected)
    service = AskQuestionService(retrieve_chunks, generate_answer)  # type: ignore[arg-type]

    result = service.execute("a question", top_k=3)

    assert result == expected
    assert retrieve_chunks.received_question == "a question"
    assert retrieve_chunks.received_top_k == 3
    assert generate_answer.received_question == "a question"
    assert generate_answer.received_search_results == search_results


def test_execute_propagates_retrieve_chunks_error_without_calling_generate_answer() -> None:
    retrieve_chunks = FakeRetrieveChunksService(error=EmptyQueryError("boom"))
    generate_answer = FakeGenerateAnswerService()
    service = AskQuestionService(retrieve_chunks, generate_answer)  # type: ignore[arg-type]

    with pytest.raises(EmptyQueryError):
        service.execute("", top_k=3)

    assert generate_answer.received_question is None


def test_execute_propagates_generate_answer_error() -> None:
    retrieve_chunks = FakeRetrieveChunksService(results=[_make_search_result()])
    generate_answer = FakeGenerateAnswerService(error=RuntimeError("llm unavailable"))
    service = AskQuestionService(retrieve_chunks, generate_answer)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError):
        service.execute("a question", top_k=3)
