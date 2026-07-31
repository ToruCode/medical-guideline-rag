"""Use case for answering a question end to end: retrieve, then generate."""

from app.application.services.generate_answer import GenerateAnswerService, GenerationResult
from app.application.services.retrieve_chunks import RetrieveChunksService


class AskQuestionService:
    """Composes RetrieveChunksService and GenerateAnswerService to answer
    a question end to end.

    Does not catch any exception raised by either step; they propagate
    to the caller unchanged (fail-fast), matching IndexDocumentService's
    approach to composing existing Application services
    (docs/adr/0007-indexing-pipeline.md, docs/adr/0009-generation-strategy.md).
    """

    def __init__(
        self, retrieve_chunks: RetrieveChunksService, generate_answer: GenerateAnswerService
    ) -> None:
        self._retrieve_chunks = retrieve_chunks
        self._generate_answer = generate_answer

    def execute(self, question: str, top_k: int) -> GenerationResult:
        search_results = self._retrieve_chunks.execute(question, top_k)
        return self._generate_answer.execute(question, search_results)
