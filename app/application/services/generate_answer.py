"""Use case for generating a citation-grounded answer from retrieved chunks."""

import logging
from dataclasses import dataclass

from app.domain.exceptions.retrieval import EmptyQueryError
from app.domain.models.search_result import SearchResult
from app.domain.ports.llm import Llm

logger = logging.getLogger(__name__)

INSUFFICIENT_EVIDENCE_ANSWER = (
    "The retrieved guideline passages do not contain sufficient evidence to "
    "answer this question. Please consult the original guideline and current "
    "clinical information."
)

_PROMPT_INSTRUCTIONS = (
    "Answer the question using only the numbered guideline passages below. "
    "Cite passages inline using their [n] number. Do not use any knowledge "
    "beyond what the passages state. If the passages do not contain enough "
    "information to answer, say so explicitly instead of guessing."
)


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Summary of one GenerateAnswerService.execute() run.

    citations reuses SearchResult (the same type SearchChunksService and
    RetrieveChunksService already return) rather than introducing a
    separate Citation model, in the same order the passages were numbered
    in the prompt. is_insufficient_evidence is True only when
    search_results was empty and the Llm was never called; it does not
    reflect any semantic judgment of the Llm's own answer text
    (answerability judgment is out of scope for this use case).
    """

    answer: str
    citations: list[SearchResult]
    is_insufficient_evidence: bool


def build_context(search_results: list[SearchResult]) -> str:
    """Formats retrieved chunks into a numbered, citable context block.

    Numbering follows search_results' existing order (descending
    similarity), so citation [n] in the Llm's answer corresponds to
    citations[n - 1] in the returned GenerationResult.
    """
    blocks = []
    for index, result in enumerate(search_results, start=1):
        chunk = result.embedded_chunk.chunk
        source = chunk.title or chunk.source_name
        blocks.append(f"[{index}] {source}, page {chunk.page_number}\n{chunk.text}")
    return "\n\n".join(blocks)


class GenerateAnswerService:
    """Generates an answer to a question, grounded only in given search results.

    Delegates the actual text generation to an injected Llm. When
    search_results is empty, the Llm is never called and a fixed
    insufficient-evidence answer is returned instead, so a
    no-evidence case can never result in the Llm inventing an answer.
    top_k is not a parameter here: validating it is RetrieveChunksService's
    responsibility for whatever search_results this method is given.
    """

    def __init__(self, llm: Llm) -> None:
        self._llm = llm

    def execute(self, question: str, search_results: list[SearchResult]) -> GenerationResult:
        if not question.strip():
            raise EmptyQueryError("question must not be empty or whitespace-only")

        if not search_results:
            logger.info("generate_answer insufficient_evidence citation_count=0")
            return GenerationResult(
                answer=INSUFFICIENT_EVIDENCE_ANSWER,
                citations=[],
                is_insufficient_evidence=True,
            )

        context = build_context(search_results)
        prompt = f"{_PROMPT_INSTRUCTIONS}\n\n{context}\n\nQuestion: {question}"
        answer = self._llm.generate(prompt)

        logger.info("generate_answer completed citation_count=%d", len(search_results))
        return GenerationResult(
            answer=answer,
            citations=search_results,
            is_insufficient_evidence=False,
        )
