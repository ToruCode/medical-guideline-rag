from app.application.services.generate_answer import GenerateAnswerService
from app.application.services.retrieve_chunks import RetrieveChunksService
from app.application.services.search_chunks import SearchChunksService
from app.domain.models.chunk import Chunk
from app.domain.models.embedding import EmbeddedChunk
from tests.support.fake_llm import FakeLlm
from tests.support.in_memory_vector_store import InMemoryVectorStore


class QueryAwareEmbedder:
    """Maps known query/chunk texts to hand-picked vectors, so retrieval
    ranking between the query and indexed chunks is predictable.
    """

    def __init__(self, vectors_by_text: dict[str, list[float]]) -> None:
        self._vectors_by_text = vectors_by_text

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors_by_text[text] for text in texts]


def _make_chunk(
    chunk_index: int, text: str, page_number: int = 1, title: str | None = "Sample Guideline"
) -> Chunk:
    return Chunk(
        document_id="doc-1",
        source_name="guideline.pdf",
        source_path="/tmp/guideline.pdf",
        page_number=page_number,
        chunk_index=chunk_index,
        text=text,
        title=title,
    )


def test_retrieve_then_generate_returns_answer_grounded_in_retrieved_chunks() -> None:
    relevant_chunk = _make_chunk(0, "recommended dosage guidance", page_number=4)
    unrelated_chunk = _make_chunk(1, "unrelated administrative text", page_number=9)

    embedder = QueryAwareEmbedder(
        {
            "what is the recommended dosage": [1.0, 0.0],
            "recommended dosage guidance": [1.0, 0.0],
            "unrelated administrative text": [0.0, 1.0],
        }
    )
    store = InMemoryVectorStore()
    store.upsert(
        [
            EmbeddedChunk(
                chunk=relevant_chunk,
                vector=embedder.embed(["recommended dosage guidance"])[0],
            ),
            EmbeddedChunk(
                chunk=unrelated_chunk,
                vector=embedder.embed(["unrelated administrative text"])[0],
            ),
        ]
    )
    retrieve_chunks = RetrieveChunksService(
        embedder=embedder, search_chunks=SearchChunksService(store)
    )
    llm = FakeLlm(answer="Follow the dosage in [1].")
    generate_answer = GenerateAnswerService(llm)

    question = "what is the recommended dosage"
    search_results = retrieve_chunks.execute(question, top_k=1)
    result = generate_answer.execute(question, search_results)

    assert result.is_insufficient_evidence is False
    assert result.answer == "Follow the dosage in [1]."
    assert len(result.citations) == 1
    assert result.citations[0].embedded_chunk.chunk.chunk_id == relevant_chunk.chunk_id
    assert llm.received_prompt is not None
    assert "recommended dosage guidance" in llm.received_prompt


def test_retrieve_then_generate_with_no_indexed_chunks_returns_insufficient_evidence() -> None:
    embedder = QueryAwareEmbedder({"a question with no matches": [1.0, 0.0]})
    store = InMemoryVectorStore()
    retrieve_chunks = RetrieveChunksService(
        embedder=embedder, search_chunks=SearchChunksService(store)
    )
    llm = FakeLlm()
    generate_answer = GenerateAnswerService(llm)

    question = "a question with no matches"
    search_results = retrieve_chunks.execute(question, top_k=5)
    result = generate_answer.execute(question, search_results)

    assert result.is_insufficient_evidence is True
    assert result.citations == []
    assert llm.received_prompt is None
