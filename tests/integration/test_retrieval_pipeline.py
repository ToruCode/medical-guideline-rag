from app.application.services.retrieve_chunks import RetrieveChunksService
from app.application.services.search_chunks import SearchChunksService
from app.domain.models.chunk import Chunk
from app.domain.models.embedding import EmbeddedChunk
from tests.support.in_memory_vector_store import InMemoryVectorStore


class QueryAwareEmbedder:
    """Maps known query/chunk texts to hand-picked vectors, so similarity
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


def test_retrieve_chunks_returns_top_k_in_similarity_order() -> None:
    close_chunk = _make_chunk(0, "closely related passage", page_number=1)
    far_chunk = _make_chunk(1, "unrelated passage", page_number=2)
    middle_chunk = _make_chunk(2, "somewhat related passage", page_number=3)

    embedder = QueryAwareEmbedder(
        {
            "query about the closely related topic": [1.0, 0.0],
            "closely related passage": [1.0, 0.0],
            "somewhat related passage": [0.7, 0.7],
            "unrelated passage": [0.0, 1.0],
        }
    )
    store = InMemoryVectorStore()
    store.upsert(
        [
            EmbeddedChunk(chunk=close_chunk, vector=embedder.embed(["closely related passage"])[0]),
            EmbeddedChunk(chunk=far_chunk, vector=embedder.embed(["unrelated passage"])[0]),
            EmbeddedChunk(
                chunk=middle_chunk, vector=embedder.embed(["somewhat related passage"])[0]
            ),
        ]
    )
    service = RetrieveChunksService(embedder=embedder, search_chunks=SearchChunksService(store))

    results = service.execute("query about the closely related topic", top_k=2)

    assert len(results) == 2
    assert results[0].embedded_chunk.chunk.chunk_id == close_chunk.chunk_id
    assert results[1].embedded_chunk.chunk.chunk_id == middle_chunk.chunk_id
    assert results[0].score >= results[1].score


def test_retrieve_chunks_preserves_citation_metadata() -> None:
    chunk = _make_chunk(0, "guideline passage", page_number=5, title="Sample Guideline")
    embedder = QueryAwareEmbedder(
        {
            "a query": [1.0, 0.0],
            "guideline passage": [1.0, 0.0],
        }
    )
    store = InMemoryVectorStore()
    store.upsert([EmbeddedChunk(chunk=chunk, vector=embedder.embed(["guideline passage"])[0])])
    service = RetrieveChunksService(embedder=embedder, search_chunks=SearchChunksService(store))

    results = service.execute("a query", top_k=5)

    assert len(results) == 1
    retrieved = results[0].embedded_chunk.chunk
    assert retrieved.document_id == "doc-1"
    assert retrieved.source_name == "guideline.pdf"
    assert retrieved.page_number == 5
    assert retrieved.chunk_index == 0
    assert retrieved.title == "Sample Guideline"


def test_retrieve_chunks_with_no_matching_documents_returns_empty_list() -> None:
    embedder = QueryAwareEmbedder({"a query": [1.0, 0.0]})
    store = InMemoryVectorStore()
    service = RetrieveChunksService(embedder=embedder, search_chunks=SearchChunksService(store))

    results = service.execute("a query", top_k=5)

    assert results == []
