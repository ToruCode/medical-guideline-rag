import dataclasses
import logging
from pathlib import Path

import pytest
from app.application.services.chunk_document import ChunkDocumentService
from app.application.services.embed_chunks import EmbedChunksService
from app.application.services.index_chunks import IndexChunksService
from app.application.services.index_document import IndexDocumentResult, IndexDocumentService
from app.application.services.load_document import LoadDocumentService
from app.domain.exceptions.document import DocumentNotFoundError
from app.domain.exceptions.embedding import EmbeddingCountMismatchError
from app.domain.exceptions.vector_store import VectorDimensionMismatchError
from app.domain.models.chunk import Chunk
from app.domain.models.document import DocumentPage
from app.domain.models.embedding import EmbeddedChunk
from app.infrastructure.embedding.fake_embedder import FakeEmbedder, FixedVectorsEmbedder
from app.infrastructure.vector_store.in_memory_vector_store import InMemoryVectorStore


class FakePdfLoader:
    def __init__(
        self, pages: list[DocumentPage] | None = None, error: Exception | None = None
    ) -> None:
        self._pages = pages if pages is not None else []
        self._error = error

    def load(self, file_path: Path) -> list[DocumentPage]:
        if self._error is not None:
            raise self._error
        return self._pages


class FakeTextSplitter:
    def __init__(self, fragments_by_text: dict[str, list[str]]) -> None:
        self._fragments_by_text = fragments_by_text

    def split(self, text: str) -> list[str]:
        return self._fragments_by_text.get(text, [])


def _make_page(
    page_number: int, text: str, document_id: str = "doc-1", title: str | None = "Guideline"
) -> DocumentPage:
    return DocumentPage(
        document_id=document_id,
        source_name="sample.pdf",
        source_path="/tmp/sample.pdf",
        page_number=page_number,
        text=text,
        title=title,
    )


def _build_service(
    pdf_loader: FakePdfLoader,
    fragments_by_text: dict[str, list[str]],
    embedder: object = None,
    vector_store: InMemoryVectorStore | None = None,
) -> tuple[IndexDocumentService, InMemoryVectorStore]:
    store = vector_store if vector_store is not None else InMemoryVectorStore()
    service = IndexDocumentService(
        load_document=LoadDocumentService(pdf_loader),
        chunk_document=ChunkDocumentService(FakeTextSplitter(fragments_by_text)),
        embed_chunks=EmbedChunksService(embedder if embedder is not None else FakeEmbedder()),
        index_chunks=IndexChunksService(store),
    )
    return service, store


def test_execute_returns_counts_for_multi_page_multi_chunk_document() -> None:
    pages = [_make_page(1, "page one text"), _make_page(2, "page two text")]
    fragments = {"page one text": ["p1-a", "p1-b"], "page two text": ["p2-a"]}
    service, store = _build_service(FakePdfLoader(pages), fragments)

    result = service.execute(Path("/tmp/sample.pdf"))

    assert result.document_id == "doc-1"
    assert result.source_name == "sample.pdf"
    assert result.page_count == 2
    assert result.chunk_count == 3
    assert result.indexed_count == 3
    assert len(store.search([0.0, 0.0, 0.0, 0.0], top_k=10)) == 3


def test_execute_with_zero_page_document_returns_zeroed_result() -> None:
    service, store = _build_service(FakePdfLoader([]), {})

    result = service.execute(Path("/tmp/sample.pdf"))

    assert result == IndexDocumentResult(
        document_id=None,
        source_name="sample.pdf",
        page_count=0,
        chunk_count=0,
        indexed_count=0,
    )
    assert store.search([0.0, 0.0, 0.0, 0.0], top_k=10) == []


def test_execute_with_all_empty_pages_produces_no_chunks() -> None:
    pages = [_make_page(1, ""), _make_page(2, "")]
    service, store = _build_service(FakePdfLoader(pages), {"": []})

    result = service.execute(Path("/tmp/sample.pdf"))

    assert result.page_count == 2
    assert result.chunk_count == 0
    assert result.indexed_count == 0
    assert store.search([0.0, 0.0, 0.0, 0.0], top_k=10) == []


def test_execute_twice_on_same_document_does_not_duplicate_entries() -> None:
    pages = [_make_page(1, "page one text")]
    fragments = {"page one text": ["p1-a", "p1-b"]}
    service, store = _build_service(FakePdfLoader(pages), fragments)

    service.execute(Path("/tmp/sample.pdf"))
    service.execute(Path("/tmp/sample.pdf"))

    assert len(store.search([0.0, 0.0, 0.0, 0.0], top_k=10)) == 2


def test_execute_propagates_embedding_count_mismatch_error() -> None:
    pages = [_make_page(1, "page one text")]
    fragments = {"page one text": ["p1-a", "p1-b"]}
    embedder = FixedVectorsEmbedder(vectors=[[0.1, 0.2]])
    service, _ = _build_service(FakePdfLoader(pages), fragments, embedder=embedder)

    with pytest.raises(EmbeddingCountMismatchError):
        service.execute(Path("/tmp/sample.pdf"))


def test_execute_propagates_vector_dimension_mismatch_error() -> None:
    store = InMemoryVectorStore()
    seed_chunk = Chunk(
        document_id="doc-0",
        source_name="seed.pdf",
        source_path="/tmp/seed.pdf",
        page_number=1,
        chunk_index=0,
        text="seed",
        title=None,
    )
    store.upsert([EmbeddedChunk(chunk=seed_chunk, vector=[0.0, 0.0])])

    pages = [_make_page(1, "page one text")]
    fragments = {"page one text": ["p1-a"]}
    embedder = FixedVectorsEmbedder(vectors=[[0.1, 0.2, 0.3]])
    service, _ = _build_service(
        FakePdfLoader(pages), fragments, embedder=embedder, vector_store=store
    )

    with pytest.raises(VectorDimensionMismatchError):
        service.execute(Path("/tmp/sample.pdf"))


def test_execute_propagates_pdf_loader_error() -> None:
    service, _ = _build_service(FakePdfLoader(error=DocumentNotFoundError("missing")), {})

    with pytest.raises(DocumentNotFoundError):
        service.execute(Path("/tmp/missing.pdf"))


def test_execute_logs_do_not_contain_chunk_text_or_vector(caplog: pytest.LogCaptureFixture) -> None:
    pages = [_make_page(1, "do-not-leak-this-text")]
    fragments = {"do-not-leak-this-text": ["do-not-leak-this-text"]}
    service, _ = _build_service(FakePdfLoader(pages), fragments)

    with caplog.at_level(logging.INFO):
        service.execute(Path("/tmp/sample.pdf"))

    log_output = "\n".join(caplog.messages)
    assert "do-not-leak-this-text" not in log_output


def test_index_document_result_is_frozen() -> None:
    result = IndexDocumentResult(
        document_id="doc-1",
        source_name="sample.pdf",
        page_count=1,
        chunk_count=1,
        indexed_count=1,
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.chunk_count = 2  # type: ignore[misc]
