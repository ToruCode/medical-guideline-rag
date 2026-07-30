from pathlib import Path

from app.application.services.chunk_document import ChunkDocumentService
from app.application.services.embed_chunks import EmbedChunksService
from app.application.services.index_chunks import IndexChunksService
from app.application.services.index_document import IndexDocumentService
from app.application.services.load_document import LoadDocumentService
from app.infrastructure.chunking.fixed_size_text_splitter import FixedSizeTextSplitter
from app.infrastructure.pdf.pypdf_loader import PypdfLoader
from tests.support.fake_embedder import FakeEmbedder
from tests.support.in_memory_vector_store import InMemoryVectorStore
from tests.support.pdf_factory import build_pdf


def _build_service(vector_store: InMemoryVectorStore) -> IndexDocumentService:
    return IndexDocumentService(
        load_document=LoadDocumentService(PypdfLoader()),
        chunk_document=ChunkDocumentService(FixedSizeTextSplitter(chunk_size=20, chunk_overlap=0)),
        embed_chunks=EmbedChunksService(FakeEmbedder()),
        index_chunks=IndexChunksService(vector_store),
    )


def test_pipeline_indexes_real_pdf_and_is_searchable(tmp_path: Path) -> None:
    pdf_path = build_pdf(
        tmp_path / "guideline.pdf",
        pages=["first page guideline content", "second page guideline content"],
        title="Sample Guideline",
    )
    store = InMemoryVectorStore()
    service = _build_service(store)

    result = service.execute(pdf_path)

    assert result.page_count == 2
    assert result.chunk_count > 0
    assert result.indexed_count == result.chunk_count
    assert result.document_id is not None

    query_vector = [0.0] * 4
    results = store.search(query_vector, top_k=result.chunk_count)
    assert len(results) == result.chunk_count
    assert all(r.embedded_chunk.chunk.document_id == result.document_id for r in results)


def test_pipeline_with_blank_page_reduces_chunk_count(tmp_path: Path) -> None:
    pdf_path = build_pdf(
        tmp_path / "guideline_with_blank_page.pdf",
        pages=["guideline text", ""],
        title="Sample Guideline",
    )
    store = InMemoryVectorStore()
    service = _build_service(store)

    result = service.execute(pdf_path)

    assert result.page_count == 2
    assert result.chunk_count == 1
    assert result.indexed_count == 1
