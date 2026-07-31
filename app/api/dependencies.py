"""Dependency providers for the API layer.

This is the only module the API layer uses to construct Infrastructure
implementations and compose them into Application services; endpoint
modules depend only on these provider functions via Depends(...), never
importing Infrastructure directly.

get_embedder/get_vector_store/get_llm are process-wide singletons (via
lru_cache), shared across all requests for the life of the process, so
a document indexed through POST /documents/index is searchable via
POST /questions/ask. Tests must clear these caches between test
functions (see tests/conftest.py) to avoid state leaking across tests.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.application.services.ask_question import AskQuestionService
from app.application.services.chunk_document import ChunkDocumentService
from app.application.services.embed_chunks import EmbedChunksService
from app.application.services.generate_answer import GenerateAnswerService
from app.application.services.index_chunks import IndexChunksService
from app.application.services.index_document import IndexDocumentService
from app.application.services.load_document import LoadDocumentService
from app.application.services.retrieve_chunks import RetrieveChunksService
from app.application.services.search_chunks import SearchChunksService
from app.core.config import Settings, get_settings
from app.domain.ports.embedder import Embedder
from app.domain.ports.llm import Llm
from app.domain.ports.pdf_loader import PdfLoader
from app.domain.ports.text_splitter import TextSplitter
from app.domain.ports.vector_store import VectorStore
from app.infrastructure.chunking.fixed_size_text_splitter import FixedSizeTextSplitter
from app.infrastructure.embedding.fake_embedder import FakeEmbedder
from app.infrastructure.llm.fake_llm import FakeLlm
from app.infrastructure.pdf.pypdf_loader import PypdfLoader
from app.infrastructure.vector_store.in_memory_vector_store import InMemoryVectorStore


def get_pdf_loader() -> PdfLoader:
    return PypdfLoader()


def get_text_splitter(settings: Annotated[Settings, Depends(get_settings)]) -> TextSplitter:
    return FixedSizeTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )


@lru_cache
def get_embedder() -> Embedder:
    """Process-wide FakeEmbedder instance.

    No real embedding model adapter exists yet (see
    docs/adr/0005-embedding-strategy.md); this wires the existing Fake
    implementation so the API is usable end to end today.
    """
    return FakeEmbedder()


@lru_cache
def get_vector_store() -> VectorStore:
    """Process-wide InMemoryVectorStore instance, shared by both endpoints.

    Data does not survive a process restart. No real vector database
    adapter exists yet (see docs/adr/0006-vector-store-strategy.md).
    """
    return InMemoryVectorStore()


@lru_cache
def get_llm() -> Llm:
    """Process-wide FakeLlm instance.

    No real LLM adapter exists yet (see
    docs/adr/0009-generation-strategy.md); this wires the existing Fake
    implementation so the API is usable end to end today.
    """
    return FakeLlm()


def get_load_document_service(
    pdf_loader: Annotated[PdfLoader, Depends(get_pdf_loader)],
) -> LoadDocumentService:
    return LoadDocumentService(pdf_loader)


def get_chunk_document_service(
    text_splitter: Annotated[TextSplitter, Depends(get_text_splitter)],
) -> ChunkDocumentService:
    return ChunkDocumentService(text_splitter)


def get_embed_chunks_service(
    embedder: Annotated[Embedder, Depends(get_embedder)],
) -> EmbedChunksService:
    return EmbedChunksService(embedder)


def get_index_chunks_service(
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
) -> IndexChunksService:
    return IndexChunksService(vector_store)


def get_index_document_service(
    load_document: Annotated[LoadDocumentService, Depends(get_load_document_service)],
    chunk_document: Annotated[ChunkDocumentService, Depends(get_chunk_document_service)],
    embed_chunks: Annotated[EmbedChunksService, Depends(get_embed_chunks_service)],
    index_chunks: Annotated[IndexChunksService, Depends(get_index_chunks_service)],
) -> IndexDocumentService:
    return IndexDocumentService(
        load_document=load_document,
        chunk_document=chunk_document,
        embed_chunks=embed_chunks,
        index_chunks=index_chunks,
    )


def get_search_chunks_service(
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
) -> SearchChunksService:
    return SearchChunksService(vector_store)


def get_retrieve_chunks_service(
    embedder: Annotated[Embedder, Depends(get_embedder)],
    search_chunks: Annotated[SearchChunksService, Depends(get_search_chunks_service)],
) -> RetrieveChunksService:
    return RetrieveChunksService(embedder=embedder, search_chunks=search_chunks)


def get_generate_answer_service(
    llm: Annotated[Llm, Depends(get_llm)],
) -> GenerateAnswerService:
    return GenerateAnswerService(llm)


def get_ask_question_service(
    retrieve_chunks: Annotated[RetrieveChunksService, Depends(get_retrieve_chunks_service)],
    generate_answer: Annotated[GenerateAnswerService, Depends(get_generate_answer_service)],
) -> AskQuestionService:
    return AskQuestionService(retrieve_chunks=retrieve_chunks, generate_answer=generate_answer)
