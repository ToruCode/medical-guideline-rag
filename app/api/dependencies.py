"""Dependency providers for the API layer.

This is the only module the API layer uses to construct Infrastructure
implementations and compose them into Application services; endpoint
modules depend only on these provider functions via Depends(...), never
importing Infrastructure directly.

get_passage_embedder/get_query_embedder/get_vector_store/get_llm are
process-wide singletons (via lru_cache), shared across all requests for
the life of the process, so a document indexed through
POST /documents/index is searchable via POST /questions/ask. Tests must
clear these caches between test functions (see tests/conftest.py) to
avoid state leaking across tests.

get_passage_embedder/get_query_embedder/get_llm read Settings by calling
get_settings() directly rather than declaring it as a Depends(...)
parameter: Settings is not hashable (it is not frozen), and lru_cache
requires hashable arguments, so these provider functions stay
zero-argument. get_settings() is itself cached, so this still reflects
whatever Settings were current the first time each provider was called.
"""

from functools import lru_cache
from typing import TYPE_CHECKING, Annotated

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
from app.infrastructure.embedding.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
    load_sentence_transformer_model,
)
from app.infrastructure.llm.fake_llm import FakeLlm
from app.infrastructure.llm.openai_llm import OpenAiLlm
from app.infrastructure.pdf.pymupdf_loader import PyMuPdfLoader
from app.infrastructure.pdf.pypdf_loader import PypdfLoader
from app.infrastructure.vector_store.in_memory_vector_store import InMemoryVectorStore

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


def get_pdf_loader(settings: Annotated[Settings, Depends(get_settings)]) -> PdfLoader:
    """Selects the PdfLoader implementation via Settings.pdf_extractor.

    Raises ValueError for an unrecognized value, mirroring get_llm's
    fail-fast behavior. Settings.pdf_extractor is Literal-typed, so
    pydantic already rejects unknown values from the environment at
    Settings() construction time; this branch is defense-in-depth for
    callers that construct/monkeypatch a Settings-like object directly.
    """
    if settings.pdf_extractor == "pymupdf":
        return PyMuPdfLoader()
    if settings.pdf_extractor == "pypdf":
        return PypdfLoader()
    raise ValueError(f"Unknown pdf_extractor: {settings.pdf_extractor!r}")


def get_text_splitter(settings: Annotated[Settings, Depends(get_settings)]) -> TextSplitter:
    return FixedSizeTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )


@lru_cache
def _get_sentence_transformer_model() -> "SentenceTransformer":
    """Process-wide loaded model, shared by both get_passage_embedder and
    get_query_embedder so the (large) model weights are loaded once, not
    once per Embedder instance.
    """
    return load_sentence_transformer_model(get_settings().embedding_model_name)


def _build_sentence_transformer_backed_embedder(settings: Settings, prefix: str) -> Embedder:
    if settings.embedding_provider == "fake":
        return FakeEmbedder()
    if settings.embedding_provider == "sentence_transformers":
        return SentenceTransformerEmbedder(_get_sentence_transformer_model(), prefix=prefix)
    raise ValueError(f"Unknown embedding_provider: {settings.embedding_provider!r}")


@lru_cache
def get_passage_embedder() -> Embedder:
    """Process-wide Embedder for indexed chunk text (POST /documents/index).

    Uses a "passage: " prefix under the sentence_transformers provider,
    matching the query:/passage: convention required by asymmetric
    retrieval models such as intfloat/multilingual-e5-*. See
    docs/adr/0011-real-embedding-and-llm-adapters.md.
    """
    return _build_sentence_transformer_backed_embedder(get_settings(), prefix="passage: ")


@lru_cache
def get_query_embedder() -> Embedder:
    """Process-wide Embedder for search queries (POST /questions/ask).

    Uses a "query: " prefix under the sentence_transformers provider; see
    get_passage_embedder and docs/adr/0011-real-embedding-and-llm-adapters.md.
    """
    return _build_sentence_transformer_backed_embedder(get_settings(), prefix="query: ")


@lru_cache
def get_vector_store() -> VectorStore:
    """Process-wide InMemoryVectorStore instance, shared by both endpoints.

    Data does not survive a process restart. No real vector database
    adapter exists yet (see docs/adr/0006-vector-store-strategy.md).
    """
    return InMemoryVectorStore()


@lru_cache
def get_llm() -> Llm:
    """Process-wide Llm instance, selected by Settings.llm_provider.

    Raises ValueError at construction time (not on the first request)
    when llm_provider is "openai" but no API key is configured, so a
    misconfiguration fails fast and clearly rather than as an opaque
    authentication error from the OpenAI SDK. See
    docs/adr/0011-real-embedding-and-llm-adapters.md.
    """
    settings = get_settings()
    if settings.llm_provider == "fake":
        return FakeLlm()
    if settings.llm_provider == "openai":
        if settings.llm_api_key is None:
            raise ValueError(
                "MEDICAL_RAG_LLM_API_KEY must be set when MEDICAL_RAG_LLM_PROVIDER=openai"
            )
        return OpenAiLlm(
            api_key=settings.llm_api_key.get_secret_value(),
            model=settings.llm_model_name,
            timeout=settings.llm_timeout_seconds,
        )
    raise ValueError(f"Unknown llm_provider: {settings.llm_provider!r}")


def get_load_document_service(
    pdf_loader: Annotated[PdfLoader, Depends(get_pdf_loader)],
) -> LoadDocumentService:
    return LoadDocumentService(pdf_loader)


def get_chunk_document_service(
    text_splitter: Annotated[TextSplitter, Depends(get_text_splitter)],
) -> ChunkDocumentService:
    return ChunkDocumentService(text_splitter)


def get_embed_chunks_service(
    embedder: Annotated[Embedder, Depends(get_passage_embedder)],
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
    embedder: Annotated[Embedder, Depends(get_query_embedder)],
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
