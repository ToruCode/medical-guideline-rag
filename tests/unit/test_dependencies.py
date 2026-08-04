"""Unit tests for the API layer's provider-selection logic
(app/api/dependencies.py).

Never loads a real sentence-transformers model and never makes a real
OpenAI network call: sentence_transformers.SentenceTransformer is
monkeypatched where needed, and constructing an OpenAI client (unlike
calling it) performs no network I/O.
"""

from pathlib import Path
from typing import Any

import pytest
from app.api import dependencies
from app.core.config import get_settings
from app.infrastructure.embedding.fake_embedder import FakeEmbedder
from app.infrastructure.embedding.sentence_transformer_embedder import SentenceTransformerEmbedder
from app.infrastructure.llm.fake_llm import FakeLlm
from app.infrastructure.llm.openai_llm import OpenAiLlm
from app.infrastructure.pdf.pymupdf_loader import PyMuPdfLoader
from app.infrastructure.pdf.pypdf_loader import PypdfLoader
from app.infrastructure.vector_store.in_memory_vector_store import InMemoryVectorStore
from app.infrastructure.vector_store.qdrant_vector_store import QdrantVectorStore


def test_get_passage_embedder_returns_fake_embedder_by_default() -> None:
    embedder = dependencies.get_passage_embedder()

    assert isinstance(embedder, FakeEmbedder)


def test_get_query_embedder_returns_fake_embedder_by_default() -> None:
    embedder = dependencies.get_query_embedder()

    assert isinstance(embedder, FakeEmbedder)


def test_passage_and_query_embedder_share_one_loaded_model_under_sentence_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed_model_names: list[str] = []

    class FakeSentenceTransformerClass:
        def __init__(self, model_name: str) -> None:
            constructed_model_names.append(model_name)

        def encode(self, texts: list[str], convert_to_numpy: bool = True) -> Any:
            raise NotImplementedError

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", FakeSentenceTransformerClass)
    monkeypatch.setenv("MEDICAL_RAG_EMBEDDING_PROVIDER", "sentence_transformers")
    monkeypatch.setenv("MEDICAL_RAG_EMBEDDING_MODEL_NAME", "fake-model-name")

    passage_embedder = dependencies.get_passage_embedder()
    query_embedder = dependencies.get_query_embedder()

    assert isinstance(passage_embedder, SentenceTransformerEmbedder)
    assert isinstance(query_embedder, SentenceTransformerEmbedder)
    assert constructed_model_names == ["fake-model-name"]


def test_build_sentence_transformer_backed_embedder_raises_for_unknown_provider() -> None:
    class _FakeSettingsWithUnknownProvider:
        embedding_provider = "not-a-real-provider"

    with pytest.raises(ValueError, match="Unknown embedding_provider"):
        dependencies._build_sentence_transformer_backed_embedder(
            _FakeSettingsWithUnknownProvider(),  # type: ignore[arg-type]
            prefix="",
        )


def test_get_llm_returns_fake_llm_by_default() -> None:
    llm = dependencies.get_llm()

    assert isinstance(llm, FakeLlm)


def test_get_llm_with_openai_provider_and_no_api_key_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEDICAL_RAG_LLM_PROVIDER", "openai")

    with pytest.raises(ValueError, match="MEDICAL_RAG_LLM_API_KEY"):
        dependencies.get_llm()


def test_get_llm_with_openai_provider_and_api_key_returns_openai_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEDICAL_RAG_LLM_PROVIDER", "openai")
    monkeypatch.setenv("MEDICAL_RAG_LLM_API_KEY", "sk-test")

    llm = dependencies.get_llm()

    assert isinstance(llm, OpenAiLlm)


def test_get_llm_raises_for_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeSettingsWithUnknownProvider:
        llm_provider = "not-a-real-provider"

    monkeypatch.setattr(dependencies, "get_settings", lambda: _FakeSettingsWithUnknownProvider())

    with pytest.raises(ValueError, match="Unknown llm_provider"):
        dependencies.get_llm()


def test_get_pdf_loader_returns_pymupdf_loader_by_default() -> None:
    pdf_loader = dependencies.get_pdf_loader(get_settings())

    assert isinstance(pdf_loader, PyMuPdfLoader)


def test_get_pdf_loader_with_pypdf_setting_returns_pypdf_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEDICAL_RAG_PDF_EXTRACTOR", "pypdf")

    pdf_loader = dependencies.get_pdf_loader(get_settings())

    assert isinstance(pdf_loader, PypdfLoader)


def test_get_pdf_loader_raises_for_unknown_extractor() -> None:
    class _FakeSettingsWithUnknownExtractor:
        pdf_extractor = "not-a-real-extractor"

    with pytest.raises(ValueError, match="Unknown pdf_extractor"):
        dependencies.get_pdf_loader(_FakeSettingsWithUnknownExtractor())  # type: ignore[arg-type]


def test_get_vector_store_returns_in_memory_vector_store_by_default() -> None:
    vector_store = dependencies.get_vector_store()

    assert isinstance(vector_store, InMemoryVectorStore)


def test_get_vector_store_with_qdrant_provider_returns_qdrant_vector_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MEDICAL_RAG_VECTOR_STORE_PROVIDER", "qdrant")
    monkeypatch.setenv("MEDICAL_RAG_VECTOR_STORE_PATH", str(tmp_path / "qdrant"))

    vector_store = dependencies.get_vector_store()

    assert isinstance(vector_store, QdrantVectorStore)
    vector_store.close()


def test_get_vector_store_raises_for_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeSettingsWithUnknownProvider:
        vector_store_provider = "not-a-real-provider"

    monkeypatch.setattr(dependencies, "get_settings", lambda: _FakeSettingsWithUnknownProvider())

    with pytest.raises(ValueError, match="Unknown vector_store_provider"):
        dependencies.get_vector_store()


def test_get_generate_answer_service_uses_configured_context_max_chars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEDICAL_RAG_LLM_CONTEXT_MAX_CHARS", "1234")

    service = dependencies.get_generate_answer_service(FakeLlm(), get_settings())

    assert service._context_max_chars == 1234
