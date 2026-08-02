"""Unit tests for the API layer's provider-selection logic
(app/api/dependencies.py).

Never loads a real sentence-transformers model and never makes a real
OpenAI network call: sentence_transformers.SentenceTransformer is
monkeypatched where needed, and constructing an OpenAI client (unlike
calling it) performs no network I/O.
"""

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
