"""Opt-in, full-stack live RAG test: real sentence-transformers Embedding
and the real OpenAI API, driven through the FastAPI app exactly as a
client would.

Skipped by default: requires both a real, downloaded sentence-
transformers model (RUN_SLOW_TESTS=1) and a real, billable OpenAI API
key (MEDICAL_RAG_LLM_API_KEY) - the same two gates already used by
tests/integration/test_live_sentence_transformer_embedder.py and
tests/integration/test_live_openai_llm.py, combined. See
docs/adr/0012-live-e2e-verification.md.

The sample PDF is generated at test-run time from self-authored,
fictional placeholder text (tests/support/pdf_factory.build_pdf); no
PDF is committed to the repository, and no real medical or patient
content is used anywhere in this file.
"""

import os
from pathlib import Path

import pytest
from app.main import app
from fastapi.testclient import TestClient
from tests.support.pdf_factory import build_pdf

# Captured at collection time (before tests/conftest.py's per-test
# _isolate_provider_settings_from_dotenv fixture strips
# MEDICAL_RAG_LLM_API_KEY from os.environ for ordinary tests), the same
# pattern tests/integration/test_live_openai_llm.py already uses. Each
# test below restores it via monkeypatch.setenv - never logged or
# printed.
_API_KEY = os.environ.get("MEDICAL_RAG_LLM_API_KEY")

pytestmark = pytest.mark.skipif(
    not (os.environ.get("RUN_SLOW_TESTS") and _API_KEY),
    reason=(
        "requires RUN_SLOW_TESTS=1 (downloads a real sentence-transformers model) "
        "and a real OpenAI API key in MEDICAL_RAG_LLM_API_KEY (billable) to run"
    ),
)


def test_live_index_then_ask_with_real_embedding_and_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _API_KEY is not None
    monkeypatch.setenv("MEDICAL_RAG_EMBEDDING_PROVIDER", "sentence_transformers")
    monkeypatch.setenv("MEDICAL_RAG_LLM_PROVIDER", "openai")
    monkeypatch.setenv("MEDICAL_RAG_LLM_API_KEY", _API_KEY)

    client = TestClient(app)
    pdf_path = build_pdf(
        tmp_path / "sample_guideline.pdf",
        ["Adults should take 500 mg of Medicamentum X twice daily with food."],
        title="Self-Authored Sample Guideline",
    )

    with pdf_path.open("rb") as f:
        index_response = client.post(
            "/api/v1/documents/index",
            files={"file": ("sample_guideline.pdf", f, "application/pdf")},
        )
    assert index_response.status_code == 201
    index_body = index_response.json()
    assert index_body["source_name"] == "sample_guideline.pdf"
    assert index_body["page_count"] == 1
    assert index_body["chunk_count"] >= 1
    assert index_body["indexed_count"] == index_body["chunk_count"]

    ask_response = client.post(
        "/api/v1/questions/ask",
        json={"question": "What is the recommended adult dosage of Medicamentum X?", "top_k": 3},
    )

    assert ask_response.status_code == 200
    ask_body = ask_response.json()
    assert ask_body["is_insufficient_evidence"] is False
    assert len(ask_body["citations"]) >= 1
    citation = ask_body["citations"][0]
    assert citation["source_name"] == "sample_guideline.pdf"
    assert citation["page_number"] == 1
    assert isinstance(ask_body["answer"], str)
    assert ask_body["answer"].strip() != ""


def test_live_ask_with_no_indexed_documents_returns_insufficient_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _API_KEY is not None
    monkeypatch.setenv("MEDICAL_RAG_EMBEDDING_PROVIDER", "sentence_transformers")
    monkeypatch.setenv("MEDICAL_RAG_LLM_PROVIDER", "openai")
    monkeypatch.setenv("MEDICAL_RAG_LLM_API_KEY", _API_KEY)

    client = TestClient(app)

    response = client.post(
        "/api/v1/questions/ask",
        json={"question": "a question with nothing indexed in this fresh store", "top_k": 3},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["is_insufficient_evidence"] is True
    assert body["citations"] == []
