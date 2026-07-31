"""End-to-end API test: index a PDF, then ask a question grounded in it."""

from pathlib import Path

from app.main import app
from fastapi.testclient import TestClient
from tests.support.pdf_factory import build_pdf

client = TestClient(app)


def test_index_then_ask_returns_grounded_answer_via_shared_vector_store(tmp_path: Path) -> None:
    pdf_path = build_pdf(
        tmp_path / "source.pdf",
        ["Recommended dosage guidance for the sample condition."],
        title="Sample Guideline",
    )

    with pdf_path.open("rb") as f:
        index_response = client.post(
            "/api/v1/documents/index",
            files={"file": ("guideline.pdf", f, "application/pdf")},
        )
    assert index_response.status_code == 201
    document_id = index_response.json()["document_id"]

    ask_response = client.post(
        "/api/v1/questions/ask",
        json={"question": "What is the recommended dosage guidance?", "top_k": 3},
    )

    assert ask_response.status_code == 200
    body = ask_response.json()
    assert body["is_insufficient_evidence"] is False
    assert len(body["citations"]) >= 1
    assert body["citations"][0]["document_id"] == document_id


def test_ask_before_any_document_is_indexed_returns_insufficient_evidence() -> None:
    response = client.post(
        "/api/v1/questions/ask",
        json={"question": "a question with nothing indexed yet", "top_k": 3},
    )

    assert response.status_code == 200
    assert response.json()["is_insufficient_evidence"] is True


def test_health_endpoint_still_works_alongside_new_endpoints() -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
