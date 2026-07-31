from pathlib import Path

import pytest
from app.main import app
from fastapi.testclient import TestClient
from tests.support.pdf_factory import build_pdf

client = TestClient(app)


def _index_sample_document(tmp_path: Path, text: str = "recommended dosage guidance") -> None:
    pdf_path = build_pdf(tmp_path / "source.pdf", [text], title="Sample Guideline")
    with pdf_path.open("rb") as f:
        response = client.post(
            "/api/v1/documents/index",
            files={"file": ("guideline.pdf", f, "application/pdf")},
        )
    assert response.status_code == 201


def test_ask_question_with_indexed_content_returns_citations(tmp_path: Path) -> None:
    _index_sample_document(tmp_path)

    response = client.post(
        "/api/v1/questions/ask", json={"question": "dosage guidance", "top_k": 3}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["is_insufficient_evidence"] is False
    assert len(body["citations"]) >= 1
    citation = body["citations"][0]
    assert citation["source_name"] == "guideline.pdf"
    assert "vector" not in citation


def test_ask_question_with_no_indexed_content_returns_insufficient_evidence() -> None:
    response = client.post("/api/v1/questions/ask", json={"question": "anything", "top_k": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["is_insufficient_evidence"] is True
    assert body["citations"] == []


def test_ask_question_rejects_empty_question() -> None:
    response = client.post("/api/v1/questions/ask", json={"question": "", "top_k": 3})

    assert response.status_code == 422


def test_ask_question_rejects_whitespace_only_question() -> None:
    response = client.post("/api/v1/questions/ask", json={"question": "   ", "top_k": 3})

    assert response.status_code == 400


@pytest.mark.parametrize("top_k", [0, -1])
def test_ask_question_rejects_non_positive_top_k(top_k: int) -> None:
    response = client.post("/api/v1/questions/ask", json={"question": "a question", "top_k": top_k})

    assert response.status_code == 422


def test_ask_question_uses_default_top_k_when_omitted() -> None:
    response = client.post("/api/v1/questions/ask", json={"question": "a question"})

    assert response.status_code == 200


def test_ask_question_logs_do_not_contain_question_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("INFO"):
        response = client.post(
            "/api/v1/questions/ask",
            json={"question": "do-not-leak-this-question", "top_k": 3},
        )

    assert response.status_code == 200
    log_output = "\n".join(caplog.messages)
    assert "do-not-leak-this-question" not in log_output


def test_ask_question_logs_do_not_contain_answer_or_passage_text(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _index_sample_document(tmp_path, text="do-not-leak-this-passage-text")

    with caplog.at_level("INFO"):
        response = client.post(
            "/api/v1/questions/ask", json={"question": "dosage guidance", "top_k": 3}
        )

    assert response.status_code == 200
    log_output = "\n".join(caplog.messages)
    assert "do-not-leak-this-passage-text" not in log_output
    assert response.json()["answer"] not in log_output
