import tempfile
from pathlib import Path

import pytest
from app.main import app
from fastapi.testclient import TestClient
from tests.support.pdf_factory import build_corrupted_pdf, build_encrypted_pdf, build_pdf

client = TestClient(app)


@pytest.fixture(autouse=True)
def _redirect_temp_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirects the endpoint's temp file writes into an isolated
    per-test directory, so cleanup can be verified without touching the
    real OS temp directory (which may contain unrelated files).
    """
    upload_temp_dir = tmp_path / "uploads"
    upload_temp_dir.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(upload_temp_dir))
    return upload_temp_dir


def test_index_document_returns_created_with_counts(
    tmp_path: Path, _redirect_temp_dir: Path
) -> None:
    pdf_path = build_pdf(
        tmp_path / "source.pdf", ["Guideline content here"], title="Sample Guideline"
    )

    with pdf_path.open("rb") as f:
        response = client.post(
            "/api/v1/documents/index",
            files={"file": ("guideline.pdf", f, "application/pdf")},
        )

    assert response.status_code == 201
    body = response.json()
    assert body["source_name"] == "guideline.pdf"
    assert body["page_count"] == 1
    assert body["chunk_count"] >= 1
    assert body["indexed_count"] == body["chunk_count"]
    assert body["document_id"] is not None
    assert list(_redirect_temp_dir.iterdir()) == []


def test_index_document_rejects_non_pdf_file(_redirect_temp_dir: Path) -> None:
    response = client.post(
        "/api/v1/documents/index",
        files={"file": ("notes.txt", b"just text", "text/plain")},
    )

    assert response.status_code == 415
    assert list(_redirect_temp_dir.iterdir()) == []


def test_index_document_rejects_empty_file(_redirect_temp_dir: Path) -> None:
    response = client.post(
        "/api/v1/documents/index",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )

    assert response.status_code == 400
    assert list(_redirect_temp_dir.iterdir()) == []


def test_index_document_rejects_encrypted_pdf(tmp_path: Path, _redirect_temp_dir: Path) -> None:
    pdf_path = build_encrypted_pdf(tmp_path / "source.pdf", password="secret")

    with pdf_path.open("rb") as f:
        response = client.post(
            "/api/v1/documents/index",
            files={"file": ("encrypted.pdf", f, "application/pdf")},
        )

    assert response.status_code == 422
    assert list(_redirect_temp_dir.iterdir()) == []


def test_index_document_rejects_corrupted_pdf(tmp_path: Path, _redirect_temp_dir: Path) -> None:
    pdf_path = build_corrupted_pdf(tmp_path / "source.pdf")

    with pdf_path.open("rb") as f:
        response = client.post(
            "/api/v1/documents/index",
            files={"file": ("corrupted.pdf", f, "application/pdf")},
        )

    assert response.status_code == 422
    assert list(_redirect_temp_dir.iterdir()) == []


def test_index_document_logs_do_not_contain_page_text(
    tmp_path: Path, _redirect_temp_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    pdf_path = build_pdf(tmp_path / "source.pdf", ["do-not-leak-this-page-text"])

    with caplog.at_level("INFO"):
        with pdf_path.open("rb") as f:
            response = client.post(
                "/api/v1/documents/index",
                files={"file": ("guideline.pdf", f, "application/pdf")},
            )

    assert response.status_code == 201
    log_output = "\n".join(caplog.messages)
    assert "do-not-leak-this-page-text" not in log_output
