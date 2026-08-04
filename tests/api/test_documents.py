import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from app.api.dependencies import get_object_storage
from app.domain.exceptions.storage import ObjectNotFoundError
from app.main import app
from fastapi.testclient import TestClient
from tests.support.pdf_factory import build_corrupted_pdf, build_encrypted_pdf, build_pdf

client = TestClient(app)


class FakeObjectStorage:
    """In-memory ObjectStorage fake, mirroring the dict-backed style of
    this codebase's other Fake* infrastructure implementations
    (e.g. InMemoryVectorStore).
    """

    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}

    def upload(self, key: str, content: bytes) -> None:
        self._objects[key] = content

    def download(self, key: str) -> bytes:
        if key not in self._objects:
            raise ObjectNotFoundError(f"Object not found: {key!r}")
        return self._objects[key]


@pytest.fixture
def fake_object_storage() -> Iterator[FakeObjectStorage]:
    storage = FakeObjectStorage()
    app.dependency_overrides[get_object_storage] = lambda: storage
    try:
        yield storage
    finally:
        app.dependency_overrides.pop(get_object_storage, None)


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


def test_upload_document_returns_created_with_key(
    tmp_path: Path, fake_object_storage: FakeObjectStorage
) -> None:
    pdf_path = build_pdf(tmp_path / "source.pdf", ["Guideline content here"])

    with pdf_path.open("rb") as f:
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("guideline.pdf", f, "application/pdf")},
        )

    assert response.status_code == 201
    assert response.json() == {"key": "guideline.pdf"}
    assert fake_object_storage.download("guideline.pdf") == pdf_path.read_bytes()


def test_upload_document_rejects_non_pdf_file(fake_object_storage: FakeObjectStorage) -> None:
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("notes.txt", b"just text", "text/plain")},
    )

    assert response.status_code == 415


def test_upload_document_rejects_empty_file(fake_object_storage: FakeObjectStorage) -> None:
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )

    assert response.status_code == 400


def test_index_document_from_s3_returns_created_with_counts(
    tmp_path: Path, _redirect_temp_dir: Path, fake_object_storage: FakeObjectStorage
) -> None:
    pdf_path = build_pdf(
        tmp_path / "source.pdf", ["Guideline content here"], title="Sample Guideline"
    )
    fake_object_storage.upload("guideline.pdf", pdf_path.read_bytes())

    response = client.post("/api/v1/documents/index-from-s3", json={"key": "guideline.pdf"})

    assert response.status_code == 201
    body = response.json()
    assert body["source_name"] == "guideline.pdf"
    assert body["page_count"] == 1
    assert body["chunk_count"] >= 1
    assert body["indexed_count"] == body["chunk_count"]
    assert body["document_id"] is not None
    assert list(_redirect_temp_dir.iterdir()) == []


def test_index_document_from_s3_returns_404_for_missing_key(
    fake_object_storage: FakeObjectStorage,
) -> None:
    response = client.post("/api/v1/documents/index-from-s3", json={"key": "missing.pdf"})

    assert response.status_code == 404


def test_index_document_from_s3_rejects_encrypted_pdf(
    tmp_path: Path, _redirect_temp_dir: Path, fake_object_storage: FakeObjectStorage
) -> None:
    pdf_path = build_encrypted_pdf(tmp_path / "source.pdf", password="secret")
    fake_object_storage.upload("encrypted.pdf", pdf_path.read_bytes())

    response = client.post("/api/v1/documents/index-from-s3", json={"key": "encrypted.pdf"})

    assert response.status_code == 422
    assert list(_redirect_temp_dir.iterdir()) == []
