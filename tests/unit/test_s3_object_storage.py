"""Unit tests for S3ObjectStorage.

boto3.client is monkeypatched with a fake client, so these tests make
no real network requests and need no real AWS credentials or bucket.
"""

from typing import Any

import pytest
from app.domain.exceptions.storage import ObjectNotFoundError, ObjectStorageUnavailableError
from app.infrastructure.storage.s3_object_storage import S3ObjectStorage
from botocore.exceptions import ClientError


class _FakeBody:
    def __init__(self, content: bytes) -> None:
        self._content = content

    def read(self) -> bytes:
        return self._content


class FakeS3Client:
    def __init__(self) -> None:
        self.put_object_calls: list[dict[str, Any]] = []
        self._objects: dict[str, bytes] = {}

    def seed(self, key: str, content: bytes) -> None:
        self._objects[key] = content

    def put_object(self, Bucket: str, Key: str, Body: bytes) -> None:  # noqa: N803
        self.put_object_calls.append({"Bucket": Bucket, "Key": Key, "Body": Body})
        self._objects[Key] = Body

    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        if Key not in self._objects:
            raise ClientError(
                error_response={"Error": {"Code": "NoSuchKey", "Message": "not found"}},
                operation_name="GetObject",
            )
        return {"Body": _FakeBody(self._objects[Key])}


class RaisingS3Client:
    def put_object(self, Bucket: str, Key: str, Body: bytes) -> None:  # noqa: N803
        raise RuntimeError("do-not-leak-this-sdk-error-detail bucket=super-secret-bucket-name")

    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        raise ClientError(
            error_response={"Error": {"Code": "AccessDenied", "Message": "nope"}},
            operation_name="GetObject",
        )


def _patch_boto3_client(monkeypatch: pytest.MonkeyPatch, client: Any) -> None:
    monkeypatch.setattr("boto3.client", lambda service_name: client)


def test_upload_calls_put_object_with_bucket_key_content(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeS3Client()
    _patch_boto3_client(monkeypatch, fake_client)

    storage = S3ObjectStorage(bucket_name="my-bucket")
    storage.upload("guideline.pdf", b"pdf-bytes")

    assert fake_client.put_object_calls == [
        {"Bucket": "my-bucket", "Key": "guideline.pdf", "Body": b"pdf-bytes"}
    ]


def test_download_returns_bytes_from_get_object(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeS3Client()
    fake_client.seed("guideline.pdf", b"pdf-bytes")
    _patch_boto3_client(monkeypatch, fake_client)

    storage = S3ObjectStorage(bucket_name="my-bucket")
    content = storage.download("guideline.pdf")

    assert content == b"pdf-bytes"


def test_download_raises_object_not_found_error_for_missing_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeS3Client()
    _patch_boto3_client(monkeypatch, fake_client)

    storage = S3ObjectStorage(bucket_name="my-bucket")

    with pytest.raises(ObjectNotFoundError) as exc_info:
        storage.download("missing.pdf")

    assert "missing.pdf" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, ClientError)


def test_download_raises_object_storage_unavailable_error_for_other_client_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_boto3_client(monkeypatch, RaisingS3Client())

    storage = S3ObjectStorage(bucket_name="my-bucket")

    with pytest.raises(ObjectStorageUnavailableError) as exc_info:
        storage.download("guideline.pdf")

    assert "guideline.pdf" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, ClientError)


def test_upload_raises_object_storage_unavailable_error_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_boto3_client(monkeypatch, RaisingS3Client())

    storage = S3ObjectStorage(bucket_name="my-bucket")

    with pytest.raises(ObjectStorageUnavailableError) as exc_info:
        storage.upload("guideline.pdf", b"pdf-bytes")

    assert "guideline.pdf" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_error_messages_do_not_contain_bucket_name_or_raw_exception_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_boto3_client(monkeypatch, RaisingS3Client())

    storage = S3ObjectStorage(bucket_name="super-secret-bucket-name")

    with pytest.raises(ObjectStorageUnavailableError) as exc_info:
        storage.upload("guideline.pdf", b"pdf-bytes")

    message = str(exc_info.value)
    assert "super-secret-bucket-name" not in message
    assert "do-not-leak-this-sdk-error-detail" not in message
