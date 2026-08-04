"""Abstract interface for storing and retrieving PDF documents in an
external object store (e.g. S3)."""

from typing import Protocol


class ObjectStorage(Protocol):
    """Stores and retrieves whole-file content by a string key."""

    def upload(self, key: str, content: bytes) -> None: ...

    def download(self, key: str) -> bytes: ...
