"""Exceptions raised while storing or retrieving objects from external
object storage (e.g. S3).

Messages must contain only the object key, never bucket contents,
credentials, or full ARNs.
"""


class ObjectStorageError(Exception):
    """Base class for object storage failures."""


class ObjectNotFoundError(ObjectStorageError):
    """Raised when the given key does not exist in the store."""


class ObjectStorageUnavailableError(ObjectStorageError):
    """Raised when the store cannot be reached or a request otherwise fails."""
