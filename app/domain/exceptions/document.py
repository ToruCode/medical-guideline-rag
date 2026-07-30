"""Exceptions raised while loading source documents.

Messages must contain only the file name, never a full path, document
content, or patient information.
"""


class DocumentLoadError(Exception):
    """Base class for document loading failures."""


class DocumentNotFoundError(DocumentLoadError):
    """Raised when the given path does not exist or is not a regular file."""


class UnsupportedDocumentTypeError(DocumentLoadError):
    """Raised when the file extension is not supported."""


class EncryptedDocumentError(DocumentLoadError):
    """Raised when the document is encrypted and cannot be read."""


class InvalidPdfError(DocumentLoadError):
    """Raised when the PDF content cannot be parsed."""
