"""S3-backed implementation of the ObjectStorage port.

Importing boto3 is deferred to __init__ (not done at module import
time), mirroring OpenAiLlm's pattern
(app/infrastructure/llm/openai_llm.py), so merely referencing this
class - as app/api/dependencies.py always does, regardless of which
storage_provider is configured - never pays the cost of importing the
SDK. See docs/adr/0028-s3-document-storage.md.

Any exception raised by the S3 client is translated to
ObjectNotFoundError/ObjectStorageUnavailableError (never the raw
botocore exception) so callers depend only on
app.domain.exceptions.storage. The translated exception's message
never includes the bucket name, credentials, or full ARN - only the
object key and the underlying exception's type - as a defensive
measure against account-ID/credential leakage into logs, mirroring
OpenAiLlm's equivalent guarantee for API keys.

Credentials and region are resolved via boto3's own standard chain
(e.g. an ECS task role's credentials and the container's AWS_REGION in
production) rather than a MEDICAL_RAG_* setting - see
docs/adr/0028-s3-document-storage.md.
"""

from typing import Any

from app.domain.exceptions.storage import ObjectNotFoundError, ObjectStorageUnavailableError

_NOT_FOUND_ERROR_CODES = {"NoSuchKey", "404"}


class S3ObjectStorage:
    def __init__(self, bucket_name: str) -> None:
        import boto3

        self._client: Any = boto3.client("s3")
        self._bucket_name = bucket_name

    def upload(self, key: str, content: bytes) -> None:
        try:
            self._client.put_object(Bucket=self._bucket_name, Key=key, Body=content)
        except Exception as exc:
            raise ObjectStorageUnavailableError(
                f"Failed to upload object {key!r}: {type(exc).__name__}"
            ) from exc

    def download(self, key: str) -> bytes:
        from botocore.exceptions import ClientError

        try:
            response = self._client.get_object(Bucket=self._bucket_name, Key=key)
            return response["Body"].read()
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code")
            if error_code in _NOT_FOUND_ERROR_CODES:
                raise ObjectNotFoundError(f"Object not found: {key!r}") from exc
            raise ObjectStorageUnavailableError(
                f"Failed to download object {key!r}: {type(exc).__name__}"
            ) from exc
        except Exception as exc:
            raise ObjectStorageUnavailableError(
                f"Failed to download object {key!r}: {type(exc).__name__}"
            ) from exc
