# 0028. S3 document storage

## Status

Accepted

## Context

Issue #21 (deploy to AWS) requires S3 storage for uploaded guideline
PDFs (upload/download/versioning/lifecycle) and the ability to index a
PDF that lives in S3. Nothing AWS-related existed in the codebase
before this: no `boto3` dependency, no object-storage abstraction, and
`PdfLoader`/`IndexDocumentService` are entirely local-filesystem-`Path`
based (`app/domain/ports/pdf_loader.py`, `load(file_path: Path)`).
`POST /documents/index` (`app/api/v1/endpoints/documents.py`) already
establishes the pattern for turning arbitrary bytes into something
`IndexDocumentService` can consume: buffer the upload into memory,
write it to a freshly-created temp directory, call
`IndexDocumentService.execute(temp_path)`, always remove the temp
directory afterward.

This issue was split into three sequential PRs (S3 application code,
Terraform infrastructure, CI/CD pipeline) rather than one, given its
overall scope; this ADR covers only the first (application-code) part.

## Decision

- **`PdfLoader` and `IndexDocumentService` are unchanged.** Both stay
  local-`Path`-based. S3 support is added as a new entry point, not a
  redesign of the existing indexing pipeline: `_download_temp_pdf()`
  (`app/api/v1/endpoints/documents.py`) downloads S3 bytes into the
  exact same kind of temp file `_save_temp_pdf()` already creates for a
  direct upload, then hands that `Path` to the same
  `IndexDocumentService.execute()` call. This keeps the existing,
  already-tested indexing pipeline's contract at "index a document
  already resolved to a local path," regardless of the path's origin.
- **A new `ObjectStorage` Protocol** (`app/domain/ports/object_storage.py`,
  `upload(key: str, content: bytes) -> None`, `download(key: str) -> bytes`)
  and its S3-backed implementation, **`S3ObjectStorage`**
  (`app/infrastructure/storage/s3_object_storage.py`), follow the exact
  shape of `OpenAiLlm`
  (`app/infrastructure/llm/openai_llm.py`): `boto3` is imported lazily
  inside `__init__`, never at module import time, so merely referencing
  the class (as `app/api/dependencies.py` always does) never pays the
  SDK's import cost unless `storage_provider="s3"` is actually
  configured; every SDK exception is translated into a domain exception
  (`ObjectNotFoundError`/`ObjectStorageUnavailableError`,
  `app/domain/exceptions/storage.py`) whose message contains only the
  object key and the underlying exception's type name - never the
  bucket name, credentials, or a full ARN - mirroring `OpenAiLlm`'s
  guarantee that its translated exception never leaks the API key.
- **Two new endpoints, not a change to `POST /documents/index`.**
  `POST /documents/upload` stores a PDF in S3 without indexing it;
  `POST /documents/index-from-s3` indexes a PDF already in S3 by key.
  `POST /documents/index` (local upload, storage-agnostic) keeps
  working byte-for-byte unchanged - the default `storage_provider="local"`
  means every existing caller, test, and the whole Docker
  Compose/local-dev workflow is entirely unaffected by this change.
- **The upload/download orchestration (temp-file lifecycle,
  exception-to-HTTP mapping) lives in `app/api/v1/endpoints/documents.py`
  itself, not in a new Application service.** The existing endpoint
  already owns this exact kind of plumbing for the local-upload path
  (`_save_temp_pdf`/`finally: shutil.rmtree`); a new Application
  service wrapping "download bytes, write a temp file, call
  `IndexDocumentService`" would have no domain logic of its own; it
  would just relay two calls, which does not fit the Application
  layer's existing role of coordinating domain logic (chunking,
  embedding, retrieval), as established throughout
  `app/application/services/`. The shared exception-mapping/response-
  building logic between `POST /documents/index` and
  `POST /documents/index-from-s3` was extracted into a private
  `_index_temp_pdf()` helper local to the endpoint module, to avoid
  duplicating that block - a same-module refactor, not a new
  architectural layer.
- **`get_object_storage()`** (`app/api/dependencies.py`, `@lru_cache`)
  follows the exact `if/elif/else raise ValueError` shape every other
  provider function uses. Unlike the others, its `"local"` branch is
  not a working alternative implementation - it raises `ValueError`
  immediately, since there is no local `ObjectStorage`. The two new
  endpoints are therefore expected to fail fast (surfacing as an
  unhandled 500, uncaught by design) when called under
  `storage_provider="local"`, exactly mirroring how `get_llm()` already
  fails fast when `llm_provider="openai"` is selected without an API
  key - a configuration error, not a request error.
- **Upload keys are the sanitized filename, not a generated UUID.**
  Uploading the same filename twice overwrites the previous S3 object
  under that key. This is a deliberate simplification, made safe by the
  S3 bucket's own versioning (enabled by the Terraform in the PR that
  follows this one) - the previous version is retained, not lost -
  rather than adding key-uniqueness logic to the application.
- **No new `Settings` field for AWS region or credentials.**
  `Settings.storage_provider: Literal["local", "s3"] = "local"` and
  `Settings.s3_bucket_name: str | None = None` are the only additions
  (`MEDICAL_RAG_` prefix, matching every other setting). Region and
  credentials are left entirely to `boto3`'s own standard resolution
  chain (environment variables, an ECS task role's injected credentials
  in production, or a local AWS CLI profile for manual testing) -
  adding a `Settings.aws_region`-style field would just shadow
  configuration boto3 already resolves correctly on its own.
- **Tests use a hand-written stub, not `moto`.** `tests/unit/test_s3_object_storage.py`
  monkeypatches `boto3.client` with a fake object exposing
  `put_object`/`get_object`, exactly like `tests/unit/test_openai_llm.py`
  monkeypatches `openai.OpenAI` - consistent with this codebase's
  existing convention of hand-written fakes/stubs over a mocking
  library, and avoids adding `moto` as a new dev dependency.
  `tests/api/test_documents.py`'s new cases use
  `app.dependency_overrides[get_object_storage]` with a small
  dict-backed in-memory fake, mirroring `tests/api/test_health.py`'s
  existing `dependency_overrides` pattern.

## Consequences

- Existing behavior is completely unaffected: `storage_provider`
  defaults to `"local"`, so `POST /documents/index`, every existing
  test, and the Docker Compose stack (Issue #19) are unchanged.
- Operators who want S3-backed storage set
  `MEDICAL_RAG_STORAGE_PROVIDER=s3` and `MEDICAL_RAG_S3_BUCKET_NAME`,
  and must run in an environment where `boto3` can resolve AWS
  credentials (an ECS task role in production; a local AWS CLI profile
  or exported environment variables for manual testing) - this PR adds
  no code path that runs `terraform apply`, calls the AWS CLI, or
  otherwise creates/touches real AWS resources; the S3 bucket itself is
  provisioned by the Terraform PR that follows this one
  (`docs/adr/0029-aws-ecs-fargate-deployment.md`, not yet written at
  the time of this ADR).
- `POST /documents/upload`'s overwrite-by-filename behavior means a
  second upload of a same-named file is only non-destructive because
  bucket versioning is enabled; a bucket created without versioning
  (e.g. manually, outside the Terraform in the next PR) would silently
  lose the previous object's content.
