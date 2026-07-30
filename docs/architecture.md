# Architecture

This project follows a layered architecture with dependencies pointing
inward.

## Layers

- **API** (`app/api`, `app/schemas`): receives HTTP requests, validates
  input, calls application use cases, converts results into HTTP
  responses. Must not call Qdrant, PostgreSQL, LangChain, or an LLM
  directly.
- **Application** (`app/application`): executes use cases and
  coordinates retrieval, reranking, answerability judgment, and
  generation.
- **Domain** (`app/domain`): defines entities, value objects, and
  interfaces independent of frameworks. Must not depend on API,
  Application, or Infrastructure, nor on any framework or SDK.
- **Infrastructure** (`app/infrastructure`): implements domain
  interfaces using external libraries (PDF loading, embeddings, LLMs,
  Qdrant, PostgreSQL, S3).
- **Core** (`app/core`): application settings, logging, common
  exceptions, security, shared constants.
  - `app/core/config.py` defines a `Settings` class (pydantic-settings)
    loaded from environment variables (`MEDICAL_RAG_` prefix) and an
    optional `.env` file, exposed via a cached `get_settings()`
    accessor usable as a FastAPI dependency.
  - `app/core/logging.py` defines `setup_logging(settings)`, which
    configures the standard library `logging` module via
    `logging.config.dictConfig`. Modules obtain loggers with
    `logging.getLogger(__name__)`.

## Dependency Rule

```
API -> Application -> Domain
Infrastructure -> Domain (implements interfaces)
Core is referenced by all layers
```

Domain must never depend on API, Application, or Infrastructure.

## Status

Minimal FastAPI setup with a health check endpoint, environment-based
settings, and standard logging (Issue #3). Retrieval, generation, and
external service integrations land in subsequent issues.

See `docs/adr/0001-project-architecture.md` and
`docs/adr/0002-configuration-and-logging.md` for the architecture
decision records.
