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

## Dependency Rule

```
API -> Application -> Domain
Infrastructure -> Domain (implements interfaces)
Core is referenced by all layers
```

Domain must never depend on API, Application, or Infrastructure.

## Status

As of this initial structure (Issue #1), all layers are empty
skeletons. Concrete implementations land in subsequent issues.

See `docs/adr/0001-project-architecture.md` for the architecture
decision record.
