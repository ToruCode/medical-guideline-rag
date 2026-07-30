# 0001. Adopt a layered architecture (API / Application / Domain / Infrastructure / Core)

## Status

Accepted

## Context

The project is a citation-grounded RAG system for medical guidelines. It
must keep retrieval, generation, and safety rules (citation
preservation, insufficient-evidence handling) testable and independent
from any specific web framework, vector database, or LLM SDK, so that
these can be swapped or mocked without touching business rules.

## Decision

Adopt a layered architecture with dependencies pointing inward:

- API layer (`app/api`, `app/schemas`)
- Application layer (`app/application`)
- Domain layer (`app/domain`)
- Infrastructure layer (`app/infrastructure`)
- Core layer (`app/core`)

The domain layer must not depend on FastAPI, LangChain, Qdrant,
SQLAlchemy, the AWS SDK, or any specific LLM SDK. External services are
accessed through interfaces defined in the domain layer and implemented
in the infrastructure layer.

## Consequences

- Business rules (e.g. insufficient-evidence handling, citation
  preservation) can be unit tested without a running database, vector
  store, or LLM.
- Swapping infrastructure (e.g. changing the vector database or LLM
  provider) should not require changes to domain or application logic.
- Adds indirection (interfaces) that is not needed for a small script,
  but is justified given the project's stated goal of
  production-oriented code quality and AWS deployment.
