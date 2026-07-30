# CLAUDE.md

## Project Overview

This project is a citation-grounded RAG system for searching medical
guidelines.

The system supports healthcare professionals in locating relevant passages
from medical guidelines.

This system is a technical demonstration. It must not provide medical
diagnoses, individual treatment decisions, or patient-specific recommendations.

## Project Goals

- Search medical guideline documents using natural language
- Return answers grounded only in retrieved passages
- Show citations with document title, edition, chapter, section, and page
- Return an insufficient-evidence response when reliable evidence is absent
- Separate retrieval quality from generation quality
- Provide a reproducible environment using Docker
- Deploy the application to AWS
- Maintain production-oriented code quality

## Technology Stack

- Python 3.12
- FastAPI
- Pydantic
- LangChain
- Qdrant
- PostgreSQL
- SQLAlchemy
- Alembic
- pytest
- Ruff
- mypy
- Docker
- Docker Compose
- GitHub Actions
- AWS

## Architecture

The project uses the following layers.

### API Layer

Location:

- `app/api`
- `app/schemas`

Responsibilities:

- Receive HTTP requests
- Validate request data
- Call application use cases
- Convert results into HTTP responses
- Select appropriate HTTP status codes

The API layer must not directly call Qdrant, PostgreSQL, LangChain, or an LLM.

### Application Layer

Location:

- `app/application`

Responsibilities:

- Execute application use cases
- Coordinate retrieval, reranking, answerability judgment, and generation
- Control the order of processing
- Coordinate domain interfaces and infrastructure implementations

### Domain Layer

Location:

- `app/domain`

Responsibilities:

- Define core entities
- Define value objects
- Define interfaces for external services
- Define business rules independent of frameworks

The domain layer must not depend on:

- FastAPI
- LangChain
- Qdrant
- SQLAlchemy
- AWS SDK
- Any specific LLM SDK

### Infrastructure Layer

Location:

- `app/infrastructure`

Responsibilities:

- PDF loading
- Text extraction and preprocessing
- Embedding model integration
- LLM integration
- Qdrant integration
- PostgreSQL integration
- S3 integration
- External library-specific implementations

### Core Layer

Location:

- `app/core`

Responsibilities:

- Application settings
- Logging
- Common exceptions
- Security
- Shared constants

## Dependency Rules

Dependencies should generally point inward.

- API may depend on Application.
- Application may depend on Domain.
- Infrastructure may implement Domain interfaces.
- Domain must not depend on API or Infrastructure.
- FastAPI endpoints must not contain business logic.
- External services must be accessed through interfaces when practical.

## Medical Safety Rules

- Generate answers only from retrieved guideline passages.
- Do not invent information that is not present in the context.
- Preserve the original source metadata.
- Return an insufficient-evidence result when evidence is missing.
- Do not provide patient-specific diagnoses.
- Do not recommend patient-specific treatments or medication changes.
- Do not store real patient information in source code, tests, logs, or samples.
- Remind users to confirm the original guideline and current clinical information.

## Data and Copyright Rules

Do not commit the following files:

- Medical guideline PDF files without confirmed redistribution permission
- Extracted guideline text
- Vector database storage
- API keys
- AWS credentials
- `.env` files
- Real patient information

Use self-authored sample documents for public tests and demonstrations.

## Coding Rules

- Use Python 3.12.
- Use type hints for public functions and methods.
- Keep functions small and focused on one responsibility.
- Use descriptive English names for files, classes, functions, and variables.
- Prefer clear code over unnecessary abstraction.
- Do not add a dependency without explaining its purpose.
- Do not place business logic inside FastAPI endpoint functions.
- Use Pydantic models for API request and response schemas.
- Use dependency injection for external services.
- Use pytest for tests.
- Use Ruff for linting and formatting.
- Use mypy for static type checking.
- Handle errors explicitly.
- Do not hide errors or report failed checks as successful.

## Testing Rules

New behavior should include appropriate tests.

Tests are divided into:

- `tests/unit`
- `tests/integration`
- `tests/api`

At minimum, consider:

- Normal cases
- Invalid input
- Empty results
- External service failures
- Insufficient evidence
- Citation metadata preservation

LLM calls should normally be mocked in unit tests.

## Logging Rules

- Use structured logging.
- Include a request ID where appropriate.
- Do not log API keys or credentials.
- Do not log real patient information.
- Record processing time for retrieval and generation separately.
- Record model and prompt versions when appropriate.

## Git Rules

- Use GitHub Issues to manage work.
- Create one branch for each Issue.
- Do not commit directly to `main` after the initial setup.
- Keep commits small and focused.
- Use Conventional Commits.

Examples:

- `feat: add guideline search endpoint`
- `fix: preserve page metadata during chunking`
- `test: add retrieval service unit tests`
- `docs: document RAG architecture`
- `chore: configure development tools`

Do not commit automatically unless the user explicitly approves it.

## Claude Code Workflow

Before changing files:

1. Read this `CLAUDE.md`.
2. Inspect the current Git branch.
3. Run `git status`.
4. Read all relevant existing files.
5. Explain the proposed change in Japanese.
6. List the files to create or modify.
7. Ask for approval before large changes.

After changing files:

1. Summarize the changes in Japanese.
2. Explain important code for a beginner.
3. Run relevant tests and quality checks.
4. Report all errors honestly.
5. Show `git diff`.
6. Recommend the next Git command.
7. Do not run `git commit` without explicit approval.

## Response Language

Explain plans, implementation decisions, errors, and code behavior in Japanese.

Source code names and technical identifiers should remain in English.