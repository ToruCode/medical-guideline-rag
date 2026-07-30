# 0002. Use pydantic-settings for configuration and stdlib logging for logs

## Status

Accepted

## Context

The application needs type-safe configuration loaded from environment
variables and an optional `.env` file, and a reusable logging setup,
before any external services (LLM, Qdrant, PostgreSQL) are integrated.
The configuration must not fail startup on unrelated environment
variables, must not leak secrets, and must be overridable in tests.

## Decision

- Use `pydantic-settings` for a single `Settings` class in
  `app/core/config.py`, sourced from environment variables prefixed
  with `MEDICAL_RAG_` (case-insensitive) and an optional `.env` file.
  Unknown environment variables are ignored (`extra="ignore"`) so the
  application does not fail to start because of unrelated variables in
  the environment.
- Expose settings through a single `get_settings()` function decorated
  with `functools.lru_cache`, usable both as a plain function and as a
  FastAPI dependency (`Depends(get_settings)`), so the API layer never
  constructs `Settings` directly and tests can override it via
  `app.dependency_overrides`.
- Use Python's standard `logging` module configured through
  `logging.config.dictConfig` in `app/core/logging.py`, exposed as
  `setup_logging(settings: Settings) -> None` and called once at
  application startup in `app/main.py`. Modules obtain a logger with
  `logging.getLogger(__name__)`.
- Do not introduce third-party logging libraries (e.g. structlog,
  loguru) or a request ID middleware at this stage; the current need is
  a minimal, dependency-free foundation.

## Consequences

- Configuration is validated and type-checked at startup; invalid
  values (e.g. an unsupported `environment`) fail fast with a clear
  `ValidationError` instead of causing unexpected behavior later.
- `get_settings()` being cached means tests must clear the cache
  (`get_settings.cache_clear()`) between runs to observe environment
  variable changes; this is handled by an autouse fixture in
  `tests/conftest.py`.
- The single `Settings` class mixes concerns that will eventually
  differ per environment (e.g. debug flags, external service
  endpoints). A future issue is expected to split this into
  environment-specific settings (Local / Test / Staging / Production)
  once those differences become concrete.
- Logging configuration stays minimal (console output, standard
  fields) until a later issue introduces structured/JSON logging or a
  request ID, when the current stack is expected to be a bottleneck.
