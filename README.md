# Medical Guideline RAG

A citation-grounded RAG system for searching medical guidelines. It
helps healthcare professionals locate relevant passages from medical
guideline documents.

This system is a technical demonstration. It must not be used to
provide medical diagnoses, individual treatment decisions, or
patient-specific recommendations. Always confirm the original guideline
and current clinical information.

## Status

Initial project structure only (Issue #1). No API endpoints,
retrieval, or generation logic is implemented yet.

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync
```

## Development commands

```bash
make lint       # ruff check
make format     # ruff format --check
make typecheck  # mypy
make test       # pytest
make check      # lint + typecheck + test
```

## Project layout

See `docs/architecture.md` for the layered architecture and
`CLAUDE.md` for the full set of project rules.

```
app/            # application source (api, application, domain, infrastructure, core, schemas)
tests/          # unit, integration, api tests
docs/           # requirements, architecture, ADRs
scripts/        # development and operational scripts
data/           # raw, processed, and sample guideline data (see data/sample/README.md)
```
