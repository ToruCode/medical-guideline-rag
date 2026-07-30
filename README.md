# Medical Guideline RAG

A citation-grounded RAG system for searching medical guidelines. It
helps healthcare professionals locate relevant passages from medical
guideline documents.

This system is a technical demonstration. It must not be used to
provide medical diagnoses, individual treatment decisions, or
patient-specific recommendations. Always confirm the original guideline
and current clinical information.

## Status

Minimal FastAPI setup with a health check endpoint, environment-based
settings, standard logging, and a PDF loading foundation (Issue #4).
No chunking, embedding, retrieval, or generation logic is implemented
yet.

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync
cp .env.example .env
```

Edit `.env` as needed. All settings are environment variables prefixed
with `MEDICAL_RAG_` (see `app/core/config.py` and `.env.example` for
the full list). `.env` is not committed to Git.

Future plan: settings will be split per environment (Local / Test /
Staging / Production) instead of a single `Settings` class.

## Development commands

```bash
make dev        # run the API with uvicorn --reload
make lint       # ruff check
make format     # ruff format --check
make typecheck  # mypy
make test       # pytest
make check      # lint + typecheck + test
```

Once the server is running, open `http://127.0.0.1:8000/docs` for the
Swagger UI, or call `GET /api/v1/health` directly.

## PDF loading

Text-layer PDFs can be loaded page by page via
`app.infrastructure.pdf.pypdf_loader.PypdfLoader`, which implements
the `app.domain.ports.pdf_loader.PdfLoader` interface and returns
immutable `DocumentPage` values. Scanned PDFs (image-only, no text
layer) and encrypted PDFs are not supported; see
`docs/adr/0003-pdf-extraction-library.md` for the reasoning and
constraints.

There is no upload API yet. Only self-authored, license-free sample
PDFs belong under `data/sample/` (see `data/sample/README.md`); real
or copyrighted guideline PDFs must never be committed, and
`data/raw/` is git-ignored for that reason.

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
