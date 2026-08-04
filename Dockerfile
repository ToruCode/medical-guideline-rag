FROM python:3.12-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# README.md and LICENSE are required here, not just for their own sake:
# hatchling (the build backend) validates pyproject.toml's `readme` and
# `license-files` references when building/installing the project
# itself in the next `uv sync`, and fails the build if either file is
# missing from the build context.
COPY app ./app
# scripts/index_documents.py is run inside the container to build/rebuild
# the persistent Qdrant index (docker compose run --rm app ...); see the
# "Docker" section in README.md.
COPY scripts ./scripts
COPY README.md LICENSE ./
RUN uv sync --frozen --no-dev

# Uses the standard library (no curl in python:3.12-slim) to call the
# existing health check endpoint (app/api/v1/endpoints/health.py).
# --frozen --no-dev: without these, `uv run` re-syncs the environment
# against pyproject.toml's default (dev-included) dependency group on
# every invocation, silently re-downloading mypy/ruff/reportlab/etc.
# into the container - keep this matching the image's own build step.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["uv", "run", "--frozen", "--no-dev", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"]

CMD ["uv", "run", "--frozen", "--no-dev", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
