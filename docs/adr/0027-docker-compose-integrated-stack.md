# 0027. Docker Compose integrated local stack (FastAPI + Streamlit + persistent Qdrant)

## Status

Accepted

## Context

`compose.yaml` previously defined a single `app` service (FastAPI).
The Streamlit demo UI (`app/ui/streamlit_app.py`,
`docs/adr/0024-streamlit-demo-ui.md`) only ran locally via `make ui`,
never through Docker Compose, and the persistent vector store
(`docs/adr/0026-persistent-vector-store.md`) was never actually
exercised in persistent mode by `compose.yaml` - the `app` service
ran with whatever `MEDICAL_RAG_VECTOR_STORE_PROVIDER` happened to be
in `.env` (`memory` by default). Issue #19 asked for a one-command
local stack (`docker compose up`) covering the API, the UI, and
persistent storage, runnable with no API key.

## Decision

- **Reuse the existing single `Dockerfile` for both services.** The
  new `ui` service in `compose.yaml` builds the same image (`build:
  .`) and only overrides `command:` to run `streamlit run
  app/ui/streamlit_app.py --server.address 0.0.0.0` instead of
  `uvicorn`. No second `Dockerfile` was added - `pyproject.toml`
  already installs `streamlit` as a production dependency (it is what
  `make ui` uses locally), so nothing further was needed in the image
  itself.
- **No Qdrant server container.** `QdrantVectorStore` remains
  embedded/local-mode only, per `docs/adr/0026-persistent-vector-store.md`'s
  explicit decision ("no Docker, no network, no server process"). This
  issue's "persistent Qdrant" requirement is satisfied by the `app`
  service writing to the existing `qdrant_storage` named volume, not
  by adding a `qdrant/qdrant` image or changing `QdrantVectorStore` to
  a URL-based client. A real Qdrant server remains a candidate for a
  future, separate production-deployment issue, as ADR 0026 already
  anticipated.
- **`app`'s `MEDICAL_RAG_VECTOR_STORE_PROVIDER` is forced to `qdrant`
  via a `compose.yaml`-level `environment:` override**, not by
  changing `.env.example`'s own default. `.env.example`'s default
  must stay `memory` (ADR 0026 - it's what tests/CI use); a
  service-level `environment:` entry in Compose Specification takes
  precedence over the same variable coming from `env_file`, so this
  gives the integrated stack persistent-by-default behavior without
  touching the meaning of a bare `uv run uvicorn ...`/`make dev`.
- **`ui` reaches `app` at `http://app:8000`**, overridden the same way
  (`environment: MEDICAL_RAG_UI_API_BASE_URL`) rather than changing
  `.env.example`'s `127.0.0.1` default, which is correct for local,
  same-host `make ui` + `make dev` and must stay that way.
- **`ui` gets its own `HEALTHCHECK`, overriding the image's.** The
  `Dockerfile`'s built-in `HEALTHCHECK` polls the FastAPI health
  endpoint on `:8000` - correct for `app`, but always-failing if
  inherited as-is by `ui` (which serves Streamlit on `:8501`, not
  FastAPI on `:8000`). `compose.yaml`'s `ui` service therefore
  declares its own `healthcheck:` against Streamlit's built-in
  `/_stcore/health` endpoint. `ui` also `depends_on: app: condition:
  service_healthy`, so it does not start against an API that isn't
  ready yet.
- **`scripts/index_documents.py` is now copied into the image**
  (`Dockerfile` gained `COPY scripts ./scripts`; `.dockerignore` no
  longer excludes `scripts/`) so it can be run as a one-off container
  (`docker compose run --rm app uv run python -m
  scripts.index_documents [--rebuild]`), matching how it is already
  run locally (`uv run python -m scripts.index_documents`). `data/`
  remains excluded from the *build context* (`.dockerignore`) - PDFs
  and extracted content must never be baked into an image layer - and
  is instead bind-mounted at runtime (`./data:/app/data` in
  `compose.yaml`) so a user can drop PDFs into `data/raw/` without
  rebuilding.
- **`Dockerfile`'s `CMD`/`HEALTHCHECK`, and `ui`'s `command:`/`healthcheck:`
  in `compose.yaml`, all invoke `uv run --frozen --no-dev`.** Verified
  by manually running the stack while implementing this issue: a bare
  `uv run <command>` (no flags) re-syncs the virtualenv against
  `pyproject.toml`'s default dependency-group selection - which
  includes the `dev` group - on every invocation, so without
  `--frozen --no-dev` the container silently re-downloaded
  `mypy`/`ruff`/`reportlab`/etc. (~30 MB) on every `docker compose up`
  and every `docker compose run --rm app ...`, even though the image
  was already built with `uv sync --frozen --no-dev`. This existed
  before this issue too (the original `Dockerfile` `CMD` had the same
  gap) but is fixed here since it directly undermines this issue's
  "starts without an API key" objective and was caught while verifying
  Requirement 4/5.
- **No CI changes.** `.github/workflows/ci.yml` continues to run only
  lint/format/typecheck/`pytest` with Fake providers, as before; no
  `docker compose up`/build smoke-test job was added. This is a
  demonstration project without a deployment pipeline yet, Compose
  behavior was manually verified while implementing this issue (see
  Consequences), and a Docker-in-Docker smoke test would add CI
  runtime and flakiness surface (image build time, port/healthcheck
  timing) disproportionate to the benefit at this stage. This can be
  revisited if `compose.yaml` starts changing often enough that manual
  verification becomes unreliable, or once a real deployment pipeline
  exists to fold it into.

## Consequences

- `docker compose up --build` brings up a working `app` (`:8000`) +
  `ui` (`:8501`) stack with Fake providers and no API key, persisting
  indexed chunks to the `qdrant_storage` volume across restarts.
- Initial indexing and `--rebuild` both require stopping/not-yet-
  starting `app` first, same one-process-per-path constraint as ADR
  0026 already documented for the non-Compose case - now documented
  for the Compose case in README's "Docker" section as two explicit,
  separate flows.
- Switching the stack back to the in-memory provider (e.g. for a
  quick, storage-free demo) means editing `compose.yaml`'s `app.environment`
  block directly, not `.env` - a minor asymmetry versus every other
  provider toggle in this project, accepted because the whole point of
  this Compose stack is to default to persistence.
- Compose changes are not covered by CI; a regression in
  `compose.yaml`/`Dockerfile` (e.g. a broken healthcheck or a missing
  `COPY`) would only be caught by manual `docker compose up` testing
  until/unless a future issue adds automated coverage.
