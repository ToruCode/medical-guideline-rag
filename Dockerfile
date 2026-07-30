FROM python:3.12-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY app ./app
RUN uv sync --frozen

# No application entrypoint exists yet (Issue #1: skeleton only).
# This CMD only verifies that the package imports correctly inside the
# image and will be replaced once the API entrypoint is implemented.
CMD ["uv", "run", "python", "-c", "import app; print('medical-guideline-rag skeleton image')"]
