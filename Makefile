.PHONY: setup lint format typecheck test check

setup:
	uv sync

lint:
	uv run ruff check .

format:
	uv run ruff format --check .

typecheck:
	uv run mypy app

test:
	uv run pytest

check: lint format typecheck test
