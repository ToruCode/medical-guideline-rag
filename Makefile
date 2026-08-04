.PHONY: setup dev ui lint format typecheck test check

setup:
	uv sync

dev:
	uv run uvicorn app.main:app --reload

ui:
	uv run streamlit run app/ui/streamlit_app.py

lint:
	uv run ruff check .

format:
	uv run ruff format --check .

typecheck:
	uv run mypy app

test:
	uv run pytest

check: lint format typecheck test
