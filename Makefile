.PHONY: install run test lint clean

install:
	uv pip install -e .

run:
	uv run uvicorn src.rag_pipeline.api.main:app --reload --port 8000

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run black --check .
	uv run mypy src/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .venv/ .pytest_cache/ .mypy_cache/ .ruff_cache/ build/ dist/ *.egg-info
