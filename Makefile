.PHONY: setup test lint format data pipeline clean

setup:
	uv sync
	uv run python -m ipykernel install --user --name smores --display-name "smores"

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

data:
	uv run python -m smores.data.florida_keys

pipeline:
	uv run python -m smores.data.pipeline

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
