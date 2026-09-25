.PHONY: setup test lint format clean

setup:
	uv sync
	uv run python -m ipykernel install --user --name smores --display-name "smores"

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
