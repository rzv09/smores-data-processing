.PHONY: setup test lint format data resample baselines clean

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

# FREQ accepts several intervals: mingw32-make resample FREQ="2min 5min"
FREQ ?= 2min

resample:
	uv run python -m smores.data.resample --freq $(FREQ)

baselines:
	uv run python -m smores.models.baselines

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
