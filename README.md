# smores-data-processing

Data processing, analysis, and modeling for the smores project.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned in `.python-version`
and uv will fetch it if needed.

```bash
uv sync                                                    # create .venv, install deps
uv run python -m ipykernel install --user --name smores    # register the Jupyter kernel
```

Copy `.env.example` to `.env` for any secrets or machine-specific settings.

## Commands

| Command | What it does |
| --- | --- |
| `uv run pytest` | Run tests |
| `uv run ruff check .` | Lint |
| `uv run ruff format .` | Format |
| `uv run jupyter lab` | Start JupyterLab |

The `Makefile` wraps these as `setup`, `test`, `lint`, `format`, and `clean`. On this
machine the binary is `mingw32-make` (from `C:\mingw64\bin`), not `make` — so
`mingw32-make test`, etc. The `uv run` commands above work regardless.

## Layout

```
src/smores/
├── config.py                               project paths + .env loading
├── data/                                   loading and cleaning raw inputs
├── features/                               transformations, feature engineering
├── models/                                 training and prediction
└── viz/                                    reusable plotting
data/                                       see data/data.md
├── data.md                                 full data reference
├── raw/                                    immutable inputs — never edited in place
│   ├── florida_keys/
│   │   └── florida_keys_data.csv           203 MB · 2,044,805 × 10   [present]
│   └── biscayne_bay/
│       └── FULL data from Harvard team/    Harvard drop             [not added yet]
├── interim/                                intermediate outputs
├── processed/                              analysis-ready datasets
└── external/                               third-party data
models/                                     serialized model artifacts
reports/figures/                            exported plots
tests/
jupyter-notebooks/                          exploratory work (gitignored)
```

Everything under `data/`, `models/`, and `reports/figures/` is gitignored — only the
directory structure is tracked, via `.gitkeep`.

**[`data/data.md`](data/data.md)** is the full data reference: the same layout plus
per-column types and ranges, the deployment breakdown, and loading notes.

## Conventions

**Import paths from `smores.config`,** never build relative paths:

```python
from smores.config import FLORIDA_KEYS_CSV, PROCESSED_DIR

df = pd.read_csv(FLORIDA_KEYS_CSV)
df.to_parquet(PROCESSED_DIR / "florida_keys_clean.parquet")
```

This works identically from a notebook, a test, or the CLI.

### Dataset paths

Every raw dataset has a constant in `smores.config` — use it instead of hardcoding a
filename, especially for Biscayne Bay, whose directory name contains spaces.

| Constant | Points to | Contents |
| --- | --- | --- |
| `FLORIDA_KEYS_CSV` | `data/raw/florida_keys/florida_keys_data.csv` | 203 MB, 2,044,805 rows × 10 cols: `deployment`, `t`, `t_increase`, `Vx`, `Vy`, `Vz`, `P`, `O2_S1`–`O2_S3` |
| `BISCAYNE_HARVARD_DIR` | `data/raw/biscayne_bay/FULL data from Harvard team/` | Harvard drop — **not added yet**, arrives later in the project |
| `FLORIDA_DIR` / `BISCAYNE_DIR` | the two site folders under `data/raw/` | for globbing or adding new files per site |

Directory constants — `DATA_DIR`, `RAW_DIR`, `INTERIM_DIR`, `PROCESSED_DIR`,
`EXTERNAL_DIR`, `MODELS_DIR`, `REPORTS_DIR`, `FIGURES_DIR` — are also exported.

A plain `pd.read_csv(FLORIDA_KEYS_CSV)` reads the whole thing in ~2.5 s for ~200 MB in
memory — no need for `chunksize` or a Parquet conversion at this size. All columns
except `deployment` parse as `float64` on their own, and there are no nulls. The file
stacks **four** deployments, so group by `deployment` before any time-series work —
`t_increase` restarts at 0 for each.

`BISCAYNE_HARVARD_DIR` is defined but its directory does not exist yet. Nothing imports
or breaks because of that — `Path` objects don't touch the filesystem. When the data
lands, drop it at that exact path and the constant starts working; no code change
needed. Guard any early Biscayne code with `if BISCAYNE_HARVARD_DIR.exists():`.

See **[`data/data.md`](data/data.md)** for per-column ranges and the deployment row
counts.

**Notebooks are gitignored** as they aren't well configured for github view

`uv.lock` is committed and is what makes the environment reproducible. Don't edit it
by hand; change `pyproject.toml` and run `uv sync`.
