# Data

Reference for everything under `data/` — layout, path constants, and what each raw
dataset actually contains.

## Layout

```
data/
├── data.md                                 this file
├── raw/                                    immutable inputs — never edited in place
│   ├── florida_keys/
│   │   └── florida_keys_data.csv           203 MB · 2,044,805 × 10   [present]
│   └── biscayne_bay/
│       └── FULL data from Harvard team/    Harvard drop             [not added yet]
├── interim/                                intermediate outputs
├── processed/                              analysis-ready datasets
└── external/                               third-party data
```

Dataset contents under `data/` are gitignored. Only the directory structure (via
`.gitkeep`) and markdown docs like this file are tracked, so a fresh clone gets the
skeleton and this reference, but no data.

## Path constants

Import from `smores.config`; never hardcode a filename or build a relative path. This
resolves identically from a notebook, a test, or the CLI.

| Constant | Points to |
| --- | --- |
| `FLORIDA_KEYS_CSV` | `data/raw/florida_keys/florida_keys_data.csv` |
| `BISCAYNE_HARVARD_DIR` | `data/raw/biscayne_bay/FULL data from Harvard team/` |
| `FLORIDA_DIR` | `data/raw/florida_keys/` |
| `BISCAYNE_DIR` | `data/raw/biscayne_bay/` |
| `RAW_DIR` | `data/raw/` |
| `INTERIM_DIR` | `data/interim/` |
| `PROCESSED_DIR` | `data/processed/` |
| `EXTERNAL_DIR` | `data/external/` |
| `DATA_DIR` | `data/` |

```python
from smores.config import FLORIDA_KEYS_CSV, PROCESSED_DIR

df = pd.read_csv(FLORIDA_KEYS_CSV)
df.to_parquet(PROCESSED_DIR / "florida_keys_clean.parquet")
```

## Florida Keys

`FLORIDA_KEYS_CSV` — 203 MB (212,737,786 bytes), 2,044,805 rows × 10 columns.

A full `pd.read_csv` takes ~2.5 s and ~198 MB in memory, so no `chunksize` or Parquet
conversion is needed at this size. Every column except `deployment` parses as `float64`
without dtype hints. **There are no nulls in any column.**

| Column | Type | Min | Max | Mean | Notes |
| --- | --- | --- | --- | --- | --- |
| `deployment` | `str` | — | — | — | 4 unique deployment IDs |
| `t` | `float64` | 10.00 | 56.00 | 27.57 | |
| `t_increase` | `float64` | 0 | 72,000 | 32,476 | elapsed time within a deployment |
| `Vx` | `float64` | −49.816 | 30.791 | −3.255 | velocity |
| `Vy` | `float64` | −49.840 | 26.650 | −3.157 | velocity |
| `Vz` | `float64` | −49.662 | 8.854 | −0.687 | velocity |
| `P` | `float64` | 77.00 | 87.00 | 82.36 | pressure |
| `O2_S1` | `float64` | 232.676 | 271.396 | 249.987 | oxygen sensor 1 |
| `O2_S2` | `float64` | 230.045 | 270.708 | 249.987 | oxygen sensor 2 |
| `O2_S3` | `float64` | 229.973 | 270.722 | 249.987 | oxygen sensor 3 |

Column meanings marked above are inferred from the names — confirm against the source
before relying on units.

### Deployments

| `deployment` | Rows |
| --- | --- |
| `3oec_2017_7_15_16` | 576,001 |
| `3oec_2017_7_13_14` | 547,202 |
| `3oec_2017_7_11_12` | 518,401 |
| `3oec_2017_7_16_17` | 403,201 |

The file stacks four separate deployments. Group by `deployment` before any time-series
operation — `t_increase` restarts at 0 for each one, so treating the file as a single
continuous series will silently produce wrong results.

## Biscayne Bay

Not added yet; arrives later in the project. `BISCAYNE_HARVARD_DIR` is already defined
and points at where it will live. A `Path` never touches the filesystem, so the missing
directory breaks no import and no test — when the data lands, drop it at that exact
path and the constant starts working with no code change.

Guard any code written ahead of the data:

```python
from smores.config import BISCAYNE_HARVARD_DIR

if BISCAYNE_HARVARD_DIR.exists():
    ...
```

`tests/test_datasets.py` has a contents check that is skipped until the directory
exists, then starts running on its own.

## Conventions

- **`raw/` is immutable.** Never write to it or edit a file in place. Derived data goes
  to `interim/` (intermediate) or `processed/` (analysis-ready).
- **Reads go through `smores.config`,** so paths work from any working directory.
- **Nothing in `data/` is committed.** Keep it reproducible from the source, or document
  where it came from here.
