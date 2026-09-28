"""The harmonized oxygen timeseries schema shared by all three SMORES sites.

One row is one sensor reading at one instant. Sites arrive in different native
formats (Florida Keys is wide with three sensor columns, others differ), and each
site's module is responsible for reshaping to this long form and passing
`validate`.

`site_id` and `data_source` carry the full project-wide category sets rather than
only the values present, so frames from different sites concatenate without
re-encoding and without silently widening to object dtype.
"""

import pandas as pd
from pandas.api.types import CategoricalDtype

SITE_IDS = ("biscayne_bay", "florida_keys", "santa_barbara")
DATA_SOURCES = ("smores_2026", "wcci_fl", "wcci_sb")

QC_FLAGS = {0: "good", 1: "interpolated", 2: "flagged", 3: "masked"}

SITE_DTYPE = CategoricalDtype(SITE_IDS)
DATA_SOURCE_DTYPE = CategoricalDtype(DATA_SOURCES)

# Column order is part of the schema: it is what makes a diff between two sites'
# outputs readable.
COLUMNS = (
    "timestamp",
    "site_id",
    "depth_m",
    "height_above_bed_m",
    "sensor_id",
    "do_sat",
    "do_mgl",
    "raw_o2_umol",
    "temperature_c",
    "qc_flag",
    "data_source",
)

DTYPES = {
    "timestamp": "datetime64[ns, UTC]",
    "site_id": SITE_DTYPE,
    "depth_m": "float32",
    "height_above_bed_m": "float32",
    "sensor_id": "category",
    "do_sat": "float32",
    "do_mgl": "float32",
    "raw_o2_umol": "float32",
    "temperature_c": "float32",
    "qc_flag": "int8",
    "data_source": DATA_SOURCE_DTYPE,
}

# Columns that may never be null. `temperature_c`, `do_mgl` and `depth_m` are
# deliberately absent: a site that did not log temperature leaves it NaN.
REQUIRED = ("timestamp", "site_id", "sensor_id", "qc_flag", "data_source")


class SchemaError(AssertionError):
    """Raised when a frame does not conform to the harmonized schema."""


def validate(df, *, extra_columns=()):
    """Check `df` against the schema, raising `SchemaError` on the first problem.

    `extra_columns` names site-specific columns permitted after the schema columns,
    in order. Florida Keys uses it to carry `deployment`.
    """
    expected = list(COLUMNS) + list(extra_columns)
    if list(df.columns) != expected:
        raise SchemaError(f"columns are {list(df.columns)!r}, expected {expected!r}")

    if not isinstance(df["timestamp"].dtype, pd.DatetimeTZDtype):
        raise SchemaError(f"timestamp must be tz-aware, got {df['timestamp'].dtype}")
    if str(df["timestamp"].dt.tz) != "UTC":
        raise SchemaError(f"timestamp must be UTC, got {df['timestamp'].dt.tz}")
    if not df["timestamp"].is_monotonic_increasing:
        raise SchemaError("timestamp must be sorted non-decreasing")

    for name, dtype in DTYPES.items():
        if name == "sensor_id":
            if not isinstance(df[name].dtype, CategoricalDtype):
                raise SchemaError(f"{name} must be categorical, got {df[name].dtype}")
            continue
        if df[name].dtype != pd.api.types.pandas_dtype(dtype):
            raise SchemaError(f"{name} is {df[name].dtype}, expected {dtype}")

    for name in REQUIRED:
        if df[name].isna().any():
            raise SchemaError(f"{name} contains nulls")

    unknown = set(df["qc_flag"].unique()) - set(QC_FLAGS)
    if unknown:
        raise SchemaError(f"unknown qc_flag values: {sorted(unknown)}")

    for name, allowed in (("site_id", SITE_IDS), ("data_source", DATA_SOURCES)):
        used = set(df[name].dropna().unique())
        if not used <= set(allowed):
            raise SchemaError(f"{name} has values outside the enum: {sorted(used - set(allowed))}")

    return df


def empty():
    """An empty frame with the schema's columns and dtypes, for tests and appends."""
    return pd.DataFrame({name: pd.Series(dtype=DTYPES[name]) for name in COLUMNS})
