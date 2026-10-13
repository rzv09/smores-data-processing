"""The harmonized oxygen timeseries schema shared by all three SMORES sites.

One row is one sensor reading at one instant. Sites arrive in different native
formats (Florida Keys is wide with three sensor columns, others differ), and each
site's module is responsible for reshaping to this long form and passing
`validate`.

`deployment_id` is a required, first-class column rather than a site-specific
bonus column: every site's raw data is chunked into discrete deployments/casts,
and time-series work must never straddle a deployment boundary.

`do_mgl` (in-situ dissolved oxygen concentration, mg/L) is the primary signal
(`PRIMARY_SIGNAL`) and is required on every row. Each site module produces it on a
comparable, seawater-referenced footing: Santa Barbara reports it directly, Florida
Keys and Biscayne Bay derive it from percent saturation and seawater solubility at an
assumed salinity (see their module docstrings for the assumptions and the
uncertainty they carry). `do_sat` (percent air saturation) is kept alongside as a
secondary signal, and `do_native_value`/`do_native_unit` preserve the original
reading (e.g. "umol/L", "pO2_mmHg", "umol/kg") so a site's correction stays
auditable and reversible.

`site_id` and `data_source` carry the full project-wide category sets rather than
only the values present, so frames from different sites concatenate without
re-encoding and without silently widening to object dtype.

`SAMPLE_INTERVAL` is the shared analysis grid. Each site's pipeline
(`smores.data.pipeline`) bins its native-rate readings onto it, so every
processed output has timestamps on exact 2-minute boundaries. Native-rate frames
(e.g. the 8 Hz Florida Keys output) still conform to everything else here.
"""

import pandas as pd
from pandas.api.types import CategoricalDtype

SITE_IDS = ("biscayne_bay", "florida_keys", "santa_barbara")
DATA_SOURCES = ("smores_2026", "wcci_fl", "wcci_sb")
# The Biscayne Bay optodes log pO2 in mmHg (In-Situ units code 26): 107.3% sat
# reads 166 mmHg, i.e. 0.20946 * (760 - Pwv) per 100%, which hPa would not fit.
DO_NATIVE_UNITS = ("umol/L", "pO2_mmHg", "umol/kg")

SAMPLE_INTERVAL = pd.Timedelta(minutes=2)

PRIMARY_SIGNAL = "do_mgl"

QC_FLAGS = {0: "good", 1: "interpolated", 2: "flagged", 3: "masked"}

SITE_DTYPE = CategoricalDtype(SITE_IDS)
DATA_SOURCE_DTYPE = CategoricalDtype(DATA_SOURCES)
DO_NATIVE_UNIT_DTYPE = CategoricalDtype(DO_NATIVE_UNITS)

# Column order is part of the schema: it is what makes a diff between two sites'
# outputs readable.
COLUMNS = (
    "timestamp",
    "site_id",
    "deployment_id",
    "sensor_id",
    "do_sat",
    "temperature_c",
    "do_mgl",
    "do_native_value",
    "do_native_unit",
    "qc_flag",
    "data_source",
    "depth_m",
)

DTYPES = {
    "timestamp": "datetime64[ns, UTC]",
    "site_id": SITE_DTYPE,
    "deployment_id": "object",
    "sensor_id": "object",
    "do_sat": "float32",
    "temperature_c": "float32",
    "do_mgl": "float32",
    "do_native_value": "float32",
    "do_native_unit": DO_NATIVE_UNIT_DTYPE,
    "qc_flag": "int8",
    "data_source": DATA_SOURCE_DTYPE,
    "depth_m": "float32",
}

# Columns that may never be null. Everything else is optional: `do_sat`,
# `temperature_c`, `do_native_value`, `do_native_unit` and `depth_m` are NaN where a
# site never logged or cannot derive them.
REQUIRED = (
    "timestamp",
    "site_id",
    "deployment_id",
    "sensor_id",
    PRIMARY_SIGNAL,
    "qc_flag",
    "data_source",
)


class SchemaError(AssertionError):
    """Raised when a frame does not conform to the harmonized schema."""


def validate(df):
    """Check `df` against the schema, raising `SchemaError` on the first problem."""
    if list(df.columns) != list(COLUMNS):
        raise SchemaError(f"columns are {list(df.columns)!r}, expected {list(COLUMNS)!r}")

    if not isinstance(df["timestamp"].dtype, pd.DatetimeTZDtype):
        raise SchemaError(f"timestamp must be tz-aware, got {df['timestamp'].dtype}")
    if str(df["timestamp"].dt.tz) != "UTC":
        raise SchemaError(f"timestamp must be UTC, got {df['timestamp'].dt.tz}")
    if not df["timestamp"].is_monotonic_increasing:
        raise SchemaError("timestamp must be sorted non-decreasing")

    for name, dtype in DTYPES.items():
        if name in ("deployment_id", "sensor_id"):
            if not pd.api.types.is_string_dtype(df[name]) and not pd.api.types.is_object_dtype(df[name]):
                raise SchemaError(f"{name} must be str, got {df[name].dtype}")
            continue
        if df[name].dtype != pd.api.types.pandas_dtype(dtype):
            raise SchemaError(f"{name} is {df[name].dtype}, expected {dtype}")

    for name in REQUIRED:
        if df[name].isna().any():
            raise SchemaError(f"{name} contains nulls")

    unknown = set(df["qc_flag"].unique()) - set(QC_FLAGS)
    if unknown:
        raise SchemaError(f"unknown qc_flag values: {sorted(unknown)}")

    for name, allowed in (
        ("site_id", SITE_IDS),
        ("data_source", DATA_SOURCES),
        ("do_native_unit", DO_NATIVE_UNITS),
    ):
        used = set(df[name].dropna().unique())
        if not used <= set(allowed):
            raise SchemaError(f"{name} has values outside the enum: {sorted(used - set(allowed))}")

    return df


def conform(df):
    """Reorder `df` to `COLUMNS` and cast to `DTYPES`, without validating."""
    out = df[list(COLUMNS)].copy()
    out["deployment_id"] = out["deployment_id"].astype(str)
    out["sensor_id"] = out["sensor_id"].astype(str)
    return out.astype(
        {name: DTYPES[name] for name in COLUMNS if name not in ("deployment_id", "sensor_id")}
    )


def empty():
    """An empty frame with the schema's columns and dtypes, for tests and appends."""
    return pd.DataFrame({name: pd.Series(dtype=DTYPES[name]) for name in COLUMNS})
