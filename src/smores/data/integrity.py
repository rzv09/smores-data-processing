"""Integrity checks on a harmonized, gridded frame.

`schema.validate` answers "is this frame shaped right": columns, dtypes, enums, no
nulls where required. It says nothing about whether the *data* is sane. The checks
here do that, and they come in two kinds:

* Generic checks (`check_frame`) that hold for every site: one row per key, every
  timestamp on the 2-minute grid, values inside physical bounds, and a sensor's
  deployments not overlapping in time.
* Site checks, written in each site module and passed in, for the things only that
  site's physics or instruments can confirm -- e.g. replicate sensors agreeing, a
  recomputed saturation matching the sensor's own, or a diel cycle in the right local
  hours. `diel_check` lives here because all three sites use it. It is the one check
  that catches a timezone error, which every range and count check passes happily.

Each check is an error or a warning. Errors fail the pipeline (`Report.raise_for_failures`);
warnings are reported but do not block, for findings that are real but expected, such as
Santa Barbara's 15-minute sampling being coarser than the grid.

The report also carries a coverage table, one row per (deployment, sensor): span, bins
present against bins expected, gaps, and share of good-quality bins.
"""

import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from smores.data import schema

ERROR = "error"
WARN = "warn"

# Generous physical bounds: wide enough for the negative pO2 floor of buried optodes and
# for real supersaturation -- Biscayne ch7, an exposed optode, peaks at 441% sat around
# noon on 2026-06-08 in a smooth diel cycle (2-min steps ~5%, back to 50% by night) --
# tight enough to catch unit mistakes and sentinel values. `do_native_value` is
# unit-dependent and not bounded here.
RANGES = {
    "do_sat": (-5.0, 600.0),
    "temperature_c": (-2.0, 40.0),
    "do_mgl": (-1.0, 30.0),
    "depth_m": (-0.5, 100.0),
}

# A run of more than this many missing bins counts as a gap in the coverage table.
GAP_BINS = 2

KEYS = ["site_id", "deployment_id", "sensor_id", "timestamp"]


class IntegrityError(AssertionError):
    """Raised when a frame fails one or more error-severity integrity checks."""


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    severity: str = ERROR


@dataclass
class Report:
    site_id: str
    checks: list = field(default_factory=list)
    coverage: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: dict = field(default_factory=dict)

    @property
    def failures(self):
        return [c for c in self.checks if not c.passed and c.severity == ERROR]

    @property
    def warnings(self):
        return [c for c in self.checks if not c.passed and c.severity == WARN]

    @property
    def ok(self):
        return not self.failures

    def raise_for_failures(self):
        if self.failures:
            lines = "\n".join(f"  {c.name}: {c.detail}" for c in self.failures)
            raise IntegrityError(f"{self.site_id} failed {len(self.failures)} check(s):\n{lines}")

    def to_dict(self):
        coverage = self.coverage.copy()
        for name in coverage.columns:
            if pd.api.types.is_datetime64_any_dtype(coverage[name]):
                coverage[name] = coverage[name].astype(str)
            elif pd.api.types.is_timedelta64_dtype(coverage[name]):
                coverage[name] = coverage[name].astype(str)
        return {
            "site_id": self.site_id,
            "ok": self.ok,
            "summary": self.summary,
            "checks": [asdict(c) for c in self.checks],
            "coverage": coverage.to_dict(orient="records"),
        }

    def write(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=_json_default))

    def __str__(self):
        lines = [f"integrity: {self.site_id} -- {'OK' if self.ok else 'FAILED'}"]
        for c in self.checks:
            mark = "pass" if c.passed else ("FAIL" if c.severity == ERROR else "warn")
            lines.append(f"  [{mark}] {c.name}: {c.detail}")
        return "\n".join(lines)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)}")


def check_frame(df, interval=schema.SAMPLE_INTERVAL):
    """The site-independent checks, on a frame that already passes `schema.validate`."""
    checks = [Check("not_empty", len(df) > 0, f"{len(df):,} rows")]

    sites = df["site_id"].dropna().unique().tolist()
    checks.append(Check("single_site", len(sites) == 1, f"site_id values {sites}"))

    duplicated = int(df.duplicated(KEYS).sum())
    checks.append(
        Check(
            "unique_keys",
            duplicated == 0,
            f"{duplicated:,} duplicate (deployment, sensor, timestamp) rows",
        )
    )

    off_grid = int((df["timestamp"] != df["timestamp"].dt.floor(interval)).sum())
    checks.append(
        Check(
            "on_grid",
            off_grid == 0,
            f"{off_grid:,} timestamps off the {interval.total_seconds() / 60:g}-minute grid",
        )
    )

    for name, (low, high) in RANGES.items():
        values = df[name].dropna()
        bad = int((~values.between(low, high)).sum())
        detail = (
            f"{bad:,} of {len(values):,} outside [{low}, {high}]" if len(values) else "all null"
        )
        if len(values):
            detail += f"; observed {values.min():.3f} to {values.max():.3f}"
        checks.append(Check(f"range_{name}", bad == 0, detail))

    checks.append(_deployments_disjoint(df))
    return checks


def _deployments_disjoint(df):
    spans = (
        df.groupby(["sensor_id", "deployment_id"], observed=True)["timestamp"]
        .agg(["min", "max"])
        .reset_index()
        .sort_values(["sensor_id", "min"])
    )
    overlaps = []
    for sensor_id, group in spans.groupby("sensor_id", observed=True):
        previous_end = group["max"].shift()
        hit = group[group["min"] <= previous_end]
        overlaps += [f"{sensor_id}/{d}" for d in hit["deployment_id"]]
    return Check(
        "deployments_disjoint",
        not overlaps,
        f"{len(overlaps)} overlapping deployment span(s)"
        + (f": {overlaps[:5]}" if overlaps else ""),
    )


def coverage(df, interval=schema.SAMPLE_INTERVAL):
    """One row per (deployment, sensor): span, fill, gaps and share of good bins."""
    rows = []
    for (deployment_id, sensor_id), group in df.groupby(
        ["deployment_id", "sensor_id"], observed=True
    ):
        stamps = group["timestamp"].sort_values()
        steps = (stamps.diff().dropna() / interval).astype("int64")
        missing = steps - 1
        long_gaps = missing[missing > GAP_BINS]
        expected = int((stamps.iloc[-1] - stamps.iloc[0]) / interval) + 1
        row = {
            "deployment_id": deployment_id,
            "sensor_id": sensor_id,
            "start": stamps.iloc[0],
            "end": stamps.iloc[-1],
            "n_bins": len(stamps),
            "expected_bins": expected,
            "coverage_pct": round(100.0 * len(stamps) / expected, 2),
            "n_gaps": len(long_gaps),
            "longest_gap": (int(missing.max()) if len(missing) else 0) * interval,
            "good_pct": round(100.0 * (group["qc_flag"] == 0).mean(), 2),
        }
        if "n_obs" in group:
            row["median_n_obs"] = float(group["n_obs"].median())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df):
    return {
        "rows": len(df),
        "start": str(df["timestamp"].min()),
        "end": str(df["timestamp"].max()),
        "deployments": int(df["deployment_id"].nunique()),
        "sensors": int(df["sensor_id"].nunique()),
        "qc_flag_counts": {
            schema.QC_FLAGS[int(k)]: int(v)
            for k, v in df["qc_flag"].value_counts().sort_index().items()
        },
        "primary_signal": schema.PRIMARY_SIGNAL,
        f"{schema.PRIMARY_SIGNAL}_mean": round(float(df[schema.PRIMARY_SIGNAL].mean()), 3),
    }


def diel_check(
    df,
    local_tz,
    name="diel_cycle",
    trough_hours=range(0, 10),
    peak_hours=range(10, 22),
    min_bins_per_hour=30,
    severity=ERROR,
):
    """Oxygen should trough in the night/early morning and peak in the daytime, local time.

    Runs on `schema.PRIMARY_SIGNAL`, good-quality rows only. Hour bins with fewer than
    `min_bins_per_hour` rows are dropped, because uneven deployment coverage leaves near-empty boundary hours that
    produce spurious extremes (see the Florida Keys note in data/README.md).
    """
    good = df[df["qc_flag"] == 0]
    if good.empty:
        return Check(name, False, "no good-quality rows to test", severity)
    hour = good["timestamp"].dt.tz_convert(local_tz).dt.hour
    grouped = good.groupby(hour)[schema.PRIMARY_SIGNAL]
    by_hour = grouped.mean()[grouped.size() >= min_bins_per_hour]
    if len(by_hour) < 12:
        return Check(name, False, f"only {len(by_hour)} local hours have enough data", severity)
    trough, peak = int(by_hour.idxmin()), int(by_hour.idxmax())
    passed = trough in trough_hours and peak in peak_hours
    return Check(
        name,
        passed,
        f"local ({local_tz}) trough {trough:02d}:00, peak {peak:02d}:00, "
        f"swing {by_hour.max() - by_hour.min():.3f} {schema.PRIMARY_SIGNAL}",
        severity,
    )


def build_report(df, site_id, site_checks=(), interval=schema.SAMPLE_INTERVAL):
    """Run the generic checks plus `site_checks` on `df` and assemble the report."""
    return Report(
        site_id=site_id,
        checks=check_frame(df, interval) + list(site_checks),
        coverage=coverage(df, interval),
        summary=summarize(df),
    )
