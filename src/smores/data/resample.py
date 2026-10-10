"""Harmonized long timeseries -> fixed-grid resampled dataset.

The Florida Keys record is 8 Hz, which is far finer than any model here needs and
62 MB of parquet to carry around. This module writes the binned version once, to
`data/processed/`, so notebooks and model code read a small file instead of
re-binning millions of rows on every import.

Nothing here is specific to that site. The target interval is a parameter, and the
*source* interval is measured from the data rather than assumed, so the same code
serves Biscayne Bay or Santa Barbara at whatever rate their instruments logged at
-- which matters because a fixed `min_obs` threshold means completely different
things at 8 Hz and at 1 Hz. See "Target interval" and "Native interval" below.

Three choices are worth stating, because none is a default worth inheriting
silently.

Median, not mean
----------------
Each bin aggregates by median. Pyroscience optodes produce isolated spikes, and a
median over the hundreds of reads in a bin discards them where a mean would carry
them through attenuated. This matches `resample_channel()` in
`src/data_processing/test_sampling_frequency.py`, which is what the sampling-interval
benchmark was run with, and `load_wide()` in `smores.models.baselines`.

Per deployment and sensor
-------------------------
Bins are formed within one `(site_id, deployment_id, sensor_id)` group only, so no
bin ever spans the gap between two deployments -- the four Florida Keys deployments
sit hours apart, and a bin straddling that boundary would average across it. Bins
that caught no reads are dropped rather than emitted as nulls: an absent row says
"no data" without a downstream consumer having to decide what a NaN `do_sat` means,
and the schema forbids a null `do_sat` anyway.

Target interval
---------------
`freq` takes any pandas offset alias, and the output path records which one was
used, so several intervals of one source can sit side by side. The default is 2
minutes, the winner of `test_sampling_frequency.py` (best in 4 of 6 runs; see
`data/PLAIN_ENGLISH_GUIDE.md`) -- but read that as "fine-grained, roughly 1-5 min"
rather than a precise optimum, since the scoring weights were chosen by hand, and
it was established on this one site. Another dataset deserves its own `freq`;
re-running that benchmark is the way to pick it.

Native interval
---------------
A dataset's own sampling interval is inferred by `native_interval()` -- the median
gap between consecutive reads within a single sensor-deployment, which is robust to
both the gaps between deployments and to occasional dropped polls. Two things use
it. A `freq` finer than the native interval is rejected: median binning aggregates,
it cannot invent reads, and asking for 1 s bins from 1-minute data silently
produces a mostly-empty grid. And `coverage` -- `n_obs` over the reads a full bin
would hold at that native rate -- gives a thinness measure that means the same
thing on every dataset, which a raw count does not. `--min-coverage 0.5` is
therefore portable in a way that `--min-obs 480` is not.

Input
-----
`normalize()` accepts either the current harmonized schema or the older on-disk
layout the processed parquet was built with (`raw_o2_umol`, `deployment`,
`height_above_bed_m`, categorical id columns), for the same reason
`smores.models.baselines._resolve` does: so this module works against the parquet
as it exists today and against a rebuilt one, without `make data` having to be
re-run first. `height_above_bed_m` is dropped -- it is a documented site constant
(`florida_keys.HEIGHT_ABOVE_BED_M`), not a shared-schema column.

Output
------
The harmonized long schema (`smores.data.schema.COLUMNS`), in order, plus `n_obs`
(raw reads behind the bin) and `coverage` (that count over a full bin's worth at
the native rate). Both are carried rather than filtered on, because they are what
makes a thin bin identifiable after the fact; `--min-obs` and `--min-coverage` are
there for callers who would rather drop them at write time. `qc_flag` is the
maximum over the bin's contributing rows, so a bin holding any interpolated read is
marked interpolated.
"""

import argparse
from pathlib import Path

import pandas as pd

from smores import config
from smores.data import schema

DEFAULT_FREQ = "2min"

# Columns aggregated by median. Everything else in the schema is either a group key
# or constant within a group.
VALUE_COLUMNS = ("do_sat", "temperature_c", "do_mgl", "do_native_value", "depth_m")

GROUP_KEYS = ("site_id", "deployment_id", "sensor_id")

# Constant within a group, carried through unchanged.
CARRIED_COLUMNS = ("do_native_unit", "data_source")

# good and interpolated. Flagged (2) and masked (3) rows are excluded from the
# aggregate by default rather than being medianed into it.
KEEP_QC_FLAGS = (0, 1)

COUNT_COLUMN = "n_obs"
COVERAGE_COLUMN = "coverage"
OUTPUT_COLUMNS = (*schema.COLUMNS, COUNT_COLUMN, COVERAGE_COLUMN)

# Older column names in the on-disk processed parquet -> current schema names.
LEGACY_RENAMES = {
    "raw_o2_umol": "do_native_value",
    "deployment": "deployment_id",
}

# The legacy oxygen column spelled its unit in its own name, so the missing
# `do_native_unit` is recoverable rather than assumed.
LEGACY_NATIVE_UNITS = {"raw_o2_umol": "umol/L"}


def normalize(df):
    """Return `df` as a valid harmonized frame, translating legacy column names.

    Raises `schema.SchemaError` if the result still does not conform, so a frame
    that is wrong in some way this does not know how to fix fails here rather than
    halfway through a resample.
    """
    renames = {
        old: new
        for old, new in LEGACY_RENAMES.items()
        if old in df.columns and new not in df.columns
    }
    out = df.rename(columns=renames)

    if "do_native_unit" not in out.columns:
        units = {LEGACY_NATIVE_UNITS[old] for old in renames if old in LEGACY_NATIVE_UNITS}
        out["do_native_unit"] = units.pop() if len(units) == 1 else None

    for name in schema.COLUMNS:
        if name not in out.columns:
            out[name] = None

    out = out[list(schema.COLUMNS)].copy()
    # The legacy file stores the id columns as categoricals; the schema wants str.
    for name in ("deployment_id", "sensor_id"):
        out[name] = out[name].astype(str)
    out = out.astype(
        {
            name: dtype
            for name, dtype in schema.DTYPES.items()
            if name not in ("deployment_id", "sensor_id")
        }
    )
    out = out.sort_values("timestamp", kind="stable", ignore_index=True)
    return schema.validate(out)


def default_out_path(freq=DEFAULT_FREQ, source=None):
    """`florida_keys_o2.parquet` + `2min` -> `florida_keys_o2_2min.parquet`.

    The interval goes in the filename because two resamplings of one source are a
    normal thing to have side by side, and a bare `_resampled` suffix would make
    them collide.
    """
    source = config.FLORIDA_KEYS_PARQUET if source is None else Path(source)
    return source.parent / f"{source.stem}_{freq}.parquet"


def native_interval(df, group_keys=GROUP_KEYS):
    """The dataset's own sampling interval, as a `pd.Timedelta`.

    Measured as the median gap between consecutive reads *within* one
    sensor-deployment series, then the median of those per-series medians. Taking it
    within a series is what keeps the hours-long gaps between deployments out of the
    estimate; taking medians is what keeps dropped polls out of it. Returns `NaT`
    when no series has two reads to compare.
    """
    gaps = []
    for _, group in df.groupby(list(group_keys), observed=True, sort=False):
        stamps = group["timestamp"].sort_values()
        if len(stamps) < 2:
            continue
        diffs = stamps.diff().dropna()
        diffs = diffs[diffs > pd.Timedelta(0)]
        if not diffs.empty:
            gaps.append(diffs.median())
    if not gaps:
        return pd.NaT
    return pd.Series(gaps).median()


def bin_width(freq):
    """A `freq` alias as a `pd.Timedelta`, or `NaT` if it has no fixed width.

    Calendar offsets ("ME", "QE", "YE") have no constant duration, so they get no
    coverage figure and skip the upsampling check rather than being rejected --
    they are still perfectly valid things to resample to.
    """
    try:
        return pd.Timedelta(pd.tseries.frequencies.to_offset(freq).nanos, unit="ns")
    except ValueError:
        return pd.NaT


def obs_per_full_bin(freq, interval):
    """How many native-rate reads a complete `freq` bin holds.

    `NaN` when either the native interval or the bin width is unknown, which is
    what makes `coverage` null rather than wrong for a dataset too short to infer a
    rate from, or a calendar-offset grid.
    """
    width = bin_width(freq)
    if pd.isna(width) or pd.isna(interval):
        return float("nan")
    return width.value / interval.value


def _carried_value(group, column):
    """The group's single value for a column that must be constant within it."""
    used = group[column].dropna().unique()
    if len(used) > 1:
        raise ValueError(
            f"{column} is not constant within a group: {sorted(map(str, used))}"
        )
    return used[0] if len(used) else None


def resample_long(
    df,
    freq=DEFAULT_FREQ,
    value_columns=VALUE_COLUMNS,
    keep_qc_flags=KEEP_QC_FLAGS,
    min_obs=1,
    min_coverage=0.0,
    interval=None,
):
    """Median-bin a harmonized long frame onto a fixed `freq` grid.

    Returns a frame in the schema's column order plus `n_obs` and `coverage`. Bin
    timestamps are left edges on a UTC-anchored grid, so a given `freq` lands on the
    same instants regardless of where a deployment happens to start.

    `interval` overrides the inferred native sampling interval (anything
    `pd.Timedelta` accepts). Pass it when a dataset's nominal rate is known and
    should be used instead of what the timestamps happen to show -- e.g. a sensor
    that dropped enough polls to skew the median gap.
    """
    df = normalize(df)
    value_columns = [name for name in value_columns if name in df.columns]
    if min_obs < 1:
        raise ValueError(f"min_obs must be at least 1, got {min_obs}")
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError(f"min_coverage must be in [0, 1], got {min_coverage}")

    kept = df[df["qc_flag"].isin(keep_qc_flags)]

    interval = native_interval(kept) if interval is None else pd.Timedelta(interval)
    width = bin_width(freq)
    if not pd.isna(width) and not pd.isna(interval) and width < interval:
        raise ValueError(
            f"freq {freq!r} ({width}) is finer than the dataset's native sampling "
            f"interval ({interval}); median binning aggregates and cannot upsample"
        )
    full_bin = obs_per_full_bin(freq, interval)
    if min_coverage > 0.0 and pd.isna(full_bin):
        raise ValueError(
            f"min_coverage={min_coverage} needs a known native interval and a "
            f"fixed-width freq; got interval={interval!r}, freq={freq!r}"
        )

    pieces = []
    for keys, group in kept.groupby(list(GROUP_KEYS), observed=True, sort=True):
        indexed = group.set_index("timestamp").sort_index()
        binned = indexed[value_columns].resample(freq).median()
        binned[COUNT_COLUMN] = indexed["do_sat"].resample(freq).count()
        binned["qc_flag"] = indexed["qc_flag"].resample(freq).max()

        binned[COVERAGE_COLUMN] = binned[COUNT_COLUMN] / full_bin

        # Empty bins carry a NaN qc_flag and cannot be cast to int8; drop before
        # the cast, not after.
        binned = binned[binned[COUNT_COLUMN] >= min_obs]
        if min_coverage > 0.0:
            binned = binned[binned[COVERAGE_COLUMN] >= min_coverage]
        if binned.empty:
            continue

        for name, value in zip(GROUP_KEYS, keys):
            binned[name] = value
        for name in CARRIED_COLUMNS:
            if name in group.columns:
                binned[name] = _carried_value(group, name)
        pieces.append(binned)

    if not pieces:
        empty = schema.empty()
        empty[COUNT_COLUMN] = pd.Series(dtype="int64")
        empty[COVERAGE_COLUMN] = pd.Series(dtype="float32")
        return empty

    out = pd.concat(pieces).rename_axis("timestamp").reset_index()
    # Stable sort on timestamp alone preserves the groupby's (deployment, sensor)
    # order within a bin, which is what makes two runs byte-identical.
    out = out.sort_values("timestamp", kind="stable", ignore_index=True)

    out["qc_flag"] = out["qc_flag"].astype("int8")
    out[COUNT_COLUMN] = out[COUNT_COLUMN].astype("int64")
    out[COVERAGE_COLUMN] = out[COVERAGE_COLUMN].astype("float32")
    for name in GROUP_KEYS + CARRIED_COLUMNS:
        if name in out.columns:
            out[name] = out[name].astype(schema.DTYPES[name])
    for name in value_columns:
        out[name] = out[name].astype(schema.DTYPES[name])
    for name in schema.COLUMNS:
        if name not in out.columns:
            out[name] = pd.Series(pd.NA, index=out.index, dtype=schema.DTYPES[name])

    out = out[list(OUTPUT_COLUMNS)]
    # Validate the schema columns on their own; `n_obs` and `coverage` are additions
    # to the schema, not part of it, and `validate` requires an exact column list.
    schema.validate(out[list(schema.COLUMNS)])
    return out


def load_source(source=None):
    """Read a harmonized parquet. Defaults to the Florida Keys one."""
    source = config.FLORIDA_KEYS_PARQUET if source is None else Path(source)
    return pd.read_parquet(source, engine="pyarrow")


def build(
    freq=DEFAULT_FREQ,
    write=True,
    source=None,
    out_path=None,
    min_obs=1,
    min_coverage=0.0,
    interval=None,
    df=None,
):
    """Read the harmonized parquet, bin it to `freq`, and write it alongside.

    `df` takes an already-loaded frame, so binning one source to several intervals
    does not read it off disk once per interval.
    """
    source = config.FLORIDA_KEYS_PARQUET if source is None else Path(source)
    frame = load_source(source) if df is None else df
    out = resample_long(
        frame,
        freq=freq,
        min_obs=min_obs,
        min_coverage=min_coverage,
        interval=interval,
    )
    if write:
        destination = out_path or default_out_path(freq, source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(destination, compression="snappy", index=False)
    return out


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--freq",
        nargs="+",
        default=[DEFAULT_FREQ],
        help=f"one or more pandas offset aliases (default: {DEFAULT_FREQ})",
    )
    parser.add_argument(
        "--source",
        type=lambda value: config.PROJECT_ROOT / value if not value.startswith("/") else value,
        default=None,
        help="harmonized parquet to read (default: config.FLORIDA_KEYS_PARQUET)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output path; only valid with a single --freq (default: alongside the source)",
    )
    parser.add_argument(
        "--min-obs",
        type=int,
        default=1,
        help="drop bins holding fewer than this many raw reads (default: 1)",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.0,
        help="drop bins less than this full at the dataset's native rate, 0-1 "
             "(default: 0, keep all; portable across datasets where --min-obs is not)",
    )
    parser.add_argument(
        "--interval",
        default=None,
        help="native sampling interval, e.g. 125ms or 1min; overrides what is "
             "inferred from the timestamps",
    )
    parser.add_argument("--no-write", action="store_true", help="summarize without writing")
    args = parser.parse_args(argv)
    if args.out and len(args.freq) > 1:
        parser.error("--out takes a single --freq")
    return args


def main(argv=None):
    args = parse_args(argv)
    source = config.FLORIDA_KEYS_PARQUET if args.source is None else Path(args.source)

    # Read once, bin many: the source can be tens of millions of rows.
    df = normalize(load_source(source))
    interval = pd.Timedelta(args.interval) if args.interval else native_interval(df)
    print(f"{source}: native sampling interval {interval} ({len(df):,} rows)")

    for freq in args.freq:
        out = build(
            freq=freq,
            write=not args.no_write,
            source=source,
            out_path=args.out,
            min_obs=args.min_obs,
            min_coverage=args.min_coverage,
            interval=interval,
            df=df,
        )
        destination = args.out or default_out_path(freq, source)
        verb = "would write" if args.no_write else "wrote"
        full_bin = obs_per_full_bin(freq, interval)
        print(f"{verb} {len(out):,} bins at {freq} to {destination}")
        if out.empty:
            continue
        print(
            f"  {out['timestamp'].min()} -> {out['timestamp'].max()}  "
            f"({out['sensor_id'].nunique()} sensors, "
            f"{out['deployment_id'].nunique()} deployments)"
        )
        print(
            f"  {COUNT_COLUMN} median {out[COUNT_COLUMN].median():.0f} of "
            f"{full_bin:.0f} per full bin  "
            f"{COVERAGE_COLUMN} min {out[COVERAGE_COLUMN].min():.3f}  "
            f"bins under half full: {out[COVERAGE_COLUMN].lt(0.5).sum():,}"
        )
        print(
            f"  do_sat mean {out['do_sat'].mean():.2f}%  range "
            f"{out['do_sat'].min():.2f}-{out['do_sat'].max():.2f}%"
        )


if __name__ == "__main__":
    main()