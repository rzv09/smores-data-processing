"""
SMORES raw-file ingest: manifest discovery, per-chunk tick<->RTC clock
fitting, poll reconstruction, and long -> wide pivot.

See ../../../SMORES_preprocessing_plan_v2.md sections 1-2.

Directory layout assumed: one directory containing files named
    SENS_<serial>_<chunk>_<Kind>.csv
where <Kind> is one of RDO, RTC, External_PTS_Bar100, Internal_PTS, IMU,
Image_Meta. The tick counter (the `timestamp` column of every file) is a
single globally-monotonic counter across all chunks in a directory; each
chunk's RTC log is the only source of wall-clock truth for that chunk and
must be fit independently (ticks/s drifts ~0.25% chunk to chunk).
"""

import glob
import os
import re

import numpy as np
import pandas as pd

TICKS_PER_SEC_NOMINAL = 1.00155e8
POLL_GAP_SECONDS = 5.0

# RDO `param` -> output column, per the plan's poll reconstruction (S1).
PARAM_COLUMNS = {1: "temp_c", 20: "do_mgl", 21: "do_pctsat_reported", 30: "po2"}
N_PARAMS_PER_POLL = len(PARAM_COLUMNS)

_CHUNK_RE = re.compile(r"SENS_(?P<serial>\d+)_(?P<chunk>\d+)_(?P<kind>.+)\.csv$")


def discover_chunks(data_dir: str):
    """Group per-chunk files in `data_dir` by chunk number.

    Returns a list of (chunk_id, {kind: path}) sorted by chunk_id, which is
    also tick order (verified against this archive's chunk numbering).
    """
    chunks = {}
    for path in glob.glob(os.path.join(data_dir, "SENS_*_*.csv")):
        m = _CHUNK_RE.search(os.path.basename(path))
        if not m:
            continue
        chunk_id = int(m.group("chunk"))
        chunks.setdefault(chunk_id, {})[m.group("kind")] = path
    return [(c, chunks[c]) for c in sorted(chunks)]


def fit_chunk_clock(rtc_df: pd.DataFrame):
    """Fit tick -> unix epoch seconds for one chunk's RTC log.

    Returns (slope_sec_per_tick, intercept_sec, resid_rms_seconds).
    """
    wall = pd.to_datetime({
        "year": rtc_df["year"] + 2000,
        "month": rtc_df["month"],
        "day": rtc_df["mday"],
        "hour": rtc_df["hours"],
        "minute": rtc_df["minutes"],
        "second": rtc_df["seconds"],
    })
    epoch = wall.astype("datetime64[ns]").astype("int64").to_numpy(dtype=np.float64) / 1e9
    ticks = rtc_df["timestamp"].to_numpy(dtype=np.float64)
    slope, intercept = np.polyfit(ticks, epoch, 1)
    resid = (slope * ticks + intercept) - epoch
    return float(slope), float(intercept), float(np.sqrt(np.mean(resid ** 2)))


def ticks_to_datetime(ticks, slope: float, intercept: float):
    epoch = slope * np.asarray(ticks, dtype=np.float64) + intercept
    return pd.to_datetime(epoch, unit="s")


def assign_poll_id(
    df: pd.DataFrame,
    tick_col: str = "timestamp",
    group_cols=("serial_id", "modbus_id"),
    gap_seconds: float = POLL_GAP_SECONDS,
    ticks_per_sec: float = TICKS_PER_SEC_NOMINAL,
) -> pd.DataFrame:
    """Assign a poll_id within each (serial_id, modbus_id) group, cutting a
    new poll wherever the gap to the previous tick exceeds gap_seconds.

    Real inter-poll gaps are ~52s; intra-poll gaps between the 4 params are
    <1s, so a 5s cut is symmetric across params (plan S1).
    """
    group_cols = list(group_cols)
    gap_ticks = gap_seconds * ticks_per_sec
    df = df.sort_values(group_cols + [tick_col]).copy()
    delta = df.groupby(group_cols)[tick_col].diff()
    new_poll = delta.isna() | (delta > gap_ticks)
    df["poll_id"] = new_poll.groupby([df[c] for c in group_cols]).cumsum()
    return df


def reconstruct_polls(
    rdo_df: pd.DataFrame,
    ticks_per_sec: float = TICKS_PER_SEC_NOMINAL,
    gap_seconds: float = POLL_GAP_SECONDS,
):
    """Pivot poll_id x param -> temp_c/do_mgl/do_pctsat_reported/po2.

    Poll timestamp is the minimum tick in the poll. Returns (wide, exceptions)
    where `exceptions` holds polls that did not see exactly 4 params.
    """
    tagged = assign_poll_id(rdo_df, gap_seconds=gap_seconds, ticks_per_sec=ticks_per_sec)
    keyed = tagged.groupby(["serial_id", "modbus_id", "poll_id"])
    poll_ts = keyed["timestamp"].min()
    n_params = keyed["param"].nunique()

    wide = tagged.pivot_table(
        index=["serial_id", "modbus_id", "poll_id"],
        columns="param",
        values="value",
        aggfunc="first",
    )
    wide = wide.rename(columns=PARAM_COLUMNS)
    for col in PARAM_COLUMNS.values():
        if col not in wide.columns:
            wide[col] = np.nan
    wide["timestamp"] = poll_ts
    wide["n_params"] = n_params
    wide = wide.reset_index()

    exceptions = wide[wide["n_params"] != N_PARAMS_PER_POLL].copy()
    return wide, exceptions


def process_chunk(chunk_id: int, chunk_files: dict, gap_seconds: float = POLL_GAP_SECONDS):
    """Ingest one chunk's RDO stream against its own RTC clock fit.

    Returns (poll_df, exceptions_df, calib_row_dict). poll_df has columns
    channel_id, datetime, temp_c, do_mgl, do_pctsat_reported, po2, n_params,
    chunk_id. Raises if the chunk has no RTC or RDO file.
    """
    if "RTC" not in chunk_files or "RDO" not in chunk_files:
        raise FileNotFoundError(f"chunk {chunk_id} missing RTC or RDO file: {chunk_files}")

    rtc_df = pd.read_csv(chunk_files["RTC"])
    slope, intercept, resid_rms = fit_chunk_clock(rtc_df)
    ticks_per_sec = 1.0 / slope

    rdo_df = pd.read_csv(chunk_files["RDO"])
    wide, exceptions = reconstruct_polls(rdo_df, ticks_per_sec=ticks_per_sec, gap_seconds=gap_seconds)
    wide["datetime"] = ticks_to_datetime(wide["timestamp"], slope, intercept)
    wide = wide.rename(columns={"modbus_id": "channel_id"})
    wide["chunk_id"] = chunk_id

    calib_row = {
        "chunk_id": chunk_id,
        "slope_sec_per_tick": slope,
        "intercept_sec": intercept,
        "ticks_per_sec": ticks_per_sec,
        "resid_rms_seconds": resid_rms,
        "n_rtc_rows": len(rtc_df),
        "tick_min": rdo_df["timestamp"].min(),
        "tick_max": rdo_df["timestamp"].max(),
        "n_polls": len(wide),
        "n_poll_exceptions": len(exceptions),
    }
    return wide, exceptions, calib_row


def ingest_timeseries_file(path: str, slope: float, intercept: float, tick_col: str = "timestamp"):
    """Ingest a non-RDO per-chunk file (Bar100, Internal PTS, IMU, Image_Meta)
    using an already-fit chunk clock. Returns the file with a `datetime`
    column appended (raw column order otherwise preserved)."""
    df = pd.read_csv(path)
    df["datetime"] = ticks_to_datetime(df[tick_col], slope, intercept)
    return df


def ingest_directory(data_dir: str, gap_seconds: float = POLL_GAP_SECONDS, verbose: bool = True):
    """Ingest every chunk in `data_dir` into one long poll-level DataFrame,
    plus the auxiliary channels and the calibration table.

    Returns a dict:
        rdo               - poll-level DataFrame, all chunks concatenated
        rdo_raw           - long (pre-pivot) RDO reads with datetime + quality,
                             one row per param-read; feeds qc.qc_failure_rate
        poll_exceptions   - polls that didn't see exactly 4 params
        calibration       - one row per chunk: fit + residual + poll counts
        external_pts      - Bar100 pressure/temperature, all chunks
        internal_pts      - internal PTS pressure/temperature, all chunks
    Missing auxiliary files per chunk are skipped, not fatal.
    """
    rdo_frames, rdo_raw_frames, exception_frames, calib_rows = [], [], [], []
    ext_pts_frames, int_pts_frames = [], []

    for chunk_id, chunk_files in discover_chunks(data_dir):
        try:
            wide, exceptions, calib_row = process_chunk(chunk_id, chunk_files, gap_seconds=gap_seconds)
        except FileNotFoundError as e:
            if verbose:
                print(f"[ingest] skipping chunk {chunk_id}: {e}")
            continue

        rdo_frames.append(wide)
        if len(exceptions):
            exception_frames.append(exceptions)
        calib_rows.append(calib_row)

        slope = calib_row["slope_sec_per_tick"]
        intercept = calib_row["intercept_sec"]
        raw = ingest_timeseries_file(chunk_files["RDO"], slope, intercept)
        rdo_raw_frames.append(raw.rename(columns={"modbus_id": "channel_id"}))
        if "External_PTS_Bar100" in chunk_files:
            ext_pts_frames.append(ingest_timeseries_file(chunk_files["External_PTS_Bar100"], slope, intercept))
        if "Internal_PTS" in chunk_files:
            int_pts_frames.append(ingest_timeseries_file(chunk_files["Internal_PTS"], slope, intercept))

    if not rdo_frames:
        raise FileNotFoundError(f"no ingestible chunks found in {data_dir}")

    calibration = pd.DataFrame(calib_rows).sort_values("chunk_id").reset_index(drop=True)
    if verbose:
        spread = calibration["ticks_per_sec"].max() / calibration["ticks_per_sec"].min() - 1
        print(f"[ingest] {len(calibration)} chunks, ticks/s spread {spread:.4%}, "
              f"resid RMS max {calibration['resid_rms_seconds'].max():.3f}s")

    return {
        "rdo": pd.concat(rdo_frames, ignore_index=True).sort_values(["channel_id", "datetime"]).reset_index(drop=True),
        "rdo_raw": pd.concat(rdo_raw_frames, ignore_index=True).sort_values(["channel_id", "datetime"]).reset_index(drop=True),
        "poll_exceptions": (pd.concat(exception_frames, ignore_index=True)
                             if exception_frames else pd.DataFrame(columns=["channel_id", "poll_id", "n_params"])),
        "calibration": calibration,
        "external_pts": (pd.concat(ext_pts_frames, ignore_index=True).sort_values("datetime").reset_index(drop=True)
                          if ext_pts_frames else pd.DataFrame()),
        "internal_pts": (pd.concat(int_pts_frames, ignore_index=True).sort_values("datetime").reset_index(drop=True)
                          if int_pts_frames else pd.DataFrame()),
    }


def resample_5min(
    poll_df: pd.DataFrame,
    channel_col: str = "channel_id",
    time_col: str = "datetime",
    value_cols=("po2", "temp_c", "do_mgl", "do_pctsat_reported"),
    freq: str = "5min",
):
    """Median-resample poll-level data onto a fixed grid, per channel,
    carrying n_obs so sparse bins can be dropped downstream (plan S3)."""
    value_cols = list(value_cols)
    out = []
    for channel_id, g in poll_df.groupby(channel_col):
        g = g.set_index(time_col).sort_index()
        agg = g[value_cols].resample(freq).median()
        agg["n_obs"] = g[value_cols[0]].resample(freq).count()
        agg[channel_col] = channel_id
        out.append(agg)
    return pd.concat(out).reset_index()


def persist_artifacts(out_dir: str, calibration, liveness, qc_rates, poll_exceptions, manifest=None):
    """Persist the reproducibility record named in plan S9: file manifest,
    per-chunk calibration table, liveness table, and per-sensor-per-day QC
    percentages."""
    os.makedirs(out_dir, exist_ok=True)
    calibration.to_csv(os.path.join(out_dir, "calibration.csv"), index=False)
    liveness.to_csv(os.path.join(out_dir, "liveness.csv"), index=False)
    qc_rates.to_csv(os.path.join(out_dir, "qc_failure_rate.csv"), index=False)
    poll_exceptions.to_csv(os.path.join(out_dir, "poll_exceptions.csv"), index=False)
    if manifest is not None:
        manifest.to_csv(os.path.join(out_dir, "fold_manifest.csv"), index=False)


def mark_long_gaps(df: pd.DataFrame, max_empty_bins: int = 2, channel_col: str = "channel_id"):
    """Flag resampled bins that follow a run of more than `max_empty_bins`
    empty bins within the same channel (plan S3: never forward-fill across
    a gap longer than one or two bins)."""
    df = df.copy()
    df["long_gap"] = False
    for _, idx in df.groupby(channel_col).groups.items():
        sub = df.loc[idx]
        empty = (sub["n_obs"] == 0)
        run = empty.groupby((~empty).cumsum()).cumsum()
        df.loc[idx, "long_gap"] = (run > max_empty_bins).to_numpy()
    return df
