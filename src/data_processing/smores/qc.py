"""
SMORES quality control: pO2 -> %sat reporting conversion, and liveness
classification (plan sections 4-5).
"""

import numpy as np
import pandas as pd

ATM_TO_MMHG = 760.0
O2_FRACTION_DRY_AIR = 0.20946


def vapor_pressure_mmhg(temp_c, salinity_psu: float):
    """Saturation water-vapor pressure (mmHg), Weiss & Price (1980) as used
    for optode %sat back-calculation. Requires a salinity assumption --
    the platform has no CTD, so callers must pass one explicitly and it
    should be recorded alongside any %sat column derived from it (plan S4)."""
    t_k = np.asarray(temp_c, dtype=np.float64) + 273.15
    pwv_atm = np.exp(
        24.4543
        - 67.4509 * (100.0 / t_k)
        - 4.8489 * np.log(t_k / 100.0)
        - 0.000544 * salinity_psu
    )
    return pwv_atm * ATM_TO_MMHG


def pct_sat_from_po2(po2, temp_c, salinity_psu: float = 35.0):
    """%sat = 100 * pO2 / (0.20946 * (760 - Pwv(T, S))). Reporting-only
    conversion (plan S4) -- pO2 stays the modeling target; do not clip
    negative pO2, it's real below-detection signal on dead channels."""
    pwv = vapor_pressure_mmhg(temp_c, salinity_psu)
    return 100.0 * np.asarray(po2, dtype=np.float64) / (O2_FRACTION_DRY_AIR * (ATM_TO_MMHG - pwv))


def add_pct_sat_column(df: pd.DataFrame, po2_col="po2", temp_col="temp_c", salinity_psu: float = 35.0):
    df = df.copy()
    df["pct_sat"] = pct_sat_from_po2(df[po2_col], df[temp_col], salinity_psu)
    df["salinity_psu_assumed"] = salinity_psu
    return df


# --- Liveness classification (plan S5) --------------------------------

LIVE_STD_THRESHOLD = 0.5       # min rolling std (po2 units) over `window` to call a day "live"
ZERO_TOL = 1e-9                # exact-zero tolerance for do_mgl / do_pctsat_reported
FLATLINE_ZERO_FRACTION = 0.5   # share of a channel-day at exact zero to call it flatlined-zero


def classify_liveness(
    df: pd.DataFrame,
    channel_col: str = "channel_id",
    time_col: str = "datetime",
    po2_col: str = "po2",
    do_mgl_col: str = "do_mgl",
    do_pct_col: str = "do_pctsat_reported",
    window: str = "4h",
    std_threshold: float = LIVE_STD_THRESHOLD,
    day_freq: str = "1D",
):
    """Classify each (channel, day) as live / flatlined-zero / stuck-constant.

    stuck-constant is detected by rolling standard deviation over a
    multi-hour window, never by raw value -- a channel frozen at a nonzero
    reading (e.g. ch10 at 1.1%, ch28 at 0.4%) passes both a nonzero test and
    a range check and must be caught here (plan S5).
    """
    work = df[[channel_col, time_col, po2_col, do_mgl_col, do_pct_col]].copy()
    work["_day"] = work[time_col].dt.floor(day_freq)

    rows = []
    for (channel_id, day), g in work.groupby([channel_col, "_day"]):
        g = g.sort_values(time_col)
        rolling_std = g.set_index(time_col)[po2_col].rolling(window, min_periods=2).std()
        max_std = rolling_std.max()

        near_zero = (g[do_mgl_col].abs() < ZERO_TOL) & (g[do_pct_col].abs() < ZERO_TOL)
        zero_fraction = near_zero.mean()

        if zero_fraction > FLATLINE_ZERO_FRACTION:
            state = "flatlined-zero"
        elif pd.isna(max_std) or max_std < std_threshold:
            state = "stuck-constant"
        else:
            state = "live"

        rows.append({
            "channel_id": channel_id,
            "day": day,
            "state": state,
            "max_rolling_std": max_std,
            "zero_fraction": zero_fraction,
            "n_obs": len(g),
        })

    return pd.DataFrame(rows).sort_values(["channel_id", "day"]).reset_index(drop=True)


FAILING_QUALITY_CODES = {3, 5, 7}


def qc_failure_rate(rdo_long_df: pd.DataFrame, channel_col: str = "channel_id", time_col: str = "datetime", day_freq: str = "1D"):
    """Per-sensor-per-day fraction of RDO reads with a failing quality code
    (3, 5, or 7). Takes the raw long-format RDO frame (one row per
    param-read, with its `quality` column), not the pivoted poll table.
    Used to persist the QC record (plan S9) and to check the deployment-wide
    failure rate against the report's 3.73% (plan S10 gate 5)."""
    work = rdo_long_df[[channel_col, time_col, "quality"]].copy()
    work["_day"] = work[time_col].dt.floor(day_freq)
    work["_failing"] = work["quality"].isin(FAILING_QUALITY_CODES)
    return (
        work.groupby([channel_col, "_day"])["_failing"]
        .mean()
        .rename("qc_failure_rate")
        .reset_index()
        .rename(columns={"_day": "day", channel_col: "channel_id"})
    )


def channel_majority_state(liveness_df: pd.DataFrame):
    """Collapse per-day liveness to one dominant state per channel, weighted
    by day count -- used to exclude channels that are stuck-constant for
    most of their record (plan S8)."""
    counts = liveness_df.groupby(["channel_id", "state"]).size().rename("n_days").reset_index()
    idx = counts.groupby("channel_id")["n_days"].idxmax()
    return counts.loc[idx].set_index("channel_id")["state"]
