"""
test_sampling_frequency.py
==========================
Determines the optimal resampling frequency for dissolved-oxygen time series
across one or more datasets (SMORES Biscayne Bay, FL Keys 3OEC, SB LTER, or
any future source that conforms to the central observations.parquet schema).

Candidates tested: 1min, 2min, 5min, 10min, 15min, 60min.

For each candidate frequency and each sensor/channel the script measures:

  1. Bin occupancy       — median reads per bin, empty-bin rate.
                          A good frequency has ≥3 reads/bin and 0 % empty bins.

  2. Spike suppression   — inter-quartile range of the residual after subtracting
                          a rolling median.  Lower = smoother (spikes suppressed).

  3. Signal retention    — Pearson correlation of the resampled series with the
                          finest-resolution reference (1-min).  Higher = less
                          information lost.

  4. Spectral fidelity   — power in the diurnal band (22–26 h) and tidal band
                          (11–14 h) as a fraction of total power. A good
                          frequency preserves both.

  5. Event resolution    — fraction of oxygen excursion events (consecutive runs
                          of ≥3 bins outside 1 std of the local mean) that survive
                          at the resampled resolution vs. the 1-min reference.

A composite score is derived from these five metrics and ranked.  All per-sensor
results are aggregated to a site-level summary.

Usage
-----
  # With the central observations.parquet (recommended):
  python test_sampling_frequency.py --parquet path/to/observations.parquet

  # With a raw SMORES TSV (fallback, uses do_pctsat_reported column):
  python test_sampling_frequency.py --tsv path/to/smores_qcd.tsv

  # Specify sites to test (default: all sites in the file):
  python test_sampling_frequency.py --parquet obs.parquet --sites biscayne_bay florida_keys

  # Save results table and figures to a directory:
  python test_sampling_frequency.py --parquet obs.parquet --outdir results/

Fits the existing repo conventions:
  - Reads either observations.parquet (central schema) or a raw TSV.
  - No dependency on repo-internal modules; can be run standalone.
  - Outputs a machine-readable CSV summary alongside human-readable figures,
    so results can be imported into the dataset_info pipeline.

Dependencies: pandas, numpy, scipy, matplotlib, statsmodels
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import acf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANDIDATE_FREQS: list[str] = ["1min", "2min", "5min", "10min", "15min", "60min"]

# Nyquist periods in hours implied by each candidate frequency
_FREQ_TO_HOURS: dict[str, float] = {
    "1min": 1 / 60,
    "2min": 2 / 60,
    "5min": 5 / 60,
    "10min": 10 / 60,
    "15min": 15 / 60,
    "60min": 1.0,
}

# Spectral bands of interest (hours)
DIURNAL_BAND_H: tuple[float, float] = (22.0, 26.0)   # ~24 h solar cycle
TIDAL_BAND_H: tuple[float, float]   = (11.0, 14.0)   # ~12.4 h M2 tidal

# Minimum reads per bin to call a bin "well-sampled"
MIN_READS_PER_BIN: int = 3

# Event detection: a bin is "anomalous" if it exceeds this many local stds
EVENT_SIGMA: float = 1.5
# Rolling window for local mean/std (in number of 1-min bins)
EVENT_LOCAL_WINDOW_1MIN: int = 60  # 1 hour


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_observations_parquet(path: Path, sites: list[str] | None) -> pd.DataFrame:
    """Load the central observations.parquet and return a tidy DataFrame."""
    df = pd.read_parquet(path)
    required = {"timestamp", "site_id", "sensor_id", "do_native_value", "qc_flag"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"observations.parquet is missing required columns: {missing}\n"
            "Expected schema: timestamp, site_id, sensor_id, do_native_value, "
            "qc_flag, do_native_unit, deployment_id"
        )
    # Keep only good rows (qc_flag 0 or 1)
    df = df[df["qc_flag"].isin([0, 1])].copy()
    if sites:
        df = df[df["site_id"].isin(sites)]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    return df


def load_raw_tsv(path: Path) -> pd.DataFrame:
    """
    Fallback loader for a raw SMORES QC'd TSV produced by the AcParse pipeline.
    Expects at minimum: a timestamp column and a DO column.
    """
    df = pd.read_csv(path, sep="\t", low_memory=False)
    # Normalise timestamp column name
    ts_candidates = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
    if not ts_candidates:
        raise ValueError("Cannot find a timestamp column in the TSV.")
    df = df.rename(columns={ts_candidates[0]: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Normalise DO column: prefer do_pctsat_reported, then do_native_value, then pO2
    do_candidates = ["do_pctsat_reported", "do_native_value", "po2", "pO2", "O2_sat", "do_sat"]
    do_col = next((c for c in do_candidates if c in df.columns), None)
    if do_col is None:
        raise ValueError(
            f"Cannot find a DO column. Expected one of {do_candidates}. "
            f"Found: {list(df.columns)}"
        )
    df = df.rename(columns={do_col: "do_native_value"})

    # Synthetic site/sensor ids for the fallback path
    if "sensor_id" not in df.columns:
        if "modbus_id" in df.columns:
            df["sensor_id"] = "ch" + df["modbus_id"].astype(str).str.zfill(2)
        else:
            df["sensor_id"] = "unknown"
    if "site_id" not in df.columns:
        df["site_id"] = "unknown_site"
    if "qc_flag" not in df.columns:
        df["qc_flag"] = 0

    df = df[df["qc_flag"].isin([0, 1])].copy()
    df = df.sort_values("timestamp")
    return df


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def resample_series(
    s: pd.Series,
    freq: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Resample a time-indexed Series to `freq` using median aggregation.

    Returns
    -------
    stats : DataFrame with columns [median, n_obs] indexed by bin timestamp.
    reference_1min : Series resampled to 1-min for consistent comparison.
    """
    s = s.dropna()
    if s.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    resampled = s.resample(freq).agg(["median", "count"]).rename(
        columns={"median": "median", "count": "n_obs"}
    )
    ref_1min = s.resample("1min").median()
    return resampled, ref_1min


def metric_bin_occupancy(
    resampled: pd.DataFrame,
) -> dict[str, float]:
    """
    Measure 1: how well does this frequency sample the raw signal?

    Returns
    -------
    dict with keys:
      median_reads_per_bin, empty_bin_pct, undersampled_bin_pct
    """
    n = resampled["n_obs"]
    empty = (n == 0).mean() * 100
    undersample = (n < MIN_READS_PER_BIN).mean() * 100
    return {
        "median_reads_per_bin": float(n[n > 0].median()),
        "empty_bin_pct": float(empty),
        "undersampled_bin_pct": float(undersample),
    }


def metric_spike_suppression(
    resampled: pd.DataFrame,
    raw: pd.Series,
) -> dict[str, float]:
    """
    Measure 2: how much optode spiking does this frequency suppress?

    Uses the IQR of the residual (signal minus 3-bin rolling median) as a
    measure of local noise.  Ratio vs. raw-1min gives suppression factor.
    """
    def _iqr_residual(s: pd.Series, window: int = 3) -> float:
        if len(s) < window + 1:
            return np.nan
        smooth = s.rolling(window, center=True, min_periods=1).median()
        resid = s - smooth
        return float(resid.quantile(0.75) - resid.quantile(0.25))

    raw_iqr = _iqr_residual(raw.resample("1min").median().dropna())
    res_iqr = _iqr_residual(resampled["median"].dropna())

    if raw_iqr == 0 or res_iqr == 0 or np.isnan(raw_iqr) or np.isnan(res_iqr):
        suppression_ratio = np.nan
    else:
        suppression_ratio = raw_iqr / res_iqr  # >1 means spikes suppressed

    return {
        "residual_iqr": float(res_iqr),
        "spike_suppression_ratio": float(suppression_ratio),
    }


def metric_signal_retention(
    resampled: pd.DataFrame,
    ref_1min: pd.Series,
) -> dict[str, float]:
    """
    Measure 3: how much information is retained relative to the 1-min reference?

    Computes Pearson r between the resampled series and the 1-min reference
    after aligning on the nearest index.
    """
    res = resampled["median"].dropna()
    ref = ref_1min.dropna()
    if len(res) < 10 or len(ref) < 10:
        return {"signal_retention_r": np.nan}

    # Align: resample reference to the coarser grid of `res`
    ref_aligned = ref.reindex(res.index, method="nearest", tolerance="5min")
    mask = ref_aligned.notna() & res.notna()
    if mask.sum() < 10:
        return {"signal_retention_r": np.nan}

    r, _ = pearsonr(res[mask].values, ref_aligned[mask].values)
    return {"signal_retention_r": float(r)}


def metric_spectral_fidelity(
    resampled: pd.DataFrame,
    freq: str,
) -> dict[str, float]:
    """
    Measure 4: does this frequency preserve the diurnal and tidal bands?

    Computes the Lomb-Scargle periodogram (handles irregular spacing / gaps)
    and returns the fraction of power in each band of interest.
    """
    s = resampled["median"].dropna()
    if len(s) < 48:  # need at least 2 diurnal cycles
        return {
            "diurnal_power_frac": np.nan,
            "tidal_power_frac": np.nan,
            "nyquist_period_h": np.nan,
        }

    freq_min = _FREQ_TO_HOURS[freq] * 60  # minutes per sample
    nyquist_h = 2 * freq_min / 60         # shortest resolvable period in hours

    # Convert index to hours since start (float)
    t_h = (s.index - s.index[0]).total_seconds() / 3600
    y = s.values - s.values.mean()

    # Angular frequencies to test (rad/hour), avoid zero
    max_f = 1 / (2 * freq_min / 60)  # Nyquist in cycles/hour
    freqs_cph = np.linspace(1 / t_h[-1], max_f, num=min(10_000, len(s) * 4))
    freqs_rad = 2 * np.pi * freqs_cph

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pgram = sp_signal.lombscargle(t_h.values, y, freqs_rad, normalize=True)

    period_h = 1 / freqs_cph  # convert to period in hours

    def _band_power(lo: float, hi: float) -> float:
        mask = (period_h >= lo) & (period_h <= hi)
        if mask.sum() == 0:
            return np.nan
        return float(pgram[mask].mean() / pgram.mean()) if pgram.mean() > 0 else np.nan

    return {
        "diurnal_power_frac": _band_power(*DIURNAL_BAND_H),
        "tidal_power_frac": _band_power(*TIDAL_BAND_H),
        "nyquist_period_h": float(nyquist_h),
    }


def metric_event_resolution(
    resampled: pd.DataFrame,
    ref_1min: pd.Series,
) -> dict[str, float]:
    """
    Measure 5: what fraction of oxygen excursion events survive resampling?

    An "event" is a run of ≥3 consecutive bins where the signal exceeds
    EVENT_SIGMA local standard deviations from the local rolling mean.
    Events are detected on the 1-min reference, then checked for survival
    in the resampled series.
    """
    ref = ref_1min.dropna()
    res = resampled["median"].dropna()

    if len(ref) < EVENT_LOCAL_WINDOW_1MIN * 2:
        return {"event_survival_pct": np.nan, "n_events_1min": 0}

    # Detect events on 1-min reference
    roll_mean = ref.rolling(EVENT_LOCAL_WINDOW_1MIN, center=True, min_periods=10).mean()
    roll_std  = ref.rolling(EVENT_LOCAL_WINDOW_1MIN, center=True, min_periods=10).std()
    is_event_ref = (ref - roll_mean).abs() > EVENT_SIGMA * roll_std

    # Label contiguous runs
    event_starts: list[pd.Timestamp] = []
    event_ends:   list[pd.Timestamp] = []
    in_event = False
    run_len = 0
    run_start: pd.Timestamp | None = None

    for ts, val in is_event_ref.items():
        if val:
            if not in_event:
                in_event = True
                run_start = ts
                run_len = 1
            else:
                run_len += 1
        else:
            if in_event and run_len >= 3:
                event_starts.append(run_start)          # type: ignore[arg-type]
                event_ends.append(ts)
            in_event = False
            run_len = 0

    n_events = len(event_starts)
    if n_events == 0:
        return {"event_survival_pct": 100.0, "n_events_1min": 0}

    # Check survival: at least one resampled bin inside the event window
    # must also be anomalous
    if len(res) < 10:
        return {"event_survival_pct": np.nan, "n_events_1min": n_events}

    res_roll_mean = res.rolling(max(3, EVENT_LOCAL_WINDOW_1MIN // 5), center=True, min_periods=3).mean()
    res_roll_std  = res.rolling(max(3, EVENT_LOCAL_WINDOW_1MIN // 5), center=True, min_periods=3).std()
    is_event_res  = (res - res_roll_mean).abs() > EVENT_SIGMA * res_roll_std

    survived = 0
    for start, end in zip(event_starts, event_ends):
        window = is_event_res.loc[start:end]
        if window.any():
            survived += 1

    return {
        "event_survival_pct": float(100 * survived / n_events),
        "n_events_1min": n_events,
    }


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

def compute_composite_score(row: pd.Series) -> float:
    """
    Weighted composite score in [0, 1].  Higher = better frequency.

    Weights reflect the priorities established in the preprocessing plan:
      - Bin occupancy (no empty bins)           : 0.20
      - Spike suppression                        : 0.20
      - Signal retention (vs 1-min reference)   : 0.25
      - Spectral fidelity (diurnal + tidal)     : 0.20
      - Event survival                           : 0.15
    """
    score = 0.0
    weight_total = 0.0

    # 1. Bin occupancy: 0 if ≥5% empty bins, 1 if 0% empty
    if not np.isnan(row.get("empty_bin_pct", np.nan)):
        s = max(0.0, 1.0 - row["empty_bin_pct"] / 5.0)
        score += 0.20 * s
        weight_total += 0.20

    # 2. Spike suppression: sigmoid-scaled suppression ratio; 1:1 = 0, 5:1 = 1
    sr = row.get("spike_suppression_ratio", np.nan)
    if not np.isnan(sr):
        s = min(1.0, max(0.0, (sr - 1.0) / 4.0))
        score += 0.20 * s
        weight_total += 0.20

    # 3. Signal retention: r=1 → 1, r<0.9 → 0 (linear)
    r = row.get("signal_retention_r", np.nan)
    if not np.isnan(r):
        s = max(0.0, (r - 0.90) / 0.10)
        score += 0.25 * s
        weight_total += 0.25

    # 4. Spectral fidelity: average of diurnal and tidal power fractions
    #    A value of 1 means the band has average power (perfect); higher is better up to 3
    dp = row.get("diurnal_power_frac", np.nan)
    tp = row.get("tidal_power_frac", np.nan)
    valid = [x for x in [dp, tp] if not np.isnan(x)]
    if valid:
        mean_frac = np.mean(valid)
        s = min(1.0, max(0.0, mean_frac / 3.0))
        score += 0.20 * s
        weight_total += 0.20

    # 5. Event survival
    es = row.get("event_survival_pct", np.nan)
    if not np.isnan(es):
        s = es / 100.0
        score += 0.15 * s
        weight_total += 0.15

    if weight_total == 0:
        return np.nan
    return score / weight_total


# ---------------------------------------------------------------------------
# Per-sensor analysis
# ---------------------------------------------------------------------------

def analyse_sensor(
    s: pd.Series,
    sensor_id: str,
    site_id: str,
) -> pd.DataFrame:
    """
    Run all five metrics for every candidate frequency on a single sensor's series.
    Returns a DataFrame with one row per candidate frequency.
    """
    records: list[dict] = []

    for freq in CANDIDATE_FREQS:
        resampled, ref_1min = resample_series(s, freq)

        if resampled.empty or resampled["median"].dropna().empty:
            records.append({"freq": freq, "sensor_id": sensor_id, "site_id": site_id})
            continue

        row: dict = {
            "freq": freq,
            "sensor_id": sensor_id,
            "site_id": site_id,
            "n_bins": int(resampled["n_obs"].gt(0).sum()),
            "duration_days": float(
                (resampled.index[-1] - resampled.index[0]).total_seconds() / 86400
            ),
        }
        row.update(metric_bin_occupancy(resampled))
        row.update(metric_spike_suppression(resampled, s))
        row.update(metric_signal_retention(resampled, ref_1min))
        row.update(metric_spectral_fidelity(resampled, freq))
        row.update(metric_event_resolution(resampled, ref_1min))
        records.append(row)

    df = pd.DataFrame(records)
    df["composite_score"] = df.apply(compute_composite_score, axis=1)
    return df


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_metric_heatmap(
    summary: pd.DataFrame,
    outdir: Path,
) -> None:
    """
    Heatmap: rows = sensors, columns = frequencies.
    Cell value = composite score.  Colour = score.
    """
    pivot = summary.pivot_table(
        index=["site_id", "sensor_id"],
        columns="freq",
        values="composite_score",
    )[CANDIDATE_FREQS]

    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.4 + 1)))
    im = ax.imshow(pivot.values, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(len(CANDIDATE_FREQS)))
    ax.set_xticklabels(CANDIDATE_FREQS, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels([f"{s}/{c}" for s, c in pivot.index], fontsize=8)
    for i in range(len(pivot)):
        for j in range(len(CANDIDATE_FREQS)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if v > 0.5 else "white")
    fig.colorbar(im, ax=ax, label="Composite score (higher = better)")
    ax.set_title("Sampling frequency composite score by sensor")
    fig.tight_layout()
    fig.savefig(outdir / "score_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {outdir / 'score_heatmap.png'}")


def plot_metric_bars(
    site_summary: pd.DataFrame,
    outdir: Path,
) -> None:
    """
    Bar chart of mean composite score per frequency, aggregated across sensors,
    with error bars showing ±1 std across sensors.
    """
    grouped = site_summary.groupby("freq")["composite_score"]
    means = grouped.mean().reindex(CANDIDATE_FREQS)
    stds  = grouped.std().reindex(CANDIDATE_FREQS)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(CANDIDATE_FREQS))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(CANDIDATE_FREQS)
    ax.set_ylabel("Mean composite score ± std across sensors")
    ax.set_title("Site-level: optimal sampling frequency")
    ax.set_ylim(0, 1.1)
    # Annotate best
    best_idx = int(means.argmax())
    ax.bar(best_idx, means.iloc[best_idx], yerr=stds.iloc[best_idx],
           capsize=5, color="forestgreen", alpha=0.9, label="best")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "score_bars.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {outdir / 'score_bars.png'}")


def plot_individual_metrics(
    summary: pd.DataFrame,
    outdir: Path,
) -> None:
    """
    One subplot per metric, each showing mean across sensors per frequency.
    """
    metrics = [
        ("empty_bin_pct",          "Empty bin % (lower = better)"),
        ("spike_suppression_ratio","Spike suppression ratio (higher = better)"),
        ("signal_retention_r",     "Signal retention r vs 1-min (higher = better)"),
        ("diurnal_power_frac",     "Diurnal band power fraction (higher = better)"),
        ("tidal_power_frac",       "Tidal band power fraction (higher = better)"),
        ("event_survival_pct",     "Event survival % (higher = better)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, (col, title) in zip(axes.flat, metrics):
        if col not in summary.columns:
            ax.set_visible(False)
            continue
        vals = summary.groupby("freq")[col].mean().reindex(CANDIDATE_FREQS)
        stds = summary.groupby("freq")[col].std().reindex(CANDIDATE_FREQS)
        ax.bar(range(len(CANDIDATE_FREQS)), vals, yerr=stds, capsize=4,
               color="steelblue", alpha=0.8)
        ax.set_xticks(range(len(CANDIDATE_FREQS)))
        ax.set_xticklabels(CANDIDATE_FREQS, rotation=45, ha="right", fontsize=8)
        ax.set_title(title, fontsize=9)
    fig.suptitle("Individual metrics by sampling frequency", fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / "individual_metrics.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {outdir / 'individual_metrics.png'}")


def plot_resampled_traces(
    df_site: pd.DataFrame,
    sensor_id: str,
    outdir: Path,
    n_days: float = 3.0,
) -> None:
    """
    Visual comparison: overlay the first n_days of resampled traces for each
    candidate frequency on a single sensor, so the smoothing effect is visible.
    """
    s = df_site[df_site["sensor_id"] == sensor_id].set_index("timestamp")["do_native_value"]
    s = s.dropna().sort_index()
    if s.empty:
        return

    t_end = s.index[0] + pd.Timedelta(days=n_days)
    s_window = s.loc[:t_end]

    fig, axes = plt.subplots(
        len(CANDIDATE_FREQS), 1,
        figsize=(12, 2.2 * len(CANDIDATE_FREQS)),
        sharex=True,
    )
    for ax, freq in zip(axes, CANDIDATE_FREQS):
        resampled, _ = resample_series(s_window, freq)
        ax.plot(s_window.index, s_window.values, color="lightgray",
                linewidth=0.5, label="raw", zorder=1)
        if not resampled.empty:
            ax.plot(resampled.index, resampled["median"].values,
                    color="steelblue", linewidth=1.2, label=freq, zorder=2)
        ax.set_ylabel(freq, fontsize=8)
        ax.legend(loc="upper right", fontsize=7)

    axes[0].set_title(
        f"Resampled DO traces — sensor {sensor_id} (first {n_days:.0f} days)",
        fontsize=10,
    )
    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    fname = outdir / f"traces_{sensor_id.replace('/', '_')}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def print_results_table(summary: pd.DataFrame) -> None:
    cols_display = [
        "site_id", "sensor_id", "freq", "n_bins",
        "empty_bin_pct", "spike_suppression_ratio",
        "signal_retention_r", "diurnal_power_frac",
        "tidal_power_frac", "event_survival_pct", "composite_score",
    ]
    cols_present = [c for c in cols_display if c in summary.columns]
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:.3f}".format)
    print(summary[cols_present].to_string(index=False))


def print_recommendation(summary: pd.DataFrame) -> None:
    """
    Print a per-site recommendation for the optimal sampling frequency.
    Also flag if 5min is not the top choice — that is a meaningful finding.
    """
    print("\n" + "=" * 60)
    print("RECOMMENDATION SUMMARY")
    print("=" * 60)

    for site in summary["site_id"].unique():
        site_df = summary[summary["site_id"] == site]
        mean_scores = (
            site_df.groupby("freq")["composite_score"]
            .mean()
            .reindex(CANDIDATE_FREQS)
        )
        best_freq = mean_scores.idxmax()
        best_score = mean_scores.max()

        print(f"\nSite: {site}")
        print(f"  Best frequency : {best_freq}  (score = {best_score:.3f})")

        for freq in CANDIDATE_FREQS:
            marker = " ← best" if freq == best_freq else (
                " ← current pipeline" if freq == "5min" else ""
            )
            score = mean_scores[freq]
            print(f"  {freq:6s}  {score:.3f}{marker}")

        if best_freq != "5min":
            score_5min = mean_scores["5min"]
            print(
                f"\n  NOTE: {best_freq} outperforms 5min "
                f"(Δ = {best_score - score_5min:+.3f}). "
                f"Consider updating the pipeline default for this site."
            )
        else:
            print(
                "\n  5-minute resampling confirmed as optimal for this site."
            )

    print("=" * 60)


def main(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Load data --------------------------------------------------------
    if args.parquet:
        print(f"Loading observations.parquet: {args.parquet}")
        df = load_observations_parquet(
            Path(args.parquet),
            sites=args.sites.split(",") if args.sites else None,
        )
        value_col = "do_native_value"
    elif args.tsv:
        print(f"Loading raw TSV: {args.tsv}")
        df = load_raw_tsv(Path(args.tsv))
        value_col = "do_native_value"
    else:
        print("ERROR: provide --parquet or --tsv", file=sys.stderr)
        sys.exit(1)

    if df.empty:
        print("No usable rows after QC filtering. Check qc_flag column.")
        sys.exit(1)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    print(
        f"Loaded {len(df):,} rows | "
        f"sites: {df['site_id'].unique().tolist()} | "
        f"sensors: {df['sensor_id'].nunique()}"
    )

    # ---- Run analysis per sensor ------------------------------------------
    all_results: list[pd.DataFrame] = []

    for site_id in df["site_id"].unique():
        site_df = df[df["site_id"] == site_id]
        sensors = site_df["sensor_id"].unique()
        print(f"\n[{site_id}] Analysing {len(sensors)} sensors ...")

        for sensor_id in sorted(sensors):
            s = site_df[site_df["sensor_id"] == sensor_id][value_col].dropna()
            if len(s) < 100:
                print(f"  Skipping {sensor_id}: fewer than 100 valid reads.")
                continue
            print(f"  {sensor_id}: {len(s):,} reads over "
                  f"{(s.index[-1] - s.index[0]).days:.0f} days")

            sensor_results = analyse_sensor(s, sensor_id, site_id)
            all_results.append(sensor_results)

            # Trace plot for this sensor (first 3 days)
            if args.plots:
                df_site_reset = site_df.reset_index()
                plot_resampled_traces(df_site_reset, sensor_id, outdir)

    if not all_results:
        print("No sensors had enough data to analyse.")
        sys.exit(1)

    summary = pd.concat(all_results, ignore_index=True)

    # ---- Print results ----------------------------------------------------
    print("\n--- Full results table ---")
    print_results_table(summary)
    print_recommendation(summary)

    # ---- Save outputs -----------------------------------------------------
    csv_path = outdir / "sampling_frequency_results.csv"
    summary.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nResults saved to: {csv_path}")

    # ---- Figures ----------------------------------------------------------
    if args.plots:
        print("\nGenerating figures ...")
        plot_metric_heatmap(summary, outdir)
        plot_metric_bars(summary, outdir)
        plot_individual_metrics(summary, outdir)
        print("Figures saved.")

    # ---- Validation gate: does 5min have 0% empty bins? ------------------
    print("\n--- Validation gate: preprocessing plan §3 ---")
    for site in summary["site_id"].unique():
        row_5min = summary[
            (summary["site_id"] == site) & (summary["freq"] == "5min")
        ]
        if row_5min.empty:
            continue
        empty_pct = row_5min["empty_bin_pct"].mean()
        status = "PASS" if empty_pct == 0 else "WARN"
        print(
            f"  [{status}] {site} @ 5min: "
            f"mean empty bin rate = {empty_pct:.1f}%"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--parquet",
        metavar="PATH",
        help="Path to observations.parquet (central schema, recommended).",
    )
    source.add_argument(
        "--tsv",
        metavar="PATH",
        help="Path to a raw SMORES QC'd TSV (fallback for pre-pipeline data).",
    )

    parser.add_argument(
        "--sites",
        metavar="SITE1,SITE2",
        default=None,
        help="Comma-separated list of site_ids to analyse. Default: all sites.",
    )
    parser.add_argument(
        "--outdir",
        metavar="DIR",
        default="sampling_freq_results",
        help="Directory to write CSV and figure outputs. Default: sampling_freq_results/",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        default=True,
        help="Generate and save figures (default: True).",
    )
    parser.add_argument(
        "--no-plots",
        dest="plots",
        action="store_false",
        help="Skip figure generation.",
    )

    args = parser.parse_args()
    main(args)
