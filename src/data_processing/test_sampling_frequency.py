"""
test_sampling_frequency.py
==========================
Determines the optimal resampling frequency for dissolved-oxygen data, run
directly against each source's raw files -- no pre-built central dataset
required.

Two raw sources are supported:
  --source smores        Raw SENS_<serial>_<chunk>_<Kind>.csv chunk files
                          (the same input `data_processing.smores.ingest
                          .ingest_directory` consumes). Site groups are
                          "biscayne_bay::<depth_class>" -- live channels
                          (per `qc.classify_liveness`) sharing a depth class
                          are combined.
  --source florida-keys   Raw Florida_Keys_data.csv (the same input
                          `data_processing.three_oec.process_3oec
                          .make_timeseries` consumes). Site groups are
                          "florida_keys::FL<n>", one per deployment window
                          (FL1-4); each deployment's three redundant O2
                          sensors (O2_S1/S2/S3) are pre-averaged into one
                          series (O2_avg, at native resolution) before any
                          resampling.

Candidates tested: 1min, 2min, 5min, 10min, 15min, 1hour.

Unlike a per-sensor comparison, every metric here is computed per SITE GROUP
-- not per individual channel/sensor. Raw per-channel series are each
resampled to a candidate frequency and then combined across the
channels/sensors sharing a group via a per-bin median, before any metric is
computed. This matches the "per-site, not per-sensor" convention the central
dataset is moving to: a channel dying, or a new one being added, shouldn't
change which frequency looks best for a site.

For each candidate frequency and each site group the script measures:

  1. Bin occupancy       -- total reads per bin (summed across channels in the
                          group), empty-bin rate. A good frequency has
                          >=3 reads/bin and 0% empty bins.

  2. Spike suppression   -- inter-quartile range of the residual after subtracting
                          a rolling median. Lower = smoother (spikes suppressed).

  3. Signal retention    -- Pearson correlation of the resampled group series with
                          the finest-resolution reference (1-min, same combine
                          logic). Higher = less information lost.

  4. Spectral fidelity   -- power in the diurnal band (22-26 h) and tidal band
                          (11-14 h) as a fraction of total power (Lomb-Scargle,
                          handles irregular spacing/gaps). A good frequency
                          preserves both.

  5. Event resolution    -- fraction of oxygen excursion events (consecutive runs
                          of >=3 bins outside 1.5 std of the local mean) that
                          survive at the resampled resolution vs. the 1-min
                          reference.

A composite score is derived from these five metrics and ranked per site group.

Usage
-----
  python test_sampling_frequency.py --source smores --data-path path/to/smores/raw/chunk/dir

  # Exclude the pre-deployment bucket-test window:
  python test_sampling_frequency.py --source smores --data-path raw/ --deployment-start "2026-05-13 21:30:00"

  python test_sampling_frequency.py --source florida-keys --data-path path/to/Florida_Keys_data.csv

  # Save results table and figures to a directory:
  python test_sampling_frequency.py --source smores --data-path raw/ --outdir results/

Dependencies: pandas, numpy, scipy, matplotlib
"""

from __future__ import annotations

import argparse
import os
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_processing.smores import ingest, qc, folds  # noqa: E402
from data_processing.three_oec import process_3oec  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# pandas-valid resample frequency strings
CANDIDATE_FREQS: list[str] = ["1min", "2min", "5min", "10min", "15min", "60min"]
# display labels (the repo/plan spells the last one "1hour", not "60min")
FREQ_LABELS: dict[str, str] = {
    "1min": "1min", "2min": "2min", "5min": "5min",
    "10min": "10min", "15min": "15min", "60min": "1hour",
}

# Nyquist periods in hours implied by each candidate frequency
_FREQ_TO_HOURS: dict[str, float] = {
    "1min": 1 / 60, "2min": 2 / 60, "5min": 5 / 60,
    "10min": 10 / 60, "15min": 15 / 60, "60min": 1.0,
}

# Spectral bands of interest (hours)
DIURNAL_BAND_H: tuple[float, float] = (22.0, 26.0)   # ~24 h solar cycle
TIDAL_BAND_H: tuple[float, float] = (11.0, 14.0)     # ~12.4 h M2 tidal

# Minimum reads per bin to call a bin "well-sampled"
MIN_READS_PER_BIN: int = 3

# Minimum raw reads for a channel to be included at all
MIN_CHANNEL_READS: int = 100

# Event detection: a bin is "anomalous" if it exceeds this many local stds
EVENT_SIGMA: float = 1.5
# Rolling window for local mean/std (in number of 1-min bins)
EVENT_LOCAL_WINDOW_1MIN: int = 60  # 1 hour

# Site ids for each supported raw source.
SMORES_SITE_ID = "biscayne_bay"
FL_SITE_ID = "florida_keys"
FL_DEPLOYMENTS = [
    ("FL1", process_3oec.get_first_piece),
    ("FL2", process_3oec.get_second_piece),
    ("FL3", process_3oec.get_third_piece),
    ("FL4", process_3oec.get_fourth_piece),
]


# ---------------------------------------------------------------------------
# Raw data loading + per-site-group combination
# ---------------------------------------------------------------------------

def load_live_channel_series(
    data_dir: Path,
    deployment_start: pd.Timestamp | None,
) -> dict[int, pd.Series]:
    """
    Ingest the raw SMORES chunk directory and return one po2 Series per LIVE
    channel (dead/flatlined/stuck channels excluded per `qc.classify_liveness`,
    same gate `folds.build_bb_folds` uses), indexed by datetime and restricted
    to `deployment_start` onward if given (excludes the pre-deployment
    bucket-test window).
    """
    print(f"Ingesting raw SMORES data from {data_dir} ...")
    result = ingest.ingest_directory(str(data_dir), verbose=False)
    rdo = result["rdo"]

    liveness = qc.classify_liveness(rdo)
    majority = qc.channel_majority_state(liveness)
    live_channels = sorted(c for c, state in majority.items() if state == "live")
    print(f"Live channels: {live_channels}")

    channel_series: dict[int, pd.Series] = {}
    for channel_id in live_channels:
        s = (
            rdo[rdo["channel_id"] == channel_id]
            .set_index("datetime")["po2"]
            .dropna()
            .sort_index()
        )
        if deployment_start is not None:
            s = s.loc[deployment_start:]
        if len(s) < MIN_CHANNEL_READS:
            print(f"  Skipping channel {channel_id}: fewer than {MIN_CHANNEL_READS} valid reads.")
            continue
        channel_series[channel_id] = s

    return channel_series


def group_channels_by_site(channel_series: dict[int, pd.Series]) -> dict[str, list[int]]:
    """
    Map each live SMORES channel to a site group `"<site_id>::<depth_class>"`.
    This is the "per site, not per sensor" grouping the central dataset uses
    -- within a group, individual channels are combined before any metric is
    computed. Channels with no resolvable depth class (plan S11: ambiguous
    addresses 20/21/22/27) are excluded, matching `folds.py`.
    """
    groups: dict[str, list[int]] = {}
    excluded = []
    for channel_id in channel_series:
        depth_class = folds.get_depth_class(channel_id)
        if depth_class is None:
            excluded.append(channel_id)
            continue
        group_id = f"{SMORES_SITE_ID}::{depth_class}"
        groups.setdefault(group_id, []).append(channel_id)
    if excluded:
        print(f"Excluded channels with unresolved depth class: {excluded}")
    return groups


def load_fl_channel_series(data_path: Path) -> tuple[dict[str, pd.Series], dict[str, list[str]]]:
    """
    Ingest the raw Florida Keys 3OEC CSV and return one series per deployment
    window, grouped per site group `"florida_keys::FL<n>"`. The three
    redundant O2 sensors (O2_S1/S2/S3) are averaged together at native (raw,
    8Hz) resolution -- `O2_avg`, the same column `process_3oec.make_timeseries`
    already computes -- rather than resampled individually and combined per
    candidate frequency: which sensor happens to be reporting shouldn't
    change which frequency looks best for a deployment, and any one candidate
    frequency should see the same averaged signal.

    Deployment windows are non-contiguous (each is its own ~18-20h dive), so
    they are kept as separate groups rather than concatenated -- merging them
    would introduce multi-day gaps that don't reflect the sensor's actual
    sampling behavior.
    """
    print(f"Loading raw Florida Keys data from {data_path} ...")
    df = process_3oec.make_timeseries(str(data_path))

    channel_series: dict[str, pd.Series] = {}
    groups: dict[str, list[str]] = {}
    for label, get_piece in FL_DEPLOYMENTS:
        piece = get_piece(df)
        s = piece["O2_avg"].dropna().sort_index()
        if len(s) < MIN_CHANNEL_READS:
            print(f"  Skipping {label}: fewer than {MIN_CHANNEL_READS} valid reads.")
            continue
        key = f"O2_avg::{label}"
        channel_series[key] = s
        groups[f"{FL_SITE_ID}::{label}"] = [key]

    return channel_series, groups


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def resample_channel(s: pd.Series, freq: str) -> pd.DataFrame:
    """Resample one channel's series to `freq` using median aggregation."""
    s = s.dropna()
    if s.empty:
        return pd.DataFrame(columns=["median", "n_obs"])
    resampled = s.resample(freq).agg(["median", "count"]).rename(columns={"count": "n_obs"})
    return resampled


def combine_group(
    channel_series: dict[int, pd.Series],
    channel_ids: list[int],
    freq: str,
) -> pd.DataFrame:
    """
    Resample every channel in a site group to `freq` independently, then
    combine into a single site-group series via a per-bin median across
    channels. `n_obs` is the total read count across all channels in the
    group for that bin, so bin-occupancy reflects the group's actual
    poll density rather than any one channel's.
    """
    per_channel = {
        c: resample_channel(channel_series[c], freq)
        for c in channel_ids
        if not channel_series[c].empty
    }
    per_channel = {c: df for c, df in per_channel.items() if not df["median"].dropna().empty}
    if not per_channel:
        return pd.DataFrame(columns=["median", "n_obs"])

    medians = pd.concat({c: df["median"] for c, df in per_channel.items()}, axis=1)
    counts = pd.concat({c: df["n_obs"] for c, df in per_channel.items()}, axis=1)

    combined = pd.DataFrame({
        "median": medians.median(axis=1, skipna=True),
        "n_obs": counts.sum(axis=1, skipna=True),
    })
    return combined


def metric_bin_occupancy(resampled: pd.DataFrame) -> dict[str, float]:
    """Measure 1: how well does this frequency sample the group's channels?"""
    n = resampled["n_obs"]
    empty = (n == 0).mean() * 100
    undersample = (n < MIN_READS_PER_BIN).mean() * 100
    return {
        "median_reads_per_bin": float(n[n > 0].median()) if (n > 0).any() else np.nan,
        "empty_bin_pct": float(empty),
        "undersampled_bin_pct": float(undersample),
    }


def metric_spike_suppression(resampled: pd.DataFrame, ref_1min: pd.Series) -> dict[str, float]:
    """
    Measure 2: how much optode spiking does this frequency suppress, relative
    to the group's finest-resolution (1-min combined) reference?
    """
    def _iqr_residual(s: pd.Series, window: int = 3) -> float:
        if len(s) < window + 1:
            return np.nan
        smooth = s.rolling(window, center=True, min_periods=1).median()
        resid = s - smooth
        return float(resid.quantile(0.75) - resid.quantile(0.25))

    raw_iqr = _iqr_residual(ref_1min.dropna())
    res_iqr = _iqr_residual(resampled["median"].dropna())

    if raw_iqr == 0 or res_iqr == 0 or np.isnan(raw_iqr) or np.isnan(res_iqr):
        suppression_ratio = np.nan
    else:
        suppression_ratio = raw_iqr / res_iqr  # >1 means spikes suppressed

    return {
        "residual_iqr": float(res_iqr),
        "spike_suppression_ratio": float(suppression_ratio),
    }


def metric_signal_retention(resampled: pd.DataFrame, ref_1min: pd.Series) -> dict[str, float]:
    """Measure 3: Pearson r between the resampled group series and its 1-min reference."""
    res = resampled["median"].dropna()
    ref = ref_1min.dropna()
    if len(res) < 10 or len(ref) < 10:
        return {"signal_retention_r": np.nan}

    ref_aligned = ref.reindex(res.index, method="nearest", tolerance="5min")
    mask = ref_aligned.notna() & res.notna()
    if mask.sum() < 10:
        return {"signal_retention_r": np.nan}

    r, _ = pearsonr(res[mask].values, ref_aligned[mask].values)
    return {"signal_retention_r": float(r)}


def metric_spectral_fidelity(resampled: pd.DataFrame, freq: str) -> dict[str, float]:
    """Measure 4: fraction of spectral power in the diurnal and tidal bands (Lomb-Scargle)."""
    s = resampled["median"].dropna()
    if len(s) < 48:  # need at least 2 diurnal cycles
        return {"diurnal_power_frac": np.nan, "tidal_power_frac": np.nan, "nyquist_period_h": np.nan}

    freq_min = _FREQ_TO_HOURS[freq] * 60  # minutes per sample
    nyquist_h = 2 * freq_min / 60

    t_h = (s.index - s.index[0]).total_seconds() / 3600
    y = s.values - s.values.mean()

    max_f = 1 / (2 * freq_min / 60)  # Nyquist in cycles/hour
    freqs_cph = np.linspace(1 / t_h[-1], max_f, num=min(10_000, len(s) * 4))
    freqs_rad = 2 * np.pi * freqs_cph

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pgram = sp_signal.lombscargle(t_h.values, y, freqs_rad, normalize=True)

    period_h = 1 / freqs_cph

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


def metric_event_resolution(resampled: pd.DataFrame, ref_1min: pd.Series) -> dict[str, float]:
    """Measure 5: fraction of oxygen excursion events (on the 1-min reference) that survive resampling."""
    ref = ref_1min.dropna()
    res = resampled["median"].dropna()

    if len(ref) < EVENT_LOCAL_WINDOW_1MIN * 2:
        return {"event_survival_pct": np.nan, "n_events_1min": 0}

    roll_mean = ref.rolling(EVENT_LOCAL_WINDOW_1MIN, center=True, min_periods=10).mean()
    roll_std = ref.rolling(EVENT_LOCAL_WINDOW_1MIN, center=True, min_periods=10).std()
    is_event_ref = (ref - roll_mean).abs() > EVENT_SIGMA * roll_std

    event_starts: list[pd.Timestamp] = []
    event_ends: list[pd.Timestamp] = []
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
                event_starts.append(run_start)  # type: ignore[arg-type]
                event_ends.append(ts)
            in_event = False
            run_len = 0

    n_events = len(event_starts)
    if n_events == 0:
        return {"event_survival_pct": 100.0, "n_events_1min": 0}

    if len(res) < 10:
        return {"event_survival_pct": np.nan, "n_events_1min": n_events}

    res_roll_mean = res.rolling(max(3, EVENT_LOCAL_WINDOW_1MIN // 5), center=True, min_periods=3).mean()
    res_roll_std = res.rolling(max(3, EVENT_LOCAL_WINDOW_1MIN // 5), center=True, min_periods=3).std()
    is_event_res = (res - res_roll_mean).abs() > EVENT_SIGMA * res_roll_std

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
    Weighted composite score in [0, 1]. Higher = better frequency.

    Weights reflect the priorities established in the preprocessing plan:
      - Bin occupancy (no empty bins)          : 0.20
      - Spike suppression                       : 0.20
      - Signal retention (vs 1-min reference)  : 0.25
      - Spectral fidelity (diurnal + tidal)    : 0.20
      - Event survival                          : 0.15
    """
    score = 0.0
    weight_total = 0.0

    if not np.isnan(row.get("empty_bin_pct", np.nan)):
        s = max(0.0, 1.0 - row["empty_bin_pct"] / 5.0)
        score += 0.20 * s
        weight_total += 0.20

    sr = row.get("spike_suppression_ratio", np.nan)
    if not np.isnan(sr):
        s = min(1.0, max(0.0, (sr - 1.0) / 4.0))
        score += 0.20 * s
        weight_total += 0.20

    r = row.get("signal_retention_r", np.nan)
    if not np.isnan(r):
        s = max(0.0, (r - 0.90) / 0.10)
        score += 0.25 * s
        weight_total += 0.25

    dp = row.get("diurnal_power_frac", np.nan)
    tp = row.get("tidal_power_frac", np.nan)
    valid = [x for x in [dp, tp] if not np.isnan(x)]
    if valid:
        mean_frac = np.mean(valid)
        s = min(1.0, max(0.0, mean_frac / 3.0))
        score += 0.20 * s
        weight_total += 0.20

    es = row.get("event_survival_pct", np.nan)
    if not np.isnan(es):
        s = es / 100.0
        score += 0.15 * s
        weight_total += 0.15

    if weight_total == 0:
        return np.nan
    return score / weight_total


# ---------------------------------------------------------------------------
# Per-site-group analysis
# ---------------------------------------------------------------------------

def analyse_group(
    channel_series: dict[int, pd.Series],
    channel_ids: list[int],
    group_id: str,
) -> pd.DataFrame:
    """Run all five metrics for every candidate frequency on one site group."""
    ref_1min = combine_group(channel_series, channel_ids, "1min")["median"]

    records: list[dict] = []
    for freq in CANDIDATE_FREQS:
        resampled = combine_group(channel_series, channel_ids, freq)

        if resampled.empty or resampled["median"].dropna().empty:
            records.append({"freq": freq, "freq_label": FREQ_LABELS[freq], "group_id": group_id})
            continue

        row: dict = {
            "freq": freq,
            "freq_label": FREQ_LABELS[freq],
            "group_id": group_id,
            "n_channels": len(channel_ids),
            "n_bins": int(resampled["n_obs"].gt(0).sum()),
            "duration_days": float((resampled.index[-1] - resampled.index[0]).total_seconds() / 86400),
        }
        row.update(metric_bin_occupancy(resampled))
        row.update(metric_spike_suppression(resampled, ref_1min))
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

def plot_metric_heatmap(summary: pd.DataFrame, outdir: Path) -> None:
    """Heatmap: rows = site groups, columns = frequencies. Cell = composite score."""
    pivot = summary.pivot_table(index="group_id", columns="freq", values="composite_score")[CANDIDATE_FREQS]

    fig, ax = plt.subplots(figsize=(10, max(3, len(pivot) * 0.6 + 1)))
    im = ax.imshow(pivot.values, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(len(CANDIDATE_FREQS)))
    ax.set_xticklabels([FREQ_LABELS[f] for f in CANDIDATE_FREQS], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(list(pivot.index), fontsize=9)
    for i in range(len(pivot)):
        for j in range(len(CANDIDATE_FREQS)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if v > 0.5 else "white")
    fig.colorbar(im, ax=ax, label="Composite score (higher = better)")
    ax.set_title("Sampling frequency composite score by site group")
    fig.tight_layout()
    fig.savefig(outdir / "score_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {outdir / 'score_heatmap.png'}")


def plot_metric_bars(summary: pd.DataFrame, outdir: Path) -> None:
    """Bar chart of mean composite score per frequency, aggregated across site groups."""
    grouped = summary.groupby("freq")["composite_score"]
    means = grouped.mean().reindex(CANDIDATE_FREQS)
    stds = grouped.std().reindex(CANDIDATE_FREQS)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(CANDIDATE_FREQS))
    ax.bar(x, means, yerr=stds, capsize=5, color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([FREQ_LABELS[f] for f in CANDIDATE_FREQS])
    ax.set_ylabel("Mean composite score +/- std across site groups")
    ax.set_title("Optimal sampling frequency across site groups")
    ax.set_ylim(0, 1.1)
    best_idx = int(means.argmax())
    ax.bar(best_idx, means.iloc[best_idx], yerr=stds.iloc[best_idx],
           capsize=5, color="forestgreen", alpha=0.9, label="best")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "score_bars.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {outdir / 'score_bars.png'}")


def plot_individual_metrics(summary: pd.DataFrame, outdir: Path) -> None:
    """One subplot per metric, mean across site groups per frequency."""
    metrics = [
        ("empty_bin_pct", "Empty bin % (lower = better)"),
        ("spike_suppression_ratio", "Spike suppression ratio (higher = better)"),
        ("signal_retention_r", "Signal retention r vs 1-min (higher = better)"),
        ("diurnal_power_frac", "Diurnal band power fraction (higher = better)"),
        ("tidal_power_frac", "Tidal band power fraction (higher = better)"),
        ("event_survival_pct", "Event survival % (higher = better)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, (col, title) in zip(axes.flat, metrics):
        if col not in summary.columns:
            ax.set_visible(False)
            continue
        vals = summary.groupby("freq")[col].mean().reindex(CANDIDATE_FREQS)
        stds = summary.groupby("freq")[col].std().reindex(CANDIDATE_FREQS)
        ax.bar(range(len(CANDIDATE_FREQS)), vals, yerr=stds, capsize=4, color="steelblue", alpha=0.8)
        ax.set_xticks(range(len(CANDIDATE_FREQS)))
        ax.set_xticklabels([FREQ_LABELS[f] for f in CANDIDATE_FREQS], rotation=45, ha="right", fontsize=8)
        ax.set_title(title, fontsize=9)
    fig.suptitle("Individual metrics by sampling frequency (site groups)", fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / "individual_metrics.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {outdir / 'individual_metrics.png'}")


def plot_resampled_traces(
    channel_series: dict[int, pd.Series],
    channel_ids: list[int],
    group_id: str,
    outdir: Path,
    n_days: float = 3.0,
) -> None:
    """Overlay the first n_days of resampled traces for each candidate frequency on one site group."""
    ref = combine_group(channel_series, channel_ids, "1min")["median"].dropna()
    if ref.empty:
        return
    t_end = ref.index[0] + pd.Timedelta(days=n_days)
    ref_window = ref.loc[:t_end]

    fig, axes = plt.subplots(len(CANDIDATE_FREQS), 1, figsize=(12, 2.2 * len(CANDIDATE_FREQS)), sharex=True)
    for ax, freq in zip(axes, CANDIDATE_FREQS):
        resampled = combine_group(channel_series, channel_ids, freq)
        resampled_window = resampled.loc[:t_end] if not resampled.empty else resampled
        ax.plot(ref_window.index, ref_window.values, color="lightgray", linewidth=0.5, label="1min", zorder=1)
        if not resampled_window.empty:
            ax.plot(resampled_window.index, resampled_window["median"].values,
                    color="steelblue", linewidth=1.2, label=FREQ_LABELS[freq], zorder=2)
        ax.set_ylabel(FREQ_LABELS[freq], fontsize=8)
        ax.legend(loc="upper right", fontsize=7)

    axes[0].set_title(f"Resampled po2 traces -- {group_id} (first {n_days:.0f} days)", fontsize=10)
    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    fname = outdir / f"traces_{group_id.replace('::', '_')}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results_table(summary: pd.DataFrame) -> None:
    cols_display = [
        "group_id", "freq_label", "n_channels", "n_bins",
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
    """Per-site-group recommendation for the optimal sampling frequency."""
    print("\n" + "=" * 60)
    print("RECOMMENDATION SUMMARY (per site group)")
    print("=" * 60)

    for group_id in summary["group_id"].unique():
        group_df = summary[summary["group_id"] == group_id]
        mean_scores = group_df.set_index("freq")["composite_score"].reindex(CANDIDATE_FREQS)
        best_freq = mean_scores.idxmax()
        best_score = mean_scores.max()

        print(f"\nSite group: {group_id}")
        print(f"  Best frequency : {FREQ_LABELS[best_freq]}  (score = {best_score:.3f})")

        for freq in CANDIDATE_FREQS:
            marker = " <- best" if freq == best_freq else (" <- current pipeline" if freq == "5min" else "")
            score = mean_scores[freq]
            print(f"  {FREQ_LABELS[freq]:6s}  {score:.3f}{marker}")

        if best_freq != "5min":
            score_5min = mean_scores["5min"]
            print(
                f"\n  NOTE: {FREQ_LABELS[best_freq]} outperforms 5min "
                f"(delta = {best_score - score_5min:+.3f}). "
                f"Consider updating the pipeline default for this site group."
            )
        else:
            print("\n  5-minute resampling confirmed as optimal for this site group.")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.source == "smores":
        deployment_start = pd.Timestamp(args.deployment_start) if args.deployment_start else None
        channel_series = load_live_channel_series(Path(args.data_path), deployment_start)
        if not channel_series:
            print("No channels had enough valid reads. Nothing to analyse.")
            sys.exit(1)
        groups = group_channels_by_site(channel_series)
        if not groups:
            print("No channels resolved to a site group (depth class). Nothing to analyse.")
            sys.exit(1)
    else:  # florida-keys
        channel_series, groups = load_fl_channel_series(Path(args.data_path))
        if not groups:
            print("No deployment windows had enough valid reads. Nothing to analyse.")
            sys.exit(1)

    print(f"\nSite groups: {list(groups.keys())}")

    all_results: list[pd.DataFrame] = []
    for group_id, channel_ids in groups.items():
        print(f"\n[{group_id}] Analysing {len(channel_ids)} channel(s): {channel_ids} ...")
        group_results = analyse_group(channel_series, channel_ids, group_id)
        all_results.append(group_results)

        if args.plots:
            plot_resampled_traces(channel_series, channel_ids, group_id, outdir)

    summary = pd.concat(all_results, ignore_index=True)

    print("\n--- Full results table ---")
    print_results_table(summary)
    print_recommendation(summary)

    csv_path = outdir / "sampling_frequency_results.csv"
    summary.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nResults saved to: {csv_path}")

    if args.plots:
        print("\nGenerating figures ...")
        plot_metric_heatmap(summary, outdir)
        plot_metric_bars(summary, outdir)
        plot_individual_metrics(summary, outdir)
        print("Figures saved.")

    print("\n--- Validation gate: preprocessing plan S3 ---")
    for group_id in summary["group_id"].unique():
        row_5min = summary[(summary["group_id"] == group_id) & (summary["freq"] == "5min")]
        if row_5min.empty:
            continue
        empty_pct = row_5min["empty_bin_pct"].iloc[0]
        status = "PASS" if empty_pct == 0 else "WARN"
        print(f"  [{status}] {group_id} @ 5min: empty bin rate = {empty_pct:.1f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=["smores", "florida-keys"],
        required=True,
        help="Which raw dataset to analyse.",
    )
    parser.add_argument(
        "--data-path",
        metavar="PATH",
        required=True,
        help="For --source smores: directory of raw SENS_<serial>_<chunk>_<Kind>.csv chunk files. "
             "For --source florida-keys: path to Florida_Keys_data.csv.",
    )
    parser.add_argument(
        "--deployment-start",
        metavar="TIMESTAMP",
        default=None,
        help="smores only. Exclude data before this timestamp (e.g. the pre-deployment bucket-test window). "
             "Format: 'YYYY-MM-DD HH:MM:SS'. Default: use all available data.",
    )
    parser.add_argument(
        "--outdir",
        metavar="DIR",
        default="sampling_freq_results",
        help="Directory to write CSV and figure outputs. Default: sampling_freq_results/",
    )
    parser.add_argument("--plots", action="store_true", default=True, help="Generate and save figures (default: True).")
    parser.add_argument("--no-plots", dest="plots", action="store_false", help="Skip figure generation.")

    args = parser.parse_args()
    main(args)
