"""Predicted vs actual for Naive, ARIMA, ARIMAX and ridge_ar, across every deployment.

The companion to `baselines.fig_trace`. One panel per deployment, panel widths
proportional to the span each covers, so an hour is the same width everywhere and the
whole record is on screen rather than a single day.

Protocol
--------
Rolling origin *within* a deployment: `StatsForecast.cross_validation` with
`step_size == h`, so successive windows abut and the predictions form one continuous
curve. Every model is refit at every cutoff, on that deployment's history up to the
cutoff and nothing else.

Forecasting starts at each deployment's 80/20 boundary, the same split
`baselines.split_frames` uses, so this figure and the 80/20 column of the benchmark are
scored over exactly the same rows. The training block doubles as AutoARIMA's warmup:
below roughly 100 training points its initialisation emits `divide by zero encountered
in scalar divide` and the exogenous arm diverges, and 80% of the shortest deployment is
337 bins, comfortably clear of that.

This is *not* the leave-one-deployment-out protocol also reported in `data/baselines/`.
Mixing protocols inside one figure is the error this module exists to avoid, which is
why `ridge_ar` is refit here rather than reusing its benchmark predictions.

Why ridge_ar needs 15 fits per cutoff
-------------------------------------
ARIMA reaches 30 min recursively: predict 2 min ahead, feed that back in, repeat. So it
produces all 15 intermediate steps for free, and the drift across them is the headline
of this figure. `ridge_ar` is *direct* -- one model per horizon -- so covering the same
window means fitting it once for each step 1..15. Ridge is a closed-form least squares
on a few hundred rows, so this costs milliseconds; the result is a genuine direct
multi-step forecast rather than a recursive one.

One consequence of the single-deployment fit: `fit_hour_climatology` only trusts an hour
covered by at least `MIN_HOUR_DEPLOYMENTS` (2) distinct deployments, which one
deployment can never satisfy, so the fitted anomaly profile is identically zero here.
`ridge_ar` still carries time of day through its `hour_sin`/`hour_cos` features; it is
the learned per-hour offset that is missing. Expect it to look slightly weaker than its
leave-one-deployment-out numbers.

Usage:
    python -m smores.viz.plot_arima_trace
    python -m smores.viz.plot_arima_trace --fraction 0.8 --horizon 15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, Naive

from smores import config
from smores.models import baselines

# Categorical slots 1-4 of the validated reference palette, in fixed order. Observed
# stays ink rather than taking a series colour.
COLORS = {
    "Naive": "#2a78d6",
    "ARIMA": "#eb6834",
    "ARIMAX": "#1baf7a",
    "ridge_ar": "#eda100",
}
SERIES = tuple(COLORS)
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

DEFAULT_FREQ = "2min"
DEFAULT_HORIZON = 15  # steps; 15 x 2 min = 30 min


DISPLAY_TZ = "America/New_York"


def load_nixtla(freq=DEFAULT_FREQ):
    """The three-sensor mean per deployment, in Nixtla's unique_id/ds/y layout."""
    path = config.PROCESSED_DIR / f"florida_keys_o2_{freq}.parquet"
    df = pd.read_parquet(path)
    d = (
        df[df["qc_flag"] == 0]
        .groupby(["deployment_id", "timestamp"], as_index=False, observed=True)[
            "do_native_value"
        ]
        .mean()
        .rename(
            columns={
                "deployment_id": "unique_id",
                "timestamp": "ds",
                "do_native_value": "y",
            }
        )
    )
    # statsforecast rejects tz-aware timestamps; the parquet stores UTC.
    d["ds"] = d["ds"].dt.tz_localize(None)
    return d.sort_values(["unique_id", "ds"], ignore_index=True)


def with_hour_features(d):
    """Hour-of-day sine/cosine on the local clock, matching ridge_ar's own features."""
    local = d["ds"] + pd.Timedelta(hours=baselines.LOCAL_UTC_OFFSET_H)
    hours = local.dt.hour + local.dt.minute / 60.0
    return d.assign(
        hour_sin=np.sin(2 * np.pi * hours / 24),
        hour_cos=np.cos(2 * np.pi * hours / 24),
    )


def ridge_window(frame, cutoff_pos, horizon):
    """Direct multi-step ridge: one fit per step, forecasting 1..horizon from `cutoff_pos`.

    `frame` is indexed by timestamp with a single `mean3` column. Training uses rows
    `0..cutoff_pos` only; the returned predictions land at positions
    `cutoff_pos+1 .. cutoff_pos+horizon`.
    """
    train = frame.iloc[: cutoff_pos + 1]
    profile = baselines.fit_hour_climatology([train], baselines.MEAN3)
    out = []
    for step in range(1, horizon + 1):
        target_pos = cutoff_pos + step
        if target_pos >= len(frame):
            break
        design = baselines._design(
            train, step, profile, baselines.MEAN3, baselines.MEAN3
        ).dropna()
        if len(design) <= baselines.AR_WINDOW:
            out.append(np.nan)
            continue
        y = design.pop("_target")
        model = Ridge(alpha=1.0).fit(design.to_numpy(), y.to_numpy())

        # Features for the target row come from `step` back, i.e. at or before the
        # cutoff -- so nothing after the origin is ever read.
        full = baselines._design(
            frame, step, profile, baselines.MEAN3, baselines.MEAN3
        ).drop(columns=["_target"])
        row = full.iloc[[target_pos]]
        out.append(
            float(model.predict(row.to_numpy())[0]) if row.notna().all(axis=1).iloc[0]
            else np.nan
        )
    return out


def run_deployment(plain, exog, horizon, freq, fraction=baselines.SPLIT_FRACTION):
    """Forecast one deployment's test block. Returns a tidy frame or None.

    The test block is the last `1 - fraction` of the deployment, matching
    `baselines.split_frames`, so this figure and the 80/20 column of the benchmark are
    scored over exactly the same rows.
    """
    cut = int(round(len(plain) * fraction))
    windows = (len(plain) - cut) // horizon
    if windows < 2:
        return None

    sf = StatsForecast(models=[AutoARIMA(season_length=1), Naive()], freq=freq, n_jobs=1)
    bare = sf.cross_validation(
        df=plain, h=horizon, n_windows=windows, step_size=horizon
    ).rename(columns={"AutoARIMA": "ARIMA"})

    sfx = StatsForecast(models=[AutoARIMA(season_length=1)], freq=freq, n_jobs=1)
    withx = sfx.cross_validation(
        df=exog, h=horizon, n_windows=windows, step_size=horizon
    ).rename(columns={"AutoARIMA": "ARIMAX"})

    cv = bare.merge(withx[["unique_id", "ds", "ARIMAX"]], on=["unique_id", "ds"],
                    how="left")

    # ridge_ar on the same cutoffs.
    frame = plain.set_index("ds")[["y"]].rename(columns={"y": baselines.MEAN3})
    positions = {ts: i for i, ts in enumerate(frame.index)}
    preds = {}
    for cutoff in sorted(cv["cutoff"].unique()):
        pos = positions.get(pd.Timestamp(cutoff))
        if pos is None:
            continue
        values = ridge_window(frame, pos, horizon)
        for step, value in enumerate(values, start=1):
            if pos + step < len(frame):
                preds[frame.index[pos + step]] = value
    cv["ridge_ar"] = cv["ds"].map(preds)
    return cv


def build(freq=DEFAULT_FREQ, horizon=DEFAULT_HORIZON,
          fraction=baselines.SPLIT_FRACTION):
    data = load_nixtla(freq)
    panels = []
    for name in sorted(data["unique_id"].unique()):
        plain = data[data["unique_id"] == name].reset_index(drop=True)
        exog = with_hour_features(plain)
        cv = run_deployment(plain, exog, horizon, freq, fraction)
        if cv is not None and not cv.empty:
            panels.append((name, cv))
    if not panels:
        raise ValueError("no deployment had a long enough test block")
    panels.sort(key=lambda item: item[1]["ds"].iloc[0])
    return panels


def rmse(pred, truth):
    pred, truth = np.asarray(pred, dtype=float), np.asarray(truth, dtype=float)
    ok = np.isfinite(pred) & np.isfinite(truth)
    return float(np.sqrt(np.mean((pred[ok] - truth[ok]) ** 2))) if ok.any() else np.nan


def plot(panels, horizon, freq, out):
    minutes = int(round(horizon * pd.Timedelta(freq).total_seconds() / 60))

    # Pooled scores across every deployment, so the legend reports the whole record
    # rather than whichever panel happens to be first.
    allcv = pd.concat([cv for _, cv in panels], ignore_index=True)
    scores = {m: rmse(allcv[m], allcv["y"]) for m in SERIES}

    spans = [(cv["ds"].iloc[-1] - cv["ds"].iloc[0]).total_seconds() for _, cv in panels]
    fig, axes = plt.subplots(
        1, len(panels), figsize=(16, 4.8), sharey=True,
        gridspec_kw={"width_ratios": spans, "wspace": 0.04},
        constrained_layout=True, facecolor=SURFACE,
    )
    axes = [axes] if len(panels) == 1 else list(axes)

    for index, (ax, (name, cv)) in enumerate(zip(axes, panels)):
        ax.set_facecolor(SURFACE)
        for cutoff in sorted(cv["cutoff"].unique())[1:]:
            ax.axvline(cutoff, color=MUTED, lw=0.6, alpha=0.35, zorder=2)

        ax.plot(cv["ds"], cv["y"], color=INK, lw=1.8, zorder=5,
                label="observed" if index == 0 else None)
        for model in SERIES:
            ax.plot(cv["ds"], cv[model], color=COLORS[model], lw=1.2, zorder=3,
                    label=model if index == 0 else None)

        start, end = cv["ds"].iloc[0], cv["ds"].iloc[-1]
        ax.set_xlim(start, end)
        ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=1)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=7)
        baselines._timestamp_axis(ax, start, end, target_ticks=4)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)

        ax.set_title(f"{name}\n{baselines._date_range_label(start, end)}",
                     color=INK, fontsize=8, loc="left", pad=6)

    axes[0].set_ylabel("O2 (umol/L)", fontsize=9, color=MUTED)
    axes[0].spines["left"].set_visible(True)
    axes[0].spines["left"].set_color(GRID)

    # Figure-level, above the panels: the curves fill the plotting area on every panel
    # and an in-axes legend lands on data whichever corner it is given.
    handles, labels = axes[0].get_legend_handles_labels()
    labels = [
        f"{text}  RMSE {scores[text]:.2f}" if text in scores else text
        for text in labels
    ]
    fig.legend(handles, labels, frameon=False, fontsize=8, labelcolor=MUTED,
               loc="upper right", ncols=len(labels), bbox_to_anchor=(0.995, 0.995))

    fig.suptitle(
        f"{minutes}-minute forecast across all deployments ({freq} bins) — "
        "rolling origin, every model refit at each cutoff; RMSE pooled over all panels",
        color=INK, fontsize=11, x=0.01, ha="left",
    )
    fig.supxlabel("UTC", color=MUTED, fontsize=9)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)

    print(f"wrote {out}\n")
    for name, cv in panels:
        per = {m: rmse(cv[m], cv["y"]) for m in SERIES}
        best = min(per, key=lambda k: per[k])
        joined = "  ".join(f"{m} {per[m]:.2f}" for m in SERIES)
        print(f"  {name:<20} {joined}   best: {best}")
    print()
    for model, value in sorted(scores.items(), key=lambda kv: kv[1]):
        print(f"  pooled  {model:<10} RMSE {value:.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freq", default=DEFAULT_FREQ)
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    p.add_argument("--fraction", type=float, default=baselines.SPLIT_FRACTION,
                   help="train fraction per deployment (default: %(default)s)")
    p.add_argument(
        "--out", type=Path,
        default=config.DATA_DIR / "processed" / "figures" / "arima_trace_2min.png",
    )
    args = p.parse_args()

    panels = build(args.freq, args.horizon, args.fraction)
    plot(panels, args.horizon, args.freq, args.out)


if __name__ == "__main__":
    main()
