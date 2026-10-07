"""Forecast baselines for the harmonized oxygen timeseries.

Every analysis in this repo so far measures *structure* — correlation, lead/lag,
autocorrelation, sampling fidelity. None of it forecasts. That leaves a specific hole:
`data/corr_results/REPORT.md` ranks features by `mean_gain_pct`, "average RMSE change vs
the DO-only AR model", but never reports that model's RMSE. A -5% gain against an
unknown baseline is uninterpretable. This module supplies the missing absolute numbers.

Scored models, all predicting y(t+h) from information available at t only:

  persistence   y(t). The reference every skill score is measured against.
  drift         y(t) plus the recent slope extrapolated h steps.
  climatology   causal running mean of y, plus an hour-of-day anomaly.
  persist_clim  y(t) + [clim(t+h) - clim(t)]. Persistence corrected by the expected
                diel change, and immune to between-deployment level drift because it
                uses only the climatology *difference*, never its level.
  ridge_ar      ridge on the last AR_WINDOW lags of y plus hour-of-day sin/cos.

Three deliberate choices, each guarding a documented hazard:

1. Resampling happens *within* a deployment, never across. The four Florida Keys
   deployments are separated by 3-28 h gaps; a lag that straddles one is fabricated.
   Every model's features come from `shift()` on a single deployment's frame, so the
   guarantee is structural rather than a filter applied afterwards.
2. The hour-of-day climatology carries a minimum-count filter. On this dataset a naive
   hour-of-day mean is confounded by uneven deployment coverage -- the 08:00 bin can
   hold a handful of rows and produce a spurious trough.
3. The pure `climatology` model anchors its level on an expanding mean of data observed
   up to the forecast origin. Anchoring on the held-out deployment's own mean would
   leak, and anchoring on the training mean would import a level offset: deployment
   means span 245-258 umol/L.

Cross-validation is leave-one-deployment-out, matching `corr_results`'
`leave-one-deployment-piece-out`. With `--freq 5min` the step horizons are 1/6/12/36,
so the first three coincide exactly with that report's `horizons (steps) = [1, 6, 12]`
and its percentage gains can finally be read in umol/L.

Target is `umol/L` as published. Note that skill scores and percentage RMSE gains are
invariant to any constant rescaling of the target, so the open question about this
site's absolute saturation level does not touch these results.

Usage:
  python -m smores.models.baselines                # full run, writes data/baselines/
  python -m smores.models.baselines --freq 5min    # one grid only
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from smores import config

# Resample grids. 5min matches corr_results; 2min is the sampling-frequency winner in
# src/test_sampling_frequency.py.
FREQS = ("2min", "5min")

# Forecast horizons in minutes. Deliberately capped at 3 h: the deployments are 14-20 h
# long, so the 6/12/24 h horizons in eda/REPORT.md section 10 have no valid samples here.
HORIZONS_MIN = (5, 30, 60, 180)

# Matches corr_results' "AR window = 12 steps".
AR_WINDOW = 12

# Minimum samples an hour-of-day bin needs before its anomaly is trusted (hazard 2).
MIN_HOUR_COUNT = 10

# ...and the minimum number of distinct deployments that must contribute to it. A count
# threshold alone is not enough here: hour-of-day coverage is badly uneven across the
# four deployments (hours 7, 8 and 10 are each covered by exactly one; hour 9 by none),
# so a single deployment's 12 samples clears MIN_HOUR_COUNT and lets one piece's local
# level masquerade as a diel feature. That is what put the fitted trough at 10:00 when
# the raw series troughs at 06:00-07:00.
MIN_HOUR_DEPLOYMENTS = 2

# The logged clock is local EDT; the harmonized parquet stores UTC. Hour-of-day features
# are a bijection of the hour either way, so this does not change any score -- it is
# here so a reported peak hour means what a reader expects.
LOCAL_UTC_OFFSET_H = -4

SENSORS = ("FL1", "FL2", "FL3")
MEAN3 = "mean3"

OUT_DIR = config.DATA_DIR / "baselines"

# Categorical slots 1-5 of the validated reference palette, in fixed order.
PALETTE = {
    "persistence": "#2a78d6",
    "drift": "#eb6834",
    "climatology": "#1baf7a",
    "persist_clim": "#eda100",
    "ridge_ar": "#e87ba4",
    "seasonal_naive": "#008300",
}
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _resolve(columns, candidates, what):
    """First present name among `candidates`.

    The on-disk parquet predates the current `schema.COLUMNS`, so the oxygen and
    deployment columns go by two different names depending on when it was built.
    Resolving at read time means this module never needs the data layer rebuilt.
    """
    for name in candidates:
        if name in columns:
            return name
    raise KeyError(f"no {what} column found; looked for {candidates}, have {list(columns)}")


def load_wide(path=None, freq="5min"):
    """Load the harmonized parquet as {deployment_id: wide DataFrame}.

    Each frame is indexed by bin timestamp with one column per sensor plus `mean3`,
    the three-sensor mean. Merikhi et al. (2021) prove that mean equals the oxygen
    concentration at the centre of the ADV measuring volume, which makes it the best
    available estimate of truth at this site.
    """
    path = path or config.FLORIDA_KEYS_PARQUET
    head = pd.read_parquet(path, engine="pyarrow").head(0)
    o2_col = _resolve(head.columns, ("do_native_value", "raw_o2_umol"), "oxygen")
    dep_col = _resolve(head.columns, ("deployment_id", "deployment"), "deployment")

    cols = ["timestamp", "sensor_id", dep_col, o2_col]
    if "qc_flag" in head.columns:
        cols.append("qc_flag")
    df = pd.read_parquet(path, columns=cols, engine="pyarrow")
    if "qc_flag" in df.columns:
        df = df[df["qc_flag"].isin([0, 1])]

    out = {}
    for dep, piece in df.groupby(dep_col, observed=True):
        # Median within each bin, per sensor, resampled inside this deployment only
        # (hazard 1). Median rather than mean to suppress optode spikes, matching
        # resample_series() in src/test_sampling_frequency.py.
        binned = (
            piece.set_index("timestamp")
            .groupby("sensor_id", observed=True)[o2_col]
            .resample(freq)
            .median()
            .unstack("sensor_id")
            .sort_index()
        )
        binned = binned.reindex(columns=[s for s in SENSORS if s in binned.columns])
        if binned.empty:
            continue
        binned[MEAN3] = binned[list(binned.columns)].mean(axis=1)
        out[str(dep)] = binned
    return out


def local_hour(index):
    """Hour of day on the local (EDT) clock, as a float."""
    shifted = index + pd.Timedelta(hours=LOCAL_UTC_OFFSET_H)
    return shifted.hour + shifted.minute / 60.0


def noise_floor(frames):
    """Mean inter-optode standard deviation, in umol/L.

    An irreducible-error reference: a model whose RMSE approaches this is at the
    instrument's own disagreement level and cannot be meaningfully improved.
    """
    spreads = [
        f[[s for s in SENSORS if s in f.columns]].std(axis=1, ddof=1).mean() for f in frames
    ]
    return float(np.nanmean(spreads))


# ---------------------------------------------------------------------------
# Climatology
# ---------------------------------------------------------------------------

def fit_hour_climatology(train_frames, column, min_count=MIN_HOUR_COUNT,
                         min_deployments=MIN_HOUR_DEPLOYMENTS):
    """Hour-of-day anomaly profile, fitted on training deployments only.

    Each deployment contributes departures from *its own* mean, so the profile carries
    diel shape without any deployment's absolute level. An hour is trusted only if it
    holds at least `min_count` samples drawn from at least `min_deployments` distinct
    deployments; otherwise its anomaly is 0 (hazard 2). Both gates are needed -- see
    MIN_HOUR_DEPLOYMENTS for why the count alone lets a single piece through.
    """
    parts = []
    for i, frame in enumerate(train_frames):
        series = frame[column].dropna()
        if series.empty:
            continue
        parts.append(
            pd.DataFrame(
                {"hour": np.floor(local_hour(series.index)).astype(int),
                 "anom": series.to_numpy() - series.mean(),
                 "piece": i}
            )
        )
    profile = pd.Series(0.0, index=range(24))
    if not parts:
        return profile
    stacked = pd.concat(parts, ignore_index=True)
    grouped = stacked.groupby("hour").agg(
        mean=("anom", "mean"), size=("anom", "size"), pieces=("piece", "nunique")
    )
    trusted = grouped[
        (grouped["size"] >= min_count) & (grouped["pieces"] >= min_deployments)
    ]["mean"]
    profile.update(trusted)
    return profile


def _anom_at(index, profile):
    hours = np.floor(local_hour(index)).astype(int)
    return profile.reindex(hours).to_numpy()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def mare(prediction, truth):
    """Mean absolute relative error.

    The metric the upstream SMORES work reports throughout, so results here drop
    straight into those comparisons.

    Original definition by **Raman Zatsarenko** (github.com/rzv09),
    `src/metrics/np/regression.py` in rzv09/smores_proj, used across 32 of its files.

    This follows the corrected form in `smores/from_smoresProj/metrics/np/regression.py`
    -- `abs(truth) + eps` rather than upstream's `abs(truth + eps)`, which is
    wrong-signed for negative truth. Identical on this dataset (O2 is ~250 umol/L
    throughout) but not on the SMORES channels that sit at a negative pO2 floor.

    RMSE and MAE stay the primary scores: MARE is scale-dependent, so unlike the skill
    scores it would move if the open question about this site's absolute level were
    ever resolved.
    """
    eps = 1e-8
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if truth.size == 0:
        return float("nan")
    return float(np.sum(np.abs(prediction - truth) / (np.abs(truth) + eps)) / len(truth))


def _season_steps(frame):
    """Number of bins in a 24 h diel season, or None if the index is too short."""
    if len(frame.index) < 2:
        return None
    step = frame.index[1] - frame.index[0]
    if step <= pd.Timedelta(0):
        return None
    return int(round(pd.Timedelta(hours=24) / step))


def predict(model, frame, steps, profile, input_col, target_col):
    """Return (y_true, y_pred) as arrays. See `predict_series` for the aligned form."""
    truth, pred = predict_series(model, frame, steps, profile, input_col, target_col)
    return truth.to_numpy(), pred.to_numpy()


def predict_series(model, frame, steps, profile, input_col, target_col):
    """Return (y_true, y_pred) as Series indexed at the forecast *target* time.

    Every feature is taken from `shift(steps)` or further back, so nothing at or after
    the forecast target is ever read. Returning the index rather than bare arrays keeps
    callers from having to assume where the unusable rows ended up.
    """
    x = frame[input_col]
    y = frame[target_col]
    origin = x.shift(steps)  # the last observation available at forecast time

    if model == "persistence":
        pred = origin
    elif model == "seasonal_naive":
        # y(t+h) = y(t+h-24h): yesterday at the same clock time. On this dataset it has
        # no valid samples at all -- the deployments are 14-20 h long, so t-24h never
        # falls inside the same one, and crossing into another would straddle a 3-28 h
        # gap. Evaluated anyway so the gap is measured rather than assumed.
        season = _season_steps(frame)
        pred = x.shift(season) if season and season < len(x) else x * np.nan
    elif model == "drift":
        slope_per_step = (x - x.shift(AR_WINDOW)) / AR_WINDOW
        pred = origin + steps * slope_per_step.shift(steps)
    elif model == "climatology":
        # Expanding mean of what has actually been seen by the forecast origin, plus
        # the expected anomaly for the target hour (hazard 3).
        level = x.expanding(min_periods=1).mean().shift(steps)
        pred = level + _anom_at(y.index, profile)
    elif model == "persist_clim":
        target_anom = _anom_at(y.index, profile)
        origin_anom = pd.Series(_anom_at(x.index, profile), index=x.index).shift(steps)
        pred = origin + (target_anom - origin_anom.to_numpy())
    elif model == "ridge_ar":
        pred = _ridge_ar_predict(frame, steps, profile, input_col, target_col)
    else:
        raise ValueError(f"unknown model {model!r}")

    pred = pd.Series(np.asarray(pred, dtype=float), index=y.index)
    mask = pred.notna() & y.notna()
    return y[mask], pred[mask]


def _design(frame, steps, profile, input_col, target_col):
    """Lag design matrix for ridge_ar. Lags start at `steps` back from the target."""
    x = frame[input_col]
    lags = {f"lag{i}": x.shift(steps + i) for i in range(AR_WINDOW)}
    hours = local_hour(frame.index)
    lags["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    lags["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    design = pd.DataFrame(lags, index=frame.index)
    design["_target"] = frame[target_col]
    design["_anom"] = _anom_at(frame.index, profile)
    return design


# Ridge coefficients are fitted on the training deployments by `run_fold` and cached
# here per (steps, input_col, target_col) so each fold fits once.
_RIDGE_CACHE = {}


def _ridge_ar_predict(frame, steps, profile, input_col, target_col):
    key = (steps, input_col, target_col)
    model = _RIDGE_CACHE.get(key)
    if model is None:
        return np.full(len(frame), np.nan)
    design = _design(frame, steps, profile, input_col, target_col)
    features = design.drop(columns=["_target"])
    out = pd.Series(np.nan, index=frame.index)
    ok = features.notna().all(axis=1)
    if ok.any():
        out[ok] = model.predict(features[ok].to_numpy())
    return out.to_numpy()


def fit_ridge(train_frames, steps, profile, input_col, target_col):
    """Fit and cache the ridge for one fold."""
    blocks = [_design(f, steps, profile, input_col, target_col) for f in train_frames]
    stacked = pd.concat(blocks, ignore_index=True).dropna()
    key = (steps, input_col, target_col)
    if len(stacked) <= AR_WINDOW:
        _RIDGE_CACHE.pop(key, None)
        return
    target = stacked.pop("_target")
    model = Ridge(alpha=1.0)
    model.fit(stacked.to_numpy(), target.to_numpy())
    _RIDGE_CACHE[key] = model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

MODELS = ("persistence", "seasonal_naive", "drift", "climatology", "persist_clim",
          "ridge_ar")


def evaluate(frames, freq, horizons_min=HORIZONS_MIN, input_col=MEAN3, target_col=MEAN3):
    """Leave-one-deployment-out scores for every model x horizon.

    `input_col` and `target_col` differ only for the single-vs-triple-optode arms.
    """
    step_min = pd.Timedelta(freq).total_seconds() / 60
    rows = []
    for horizon in horizons_min:
        steps = int(round(horizon / step_min))
        if steps < 1:
            continue
        for held_out in frames:
            train = [f for name, f in frames.items() if name != held_out]
            test = frames[held_out]
            profile = fit_hour_climatology(train, input_col)
            fit_ridge(train, steps, profile, input_col, target_col)
            for model in MODELS:
                y_true, y_pred = predict(model, test, steps, profile, input_col, target_col)
                if len(y_true) == 0:
                    continue
                err = y_pred - y_true
                rows.append(
                    {
                        "freq": freq,
                        "horizon_min": horizon,
                        "steps": steps,
                        "fold": held_out,
                        "model": model,
                        "input": input_col,
                        "target": target_col,
                        "n": len(err),
                        "rmse": float(np.sqrt(np.mean(err**2))),
                        "mae": float(np.mean(np.abs(err))),
                        "mare": mare(y_pred, y_true),
                    }
                )
            _RIDGE_CACHE.clear()
    return pd.DataFrame(rows)


def pool(per_fold):
    """Pool folds by sample-weighted RMSE and add skill vs persistence."""
    if per_fold.empty:
        return per_fold
    keys = ["freq", "horizon_min", "steps", "model", "input", "target"]

    def _agg(g):
        n = g["n"].sum()
        return pd.Series(
            {
                "n": int(n),
                "n_folds": g["fold"].nunique(),
                "rmse": float(np.sqrt(np.average(g["rmse"] ** 2, weights=g["n"]))),
                "mae": float(np.average(g["mae"], weights=g["n"])),
                "mare": float(np.average(g["mare"], weights=g["n"])),
                "rmse_fold_min": float(g["rmse"].min()),
                "rmse_fold_max": float(g["rmse"].max()),
            }
        )

    pooled = per_fold.groupby(keys, observed=True).apply(_agg, include_groups=False).reset_index()
    ref = pooled[pooled["model"] == "persistence"].set_index(
        ["freq", "horizon_min", "input", "target"]
    )["rmse"]
    idx = pd.MultiIndex.from_frame(pooled[["freq", "horizon_min", "input", "target"]])
    pooled["skill_vs_persistence"] = 1.0 - pooled["rmse"].to_numpy() / ref.reindex(idx).to_numpy()
    # Per-fold skill sign, which is what exposes a model that only wins on average.
    beats = (
        per_fold.merge(
            per_fold[per_fold["model"] == "persistence"][
                ["freq", "horizon_min", "fold", "input", "target", "rmse"]
            ].rename(columns={"rmse": "rmse_persist"}),
            on=["freq", "horizon_min", "fold", "input", "target"],
        )
        .assign(beat=lambda d: d["rmse"] < d["rmse_persist"])
        .groupby(["freq", "horizon_min", "model", "input", "target"], observed=True)["beat"]
        .agg(["sum", "size"])
    )
    merged = pooled.set_index(["freq", "horizon_min", "model", "input", "target"]).join(beats)
    pooled["folds_beating_persistence"] = (
        merged["sum"].astype("Int64").to_numpy().astype(object)
    )
    return pooled.sort_values(["freq", "horizon_min", "rmse"]).reset_index(drop=True)


def optode_arms(frames, freq):
    """Single-optode input vs three-sensor-mean input, predicting the SAME target.

    Merikhi et al. (2021) show the three-sensor mean cuts signal variance 1.9-3.4x and
    prove it equals the concentration at the ADV centre, but only ever evaluate that
    for flux. Whether it buys *forecast* skill is open.

    Both arms must predict `mean3`. Scoring a single sensor against itself instead
    would hand the averaged arm a free win, because its target would be the smoother
    of the two series rather than the same one.
    """
    single = SENSORS[0] if SENSORS[0] in next(iter(frames.values())).columns else None
    arms = [evaluate(frames, freq, input_col=MEAN3, target_col=MEAN3)]
    if single:
        arms.append(evaluate(frames, freq, input_col=single, target_col=MEAN3))
    return pd.concat(arms, ignore_index=True)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=1.0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    return ax


# Skill axis clip. `climatology` reaches about -6 at 5 min because it deliberately
# ignores the latest observation; letting that set the scale would compress the band
# where every other model actually lives into a sliver. Off-scale points are drawn at
# the boundary and annotated with their true value.
SKILL_YLIM = (-1.35, 0.45)


def fig_skill_vs_horizon(pooled, path):
    grids = sorted(pooled["freq"].unique())
    fig, axes = plt.subplots(1, len(grids), figsize=(5.8 * len(grids), 4.4), squeeze=False)
    lo, hi = SKILL_YLIM
    for ax, freq in zip(axes[0], grids):
        _style(ax)
        sub = pooled[(pooled["freq"] == freq) & (pooled["input"] == MEAN3)]
        for model in MODELS:
            s = sub[sub["model"] == model].sort_values("horizon_min")
            if s.empty:
                continue
            skill = s["skill_vs_persistence"].to_numpy()
            ax.plot(s["horizon_min"], np.clip(skill, lo, hi), marker="o", markersize=5,
                    linewidth=2, color=PALETTE[model], label=model, zorder=3)
            for x, value in zip(s["horizon_min"], skill):
                if value < lo:
                    ax.annotate(f"{value:.1f}", (x, lo), textcoords="offset points",
                                xytext=(0, 7), fontsize=7, color=PALETTE[model],
                                ha="center", fontweight="bold")
        ax.axhline(0, color="#c3c2b7", linewidth=1.2, zorder=2)
        ax.set_ylim(*SKILL_YLIM)
        ax.set_title(f"{freq} bins", fontsize=11, color=INK, loc="left")
        ax.set_xlabel("horizon (minutes)", fontsize=9, color=MUTED)
        ax.set_xscale("log")
        ax.set_xticks(list(HORIZONS_MIN))
        ax.set_xticklabels([str(h) for h in HORIZONS_MIN])
        # Headroom so an off-scale label on the last horizon is not clipped.
        ax.set_xlim(4.4, 230)
    axes[0][0].set_ylabel("skill vs persistence  (1 - RMSE/RMSE_persist)", fontsize=9, color=MUTED)
    axes[0][0].legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    fig.suptitle("Forecast skill against persistence — above the line beats it"
                 "   (axis clipped; off-scale values labelled)",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_trace(frames, freq, pooled, path):
    """Predicted vs actual for the best model at 30 min on one held-out deployment."""
    sub = pooled[(pooled["freq"] == freq) & (pooled["horizon_min"] == 30)
                 & (pooled["input"] == MEAN3) & (pooled["model"] != "persistence")]
    if sub.empty:
        return
    best = sub.sort_values("rmse")["model"].iloc[0]
    held_out = sorted(frames)[0]
    train = [f for name, f in frames.items() if name != held_out]
    test = frames[held_out]
    steps = int(round(30 / (pd.Timedelta(freq).total_seconds() / 60)))
    profile = fit_hour_climatology(train, MEAN3)
    fit_ridge(train, steps, profile, MEAN3, MEAN3)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    _style(ax)
    ax.plot(test.index, test[MEAN3], linewidth=2, color=INK, label="observed", zorder=4)
    for model in ("persistence", best):
        _, y_pred = predict_series(model, test, steps, profile, MEAN3, MEAN3)
        ax.plot(y_pred.index, y_pred.to_numpy(), linewidth=2, color=PALETTE[model],
                label=f"{model} (+30 min)", alpha=0.9, zorder=3)
    _RIDGE_CACHE.clear()
    ax.set_xlabel("UTC", fontsize=9, color=MUTED)
    ax.set_title(f"30-minute forecast, held-out deployment {held_out} ({freq} bins)",
                 fontsize=11, color=INK, loc="left")
    ax.set_ylabel("O2 (umol/L)", fontsize=9, color=MUTED)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_per_fold(per_fold, freq, path):
    sub = per_fold[(per_fold["freq"] == freq) & (per_fold["horizon_min"] == 30)
                   & (per_fold["input"] == MEAN3)]
    if sub.empty:
        return
    folds = sorted(sub["fold"].unique())
    fig, ax = plt.subplots(figsize=(9, 4.2))
    _style(ax)
    width = 0.8 / len(MODELS)
    for i, model in enumerate(MODELS):
        vals = [
            sub[(sub["fold"] == f) & (sub["model"] == model)]["rmse"].mean() for f in folds
        ]
        # 2px surface gap between adjacent bars.
        ax.bar(np.arange(len(folds)) + i * width, vals, width * 0.88, label=model,
               color=PALETTE[model], edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.set_xticks(np.arange(len(folds)) + 0.4 - width / 2)
    ax.set_xticklabels(folds, fontsize=8)
    ax.set_ylabel("RMSE (umol/L), 30 min", fontsize=9, color=MUTED)
    ax.set_title(f"Per-fold RMSE at 30 min ({freq} bins) — fold spread exposes unstable models",
                 fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=len(MODELS))
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_residuals(frames, freq, path):
    steps = int(round(30 / (pd.Timedelta(freq).total_seconds() / 60)))
    held_out = sorted(frames)[0]
    train = [f for name, f in frames.items() if name != held_out]
    profile = fit_hour_climatology(train, MEAN3)
    fit_ridge(train, steps, profile, MEAN3, MEAN3)
    data, labels, colors = [], [], []
    for model in MODELS:
        y_true, y_pred = predict(model, frames[held_out], steps, profile, MEAN3, MEAN3)
        if len(y_true) == 0:
            continue
        data.append(y_pred - y_true)
        labels.append(model)
        colors.append(PALETTE[model])
    _RIDGE_CACHE.clear()
    if not data:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    _style(ax)
    parts = ax.violinplot(data, showextrema=False, showmedians=True)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(SURFACE)
        body.set_linewidth(1.5)
        body.set_alpha(0.95)
    parts["cmedians"].set_color(INK)
    ax.axhline(0, color="#c3c2b7", linewidth=1.2)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("residual (umol/L)", fontsize=9, color=MUTED)
    ax.set_title(f"Residual distribution at 30 min, deployment {held_out} ({freq} bins)",
                 fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_optode_arms(pooled, freq, path):
    sub = pooled[pooled["freq"] == freq]
    arms = [c for c in sub["input"].unique() if c != MEAN3]
    if not arms:
        return
    single = arms[0]
    # Plot the *penalty*, not the two RMSE curves. The arms differ by 0.2-7%, which is
    # invisible on an absolute axis dominated by the horizon effect; the quantity the
    # experiment is actually about is the gap between them.
    wide = sub.pivot_table(index=["model", "horizon_min"], columns="input", values="rmse")
    wide["penalty_pct"] = 100 * (wide[single] / wide[MEAN3] - 1)

    fig, ax = plt.subplots(figsize=(8, 4.4))
    _style(ax)
    for model in MODELS:
        if model not in wide.index.get_level_values("model"):
            continue
        s = wide.xs(model, level="model").sort_index()
        ax.plot(s.index, s["penalty_pct"], marker="o", markersize=6, linewidth=2,
                color=PALETTE[model], label=model, zorder=3)
    ax.axhline(0, color="#c3c2b7", linewidth=1.2, zorder=2)
    ax.set_xscale("log")
    ax.set_xticks(list(HORIZONS_MIN))
    ax.set_xticklabels([str(h) for h in HORIZONS_MIN])
    ax.set_xlim(4.4, 215)
    ax.set_xlabel("horizon (minutes)", fontsize=9, color=MUTED)
    ax.set_ylabel(f"RMSE penalty from using {single} alone (%)", fontsize=9, color=MUTED)
    ax.set_title(
        f"Cost of one optode instead of three ({freq} bins)\n"
        f"both arms predict the same target (mean3); only the input series differs",
        fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(freqs=FREQS, path=None, out_dir=OUT_DIR, make_figures=True):
    out_dir = out_dir
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    per_fold_all, pooled_all, floors = [], [], {}
    for freq in freqs:
        frames = load_wide(path, freq)
        if not frames:
            continue
        floors[freq] = noise_floor(list(frames.values()))
        per_fold = optode_arms(frames, freq)
        per_fold_all.append(per_fold)
        pooled = pool(per_fold)
        pooled_all.append(pooled)
        if make_figures:
            figs = out_dir / "figures"
            fig_trace(frames, freq, pooled, figs / f"trace_{freq}.png")
            fig_per_fold(per_fold, freq, figs / f"per_fold_{freq}.png")
            fig_residuals(frames, freq, figs / f"residuals_{freq}.png")
            fig_optode_arms(pooled, freq, figs / f"optode_arms_{freq}.png")

    per_fold = pd.concat(per_fold_all, ignore_index=True)
    pooled = pd.concat(pooled_all, ignore_index=True)
    if make_figures and not pooled.empty:
        fig_skill_vs_horizon(pooled, out_dir / "figures" / "skill_vs_horizon.png")
    per_fold.to_csv(out_dir / "results_per_fold.csv", index=False)
    pooled.to_csv(out_dir / "results.csv", index=False)
    return per_fold, pooled, floors


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--freq", action="append", choices=list(FREQS),
                        help="resample grid; repeatable. Default: both.")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    per_fold, pooled, floors = run(
        freqs=tuple(args.freq) if args.freq else FREQS, make_figures=not args.no_figures
    )
    for freq, floor in floors.items():
        print(f"{freq}: inter-optode noise floor {floor:.3f} umol/L")
    shown = pooled[pooled["input"] == MEAN3][
        ["freq", "horizon_min", "model", "rmse", "mae", "skill_vs_persistence",
         "folds_beating_persistence", "n_folds"]
    ]
    print(shown.to_string(index=False))
    print(f"\nwrote {OUT_DIR / 'results.csv'} ({len(pooled)} rows)")


if __name__ == "__main__":
    main()
