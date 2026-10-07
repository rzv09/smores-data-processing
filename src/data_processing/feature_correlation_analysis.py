"""
Feature / dissolved-oxygen correlation analysis for the Florida Keys (3OEC)
and Santa Barbara (SBH) datasets.

Goal: decide which feature is the best *main signal* for forecasting DO, and
which other features are worth carrying along as accompanying inputs.

Methods (each one answers a different question; see the generated REPORT.md):

  M1  Contemporaneous correlation   Pearson + Spearman of DO vs every feature,
                                    per deployment piece and pooled after
                                    per-piece z-scoring (as in the project PDF's
                                    "Standardized" plots). Moving-block bootstrap
                                    CIs, because autocorrelated series make naive
                                    p-values far too optimistic.
  M2  First-difference correlation  Same on d/dt. Strips the shared slow
                                    trend/diurnal cycle that makes two unrelated
                                    smooth curves look correlated.
  M3  Lagged cross-correlation      corr(feature(t-k), DO(t)). Peak at k>0 means
                                    the feature LEADS DO -> usable for prediction.
  M4  Mutual information            Catches non-linear dependence Pearson misses.
  M5  Forecast skill (incremental)  Leave-one-piece-out ridge forecast of
                                    DO(t+h): AR baseline (DO history only) vs
                                    AR + each feature vs AR + all. The drop in
                                    RMSE over the AR baseline is the
                                    "does it help a model" answer. This is the
                                    primary ranking criterion.
  M6  Feature redundancy            Feature-feature correlation: two features
                                    that are highly correlated with each other
                                    add little when used together.
  M7  Cross-dataset                 FL DO vs SB DO (and SB feature vs FL DO) on
                                    the overlapping July 2017 windows, with a lag
                                    sweep to expose clock/timezone offsets.

Usage:
  python feature_correlation_analysis.py --sb-path DO_allsites_allyears_20250611.csv \
      --fl-path Florida_Keys_data.csv --freq 5min --outdir corr_results
  python feature_correlation_analysis.py --synthetic      # smoke test, no data needed
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

UMOL_TO_MG_O2 = 0.031998  # umol/L -> mg/L for O2 (32 g/mol)

# Deployment pieces. Same windows as process_3oec.py / process_sb.py.
FL_PIECES = {
    "FL1": ("2017-07-11 14:00", "2017-07-12 08:00"),
    "FL2": ("2017-07-13 11:00", "2017-07-14 06:00"),
    "FL3": ("2017-07-15 10:00", "2017-07-16 06:00"),
    "FL4": ("2017-07-16 16:00", "2017-07-17 06:00"),
}
SB_PIECES = {
    "SB1": ("2017-07-11 14:00", "2017-07-12 08:00"),
    "SB2": ("2017-07-13 09:00", "2017-07-14 06:00"),
    "SB3": ("2017-07-15 09:00", "2017-07-16 06:00"),
    "SB4": ("2017-07-16 15:00", "2017-07-17 06:00"),
}
TARGET = "DO"


# ----------------------------------------------------------------------------
# Loading / feature construction
# ----------------------------------------------------------------------------

def add_calendar_features(df):
    """Hour-of-day sin/cos: known in advance at forecast time (a free covariate)."""
    h = df.index.hour + df.index.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * h / 24)
    df["hour_cos"] = np.cos(2 * np.pi * h / 24)
    return df


def build_fl(raw: pd.DataFrame, freq: str) -> dict:
    """raw: output of process_3oec.make_timeseries (8 Hz, DatetimeIndex)."""
    out = {}
    for label, (a, b) in FL_PIECES.items():
        p = raw.loc[a:b]
        f = pd.DataFrame(index=p.index)
        f[TARGET] = p["O2_avg"] * UMOL_TO_MG_O2
        f["Vx"], f["Vy"], f["Vz"] = p["Vx"], p["Vy"], p["Vz"]
        f["speed"] = np.sqrt(p["Vx"] ** 2 + p["Vy"] ** 2 + p["Vz"] ** 2)
        f["speed_h"] = np.sqrt(p["Vx"] ** 2 + p["Vy"] ** 2)
        f["P"] = p["P"]
        f = f.resample(freq).mean().interpolate(limit=3, limit_area="inside")
        f["dP"] = f["P"].diff()  # tide rising/falling (tidal pumping proxy)
        # Spread between the 3 optodes at native rate, averaged: sensor noise proxy
        spread = p[["O2_S1", "O2_S2", "O2_S3"]].std(axis=1) * UMOL_TO_MG_O2
        f["O2_spread"] = spread.resample(freq).mean()
        out[label] = add_calendar_features(f).dropna(subset=[TARGET])
    return out


def build_sb(raw: pd.DataFrame, freq: str) -> dict:
    """raw: output of process_sb.make_timeseries (DatetimeIndex, local time)."""
    out = {}
    for label, (a, b) in SB_PIECES.items():
        p = raw.loc[a:b]
        f = pd.DataFrame(index=p.index)
        f[TARGET] = p["DO_mgl"]
        f["temperature_C"] = p["temperature_C"]
        f = f.resample(freq).mean().interpolate(limit=3, limit_area="inside")
        f["dT"] = f["temperature_C"].diff()
        # Saturation-style term: DO that temperature alone would not explain.
        out[label] = add_calendar_features(f).dropna(subset=[TARGET])
    return out


def synthetic(freq: str):
    """Smoke-test data with the same schema as the real files (NOT real results)."""
    rng = np.random.default_rng(0)
    fl_rows, sb_rows = [], []
    for lab, (a, b) in FL_PIECES.items():
        idx = pd.date_range(a, b, freq="125ms")
        t = np.arange(len(idx)) / 8 / 3600.0
        tide = np.sin(2 * np.pi * t / 12.4)
        o2 = 250 + 8 * np.sin(2 * np.pi * (t - 3) / 24) + 5 * tide + rng.normal(0, 0.5, len(t))
        fl_rows.append(pd.DataFrame({
            "O2_S1": o2 + rng.normal(0, .3, len(t)), "O2_S2": o2 + rng.normal(0, .3, len(t)),
            "O2_S3": o2 + rng.normal(0, .3, len(t)), "Vx": 0.1 * np.roll(tide, 40) + rng.normal(0, .02, len(t)),
            "Vy": rng.normal(0, .05, len(t)), "Vz": rng.normal(0, .01, len(t)),
            "P": 1.5 + tide}, index=idx))
    fl = pd.concat(fl_rows)
    fl["O2_avg"] = fl[["O2_S1", "O2_S2", "O2_S3"]].mean(axis=1)
    sb_idx = pd.date_range("2017-07-11", "2017-07-18", freq="10min")
    t = np.arange(len(sb_idx)) / 6 / 24.0
    temp = 17 + 1.5 * np.sin(2 * np.pi * (t - .3)) + rng.normal(0, .1, len(t))
    do = 9.5 + 1.0 * np.sin(2 * np.pi * (t - .4)) + rng.normal(0, .15, len(t))
    sb = pd.DataFrame({"DO_mgl": do, "temperature_C": temp}, index=sb_idx)
    return fl, sb


# ----------------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------------

def zscore(df):
    return (df - df.mean()) / df.std(ddof=0).replace(0, np.nan)


def block_bootstrap_ci(x, y, block=12, n_boot=300, seed=0, method="pearson"):
    """Moving-block bootstrap CI for correlation under autocorrelation."""
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < 2 * block:
        return (np.nan, np.nan)
    starts = np.arange(n - block + 1)
    k = int(np.ceil(n / block))
    rs = []
    for _ in range(n_boot):
        s = rng.choice(starts, k)
        ix = (s[:, None] + np.arange(block)).ravel()[:n]
        xs, ys = x[ix], y[ix]
        if xs.std() == 0 or ys.std() == 0:
            continue
        rs.append(np.corrcoef(xs, ys)[0, 1])
    return tuple(np.percentile(rs, [2.5, 97.5])) if rs else (np.nan, np.nan)


def pooled_corr(pieces: dict, feats, method="pearson", diff=False, block=12):
    """Pool per-piece z-scored data; return corr + block-bootstrap CI."""
    rows = []
    for f in feats:
        xs, ys = [], []
        for p in pieces.values():
            d = p[[TARGET, f]].dropna()
            if diff:
                d = d.diff().dropna()
            if len(d) < 10 or d[f].std() == 0:
                continue
            d = zscore(d)
            xs.append(d[f].to_numpy()); ys.append(d[TARGET].to_numpy())
        if not xs:
            rows.append(dict(feature=f, r=np.nan, lo=np.nan, hi=np.nan)); continue
        x, y = np.concatenate(xs), np.concatenate(ys)
        if method == "spearman":
            x, y = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
        r = np.corrcoef(x, y)[0, 1]
        lo, hi = block_bootstrap_ci(x, y, block=block)
        rows.append(dict(feature=f, r=r, lo=lo, hi=hi))
    return pd.DataFrame(rows).set_index("feature")


def per_piece_corr(pieces: dict, feats):
    """Pearson per piece + sign consistency (stability across deployments)."""
    tab = {}
    for lab, p in pieces.items():
        tab[lab] = {f: p[[TARGET, f]].dropna().corr().iloc[0, 1] if p[f].std() > 0 else np.nan
                    for f in feats}
    t = pd.DataFrame(tab)
    t["mean_r"] = t.mean(axis=1)
    t["sign_consistent"] = t.drop(columns="mean_r").apply(
        lambda r: bool(np.all(np.sign(r.dropna()) == np.sign(r.dropna().iloc[0]))) if r.notna().any() else False, axis=1)
    return t


def lagged_xcorr(pieces: dict, feats, max_lag: int):
    """corr(feature(t-k), DO(t)) for k in [-max_lag, max_lag]; k>0 = feature leads."""
    lags = np.arange(-max_lag, max_lag + 1)
    res = {}
    for f in feats:
        vals = []
        for k in lags:
            xs, ys = [], []
            for p in pieces.values():
                d = zscore(p[[TARGET, f]].dropna())
                x = d[f].shift(k)  # value k steps earlier
                m = x.notna() & d[TARGET].notna()
                xs.append(x[m].to_numpy()); ys.append(d[TARGET][m].to_numpy())
            x, y = np.concatenate(xs), np.concatenate(ys)
            vals.append(np.corrcoef(x, y)[0, 1] if len(x) > 10 and x.std() > 0 else np.nan)
        res[f] = vals
    return pd.DataFrame(res, index=lags)


def _mi_hist(x, y, bins=12):
    """Histogram MI estimate (nats) on equal-frequency bins."""
    qx = pd.qcut(pd.Series(x).rank(method="first"), bins, labels=False)
    qy = pd.qcut(pd.Series(y).rank(method="first"), bins, labels=False)
    pxy = pd.crosstab(qx, qy).to_numpy() / len(x)
    px, py = pxy.sum(1, keepdims=True), pxy.sum(0, keepdims=True)
    nz = pxy > 0
    return float((pxy[nz] * np.log(pxy[nz] / (px @ py)[nz])).sum())


def _ridge_fit_predict(Xtr, ytr, Xte, alpha):
    A = np.c_[np.ones(len(Xtr)), Xtr.to_numpy()]
    reg = alpha * np.eye(A.shape[1]); reg[0, 0] = 0
    w = np.linalg.solve(A.T @ A + reg, A.T @ ytr.to_numpy())
    return np.c_[np.ones(len(Xte)), Xte.to_numpy()] @ w


def _md(df, index=True):
    d = df.reset_index() if index else df
    lines = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)]
    return chr(10).join(lines)


def mutual_info(pieces: dict, feats):
    out = {}
    for f in feats:
        d = pd.concat([zscore(p[[TARGET, f]].dropna()) for p in pieces.values()]).dropna()
        out[f] = _mi_hist(d[f].to_numpy(), d[TARGET].to_numpy()) if len(d) > 20 else np.nan
    return pd.Series(out, name="MI")


# ----------------------------------------------------------------------------
# Forecast skill (M5)
# ----------------------------------------------------------------------------

def make_xy(piece: pd.DataFrame, extra: list, n_lags: int, horizon: int):
    """Row t: DO[t-n_lags+1..t] (+ feature lags) -> DO[t+horizon]. Built inside one
    piece so windows never straddle a deployment gap."""
    cols = {}
    for k in range(n_lags):
        cols[f"{TARGET}_l{k}"] = piece[TARGET].shift(k)
    for f in extra:
        for k in range(n_lags):
            cols[f"{f}_l{k}"] = piece[f].shift(k)
    X = pd.DataFrame(cols, index=piece.index)
    y = piece[TARGET].shift(-horizon)
    m = X.notna().all(axis=1) & y.notna()
    return X[m], y[m]


def forecast_skill(pieces: dict, feats, n_lags: int, horizon: int, alpha: float = 1.0,
                   use_target_history: bool = True):
    """Leave-one-piece-out RMSE (in z-units of the held-out piece's training stats)."""
    labels = list(pieces)

    def run(extra):
        errs, pers = [], []
        for hold in labels:
            tr = [l for l in labels if l != hold]
            # standardise with TRAIN statistics only (no leakage)
            cols = [TARGET] + list(extra)
            mu = pd.concat([pieces[l][cols] for l in tr]).mean()
            sd = pd.concat([pieces[l][cols] for l in tr]).std().replace(0, 1)
            Xtr, ytr, Xte, yte = [], [], None, None
            for l in tr:
                z = (pieces[l][cols] - mu) / sd
                x, y = make_xy(z, extra, n_lags, horizon)
                Xtr.append(x); ytr.append(y)
            z = (pieces[hold][cols] - mu) / sd
            Xte, yte = make_xy(z, extra, n_lags, horizon)
            Xtr, ytr = pd.concat(Xtr), pd.concat(ytr)
            if not use_target_history:
                keep = [c for c in Xtr.columns if not c.startswith(f"{TARGET}_l")]
                Xtr, Xte = Xtr[keep], Xte[keep]
            if Xtr.shape[1] == 0 or len(Xte) == 0:
                continue
            errs.append((_ridge_fit_predict(Xtr, ytr, Xte, alpha) - yte.to_numpy()) ** 2)
            pers.append((z[TARGET].reindex(yte.index).to_numpy() - yte.to_numpy()) ** 2)
        return (np.sqrt(np.mean(np.concatenate(errs))), np.sqrt(np.mean(np.concatenate(pers)))) if errs else (np.nan, np.nan)

    base, persistence = run([])
    rows = [dict(model="persistence", rmse=persistence),
            dict(model="AR(DO only)", rmse=base)]
    for f in feats:
        rows.append(dict(model=f"AR + {f}", rmse=run([f])[0], feature=f))
    rows.append(dict(model="AR + ALL", rmse=run(list(feats))[0]))
    df = pd.DataFrame(rows)
    df["delta_vs_AR"] = df["rmse"] - base
    df["pct_vs_AR"] = 100 * df["delta_vs_AR"] / base
    return df


def feature_only_skill(pieces, feats, n_lags, horizon):
    """Each feature's own history (NO DO history) -> DO(t+h). Shows standalone power."""
    rows = []
    for f in feats:
        s = forecast_skill(pieces, [f], n_lags, horizon, use_target_history=False)
        rows.append(dict(feature=f, rmse=s.loc[s.model == f"AR + {f}", "rmse"].iloc[0]))
    return pd.DataFrame(rows).set_index("feature")


# ----------------------------------------------------------------------------
# Cross-dataset (M7)
# ----------------------------------------------------------------------------

def cross_dataset(fl: dict, sb: dict, freq: str, max_lag_h: int = 6):
    """Pair FL_i with SB_j wherever their windows overlap in clock time."""
    step = pd.Timedelta(freq)
    n_lag = int(pd.Timedelta(hours=max_lag_h) / step)
    fl_feats = [c for c in next(iter(fl.values())).columns]
    sb_feats = [c for c in next(iter(sb.values())).columns]
    pair_rows, lag_rows = [], []
    for fl_lab, f in fl.items():
        for sb_lab, s in sb.items():
            j = f.join(s, how="inner", lsuffix="_FL", rsuffix="_SB").dropna(subset=[f"{TARGET}_FL", f"{TARGET}_SB"])
            if len(j) < 20:
                continue
            r = j[f"{TARGET}_FL"].corr(j[f"{TARGET}_SB"])
            rd = j[f"{TARGET}_FL"].diff().corr(j[f"{TARGET}_SB"].diff())
            pair_rows.append(dict(fl=fl_lab, sb=sb_lab, n=len(j), r_DO=r, r_dDO=rd))
    pair = pd.DataFrame(pair_rows)

    # Lag sweep on the union of overlapping data: shift SB by k steps relative to FL.
    fl_all = pd.concat(fl.values()); sb_all = pd.concat(sb.values())
    lags = np.arange(-n_lag, n_lag + 1)
    for tgt_name, tgt in ((f"{TARGET}_FL", fl_all[TARGET]),):
        for sb_f in [c for c in sb_all.columns if c not in ("hour_sin", "hour_cos")]:
            vals = []
            for k in lags:
                x = sb_all[sb_f].copy(); x.index = x.index + k * step
                j = pd.concat([zscore(tgt.to_frame("y")), x.rename("x")], axis=1, join="inner").dropna()
                vals.append(j["y"].corr(zscore(j[["x"]])["x"]) if len(j) > 20 else np.nan)
            lag_rows.append(pd.Series(vals, index=lags, name=f"SB.{sb_f} -> FL.DO"))
    # reverse direction: FL features -> SB DO
    for fl_f in [c for c in fl_all.columns if c not in ("hour_sin", "hour_cos")]:
        vals = []
        for k in lags:
            x = fl_all[fl_f].copy(); x.index = x.index + k * step
            j = pd.concat([zscore(sb_all[TARGET].to_frame("y")), x.rename("x")], axis=1, join="inner").dropna()
            vals.append(j["y"].corr(zscore(j[["x"]])["x"]) if len(j) > 20 else np.nan)
        lag_rows.append(pd.Series(vals, index=lags, name=f"FL.{fl_f} -> SB.DO"))
    return pair, pd.DataFrame(lag_rows)


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------

def plot_corr_bars(tabs: dict, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(8, 0.5 * max(len(t) for t in tabs.values()) + 1.5))
    names = list(tabs)
    feats = tabs[names[0]].sort_values("r", key=np.abs).index
    h = 0.8 / len(names)
    for i, n in enumerate(names):
        t = tabs[n].loc[feats]
        y = np.arange(len(feats)) + i * h
        ax.barh(y, t["r"], height=h, xerr=[t["r"] - t["lo"], t["hi"] - t["r"]], label=n, capsize=2)
    ax.set_yticks(np.arange(len(feats)) + h * (len(names) - 1) / 2); ax.set_yticklabels(feats)
    ax.axvline(0, c="k", lw=.6); ax.set_xlabel("correlation with DO (95% block-bootstrap CI)")
    ax.set_title(title); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_lag(df: pd.DataFrame, step_min: float, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for c in df.columns:
        ax.plot(df.index * step_min, df[c], label=c)
    ax.axvline(0, c="k", lw=.6); ax.axhline(0, c="k", lw=.6)
    ax.set_xlabel("lag (min); positive = feature leads DO"); ax.set_ylabel("corr")
    ax.set_title(title); ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_heatmap(m: pd.DataFrame, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(0.7 * len(m) + 3, 0.7 * len(m) + 2))
    im = ax.imshow(m.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(m))); ax.set_xticklabels(m.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(m))); ax.set_yticklabels(m.index)
    for i in range(len(m)):
        for j in range(len(m)):
            ax.text(j, i, f"{m.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im); ax.set_title(title); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_skill(skills: dict, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for lab, s in skills.items():
        s = s[s.model.str.startswith("AR +") & (s.model != "AR + ALL")]
        ax.plot(s.model.str.replace("AR + ", "", regex=False), s.pct_vs_AR, "o-", label=lab)
    ax.axhline(0, c="k", lw=.6); ax.set_ylabel("RMSE change vs AR(DO only), %  (negative = helps)")
    ax.set_title(title); ax.legend(title="horizon"); plt.xticks(rotation=45, ha="right")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------

def analyse_dataset(name, pieces, freq, outdir, n_lags, horizons, max_lag):
    feats = [c for c in next(iter(pieces.values())).columns if c != TARGET]
    step_min = pd.Timedelta(freq).total_seconds() / 60
    res = {}
    print(f"[{name}] features: {feats}; pieces: { {k: len(v) for k, v in pieces.items()} }")

    res["pearson"] = pooled_corr(pieces, feats, "pearson")
    res["spearman"] = pooled_corr(pieces, feats, "spearman")
    res["diff_pearson"] = pooled_corr(pieces, feats, "pearson", diff=True)
    res["per_piece"] = per_piece_corr(pieces, feats)
    res["mi"] = mutual_info(pieces, feats)
    res["lag"] = lagged_xcorr(pieces, feats, max_lag)
    allz = pd.concat([zscore(p) for p in pieces.values()])
    res["feat_corr"] = allz[[TARGET] + feats].corr()

    skills = {}
    for h in horizons:
        s = forecast_skill(pieces, feats, n_lags, h)
        s["horizon_steps"] = h
        skills[f"h={h} ({h * step_min:.0f} min)"] = s
        res[f"skill_h{h}"] = s
    res["feature_only"] = pd.concat(
        {f"h={h}": feature_only_skill(pieces, feats, n_lags, h)["rmse"] for h in horizons}, axis=1)

    for k, v in res.items():
        v.to_csv(outdir / f"{name}_{k}.csv")
    plot_corr_bars({"levels": res["pearson"], "first diff": res["diff_pearson"]},
                   f"{name}: correlation with DO", outdir / f"{name}_corr_bars.png")
    plot_lag(res["lag"], step_min, f"{name}: lagged cross-correlation with DO", outdir / f"{name}_lagged_xcorr.png")
    plot_heatmap(res["feat_corr"], f"{name}: feature-feature correlation", outdir / f"{name}_feature_heatmap.png")
    plot_skill(skills, f"{name}: forecast skill gain per added feature", outdir / f"{name}_forecast_skill.png")
    res["skills"] = skills
    res["feats"] = feats
    return res


def rank_features(res, horizons):
    """Composite ranking: forecast gain (primary), then |r_diff| and best lagged |r|."""
    feats = res["feats"]
    rows = []
    for f in feats:
        gains = [res[f"skill_h{h}"].set_index("model").loc[f"AR + {f}", "pct_vs_AR"] for h in horizons]
        lag = res["lag"][f]
        best_k = lag.abs().idxmax()
        rows.append(dict(feature=f, mean_gain_pct=np.mean(gains),
                         r=res["pearson"].loc[f, "r"], r_diff=res["diff_pearson"].loc[f, "r"],
                         mi=res["mi"][f], peak_lag_steps=best_k, peak_lag_r=lag[best_k],
                         sign_consistent=res["per_piece"].loc[f, "sign_consistent"],
                         max_abs_corr_other_feat=res["feat_corr"].loc[f].drop([f, TARGET]).abs().max()))
    return pd.DataFrame(rows).set_index("feature").sort_values("mean_gain_pct")


def write_report(outdir, args, ranks, cross_pair, cross_lag, step_min):
    L = ["# DO feature correlation analysis\n",
         f"Sampling: `{args.freq}`; AR window = {args.n_lags} steps; horizons (steps) = {args.horizons}; "
         f"CV = leave-one-deployment-piece-out.\n",
         "**Reading guide.** `mean_gain_pct` = average RMSE change vs the DO-only AR model when this feature "
         "is added (negative = helps; this is the main ranking). `r` / `r_diff` = pooled Pearson on levels / "
         "first differences. `peak_lag_steps > 0` = feature leads DO. `max_abs_corr_other_feat` high = redundant "
         "with another feature.\n",
         "Caveats: only 4 pieces per site (~18 h each) and a single week; correlations on levels are inflated by "
         "the shared diurnal cycle (compare with `r_diff`); the two sites share no common feature except DO and "
         "hour-of-day, so cross-site transfer relies on DO itself.\n"]
    for name, r in ranks.items():
        L.append(f"\n## {name}\n\n{_md(r.round(3))}\n")
    if cross_pair is not None and len(cross_pair):
        L.append(f"\n## Cross-dataset (FL vs SB, clock-time aligned)\n\n{_md(cross_pair.round(3), index=False)}\n")
        peak = cross_lag.apply(lambda s: pd.Series(dict(peak_lag_min=s.abs().idxmax() * step_min,
                                                         peak_r=s[s.abs().idxmax()], r_at_0=s[0])), axis=1)
        L.append(f"\nLag sweep (shift of the first-named series, min; positive = leads):\n\n{_md(peak.round(3))}\n")
        L.append("\nNote: FL timestamps are local (EDT) while SB is converted to America/Los_Angeles; "
                 "a peak lag near +/-180 min suggests a timezone-driven phase offset rather than a physical lead.\n")
    (outdir / "REPORT.md").write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sb-path"); ap.add_argument("--fl-path")
    ap.add_argument("--freq", default="5min")
    ap.add_argument("--n-lags", type=int, default=12, help="AR window length (steps)")
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 6, 12])
    ap.add_argument("--max-lag", type=int, default=36, help="lag sweep half-width (steps)")
    ap.add_argument("--outdir", default="corr_results")
    ap.add_argument("--synthetic", action="store_true", help="smoke test on fake data")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    if args.synthetic:
        fl_raw, sb_raw = synthetic(args.freq)
    else:
        if not args.fl_path:
            ap.error("--fl-path is required unless --synthetic")
        from data_processing.three_oec import process_3oec
        fl_raw = process_3oec.make_timeseries(args.fl_path)
        sb_raw = None
        if args.sb_path:
            from data_processing.santa_barbara import process_sb
            sb_raw = process_sb.make_timeseries(args.sb_path)
    fl = build_fl(fl_raw, args.freq)
    sb = build_sb(sb_raw, args.freq) if sb_raw is not None else None

    step_min = pd.Timedelta(args.freq).total_seconds() / 60
    ranks = {}
    for name, pieces in (("FL", fl), ("SB", sb)):
        if pieces is None:
            print(f"[{name}] skipped (no data path given)"); continue
        res = analyse_dataset(name, pieces, args.freq, outdir, args.n_lags, args.horizons, args.max_lag)
        ranks[name] = rank_features(res, args.horizons)
        ranks[name].to_csv(outdir / f"{name}_feature_ranking.csv")
        print(f"\n[{name}] ranking (best first):\n{ranks[name].round(3)}\n")

    pair, lag = (cross_dataset(fl, sb, args.freq) if sb is not None else (None, pd.DataFrame()))
    if pair is not None:
        pair.to_csv(outdir / "cross_pairs.csv", index=False); lag.to_csv(outdir / "cross_lag.csv")
    if len(lag):
        plot_lag(lag.T, step_min, "Cross-dataset lagged correlation", outdir / "cross_lagged_xcorr.png")
    write_report(outdir, args, ranks, pair, lag, step_min)
    print(f"Wrote results to {outdir.resolve()}")


if __name__ == "__main__":
    main()
