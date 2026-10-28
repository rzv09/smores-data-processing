"""
Time of day vs oxygen: Harvard SMORES (Biscayne Bay, 2026) and Florida Keys 3OEC (2017).

    python eda/time_of_day_compare.py

Both datasets are binned to the same 15-min medians and scored with the same three measures:
  * day/night r     point-biserial r with a day indicator (DAY_HOURS local clock window)
  * 24-h harmonic R multiple correlation of O2 with cos/sin of hour of day; its phase gives the peak hour
  * hour-of-day R²  variance explained by the mean of each clock hour (overfits short records; reference only)
Each measure is computed on raw values and on values with each day's (Harvard) or each
deployment's (Florida Keys) mean removed. CIs: Fisher z with Bretherton (1999) N_eff for both
datasets, plus a 1-day moving-block bootstrap for Harvard day/night r. Florida Keys deployments
are each shorter than a day, so a 1-day block bootstrap is not possible there.

Both datasets are compared as O2 concentration in umol/L. Harvard optodes measure pO2; it is converted with
Garcia & Gordon (1992) solubility at the optode temperature and an assumed salinity (HARVARD_SALINITY, not in
the dump), because the sensor's own mg/L (param 20) assumes S = 0. pO2 and the sensor mg/L are kept in the
table as checks. Florida Keys O2 units are not stated in the CSV; umol/L is assumed.
"""

import os
import sys

import numpy as np
import pandas as pd

# ============================ CONFIG ============================
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
FL_CSV = os.path.join(ROOT, "resources", "Florida_Keys_data.csv")
FIG_DIR = os.path.join(HERE, "figures")
TAB_DIR = os.path.join(HERE, "tables")
CACHE_DIR = os.path.join(HERE, "cache")

BIN = "15min"
DAY_HOURS = (7, 20)        # local clock, [start, end)
HARVARD_SIGNAL = "o2_umol"                 # concentration from pO2 (see o2_umol_per_l)
HARVARD_CHECK_SIGNALS = ["po2", "do_mgl"]  # table only
HARVARD_SALINITY = 35.0    # assumed (PSU); no salinity in the dump. Biscayne Bay is typically ~30-36
SW_DENSITY = 1.022         # kg/L, seawater at ~29 C and S = 35, to turn umol/kg into umol/L
UNIT = "O2 (µmol/L)"
FL_SENSORS = ["O2_S1", "O2_S2", "O2_S3"]
# Florida Keys `t` is decimal hour of day (continuing past 24 after midnight). The time zone is not
# stated (BCO-DMO dataset 849934, ~9 km south of Long Key, 24.7253 N, -80.8308 W). Treated as local (EDT): the
# metadata says "measurements could not start before 14:00 due to bad weather" and FL1 starts at t = 14:00,
# and O2 bottoms out near dawn. If t were UTC, every Florida Keys hour below would shift 4 h earlier.
# Deployment windows follow src/data_processing/three_oec/process_3oec.get_*_piece (settling trimmed).
FL_WINDOWS = [("FL1", "2017-07-11", "2017-07-12 06:00:00"), ("FL2", "2017-07-13 12:00:00", "2017-07-14 06:00:00"),
              ("FL3", "2017-07-15 12:00:00", "2017-07-16 06:00:00"), ("FL4", "2017-07-16 16:00:00", "2017-07-17")]
DPI = 150
# ================================================================

sys.path.insert(0, HERE)
import correlation_eda as ce  # noqa: E402  (Harvard grid, stats helpers, palette, plot style)

plt, sns, stats = ce.plt, ce.sns, ce.stats
C_HARV, C_FL = "#2a78d6", "#eb6834"
LS = {2: "-", 3: "--", 25: ":", 7: "-", 26: "--", 8: "-"}   # separates channels sharing a depth colour
C_DAY, C_NIGHT = ce.C_DAY, ce.C_NIGHT


def savefig(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ------------------------- data -------------------------

def load_fl():
    """Florida Keys at BIN medians, local naive time index, one column per sensor plus O2_avg."""
    path = os.path.join(CACHE_DIR, f"fl_{BIN}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    raw = pd.read_csv(FL_CSV, usecols=["deployment", "t"] + FL_SENSORS)
    day0 = pd.to_datetime(raw["deployment"].str.extract(r"3oec_(\d+)_(\d+)_(\d+)_")
                          .astype(int).astype(str).agg("-".join, axis=1))
    # t counts hours from midnight of the deployment's first day.
    raw["time"] = day0 + pd.to_timedelta(raw["t"] - raw.groupby("deployment")["t"].transform("min").floordiv(24) * 24,
                                         unit="h")
    raw["O2_avg"] = raw[FL_SENSORS].mean(axis=1)
    parts = []
    for name, lo, hi in FL_WINDOWS:
        s = raw.set_index("time").sort_index().loc[lo:hi, FL_SENSORS + ["O2_avg"]]
        b = s.resample(BIN).median().dropna(how="all")
        parts.append(b.assign(deployment=name))
    fl = pd.concat(parts)
    fl.to_parquet(path)
    return fl


def o2_umol_per_l(po2_mmhg, temp_c, salinity=HARVARD_SALINITY):
    """O2 concentration (umol/L) from pO2: C = C_sat(T, S) * pO2 / pO2_sat(T, S).
    C_sat: Garcia & Gordon (1992) combined fit (Benson & Krause data), umol/kg, moist air at 1 atm.
    pO2_sat = 0.20946 (760 - Pwv), Pwv from Weiss & Price (1980) with the salinity term."""
    T = np.asarray(temp_c, float)
    ts = np.log((298.15 - T) / (273.15 + T))
    A = (5.80871, 3.20291, 4.17887, 5.10006, -9.86643e-2, 3.80369)
    B = (-7.01577e-3, -7.70028e-3, -1.13864e-2, -9.51519e-3)
    ln_c = sum(a * ts ** i for i, a in enumerate(A)) + salinity * sum(b * ts ** i for i, b in enumerate(B))         - 2.75915e-7 * salinity ** 2
    tk = T + 273.15
    pwv = np.exp(24.4543 - 67.4509 * (100 / tk) - 4.8489 * np.log(tk / 100) - 0.000544 * salinity) * 760
    return np.exp(ln_c) * SW_DENSITY * np.asarray(po2_mmhg, float) / (0.20946 * (760 - pwv))


def harvard_series():
    """{(signal, period): DataFrame} of Harvard channels at BIN, index in local EDT."""
    g = ce.build_grid(ce.load_polls(), BIN)
    g["o2_umol"] = pd.DataFrame(o2_umol_per_l(g["po2"], g["temp_c"][g["po2"].columns]),
                                index=g["po2"].index, columns=g["po2"].columns)
    g["do_mgl"] = g["do_mgl"] * 31.25   # mg/L -> umol/L, so the check is in the same units
    out = {}
    for sig in [HARVARD_SIGNAL] + HARVARD_CHECK_SIGNALS:
        S = g[sig][ce.CHANNELS]
        for pname, d in ce.periods(S).items():
            out[(sig, pname)] = d.set_axis(d.index - ce.RTC_TO_UTC)
    return out


# ------------------------- measures -------------------------

def fisher_ci(r, neff):
    if not np.isfinite(r) or neff <= 3:
        return np.nan, np.nan
    z, se = np.arctanh(np.clip(r, -0.999999, 0.999999)), 1 / np.sqrt(neff - 3)
    return np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)


def tod_measures(x, idx, groups, demean, boot_L=None, n_boot=ce.N_BOOT):
    """x: values, idx: local DatetimeIndex, groups: day/deployment labels used for demeaning."""
    x = pd.Series(np.asarray(x, float), index=idx)
    if demean:
        x = x - x.groupby(groups).transform("mean")
    m = x.notna().to_numpy()
    hour = idx.hour + idx.minute / 60
    day = ((idx.hour >= DAY_HOURS[0]) & (idx.hour < DAY_HOURS[1])).astype(float)
    xv = x.to_numpy()

    r_dn = np.corrcoef(day[m], xv[m])[0, 1]
    ne_dn = ce.neff_bretherton(int(m.sum()), ce.lag1(pd.Series(day, index=idx)), ce.lag1(x))
    lo, hi = fisher_ci(r_dn, ne_dn)

    w = 2 * np.pi * np.asarray(hour) / 24
    X = np.column_stack([np.ones(len(w)), np.cos(w), np.sin(w)])
    if demean:   # fit the harmonic jointly with group offsets
        X = np.column_stack([X[:, 1:], pd.get_dummies(groups).to_numpy(float)])
    beta, *_ = np.linalg.lstsq(X[m], xv[m], rcond=None)
    fit = X @ beta
    R = np.corrcoef(fit[m], xv[m])[0, 1] if np.nanstd(fit[m]) > 0 else np.nan
    a, b = (beta[0], beta[1]) if demean else (beta[1], beta[2])
    peak = (np.degrees(np.arctan2(b, a)) % 360) / 15
    harm = pd.Series(a * np.cos(w) + b * np.sin(w), index=idx)
    ne_h = ce.neff_bretherton(int(m.sum()), ce.lag1(harm), ce.lag1(x))
    R_lo, R_hi = fisher_ci(R, ne_h)

    hr = pd.Series(idx.hour, index=idx)
    clim = x.groupby(hr).transform("mean")
    r2_hour = 1 - np.nanvar((x - clim)[m]) / np.nanvar(xv[m])

    row = dict(n=int(m.sum()), hours_covered=int(hr[m].nunique()),
               mean_day=np.nanmean(xv[(day == 1) & m]), mean_night=np.nanmean(xv[(day == 0) & m]),
               r_daynight=r_dn, r_daynight_lo=lo, r_daynight_hi=hi, n_eff_daynight=ne_dn,
               harmonic_R=R, harmonic_R_lo=R_lo, harmonic_R_hi=R_hi, harmonic_amp=np.hypot(a, b),
               harmonic_peak_hour=peak, n_eff_harmonic=ne_h, hour_of_day_r2=r2_hour)
    if boot_L:
        bs = np.array([ce.nancorr_cols(xv[ii], day[ii][:, None])[0] for ii in ce.block_indices(len(xv), boot_L, n_boot)])
        row.update(r_daynight_boot_lo=np.nanpercentile(bs, 2.5), r_daynight_boot_hi=np.nanpercentile(bs, 97.5))
    return row


def run():
    for d in (FIG_DIR, TAB_DIR, CACHE_DIR):
        os.makedirs(d, exist_ok=True)
    H = harvard_series()
    fl = load_fl()
    L = ce.bins_per(BIN, ce.BOOT_BLOCK)
    rows = []
    for (sig, pname), d in H.items():
        for c in ce.CHANNELS:
            for demean in (False, True):
                r = tod_measures(d[c], d.index, d.index.floor("1D"), demean, boot_L=L)
                rows.append(dict(dataset="Harvard", series=f"ch{c}", depth=ce.DEPTH[c], signal=sig, period=pname,
                                 variant="demeaned" if demean else "raw", **r))
    for col in ["O2_avg"] + FL_SENSORS:
        for demean in (False, True):
            r = tod_measures(fl[col], fl.index, fl["deployment"].to_numpy(), demean)
            rows.append(dict(dataset="Florida Keys", series=col, depth="", signal="O2", period="all 4 deployments",
                             variant="demeaned" if demean else "raw", **r))
        for name, g in fl.groupby("deployment"):
            r = tod_measures(g[col], g.index, np.zeros(len(g)), False)
            rows.append(dict(dataset="Florida Keys", series=col, depth="", signal="O2", period=name, variant="raw", **r))
    t = pd.DataFrame(rows)
    t.round(4).to_csv(os.path.join(TAB_DIR, "tod_compare.csv"), index=False)
    fig_profiles(H, fl)
    fig_heatmap(t)
    fig_timeseries(H, fl, t)
    return t, H, fl


# ------------------------- figures -------------------------

def fig_profiles(H, fl):
    """Hour-of-day mean profiles, z-scored per series so both datasets share an axis."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    z = lambda s: (s - s.mean()) / s.std()
    for ax, pname, title in ((axes[0], "pre", "Harvard, before Jun 1 2026"), (axes[1], "post", "Harvard, from Jun 1 2026")):
        d = H[(HARVARD_SIGNAL, pname)]
        for c in ce.CHANNELS:
            p = d[c].groupby(d.index.hour).mean()
            if p.std() > 0:
                ax.plot(p.index, z(p), color=ce.C_DEPTH[ce.DEPTH[c]], lw=1.4, ls=LS[c], label=f"ch{c} {ce.DEPTH[c]}")
        ax.set_title(f"{title} (O2, µmol/L)", loc="left")
    ax = axes[2]
    for name, g in fl.groupby("deployment"):
        p = g["O2_avg"].groupby(g.index.hour).mean()
        allp = fl["O2_avg"].groupby(fl.index.hour).mean()
        ax.plot(p.index, (p - allp.mean()) / allp.std(), "o", ms=2.5, color=ce.C_MUTED, alpha=0.7)
    p = fl["O2_avg"].groupby(fl.index.hour).mean()
    p = z(p).reindex(range(24))   # NaN breaks the line across the hours with no data
    ax.plot(p.index, p, "o-", color=C_FL, lw=1.6, ms=3.5, label="all deployments")
    ax.plot([], [], "o", color=ce.C_MUTED, ms=3, label="single deployment")
    ax.set_title("Florida Keys, Jul 2017 (O2_avg, 4 deployments)", loc="left")
    for ax in axes:
        ax.axvspan(DAY_HOURS[0], DAY_HOURS[1], color="#f4e9c9", lw=0, zorder=0)
        ax.set_xlim(0, 23)
        ax.set_xticks(range(0, 24, 3))
        ax.set_xlabel("hour of day (local, EDT)")
        ax.legend(fontsize=6.5, loc="lower left")
    axes[0].set_ylabel("hourly mean O2, z-scored per series")
    fig.suptitle(f"Time-of-day profile of oxygen. Shaded = day window {DAY_HOURS[0]:02d}:00–{DAY_HOURS[1]:02d}:00. "
                 "Florida Keys has no data 07:00–11:59", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    savefig(fig, "tod_01_profiles.png")


def fig_heatmap(t):
    sel = pd.concat([
        t[(t.dataset == "Harvard") & (t.signal == HARVARD_SIGNAL)],
        t[(t.dataset == "Florida Keys") & (t.period == "all 4 deployments")]])
    plab = {"full": "full", "pre": "before Jun 1", "post": "from Jun 1", "all 4 deployments": "all deployments"}
    sel = sel.assign(row=sel.dataset.str.replace("Florida Keys", "FL Keys") + " " + sel.series + " · " + sel.period.map(plab))
    order = [f"Harvard ch{c} · {plab[p]}" for p in ("full", "pre", "post") for c in ce.CHANNELS] + \
            ["FL Keys O2_avg · all deployments"]   # S1-S3 match O2_avg within 0.02 on every measure
    cols, M, A = [], [], []
    for var in ("raw", "demeaned"):
        d = sel[sel.variant == var].set_index("row").loc[order]
        cols += [f"day/night r\n{var}", f"24-h harmonic R\n{var}", f"hour-of-day R²\n{var}"]
        sig_dn = (d.r_daynight_lo > 0) | (d.r_daynight_hi < 0)
        sig_h = d.harmonic_R_lo > 0
        M += [d.r_daynight, d.harmonic_R, d.hour_of_day_r2]
        A += [[f"{v:.2f}{'*' if s else ''}" for v, s in zip(d.r_daynight, sig_dn)],
              [f"{v:.2f}{'*' if s else ''}\npeak {pk:04.1f}h" for v, s, pk in zip(d.harmonic_R, sig_h, d.harmonic_peak_hour)],
              [f"{v:.2f}" for v in d.hour_of_day_r2]]
    M, A = np.array(M, float).T, np.array(A).T
    fig, ax = plt.subplots(figsize=(11, 10))
    sns.heatmap(pd.DataFrame(M, index=order, columns=cols), ax=ax, cmap=ce.DIVERGING, vmin=-1, vmax=1, annot=A, fmt="",
                linewidths=1.0, linecolor="#fcfcfb", annot_kws={"fontsize": 7}, cbar_kws={"label": "r  /  R  /  R²", "shrink": 0.6})
    ax.axvline(3, color=ce.INK2, lw=1.4)
    for y in (6, 12, 18):
        ax.axhline(y, color=ce.INK2, lw=1.4 if y == 18 else 0.6)
    ax.tick_params(axis="x", labelsize=7.5, rotation=0)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.grid(False)
    ax.set_title(f"Oxygen vs time of day: Harvard O2 (Biscayne Bay 2026) and Florida Keys O2 (3OEC 2017), {BIN} bins.\n"
                 f"day/night r: day = {DAY_HOURS[0]:02d}:00–{DAY_HOURS[1]:02d}:00 local. harmonic R: fit to cos/sin of hour, peak = "
                 "phase. * = 95% CI (Fisher z, Bretherton N_eff) excludes 0.\ndemeaned = each day's (Harvard) or deployment's "
                 "(FL) mean removed. Hour-of-day R² overfits short records (FL: 4 partial days)", loc="left", fontsize=8.5)
    fig.tight_layout()
    savefig(fig, "tod_02_heatmap.png")


def fig_timeseries(H, fl, t):
    """O2 concentration through time, each line coloured by day (DAY_HOURS) vs night, with the
    time-of-day correlation of that series in its panel title."""
    def split_plot(ax, idx, y, lw):
        hour = idx.hour
        day = (hour >= DAY_HOURS[0]) & (hour < DAY_HOURS[1])
        y = np.asarray(y, float)
        ax.plot(idx, np.where(day, y, np.nan), color=C_DAY, lw=lw)
        ax.plot(idx, np.where(~day, y, np.nan), color=C_NIGHT, lw=lw)

    def stats_text(row):
        dn = "*" if (row.r_daynight_lo > 0) or (row.r_daynight_hi < 0) else ""
        hm = "*" if row.harmonic_R_lo > 0 else ""
        h = row.harmonic_peak_hour
        return (f"day/night r = {row.r_daynight:.2f}{dn}   ·   24-h cycle R = {row.harmonic_R:.2f}{hm}, "
                f"peak {int(h):02d}:{int(round(h % 1 * 60)) % 60:02d}")

    pick = lambda **kw: t[np.logical_and.reduce([t[k] == v for k, v in kw.items()])].iloc[0]
    H_full = H[(HARVARD_SIGNAL, "full")]
    n = len(ce.CHANNELS)
    fig = plt.figure(figsize=(13, 2.0 * (n + 1) + 0.6))
    gs = fig.add_gridspec(n + 1, 4, hspace=0.95, wspace=0.12, top=0.935, bottom=0.04)
    ax0 = None
    for i, c in enumerate(ce.CHANNELS):
        ax = fig.add_subplot(gs[i, :], sharex=ax0)
        ax0 = ax0 or ax
        split_plot(ax, H_full.index, H_full[c], 0.6)
        ax.axvline(ce.REGIME_SPLIT_UTC - ce.RTC_TO_UTC, color=ce.INK2, lw=0.8, ls="--")
        row = pick(dataset="Harvard", series=f"ch{c}", signal=HARVARD_SIGNAL, period="full", variant="raw")
        ax.set_title(f"Harvard {ce.lab(c)} ({ce.DEPTH[c]})   —   {stats_text(row)}", loc="left", fontsize=8.5)
        ax.set_ylabel("µmol/L")
        ce.utc_axis(ax)
        if i < n - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(f"date (local, {ce.LOCAL_LABEL}); dashed = Jun 1", fontsize=8)
    row = pick(dataset="Florida Keys", series="O2_avg", period="all 4 deployments", variant="raw")
    axs = [fig.add_subplot(gs[n, k]) for k in range(4)]
    for k, (ax, (name, g)) in enumerate(zip(axs, fl.groupby("deployment"))):
        split_plot(ax, g.index, g["O2_avg"], 1.4)
        ax.set_title(f"{name}: {g.index[0]:%b %d %H:%M} → {g.index[-1]:%b %d %H:%M}", loc="left", fontsize=8)
        ax.xaxis.set_major_locator(ce.mdates.HourLocator(byhour=range(0, 24, 6)))
        ax.xaxis.set_major_formatter(ce.mdates.DateFormatter("%H:%M"))
        ax.set_ylim(fl["O2_avg"].min() - 2, fl["O2_avg"].max() + 2)
        if k:
            ax.tick_params(labelleft=False)
    axs[0].set_ylabel("µmol/L")
    fig.text(0.125, axs[0].get_position().y1 + 0.022,
             f"Florida Keys 3OEC, Jul 2017 (mean of 3 sensors), all deployments pooled   —   {stats_text(row)}",
             fontsize=8.5, ha="left")
    fig.legend(handles=[plt.Line2D([], [], color=C_DAY, lw=2), plt.Line2D([], [], color=C_NIGHT, lw=2)],
               labels=[f"day {DAY_HOURS[0]:02d}:00–{DAY_HOURS[1]:02d}:00 local", "night"],
               loc="upper left", ncol=2, fontsize=8, bbox_to_anchor=(0.12, 0.982))
    fig.suptitle(f"Oxygen concentration over time, coloured by time of day ({BIN} medians). Harvard: pO2 converted "
                 f"at S = {HARVARD_SALINITY:g}. * = 95% CI excludes 0 (Bretherton N_eff)",
                 x=0.125, y=0.995, ha="left", fontsize=10)
    savefig(fig, "tod_03_timeseries.png")


if __name__ == "__main__":
    tab, _, _ = run()
    pd.set_option("display.width", 250)
    cols = ["dataset", "series", "period", "variant", "n", "hours_covered", "r_daynight", "r_daynight_lo", "r_daynight_hi",
            "harmonic_R", "harmonic_R_lo", "harmonic_peak_hour", "hour_of_day_r2"]
    print(tab[tab.signal.isin([HARVARD_SIGNAL, "O2"])][cols].round(2).to_string())
