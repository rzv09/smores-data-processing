"""Compare every baseline on one axis, including the two added after the first run.

`smores.models.baselines` originally scored six models. `window_average` and
`historic_average` were used in exploratory notebooks but never scored under the
same protocol, so they could not be ranked against the rest -- a trailing mean
evaluated on a single holdout is not comparable to a ridge scored
leave-one-deployment-out. Both are now in `baselines.MODELS`, which is what makes
this figure a like-for-like comparison rather than a collage.

Two figures, never one: RMSE is in umol/L and MARE is a unitless ratio, so they
cannot share an axis and must not be drawn as a dual-axis pair.

Form
----
Eight models is past the point where color can carry identity -- only the first
three categorical slots validate all-pairs for colorblind separation. Identity
therefore lives in the y-axis labels and every bar takes a single hue; the one
thing color marks is the `persistence` reference, since "beats persistence" is
the question each panel is really answering.

Usage:
    python -m smores.viz.plot_baseline_comparison
    python -m smores.viz.plot_baseline_comparison --freq 2min
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from smores import config
from smores.models import baselines

# Slot 1 of the validated reference palette, plus the repo's neutral ink ramp.
BAR = "#2a78d6"
REFERENCE = "#898781"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

# label, long name, and a formatter for the direct labels. MARE is a ratio in the
# 0.006-0.02 range here, so it is written as a percentage; two decimals of the raw
# ratio would collapse every model onto "0.01".
METRICS = {
    "rmse": ("RMSE (umol/L)", "root mean squared error", lambda v: f"{v:.2f}"),
    "mare": ("MARE (%)", "mean absolute relative error", lambda v: f"{v * 100:.2f}%"),
}


def panel(ax, sub, order, metric, ref_value, fmt):
    """One horizon's ranked bars, drawn in the shared `order`."""
    values = sub.set_index("model")[metric].reindex(order)
    y = np.arange(len(order))

    colors = [REFERENCE if name == "persistence" else BAR for name in order]
    ax.barh(y, values.to_numpy(), height=0.72, color=colors, zorder=3)

    if np.isfinite(ref_value):
        ax.axvline(ref_value, color=REFERENCE, lw=1.0, ls="--", zorder=4)

    span = np.nanmax(values.to_numpy()) or 1.0
    for yi, value in zip(y, values.to_numpy()):
        if not np.isfinite(value):
            ax.annotate("no valid samples", xy=(span * 0.02, yi), va="center",
                        fontsize=7, color=MUTED, style="italic")
            continue
        ax.annotate(fmt(value), xy=(value, yi), xytext=(4, 0),
                    textcoords="offset points", va="center",
                    fontsize=7, color=MUTED)

    if metric == "mare":
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))

    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=8)
    ax.invert_yaxis()
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="x", color=GRID, lw=0.8, zorder=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8)
    # Headroom so the right-most value label is not clipped by the spine.
    ax.set_xlim(0, span * 1.22)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def figure(pooled, freq, metric, out):
    label, long_name, fmt = METRICS[metric]
    sub = pooled[(pooled["freq"] == freq) & (pooled["input"] == baselines.MEAN3)]
    horizons = sorted(sub["horizon_min"].unique())

    # One shared model order across panels, ranked at the horizon the report reads
    # first, so a bar keeps its row and the eye can track a model across horizons.
    anchor = 30 if 30 in horizons else horizons[0]
    order = (sub[sub["horizon_min"] == anchor]
             .sort_values(metric)["model"].tolist())
    for name in baselines.MODELS:
        if name not in order:
            order.append(name)

    fig, axes = plt.subplots(
        1, len(horizons), figsize=(4.0 * len(horizons), 4.6),
        sharey=True, facecolor=SURFACE,
    )
    axes = [axes] if len(horizons) == 1 else list(axes)

    for ax, horizon in zip(axes, horizons):
        rows = sub[sub["horizon_min"] == horizon]
        ref = rows[rows["model"] == "persistence"][metric]
        panel(ax, rows, order, metric, float(ref.iloc[0]) if len(ref) else np.nan, fmt)
        ax.set_title(f"h = {horizon} min", fontsize=9, color=INK, loc="left")
        ax.set_xlabel(label, fontsize=8, color=MUTED)

    fig.suptitle(
        f"Baseline {long_name} on {freq} bins, leave-one-deployment-out "
        "(dashed line = persistence)",
        fontsize=11, color=INK, x=0.01, ha="left",
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def protocol_figure(pooled, split, freq, metric, out):
    """The same models under both protocols, side by side.

    The two answer different questions -- leave-one-deployment-out asks whether a model
    transfers to a deployment it has never seen, the 80/20 split asks whether it can
    forecast the tail of a deployment it has already learned from. A model that wins
    under both is not winning on an artifact of either, which is the only claim this
    figure is making.

    Bars are grouped per model rather than ranked separately per protocol, because the
    comparison of interest runs across the pair, not down either column.
    """
    # Skill, not absolute error. The two protocols score different rows -- the 80/20 test
    # blocks are the tail ends of deployments, which are flatter than the record as a
    # whole, so every model's RMSE drops under the split for reasons that have nothing to
    # do with the model. Skill is measured against the persistence run under the *same*
    # protocol, which is the only quantity that survives the comparison.
    metric = "skill_vs_persistence"
    label = "skill vs persistence  (1 - RMSE/RMSE_persist)"
    long_name = "forecast skill"
    def fmt(v):
        return f"{v:+.0%}"

    arms = []
    for name, frame in (("leave-one-deployment-out", pooled), ("80/20 split", split)):
        sub = frame[(frame["freq"] == freq) & (frame["horizon_min"] == 30)
                    & (frame["input"] == baselines.MEAN3)]
        if sub.empty:
            return
        arms.append((name, sub.set_index("model")[metric]))

    order = arms[0][1].sort_values(ascending=False).index.tolist()
    for name in baselines.MODELS:
        if name not in order:
            order.append(name)
    y = np.arange(len(order))
    height = 0.8 / len(arms)

    fig, ax = plt.subplots(figsize=(9, 5.0), facecolor=SURFACE)
    for i, (name, values) in enumerate(arms):
        offset = (i - (len(arms) - 1) / 2) * height
        shades = [BAR, "#8fb9ea"][i]
        ax.barh(y + offset, values.reindex(order).to_numpy(), height * 0.88,
                color=shades, label=name, zorder=3)

    for i, (_, values) in enumerate(arms):
        offset = (i - (len(arms) - 1) / 2) * height
        for yi, value in zip(y, values.reindex(order).to_numpy()):
            if np.isfinite(value):
                right = value >= 0
                ax.annotate(fmt(value), xy=(value, yi + offset),
                            xytext=(3 if right else -3, 0),
                            textcoords="offset points", va="center",
                            ha="left" if right else "right",
                            fontsize=6.5, color=MUTED)

    ax.axvline(0, color=MUTED, lw=1.0, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=8)
    ax.invert_yaxis()
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="x", color=GRID, lw=0.8, zorder=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel(label, fontsize=9, color=MUTED)
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="upper left")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    fig.suptitle(
        f"Baseline {long_name} at h = 30 min on {freq} bins — right of the line beats "
        "persistence under that protocol",
        fontsize=11, color=INK, x=0.01, ha="left",
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freq", default="2min")
    p.add_argument("--out-dir", type=Path,
                   default=config.DATA_DIR / "processed" / "figures")
    args = p.parse_args()

    frames = baselines.load_wide(freq=args.freq)
    per_fold = baselines.optode_arms(frames, args.freq)
    pooled = baselines.pool(per_fold)
    split = baselines.pool(baselines.evaluate_split(frames, args.freq))

    for metric in METRICS:
        figure(pooled, args.freq, metric,
               args.out_dir / f"baseline_comparison_{metric}_{args.freq}.png")
    if not split.empty:
        protocol_figure(pooled, split, args.freq, "rmse",
                        args.out_dir / f"protocol_comparison_skill_{args.freq}.png")

    shown = pooled[(pooled["input"] == baselines.MEAN3)
                   & (pooled["horizon_min"] == 30)]
    print()
    print(shown[["model", "rmse", "mae", "mare", "skill_vs_persistence",
                 "folds_beating_persistence"]].to_string(index=False))


if __name__ == "__main__":
    main()
