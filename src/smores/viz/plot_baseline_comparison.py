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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freq", default="2min")
    p.add_argument("--out-dir", type=Path,
                   default=config.DATA_DIR / "processed" / "figures")
    args = p.parse_args()

    frames = baselines.load_wide(freq=args.freq)
    per_fold = baselines.optode_arms(frames, args.freq)
    pooled = baselines.pool(per_fold)

    for metric in METRICS:
        figure(pooled, args.freq, metric,
               args.out_dir / f"baseline_comparison_{metric}_{args.freq}.png")

    shown = pooled[(pooled["input"] == baselines.MEAN3)
                   & (pooled["horizon_min"] == 30)]
    print()
    print(shown[["model", "rmse", "mae", "mare", "skill_vs_persistence",
                 "folds_beating_persistence"]].to_string(index=False))


if __name__ == "__main__":
    main()
