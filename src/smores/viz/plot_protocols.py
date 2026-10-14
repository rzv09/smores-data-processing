"""Diagram of the two evaluation protocols, drawn to scale from the real deployments.

Both protocols are reported throughout `data/baselines/`, and the difference between
them is the single thing most often misread: leave-one-deployment-out holds back a whole
recording, the 80/20 split holds back the tail of every recording. Those answer
different questions, and their RMSE values are not interchangeable.

Block widths are the actual bin counts, so the figure also shows what the tables do not
-- that the deployments are uneven (421 to 601 bins), which is why the 80/20 test blocks
differ in size and why a single pooled number can hide a bad fold.

Usage:
    python -m smores.viz.plot_protocols
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from smores import config
from smores.models import baselines

TRAIN = "#c3d6f0"  # slot 1, stepped light: recedes behind the test blocks
TEST = "#2a78d6"  # slot 1
INK = "#0b0b0b"
MUTED = "#898781"
SURFACE = "#fcfcfb"

BAR_H = 0.62


def short(name):
    """`3oec_2017_7_11_12` -> `11_12`, which is how the group refers to them."""
    parts = name.split("_")
    return "_".join(parts[-2:]) if len(parts) >= 2 else name


def draw_lodo(ax, names, sizes):
    """One row per round; the held-out deployment is the only test block in that row."""
    total = sum(sizes)
    for round_index in range(len(names)):
        y = len(names) - 1 - round_index
        x = 0.0
        for col, (name, size) in enumerate(zip(names, sizes)):
            is_test = col == round_index
            ax.barh(y, size, left=x, height=BAR_H, zorder=3,
                    color=TEST if is_test else TRAIN)
            ax.annotate(short(name), xy=(x + size / 2, y), ha="center", va="center",
                        fontsize=7, color="#ffffff" if is_test else MUTED, zorder=4)
            x += size
        ax.annotate(f"round {round_index + 1}", xy=(total * 1.01, y), va="center",
                    fontsize=7.5, color=MUTED)

    ax.set_yticks([])
    ax.set_xlim(0, total * 1.14)
    ax.set_title("leave-one-deployment-out\nhold back a whole recording, rotate",
                 fontsize=9, color=INK, loc="left", pad=8)


def draw_split(ax, names, sizes, fraction):
    """One row per deployment, each cut chronologically at `fraction`."""
    widest = max(sizes)
    for index, (name, size) in enumerate(zip(names, sizes)):
        y = len(names) - 1 - index
        cut = int(round(size * fraction))
        ax.barh(y, cut, left=0, height=BAR_H, color=TRAIN, zorder=3)
        ax.barh(y, size - cut, left=cut, height=BAR_H, color=TEST, zorder=3)
        ax.annotate(f"train {cut}", xy=(cut / 2, y), ha="center", va="center",
                    fontsize=7, color=MUTED, zorder=4)
        ax.annotate(f"{size - cut}", xy=(cut + (size - cut) / 2, y), ha="center",
                    va="center", fontsize=7, color="#ffffff", zorder=4)
        ax.annotate(short(name), xy=(widest * 1.02, y), va="center",
                    fontsize=7.5, color=MUTED)

    ax.set_yticks([])
    ax.set_xlim(0, widest * 1.16)
    ax.set_title("80/20 split\nhold back the tail of every recording",
                 fontsize=9, color=INK, loc="left", pad=8)


def plot(out, freq="2min", fraction=baselines.SPLIT_FRACTION):
    frames = baselines.load_wide(freq=freq)
    names = sorted(frames, key=lambda n: frames[n].index[0])
    sizes = [len(frames[n]) for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), facecolor=SURFACE)
    draw_lodo(axes[0], names, sizes)
    draw_split(axes[1], names, sizes, fraction)

    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.set_ylim(-0.6, len(names) - 0.4)
        ax.tick_params(colors=MUTED, labelsize=7)
        ax.set_xlabel(f"bins ({freq})", fontsize=8, color=MUTED)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#e1e0d9")

    fig.legend(
        handles=[Patch(facecolor=TRAIN, label="train"),
                 Patch(facecolor=TEST, label="test")],
        frameon=False, fontsize=8, labelcolor=MUTED,
        loc="upper right", ncols=2, bbox_to_anchor=(0.995, 0.995),
    )
    fig.suptitle(
        "Two evaluation protocols, block widths to scale "
        "— skill is comparable across them, raw RMSE is not",
        fontsize=11, color=INK, x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")
    for name, size in zip(names, sizes):
        cut = int(round(size * fraction))
        print(f"  {name:<20} {size:>4} bins -> train {cut:>3} / test {size - cut:>3}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freq", default="2min")
    p.add_argument("--fraction", type=float, default=baselines.SPLIT_FRACTION)
    p.add_argument(
        "--out", type=Path,
        default=config.DATA_DIR / "processed" / "figures" / "protocols_2min.png",
    )
    args = p.parse_args()
    plot(args.out, args.freq, args.fraction)


if __name__ == "__main__":
    main()