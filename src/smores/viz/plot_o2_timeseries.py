"""Plot a dissolved-oxygen timeseries from a resampled SMORES parquet file.

One panel per deployment, side by side in a single figure, so the hours when no
instrument was in the water are not drawn as blank axis. Panel widths are
proportional to deployment duration, so an hour is the same width everywhere and
the diel slopes stay comparable between panels.

Time
----
The parquet stores UTC (the schema requires it), but the instrument logged local
wall-clock and the signal being plotted is a photosynthetic diel cycle. Axes are
therefore converted to `DISPLAY_TZ` for display only; nothing is written back.

Timestamps are resample bin *left edges* -- `smores.data.resample.resample_long`
labels each bin with its start and fills it with the median over the bin. At the
2 min default that is a 1 minute lead, well below anything readable here. It is
worth remembering at coarser `--freq`: on hourly bins the same convention puts a
point labelled 13:00 at the centre of 13:00-14:00, a 30 minute lead.

Oxygen
------
`--column` selects which column is drawn; `do_mgl` (default) and `do_sat` are
both understood. Plotting both as separate curves would be misleading: per
`smores.data.units`, both are linear in the raw umol/L reading with zero
intercept, so the two differ by a constant factor alone and the curves would be
identical in shape. Whichever is *not* plotted is therefore shown as a secondary
axis on the right -- a unit mirror of one quantity, not a second scale. The
choice of column only changes which unit gets the ticks the eye reads first.

Any other numeric column may be passed too; it is plotted without a mirror axis.

Usage:
    python -m smores.viz.plot_o2_timeseries data/processed/florida_keys_o2_2min.parquet
    python -m smores.viz.plot_o2_timeseries <parquet> --column do_sat
    python -m smores.viz.plot_o2_timeseries <parquet> --out fig.png --dark
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from smores.data.florida_keys import O2SOL_SITE_UMOL_L
from smores.data.units import O2_MG_PER_UMOL

# The clock the instrument actually logged; see the module docstring.
DISPLAY_TZ = "America/New_York"

# Hours bounding the shaded night band, in DISPLAY_TZ.
#
# Not civil twilight: mid-July Keys sunset is ~20:20 and sunrise ~06:40, so 19:00
# shades ~80 min of real daylight. These are instead the observed turning points of
# this record -- oxygen peaks in the 19:00-20:00 bin and bottoms at 06:00-07:00 --
# which is the more useful boundary for reading the curve.
NIGHT_START = 19
NIGHT_END = 7

# Categorical slots 1-3 (blue / orange / aqua), validated for all-pairs CVD
# separation in both light and dark modes.
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70"]

THEME_LIGHT = {
    "series": SERIES_LIGHT,
    "surface": "#fcfcfb",
    "text": "#141413",
    "muted": "#6e6d66",
    "grid": "#e5e4df",
    "night": ("#141413", 0.06),  # neutral, never a hue: must not read as a series
}
THEME_DARK = {
    "series": SERIES_DARK,
    "surface": "#1a1a19",
    "text": "#ffffff",
    "muted": "#c3c2b7",
    "grid": "#36352f",
    "night": ("#ffffff", 0.05),
}

VALUE_COLUMN = "do_mgl"

# do_mgl -> % air saturation, from units.mgl_from_saturation inverted.
MGL_PER_PERCENT = O2SOL_SITE_UMOL_L * O2_MG_PER_UMOL / 100.0

# Per-column presentation. `mirror_per_unit` is how many of this column's units
# make up one unit of the mirror axis, so the secondary axis is value /
# mirror_per_unit; `mirror_tick` is the spacing of its ticks in mirror units.
# Since do_mgl and do_sat are the same measurement in different units, each is
# the other's mirror -- see the module docstring.
COLUMN_SPECS = {
    "do_mgl": {
        "label": "Dissolved oxygen (mg/L)",
        "unit": "mg/L",
        "decimals": 2,
        "mirror_label": "% air saturation",
        "mirror_per_unit": MGL_PER_PERCENT,
        "mirror_tick": 5,
        "mirror_decimals": 0,
    },
    "do_sat": {
        "label": "Dissolved oxygen (% air saturation)",
        "unit": "%",
        "decimals": 1,
        "mirror_label": "mg/L",
        "mirror_per_unit": 1.0 / MGL_PER_PERCENT,
        "mirror_tick": 0.25,
        "mirror_decimals": 2,
    },
}


def column_spec(value_col: str) -> dict:
    """Presentation for `value_col`, falling back to a bare unitless spec."""
    if value_col in COLUMN_SPECS:
        return COLUMN_SPECS[value_col]
    return {"label": value_col, "unit": "", "decimals": 3, "mirror_label": None}


def load_panels(path: Path, value_col: str = VALUE_COLUMN):
    """Yield `(deployment_id, wide_frame)` in chronological order.

    Each frame is indexed by local-time timestamp with one column per sensor. No
    reindexing onto a full grid is needed: within a deployment the 8 Hz source has
    no gaps, so the resampled grid is already complete.
    """
    df = pd.read_parquet(path, columns=["timestamp", "deployment_id", "sensor_id", value_col])
    df["timestamp"] = df["timestamp"].dt.tz_convert(DISPLAY_TZ)

    panels = []
    for name, group in df.groupby("deployment_id", observed=True, sort=False):
        wide = group.pivot_table(
            index="timestamp", columns="sensor_id", values=value_col, observed=True
        ).sort_index()
        # pivot_table drops all-NaN rows, so a column the pipeline never filled
        # arrives here as an empty frame. Skipping it keeps the failure legible
        # rather than an IndexError off index[0] further down.
        if not wide.empty:
            panels.append((name, wide))
    if not panels:
        raise ValueError(f"{value_col!r} has no non-null values in {path}")
    panels.sort(key=lambda item: item[1].index[0])
    return panels


def shade_night(ax, start, end, theme):
    """Shade NIGHT_START->NIGHT_END on every night the panel touches."""
    color, alpha = theme["night"]
    day = start.normalize() - pd.Timedelta(days=1)
    bands = []
    while day <= end.normalize() + pd.Timedelta(days=1):
        dusk = day + pd.Timedelta(hours=NIGHT_START)
        dawn = day + pd.Timedelta(days=1, hours=NIGHT_END)
        if dawn > start and dusk < end:
            ax.axvspan(dusk, dawn, color=color, alpha=alpha, lw=0, zorder=0)
            bands.append(max(dusk, start))
        day += pd.Timedelta(days=1)

    if bands:
        # Anchored at the bottom: the top of a panel is where the evening peak and
        # its annotation live, and this label must not land on them.
        ax.annotate(
            "night",
            xy=(mdates.date2num(min(bands)), 0.0),
            xycoords=("data", "axes fraction"),
            xytext=(4, 5),
            textcoords="offset points",
            color=theme["muted"],
            fontsize=7,
        )


def annotate_extremes(ax, wide, theme, spec):
    """Mark the panel's min and max with their exact local time and value."""
    centre = wide.mean(axis=1)
    unit = f" {spec['unit']}" if spec["unit"] else ""
    for which, when in (("max", centre.idxmax()), ("min", centre.idxmin())):
        value = centre.loc[when]
        above = which == "max"
        ax.plot(
            [when], [value], marker="o", ms=4, mfc="none",
            mec=theme["muted"], mew=1.0, zorder=5,
        )
        ax.annotate(
            f"{value:.{spec['decimals']}f}{unit}\n{when:%b %d %H:%M}",
            xy=(when, value),
            xytext=(0, 16 if above else -28),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color=theme["muted"],
            arrowprops=dict(arrowstyle="-", color=theme["grid"], lw=0.8),
        )


def label_right(ax, wide, colors, min_gap_frac: float = 0.06):
    """Direct-label each series at its last observation, nudged apart vertically.

    The sensors are co-located replicates reading within ~0.01 mg/L of each other,
    so plain labels would stack on top of one another.
    """
    lo, hi = ax.get_ylim()
    min_gap = (hi - lo) * min_gap_frac
    ends = []
    for name, color in zip(wide.columns, colors):
        series = wide[name].dropna()
        if series.empty:
            continue
        ends.append([series.index[-1], float(series.iloc[-1]), name, color])
    ends.sort(key=lambda row: row[1])
    for prev, cur in zip(ends, ends[1:]):
        cur[1] = max(cur[1], prev[1] + min_gap)
    for x, y, name, color in ends:
        ax.annotate(
            name, xy=(x, y), xytext=(4, 0), textcoords="offset points",
            color=color, fontsize=7, va="center", clip_on=False,
        )


def print_extrema(panels, spec):
    """Per-deployment extrema with exact local timestamps, and sensor spread."""
    unit = f" ({spec['unit']})" if spec["unit"] else ""
    header = (
        f"{'deployment':<20} {'min' + unit:>22} {'max' + unit:>22} "
        f"{'range':>7} {'spread':>7}"
    )
    print(header)
    print("-" * len(header))
    for name, wide in panels:
        centre = wide.mean(axis=1)
        lo_at, hi_at = centre.idxmin(), centre.idxmax()
        lo, hi = centre.loc[lo_at], centre.loc[hi_at]
        spread = (wide.max(axis=1) - wide.min(axis=1)).mean()
        print(
            f"{name:<20} "
            f"{lo:>8.3f} @ {lo_at:%m-%d %H:%M} "
            f"{hi:>8.3f} @ {hi_at:%m-%d %H:%M} "
            f"{hi - lo:>7.3f} {spread:>7.4f}"
        )


def plot(path: Path, out: Path, dark: bool = False, value_col: str = VALUE_COLUMN):
    theme = THEME_DARK if dark else THEME_LIGHT
    colors = theme["series"]
    spec = column_spec(value_col)
    panels = load_panels(path, value_col)

    # One hour is the same width in every panel; without this the shortest
    # deployment is stretched and its diel slopes read as steeper than they are.
    spans = [(wide.index[-1] - wide.index[0]).total_seconds() for _, wide in panels]

    fig, axes = plt.subplots(
        1, len(panels), figsize=(14, 4.5), sharey=True,
        gridspec_kw={"width_ratios": spans, "wspace": 0.06},
        constrained_layout=True,
    )
    axes = [axes] if len(panels) == 1 else list(axes)
    fig.patch.set_facecolor(theme["surface"])

    # Fix the shared y-range up front, with headroom for the extrema annotations:
    # the evening maximum sits near the top of every panel and its label needs
    # somewhere to go that is not the title. Set before the loop so label_right
    # sizes its anti-collision gap against the final range.
    lo = min(wide.min().min() for _, wide in panels)
    hi = max(wide.max().max() for _, wide in panels)
    pad = (hi - lo) * 0.12
    axes[0].set_ylim(lo - pad, hi + pad * 1.4)

    for index, (ax, (name, wide)) in enumerate(zip(axes, panels)):
        start, end = wide.index[0], wide.index[-1]
        ax.set_facecolor(theme["surface"])
        shade_night(ax, start, end, theme)

        for sensor, color in zip(wide.columns, colors):
            ax.plot(wide.index, wide[sensor], color=color, lw=1.2, label=sensor, zorder=3)

        ax.set_xlim(start, end)
        # tz= on every locator and formatter is load-bearing: matplotlib's date
        # machinery works in UTC and ignores the tz carried by the data, so without
        # it the ticks read UTC under local-time data -- a silent 4 h lie.
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 3), tz=DISPLAY_TZ))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=DISPLAY_TZ))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1, tz=DISPLAY_TZ))
        ax.grid(True, which="major", color=theme["grid"], lw=0.7, zorder=1)
        ax.grid(True, which="minor", color=theme["grid"], lw=0.3, alpha=0.6, zorder=1)
        # A heavier rule at midnight, so the date boundary is visible inside the panel.
        for midnight in pd.date_range(start.ceil("D"), end, freq="D", tz=DISPLAY_TZ):
            ax.axvline(midnight, color=theme["muted"], lw=0.8, alpha=0.5, zorder=2)

        ax.set_axisbelow(True)
        ax.tick_params(colors=theme["muted"], labelsize=7)
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment("right")

        # Interior spines hidden so the four read as one broken axis.
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(theme["grid"])

        # %-d is a glibc extension and raises on Windows; strip the zero by hand.
        if start.date() == end.date():
            dates = f"{start:%b} {start.day}"
        elif start.month == end.month:
            dates = f"{start:%b} {start.day}–{end.day}"
        else:
            dates = f"{start:%b} {start.day} – {end:%b} {end.day}"
        ax.set_title(f"{name}\n{dates}", color=theme["text"], fontsize=8, loc="left", pad=8)

        annotate_extremes(ax, wide, theme, spec)
        label_right(ax, wide, colors)

        if index == 0:
            ax.set_ylabel(spec["label"], color=theme["text"], fontsize=9)
            ax.spines["left"].set_visible(True)
            ax.spines["left"].set_color(theme["grid"])
            # Upper left: the lower left is where the night label sits, and the
            # pre-dawn minimum is the one part of the curve that reaches down there.
            ax.legend(frameon=False, ncols=3, loc="upper left", fontsize=7,
                      labelcolor=theme["muted"], columnspacing=1.0, handlelength=1.4)

    # Unit mirror of the same quantity -- not a second measure on a second scale.
    # Skipped for columns with no known second unit.
    if spec["mirror_label"]:
        per_unit = spec["mirror_per_unit"]
        secondary = axes[-1].secondary_yaxis(
            "right",
            functions=(lambda v: v / per_unit, lambda m: m * per_unit),
        )
        secondary.set_ylabel(spec["mirror_label"], color=theme["muted"], fontsize=8)
        secondary.yaxis.set_major_locator(mticker.MultipleLocator(spec["mirror_tick"]))
        secondary.yaxis.set_major_formatter(
            mticker.FormatStrFormatter(f"%.{spec['mirror_decimals']}f")
        )
        secondary.tick_params(colors=theme["muted"], labelsize=7)
        secondary.spines["right"].set_color(theme["grid"])

    fig.suptitle(
        f"{path.stem} — {spec['label'].lower()} by sensor, shaded {NIGHT_START}:00–"
        f"{NIGHT_END:02d}:00 local",
        color=theme["text"], fontsize=11, x=0.01, ha="left",
    )
    fig.supxlabel("Local time (EDT)", color=theme["muted"], fontsize=9)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=theme["surface"])
    print(f"wrote {out}\n")
    print_extrema(panels, spec)
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("parquet", type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--column",
        default=VALUE_COLUMN,
        help=f"value column to plot; {' or '.join(COLUMN_SPECS)} carry units (default: %(default)s)",
    )
    p.add_argument("--dark", action="store_true")
    args = p.parse_args()

    # Suffix the non-default column so a do_sat run does not overwrite the do_mgl figure.
    stem = args.parquet.stem
    if args.column != VALUE_COLUMN:
        stem = f"{stem}_{args.column}"
    out = args.out or Path("data/processed/figures") / f"{stem}.png"
    plot(args.parquet, out, dark=args.dark, value_col=args.column)


if __name__ == "__main__":
    main()
