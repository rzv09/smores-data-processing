"""Biscayne Bay (SMORES 2026): raw Harvard chunk dump -> harmonized schema.

Source
------
`data/raw/biscayne_bay/FULL data from Harvard team/`, the SMORES pier array at the
FIU Biscayne Bay campus (25.9113 N, -80.1373 W), May 13 - June 21, 2026. One logger
(serial_id 0) polls 26 In-Situ RDO optodes on Modbus addresses 2-28 about every 52 s,
plus a Bar100 pressure/temperature sensor at 5 Hz, an IMU, a camera and an internal
housing sensor. Only the RDO and Bar100 streams feed the schema.

Files are `SENS_<prefix>_<chunk>_<Kind>.csv`. `prefix` is a logger session counter and
chunk numbers repeat across prefixes, so a chunk is keyed by (prefix, chunk), never by
chunk alone. Every file's `timestamp` is a raw tick counter (~1.003e8 ticks/s), not
wall time.

Clock
-----
The handling here is the one measured and validated in `data/eda/REPORT.md`:

* A chunk with its own `_RTC.csv` gets its own linear tick -> wall-time fit. Tick rate
  drifts ~0.25% chunk to chunk, so one global fit would not do. Residuals are <= 0.33 s.
* A chunk without an RTC file borrows the fit of the nearest own-RTC chunk in the same
  prefix, but only if it is tick-continuous with it (no counter reset, within 2 h) and
  the extrapolated span lands between its neighbours in wall time.
* Chunks that cannot be placed (~0.5% of reads) are dropped and counted.

The RTC records local EDT, not UTC: the Bar100 submersion step falls at 20:45 RTC
against a logged 21:30 EDT deployment, water temperature peaks at 16:00 RTC, and
midwater pO2 bottoms out at 04:00-07:00 RTC. Output is UTC (RTC + 4 h).

Oxygen
------
Each poll reads four parameters: temperature (param 1, C), mg/L (20), % saturation (21)
and pO2 (30, mmHg). pO2 is the direct measurement and the native value. The sensor
derives % sat from it as `100 * pO2 / (0.20946 * (760 - Pwv(T)))` with Weiss & Price
(1980) vapour pressure at salinity 0, then clips at 0. `do_sat` recomputes that same
formula from unclipped pO2, so it matches the sensor to a ratio of 1.0000 wherever the
sensor is above 0 (checked in `site_checks`), and keeps the real negative floor of
anoxic buried sensors instead of tying it at 0.

The sensor's own mg/L assumes salinity 0, which overstates concentration in bay water
by ~25%. `do_mgl`, the schema's primary signal, is instead the in-situ concentration
implied by `do_sat` and seawater solubility at the optode's temperature and an assumed
salinity of 35 (there is no CTD on the array; this matches the SMORES plan's default).
Each salinity unit off moves `do_mgl` by ~0.6%, so a bay reading of 30 instead of 35
would put it ~3% low; relative structure is unaffected. `do_sat` does not depend on
the assumption, and `do_native_value` keeps the raw pO2.

Quality
-------
`qc_flag` is per poll:

* 3 masked -- the in-situ (channel, local day) is not "live" by the SMORES plan's liveness
  classification (`smores.ingest_smores.qc.classify_liveness`): flatlined at zero or
  stuck constant. 18 of 26 optodes died within an hour of burial, so most channels
  are masked for most of the record. Values are kept, not deleted.
* 2 flagged -- an oxygen parameter of the poll carries a failing sensor quality code
  (3, 5 or 7). ch13 fails on every read from May 15.
* 0 good otherwise.

Deployments and depth
---------------------
Readings before the logged 21:30 EDT deployment on May 13 are the pre-deployment
bucket test: optodes in air-saturated water, the end-to-end check of the pO2 -> % sat
conversion. They are kept, under their own `deployment_id`, so they never mix with
in-situ data and can serve as a calibration segment.

`depth_m` is water depth over the Bar100 on the lander, from its gauge pressure (it
reads 0 bar in air), binned to the same 2-minute grid and shared by every channel. It
is not each optode's own depth: buried optodes sit 8.3 or 16.5 cm below the seafloor
and the midwater ones above it, and the file metadata carries no per-sensor depth.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from smores import config
from smores.data import integrity, resample, schema
from smores.data.units import mgl_from_saturation, o2sol_umol_l, pressure_kpa_to_depth_m
from smores.ingest_smores.qc import classify_liveness, pct_sat_from_po2

SITE_ID = "biscayne_bay"
DATA_SOURCE = "smores_2026"
SOURCE_TZ = "America/New_York"

SITE_LATITUDE = 25.9113
SITE_LONGITUDE = -80.1373

DEPLOYMENT_START_LOCAL = pd.Timestamp("2026-05-13 21:30:00")
DEPLOYMENT_ID = "smores_2026_05_13"
BUCKET_DEPLOYMENT_ID = "smores_2026_05_13_bucket"

ASSUMED_SALINITY = 35.0
SENSOR_SALINITY = 0.0  # the optodes' own % sat setting
DENSITY_SEAWATER_KG_M3 = 1022.0
DO_NATIVE_UNIT = "pO2_mmHg"

# param -> (column, expected In-Situ units code)
PARAMS = {
    1: ("temp_c", 1),
    20: ("do_mgl_reported", 117),
    21: ("pctsat_reported", 177),
    30: ("po2", 26),
}
OXYGEN_PARAMS = (20, 21, 30)
FAILING_QUALITY_CODES = (3, 5, 7)

NOMINAL_TICKS_PER_S = 1.003e8
MAX_BORROW_GAP_S = 2 * 3600
POLL_GAP_S = 5.0  # params within a poll are < 1 s apart; polls are ~52 s apart

MIDWATER_CHANNELS = (2, 3, 25)
# The optodes that stayed live in situ (data/eda/REPORT.md): midwater ch2, 3, 25,
# upper-sediment ch7, 26, and ch8. `pipeline` writes them to their own file as well.
VALID_CHANNELS = (2, 3, 25, 7, 8, 26)

_FILE_RE = re.compile(r"SENS_(?P<prefix>\d+)_(?P<chunk>\d+)_(?P<kind>.+)\.csv")

RDO_DTYPES = {
    "timestamp": np.int64,
    "serial_id": np.int16,
    "modbus_id": np.int16,
    "param": np.int16,
    "value": np.float64,
    "quality": np.int16,
    "units": np.int16,
}


def sensor_id(channel):
    return f"ch{int(channel)}"


# --- Files and clocks ------------------------------------------------------------


def discover(data_dir=None):
    """{(prefix, chunk): {kind: path}} for every SENS_* file in `data_dir`."""
    files = {}
    for path in Path(data_dir or config.BISCAYNE_HARVARD_DIR).glob("SENS_*_*_*.csv"):
        m = _FILE_RE.fullmatch(path.name)
        if m:
            files.setdefault((int(m["prefix"]), int(m["chunk"])), {})[m["kind"]] = path
    return files


def fit_rtc(rtc):
    """Linear tick -> epoch-seconds fit of one chunk's RTC log, or None if too short."""
    if len(rtc) < 2:
        return None
    wall = pd.to_datetime(rtc["timestr"].astype(str), format="%Y%m%dT%H%M%S")
    epoch = (wall - pd.Timestamp(0)).dt.total_seconds().to_numpy()
    ticks = rtc["timestamp"].to_numpy(np.float64)
    slope, icpt = np.polyfit(ticks, epoch, 1)
    rms = float(np.sqrt(np.mean((slope * ticks + icpt - epoch) ** 2)))
    return {
        "slope": slope,
        "icpt": icpt,
        "rms": rms,
        "t0": ticks[0],
        "t1": ticks[-1],
        "e0": epoch[0],
        "e1": epoch[-1],
    }


def clock_table(files):
    """One row per (prefix, chunk) with its own RTC fit where it has one."""
    rows = []
    for (prefix, chunk), kinds in files.items():
        fit = fit_rtc(pd.read_csv(kinds["RTC"])) if "RTC" in kinds else None
        rows.append(
            {
                "prefix": prefix,
                "chunk": chunk,
                **(fit or {}),
                "clock_source": "own_rtc" if fit else None,
            }
        )
    columns = ["prefix", "chunk", "slope", "icpt", "rms", "t0", "t1", "e0", "e1", "clock_source"]
    return pd.DataFrame(rows, columns=columns).sort_values(["prefix", "chunk"], ignore_index=True)


def _borrow(t0, t1, donor):
    """Donor's fit extrapolated to ticks [t0, t1], or None if not tick-continuous."""
    if t0 > donor["t1"]:
        gap_s = (t0 - donor["t1"]) / NOMINAL_TICKS_PER_S
    elif t1 < donor["t0"]:
        gap_s = (donor["t0"] - t1) / NOMINAL_TICKS_PER_S
    else:
        return None
    if gap_s > MAX_BORROW_GAP_S:
        return None
    return donor["slope"] * t0 + donor["icpt"], donor["slope"] * t1 + donor["icpt"]


def resolve_missing_clocks(clock, tick_spans):
    """Give RTC-less chunks a neighbour's fit where that is provably safe.

    `tick_spans` maps (prefix, chunk) -> (tick_min, tick_max). A borrowed fit is kept
    only if its predicted wall span sits between the previous own-RTC chunk's end and
    the next one's start; otherwise the chunk is marked "unresolved".
    """
    clock = clock.copy()
    for prefix, group in clock.groupby("prefix"):
        own = group[group["clock_source"] == "own_rtc"]
        for idx in group.index[group["clock_source"].isna()]:
            chunk = clock.at[idx, "chunk"]
            if (prefix, chunk) not in tick_spans:
                continue
            t0, t1 = tick_spans[(prefix, chunk)]
            prev = own[own["chunk"] < chunk].tail(1)
            nxt = own[own["chunk"] > chunk].head(1)
            low = prev["e1"].iloc[0] if len(prev) else -np.inf
            high = nxt["e0"].iloc[0] if len(nxt) else np.inf
            clock.at[idx, "clock_source"] = "unresolved"
            for donor_rows, tag in ((prev, "borrowed_prev"), (nxt, "borrowed_next")):
                if not len(donor_rows):
                    continue
                donor = donor_rows.iloc[0]
                predicted = _borrow(t0, t1, donor)
                if predicted and low - 1 <= predicted[0] and predicted[1] <= high + 1:
                    clock.at[idx, "slope"] = donor["slope"]
                    clock.at[idx, "icpt"] = donor["icpt"]
                    clock.at[idx, "clock_source"] = tag
                    break
    return clock


def _local_time(ticks, slope, icpt):
    epoch = np.asarray(slope * np.asarray(ticks, np.float64) + icpt)
    return pd.to_datetime(epoch, unit="s").round("ms")


# --- Raw streams -----------------------------------------------------------------


def load_rdo(files):
    """All RDO reads, long format, plus each chunk's tick span for clock borrowing."""
    frames, spans = [], {}
    for key in sorted(k for k, kinds in files.items() if "RDO" in kinds):
        df = pd.read_csv(files[key]["RDO"], dtype=RDO_DTYPES).drop(columns="index")
        if df.empty:
            continue
        df["prefix"], df["chunk"] = key
        spans[key] = (df["timestamp"].min(), df["timestamp"].max())
        frames.append(df)
    return pd.concat(frames, ignore_index=True), spans


def place_in_time(reads, clock):
    """Attach `local_time` (naive EDT) and `clock_source` to per-chunk reads."""
    fits = clock[["prefix", "chunk", "slope", "icpt", "clock_source"]]
    placed = reads.merge(fits, on=["prefix", "chunk"], how="left")
    placed["clock_source"] = placed["clock_source"].fillna("unresolved")
    placed["local_time"] = _local_time(placed["timestamp"], placed["slope"], placed["icpt"])
    return placed.drop(columns=["slope", "icpt"])


def load_bar100(files, clock):
    """Bar100 gauge pressure (bar), placed in local time; unplaceable chunks skipped."""
    fits = clock.dropna(subset=["slope"]).set_index(["prefix", "chunk"])
    frames = []
    for key, kinds in files.items():
        if "External_PTS_Bar100" not in kinds or key not in fits.index:
            continue
        df = pd.read_csv(kinds["External_PTS_Bar100"], usecols=["timestamp", "pressure_bar"])
        if df.empty:
            continue
        slope, icpt = fits.loc[key, ["slope", "icpt"]]
        df["local_time"] = _local_time(df["timestamp"], slope, icpt)
        frames.append(df[["local_time", "pressure_bar"]])
    if not frames:
        return pd.DataFrame(columns=["local_time", "pressure_bar"])
    return pd.concat(frames, ignore_index=True)


def to_utc(local_time):
    return local_time.dt.tz_localize(
        SOURCE_TZ, nonexistent="raise", ambiguous="raise"
    ).dt.tz_convert("UTC")


def depth_on_grid(bar100, interval=schema.SAMPLE_INTERVAL):
    """Median Bar100 depth (m) per UTC grid bin, as a Series indexed by bin."""
    if bar100.empty:
        return pd.Series(dtype="float64", name="depth_m")
    bins = to_utc(bar100["local_time"]).dt.floor(interval)
    pressure = bar100["pressure_bar"].groupby(bins).median()
    depth = pressure_kpa_to_depth_m(pressure * 100.0, DENSITY_SEAWATER_KG_M3, SITE_LATITUDE)
    return depth.rename("depth_m")


# --- Polls -> schema -------------------------------------------------------------


def known_reads(reads):
    """Mask selecting reads whose (param, units) pair is one the optode defines."""
    expected = pd.Series({param: units for param, (_, units) in PARAMS.items()})
    return reads["units"].to_numpy() == reads["param"].map(expected).to_numpy()


def reconstruct_polls(reads):
    """One row per (channel, poll), pivoting the four param reads into columns.

    A new poll starts at a channel change or a > 5 s gap in wall time. The poll's time
    is its first read; it is failing if any oxygen param carries a failing code.
    """
    reads = reads.sort_values(["modbus_id", "local_time"], kind="stable", ignore_index=True)
    new_poll = (reads["modbus_id"].diff() != 0) | (
        reads["local_time"].diff() > pd.Timedelta(seconds=POLL_GAP_S)
    )
    reads["poll"] = new_poll.cumsum()

    wide = (
        reads.drop_duplicates(["poll", "param"])
        .pivot(index="poll", columns="param", values="value")
        .rename(columns={param: column for param, (column, _) in PARAMS.items()})
    )
    for column, _ in PARAMS.values():
        if column not in wide:
            wide[column] = np.nan

    failing = reads["quality"].isin(FAILING_QUALITY_CODES) & reads["param"].isin(OXYGEN_PARAMS)
    grouped = reads.groupby("poll")
    polls = pd.DataFrame(
        {
            "channel": grouped["modbus_id"].first(),
            "local_time": grouped["local_time"].min(),
            "n_params": grouped["param"].nunique(),
            "failing": failing.groupby(reads["poll"]).any(),
        }
    ).join(wide[[column for column, _ in PARAMS.values()]])
    return polls.reset_index(drop=True)


def liveness(polls):
    """Per (channel, local day) liveness state, from the SMORES plan's classifier.

    Only in-situ polls are classified. On May 13 every optode reads ~100% sat in the
    bucket and then drops to its floor on burial, and that drop alone is enough
    variance to call the whole day "live" -- for ch10 and ch28, the only day they
    would ever pass.
    """
    deployed = polls[polls["local_time"] >= DEPLOYMENT_START_LOCAL]
    return classify_liveness(
        deployed,
        channel_col="channel",
        time_col="local_time",
        po2_col="po2",
        do_mgl_col="do_mgl_reported",
        do_pct_col="pctsat_reported",
    ).rename(columns={"channel_id": "channel"})


def to_schema(polls, live):
    """Poll-rate schema frame (without depth) from polls and their liveness states."""
    polls = polls.dropna(subset=["po2", "temp_c"]).copy()
    polls["day"] = polls["local_time"].dt.floor("1D")
    polls = polls.merge(live[["channel", "day", "state"]], on=["channel", "day"], how="left")

    do_sat = pct_sat_from_po2(polls["po2"], polls["temp_c"], SENSOR_SALINITY)
    solubility = o2sol_umol_l(polls["temp_c"], ASSUMED_SALINITY, DENSITY_SEAWATER_KG_M3)
    bucket = polls["local_time"] < DEPLOYMENT_START_LOCAL
    # Liveness is an in-situ judgement; bucket-test polls are only flagged on quality.
    masked = polls["state"].ne("live") & ~bucket
    qc_flag = np.select([masked, polls["failing"]], [3, 2], default=0)

    out = pd.DataFrame(
        {
            "timestamp": to_utc(polls["local_time"]),
            "site_id": SITE_ID,
            "deployment_id": np.where(bucket, BUCKET_DEPLOYMENT_ID, DEPLOYMENT_ID),
            "sensor_id": polls["channel"].map(sensor_id),
            "do_sat": do_sat,
            "temperature_c": polls["temp_c"],
            "do_mgl": mgl_from_saturation(do_sat, solubility),
            "do_native_value": polls["po2"],
            "do_native_unit": DO_NATIVE_UNIT,
            "qc_flag": qc_flag,
            "data_source": DATA_SOURCE,
            "depth_m": np.nan,
        }
    )
    out = out.sort_values(["timestamp", "sensor_id"], kind="stable", ignore_index=True)
    return schema.validate(schema.conform(out))


# --- Pipeline entry point --------------------------------------------------------


def site_checks(grid, polls, clock, reads, n_unknown):
    """Checks only this site's instruments can confirm."""
    checks = []

    own = clock[clock["clock_source"] == "own_rtc"]
    worst = float(own["rms"].max()) if len(own) else np.nan
    checks.append(
        integrity.Check(
            "clock_fit_residual",
            bool(worst < 1.0),
            f"worst own-RTC fit residual {worst:.3f} s over {len(own):,} chunks",
        )
    )

    unresolved = float((reads["clock_source"] == "unresolved").mean())
    sources = reads["clock_source"].value_counts().to_dict()
    checks.append(
        integrity.Check(
            "clock_resolved",
            unresolved < 0.02,
            f"{unresolved:.2%} of reads unplaceable and dropped; by source {sources}",
        )
    )

    unknown_share = n_unknown / max(len(reads), 1)
    checks.append(
        integrity.Check(
            "known_params",
            unknown_share < 1e-4,
            f"{n_unknown} read(s) with an unknown (param, units) pair dropped",
        )
    )

    complete = polls.dropna(subset=["po2", "temp_c", "pctsat_reported"])
    usable = complete[(complete["pctsat_reported"] > 1.0) & ~complete["failing"]]
    recomputed = pct_sat_from_po2(usable["po2"], usable["temp_c"], SENSOR_SALINITY)
    ratio = np.abs(recomputed / usable["pctsat_reported"] - 1.0)
    worst_ratio = float(ratio.quantile(0.999)) if len(ratio) else np.nan
    checks.append(
        integrity.Check(
            "saturation_matches_sensor",
            bool(worst_ratio < 1e-3),
            f"recomputed vs sensor % sat, 99.9th pct |ratio - 1| = {worst_ratio:.2e} "
            f"over {len(usable):,} polls",
        )
    )

    deployed = grid[grid["deployment_id"] == DEPLOYMENT_ID]
    live_sensors = sorted(deployed.loc[deployed["qc_flag"] == 0, "sensor_id"].unique())
    bucket = grid[
        (grid["deployment_id"] == BUCKET_DEPLOYMENT_ID) & grid["sensor_id"].isin(live_sensors)
    ]
    medians = bucket.groupby("sensor_id")["do_sat"].median()
    in_band = medians.between(85.0, 108.0)
    checks.append(
        integrity.Check(
            "bucket_test_saturation",
            bool(len(medians) and in_band.all()),
            "bucket-test median % sat of in-situ live channels: "
            + ", ".join(f"{s} {v:.1f}" for s, v in medians.items()),
            integrity.WARN,
        )
    )
    checks.append(
        integrity.Check(
            "live_channels",
            bool(live_sensors),
            f"{len(live_sensors)} channel(s) with good in-situ data: {live_sensors}",
        )
    )

    midwater = deployed[deployed["sensor_id"].isin([sensor_id(c) for c in MIDWATER_CHANNELS])]
    checks.append(integrity.diel_check(midwater, SOURCE_TZ, name="diel_cycle_midwater"))
    return checks


def pipeline(data_dir=None, verbose=True):
    """Raw dump -> 2-minute grid (with `n_obs`) and this site's integrity checks."""
    files = discover(data_dir)
    if not files:
        raise FileNotFoundError(f"no SENS_* files in {data_dir or config.BISCAYNE_HARVARD_DIR}")
    if verbose:
        print(f"[biscayne_bay] {len(files):,} chunks; reading RDO")

    raw, spans = load_rdo(files)
    clock = resolve_missing_clocks(clock_table(files), spans)
    reads = place_in_time(raw, clock)

    known = known_reads(reads)
    n_unknown = int((~known).sum())
    placed = reads[known & (reads["clock_source"] != "unresolved")]

    polls = reconstruct_polls(placed)
    native = to_schema(polls, liveness(polls))
    grid = resample.to_grid(native, min_obs=1)

    if verbose:
        print("[biscayne_bay] reading Bar100")
    depth = depth_on_grid(load_bar100(files, clock))
    grid["depth_m"] = grid["timestamp"].map(depth).astype("float32")

    return grid, site_checks(grid, polls, clock, reads, n_unknown)
