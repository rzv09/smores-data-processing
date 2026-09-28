"""Florida Keys: raw BCO-DMO export -> harmonized schema.

Source
------
`data/raw/florida_keys/florida_keys_data.csv`, an aquatic eddy-covariance deployment.
A Nortek Vector ADV logging at 8 Hz with three co-located Pyroscience oxygen sensors,
July 2017, ~9 km south of Long Key (24 43.52'N, 80 49.85'W), 9 +/- 1 m water depth on a
coral-sand carbonate platform. The ADV measuring volume sat ~35 cm above the
sediment-water interface. Site conditions: S 35-36, T 28-31 C.

Submitter-supplied units: `t` hours, `t_increase` seconds, `Vx/Vy/Vz` cm/s, `P` kPa,
`O2_S1..S3` umol/L. `deployment` is the source filename, encoding the deployment's
date range as `YYYY_M_dayStart_dayEnd`.

Three transforms need justifying, because none of them is a plain rename.

Timestamps
----------
There is no timestamp column; it is reconstructed. `t` is elapsed hours and equals
`start_hour + t_increase/3600` exactly, so the per-deployment `start_hour` (38, 11, 10,
16) recovers as `round(t - t_increase/3600)`. Timestamps are built from `t_increase`
rather than `t` because `t` carries float noise of ~10 ms while `t_increase` is exact
multiples of 0.125 s.

Each deployment's clock starts at midnight on the *first* day named in `deployment`.
Three of the four then sit inside their named day range. The fourth,
`3oec_2017_7_11_12`, starts at t=38 h and so begins on July 12 and ends 8 h into July
13, past its named range. That is not a defect: the source records "measurements could
not start before 14:00 due to bad weather preventing earlier deployment" for this
deployment, and 38 h after midnight July 11 is exactly 14:00 on July 12. The delayed
start is documented, so `t` is used as recorded and the anchor rule stays uniform.

The logged clock is local (EDT), not UTC. Binning oxygen by reconstructed hour puts the
minimum at 06:00-07:00 and the maximum at 19:00-20:00; mid-July sunrise/sunset in the
Keys is ~06:40/~20:20 EDT. A pre-dawn respiration trough and a late-afternoon
photosynthetic peak is the expected benthic diel cycle. Read as UTC the trough would
fall at 02:00 local, which is not physical.

Oxygen
------
The reported umol/L values average 250, which against seawater solubility at the
documented S/T would be 127% air saturation sustained for five days. That is not
physical. The sensors were two-point calibrated in air-saturated seawater (100% a.s.)
and sulfite-stripped seawater (0% a.s.), and their accuracy is specified in % a.s., so
percent air saturation is the native, correctly salinity-referenced quantity. The
umol/L figures were derived from it using *freshwater* solubility: 105% a.s. x 238.3
umol/L (fresh) = 250 umol/L, where the true concentration is 105% x 195.9 = 206 umol/L.

So dividing the reported umol/L by freshwater solubility recovers the sensor's original
calibrated reading rather than inventing a correction. `do_sat` is that recovered
percentage and `do_mgl` the true in-situ mg/L. `do_native_value` (with
`do_native_unit` "umol/L") keeps the reported value untouched so the published
figures stay traceable and the correction stays reversible.

The corrected series corroborates itself: it runs 99.9% pre-dawn to 108.9% in the
evening, crossing saturation daily, which is what a net-autotrophic carbonate platform
does. Referenced against seawater solubility the same data sits at 121-133% and never
crosses 100%, which no real system does.

Caveat: temperature was never logged, so solubility uses the 29.5 C midpoint of the
documented range. Across 28-31 C that moves absolute `do_sat` by about +/-2.7%, which
exceeds the sensors' +/-1% a.s. accuracy and is the dominant uncertainty on the absolute
level. The fresh-to-seawater solubility *ratio* varies only 0.4% over the same range, so
relative structure -- the diel cycle, gradients, flux covariances -- is unaffected.

Depth
-----
`P` is gauge pressure in kPa, necessarily gauge: 82 kPa absolute would place the sensor
above the waterline. Depth is the hydrostatic head `P / (rho * g)`, giving a mean of
8.23 m against the 8.65 m expected from the 9 m site depth less the 0.35 m sensor
height, with per-deployment swings of 0.58-0.89 m -- a plausible Keys tidal range.

Dropped
-------
`Vx/Vy/Vz` are dropped; the harmonized schema is oxygen-only. Note for anyone extending
this: the benthic O2 flux this deployment exists to measure is the <w'C'> covariance of
those velocities with the oxygen signal, so flux work must go back to the raw CSV, which
stays immutable in `data/raw/`. `P` survives only as `depth_m`. The height of the ADV
above the seabed (0.35 m, `HEIGHT_ABOVE_BED_M`) is not part of the shared schema; it is
recorded here as a constant for context on `depth_m` rather than as an output column.
"""

import numpy as np
import pandas as pd

from smores import config
from smores.data import schema
from smores.data.units import (
    DENSITY_FRESH_KG_M3,
    mgl_from_saturation,
    o2sol_umol_l,
    pressure_kpa_to_depth_m,
    saturation_percent,
)

SITE_ID = "florida_keys"
DATA_SOURCE = "wcci_fl"

# Instruments logged local wall-clock time. No DST transition falls inside July 2017,
# so localization is unambiguous -- but it is still asserted, not assumed, below.
SOURCE_TZ = "America/New_York"

SITE_LATITUDE = 24 + 43.52 / 60
SITE_LONGITUDE = -(80 + 49.85 / 60)
SITE_WATER_DEPTH_M = 9.0
HEIGHT_ABOVE_BED_M = 0.35

# Midpoint of the documented ranges; see the module docstring on the uncertainty this
# carries for absolute `do_sat`.
ASSUMED_TEMPERATURE_C = 29.5
ASSUMED_SALINITY = 35.5
DENSITY_SEAWATER_KG_M3 = 1022.0

O2SOL_SITE_UMOL_L = float(
    o2sol_umol_l(ASSUMED_TEMPERATURE_C, ASSUMED_SALINITY, DENSITY_SEAWATER_KG_M3)
)
O2SOL_FRESH_UMOL_L = float(
    o2sol_umol_l(ASSUMED_TEMPERATURE_C, 0.0, DENSITY_FRESH_KG_M3)
)

DO_NATIVE_UNIT = "umol/L"

SAMPLE_INTERVAL_S = 0.125
SENSOR_MAP = {"O2_S1": "FL1", "O2_S2": "FL2", "O2_S3": "FL3"}
# Ordered categorical used only internally, to keep sort order stable; the schema
# requires `sensor_id` to come out as plain str.
SENSOR_SORT_DTYPE = pd.CategoricalDtype(tuple(SENSOR_MAP.values()), ordered=True)

RAW_COLUMNS = ["deployment", "t", "t_increase", "P", *SENSOR_MAP]


def load_raw(path=None):
    """Read the raw CSV, keeping only the columns the transform needs."""
    return pd.read_csv(path or config.FLORIDA_KEYS_CSV, usecols=RAW_COLUMNS)


def deployment_origin(name):
    """Local midnight on the first day named in a `deployment` label.

    `3oec_2017_7_11_12` -> 2017-07-11 00:00. See the module docstring for why the
    first named day is the anchor even where the record runs past the last.
    """
    _, year, month, day_start, _day_end = name.split("_")
    return pd.Timestamp(int(year), int(month), int(day_start))


def start_hour(frame):
    """Whole hours between a deployment's origin and its first sample.

    `t - t_increase/3600` is constant within a deployment up to float noise, so the
    median then rounds to an exact integer. Asserting that it *is* within noise of an
    integer is what would catch a source file whose `t` and `t_increase` disagree.
    """
    offsets = frame["t"] - frame["t_increase"] / 3600.0
    hours = float(offsets.median())
    if abs(hours - round(hours)) > 1e-3:
        raise ValueError(f"start hour {hours!r} is not a whole number of hours")
    return int(round(hours))


def reconstruct_timestamps(df):
    """UTC timestamps for every row, from `deployment` and `t_increase`."""
    naive = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for name, group in df.groupby("deployment", observed=True):
        origin = deployment_origin(name) + pd.Timedelta(hours=start_hour(group))
        naive.loc[group.index] = origin + pd.to_timedelta(group["t_increase"], unit="s")

    if naive.isna().any():
        raise ValueError("some rows got no timestamp")
    return naive.dt.tz_localize(
        SOURCE_TZ, nonexistent="raise", ambiguous="raise"
    ).dt.tz_convert("UTC")


def depth_from_pressure(pressure_kpa):
    """Metres of water above the sensor, from gauge pressure in kPa."""
    return pressure_kpa_to_depth_m(pressure_kpa, DENSITY_SEAWATER_KG_M3, SITE_LATITUDE)


def to_schema(df):
    """Reshape a raw Florida Keys frame to the harmonized long schema."""
    timestamp = reconstruct_timestamps(df)
    depth_m = depth_from_pressure(df["P"])

    frames = []
    for raw_column, sensor_id in SENSOR_MAP.items():
        raw_o2 = df[raw_column]
        do_sat = saturation_percent(raw_o2, O2SOL_FRESH_UMOL_L)
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": timestamp,
                    "depth_m": depth_m,
                    "sensor_id": sensor_id,
                    "do_sat": do_sat,
                    "do_mgl": mgl_from_saturation(do_sat, O2SOL_SITE_UMOL_L),
                    "do_native_value": raw_o2,
                    "deployment_id": df["deployment"],
                }
            )
        )

    long = pd.concat(frames, ignore_index=True)
    # Sort with sensor_id temporarily categorical, for stable sensor ordering within
    # a timestamp; the schema requires the column to come out as plain str.
    long["_sensor_sort"] = long["sensor_id"].astype(SENSOR_SORT_DTYPE)
    long = long.sort_values(["timestamp", "_sensor_sort"], kind="stable", ignore_index=True)
    long = long.drop(columns="_sensor_sort")

    long["site_id"] = SITE_ID
    long["data_source"] = DATA_SOURCE
    long["temperature_c"] = np.nan  # never logged; only a 28-31 C range is documented
    long["qc_flag"] = 0  # no nulls, no gaps and no duplicates in the source
    long["do_native_unit"] = DO_NATIVE_UNIT
    long["deployment_id"] = long["deployment_id"].astype(str)
    long["sensor_id"] = long["sensor_id"].astype(str)

    out = long[list(schema.COLUMNS)].astype(
        {name: schema.DTYPES[name] for name in schema.COLUMNS if name not in ("deployment_id", "sensor_id")}
    )
    return schema.validate(out)


def build(write=True, path=None, out_path=None):
    """Transform the raw CSV and, by default, write it to `processed/`."""
    out = to_schema(load_raw(path))
    if write:
        destination = out_path or config.FLORIDA_KEYS_PARQUET
        destination.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(destination, compression="snappy", index=False)
    return out


def main():
    out = build()
    print(f"wrote {len(out):,} rows to {config.FLORIDA_KEYS_PARQUET}")
    print(
        f"  {out['timestamp'].min()} -> {out['timestamp'].max()}  "
        f"({out['sensor_id'].nunique()} sensors, {out['deployment_id'].nunique()} deployments)"
    )
    print(f"  do_sat  mean {out['do_sat'].mean():.2f}%  range "
          f"{out['do_sat'].min():.2f}-{out['do_sat'].max():.2f}%")
    print(f"  depth_m mean {out['depth_m'].mean():.3f} m  range "
          f"{out['depth_m'].min():.3f}-{out['depth_m'].max():.3f} m")


if __name__ == "__main__":
    main()
