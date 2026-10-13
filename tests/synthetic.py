"""Small synthetic inputs shaped like each site's real raw files.

Shared by the pipeline tests so they run without any data present. Every builder
mirrors the real file's columns exactly; the values are chosen so each transform has
something checkable to get right.
"""

import numpy as np
import pandas as pd

from smores.data import schema
from smores.ingest_smores.qc import pct_sat_from_po2


def schema_frame(n=4, sensor="S1", deployment="d1", start="2026-01-01", freq="30s", **values):
    """A native-rate frame conforming to the schema, with overridable columns."""
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=n, freq=freq, tz="UTC"),
            "site_id": "florida_keys",
            "deployment_id": deployment,
            "sensor_id": sensor,
            "do_sat": 100.0,
            "temperature_c": 29.0,
            "do_mgl": 6.5,
            "do_native_value": 250.0,
            "do_native_unit": "umol/L",
            "qc_flag": 0,
            "data_source": "wcci_fl",
            "depth_m": 8.0,
        }
    )
    for name, value in values.items():
        frame[name] = value
    return schema.conform(frame)


# --- Biscayne Bay ----------------------------------------------------------------

TICKS_PER_S = 1.003e8
BB_START_LOCAL = pd.Timestamp("2026-05-13 20:00:00")
BB_SPLIT_LOCAL = pd.Timestamp("2026-05-14 10:00:00")
BB_END_LOCAL = pd.Timestamp("2026-05-15 00:00:00")
BB_SUBMERGED_LOCAL = pd.Timestamp("2026-05-13 20:45:00")
BB_FLAGGED_LOCAL = (pd.Timestamp("2026-05-14 12:00:00"), pd.Timestamp("2026-05-14 13:00:00"))
LIVE_CHANNEL, STUCK_CHANNEL, ORPHAN_CHANNEL = 2, 10, 25


def _ticks(local):
    seconds = (pd.DatetimeIndex(local) - BB_START_LOCAL).total_seconds().to_numpy()
    return (1_000_000_000 + seconds * TICKS_PER_S).astype(np.int64)


def _live_po2(local):
    """Midwater-like pO2 (mmHg) and temperature (C) peaking at 15:00 local."""
    hours = local.hour + local.minute / 60.0
    phase = np.cos(2 * np.pi * (hours - 15.0) / 24.0)
    return 150.0 + 30.0 * phase, 28.0 + 1.0 * phase


def _rdo_rows(channel, local, po2, temp, quality=0):
    pct = np.clip(pct_sat_from_po2(po2, temp, 0.0), 0.0, None)
    reads = []
    for offset, (param, units, value) in enumerate(
        [(1, 1, temp), (20, 117, pct * 0.07), (21, 177, pct), (30, 26, po2)]
    ):
        reads.append(
            pd.DataFrame(
                {
                    "timestamp": _ticks(local + pd.Timedelta(milliseconds=200 * offset)),
                    "serial_id": 0,
                    "modbus_id": channel,
                    "param": param,
                    "value": value,
                    "quality": 0 if param == 1 else quality,
                    "units": units,
                }
            )
        )
    return pd.concat(reads, ignore_index=True)


def _write(frame, path):
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    frame.insert(0, "index", np.arange(len(frame)))
    frame.to_csv(path, index=False)


def write_biscayne_dump(directory):
    """Write a two-chunk SMORES dump plus one unplaceable chunk into `directory`.

    * chunk (5, 1) has an RTC log; chunk (5, 2) has none and must borrow its clock.
    * ch2 is live, with failing quality codes for one hour; ch10 is stuck constant.
    * chunk (6, 1) holds a few ch25 polls in a prefix with no RTC anywhere, so they
      cannot be placed in time and must be dropped.
    * one corrupt read with an unknown (param, units) pair.
    """
    polls = pd.date_range(BB_START_LOCAL, BB_END_LOCAL, freq="52s", inclusive="left")
    for chunk, mask in ((1, polls < BB_SPLIT_LOCAL), (2, polls >= BB_SPLIT_LOCAL)):
        local = polls[mask]
        po2, temp = _live_po2(local)
        flagged = (local >= BB_FLAGGED_LOCAL[0]) & (local < BB_FLAGGED_LOCAL[1])
        live = pd.concat(
            [
                _rdo_rows(LIVE_CHANNEL, local[~flagged], po2[~flagged], temp[~flagged]),
                _rdo_rows(LIVE_CHANNEL, local[flagged], po2[flagged], temp[flagged], quality=7),
            ]
        )
        stuck = _rdo_rows(STUCK_CHANNEL, local, np.full(len(local), 1.7), np.full(len(local), 28.0))
        rdo = pd.concat([live, stuck], ignore_index=True)
        if chunk == 1:
            corrupt = rdo.iloc[[0]].assign(param=16875, units=1)
            rdo = pd.concat([rdo, corrupt], ignore_index=True)
        _write(rdo, directory / f"SENS_5_{chunk}_RDO.csv")

        seconds = pd.date_range(local[0], local[-1], freq="1s")
        _write(
            pd.DataFrame(
                {
                    "timestamp": _ticks(seconds),
                    "pressure_bar": np.where(seconds < BB_SUBMERGED_LOCAL, 0.0, 0.09),
                    "temperature_c": 28.0,
                }
            ),
            directory / f"SENS_5_{chunk}_External_PTS_Bar100.csv",
        )

    rtc_times = pd.date_range(BB_START_LOCAL, BB_SPLIT_LOCAL, freq="60s", inclusive="left")
    _write(
        pd.DataFrame(
            {
                "timestamp": _ticks(rtc_times),
                "seconds": rtc_times.second,
                "minutes": rtc_times.minute,
                "hours": rtc_times.hour,
                "wday": rtc_times.dayofweek,
                "mday": rtc_times.day,
                "month": rtc_times.month,
                "year": rtc_times.year - 2000,
                "timestr": rtc_times.strftime("%Y%m%dT%H%M%S"),
            }
        ),
        directory / "SENS_5_1_RTC.csv",
    )

    orphan = pd.date_range("2026-05-14 03:00", periods=5, freq="52s")
    po2, temp = _live_po2(orphan)
    _write(_rdo_rows(ORPHAN_CHANNEL, orphan, po2, temp), directory / "SENS_6_1_RDO.csv")
    return directory


# --- Santa Barbara ---------------------------------------------------------------


def santa_barbara_raw(days=2, freq="15min", start="2017-07-11 00:00", sites=("SBH",)):
    """Rows shaped like the SBC LTER CSV, with a diel cycle peaking at 15:00 local."""
    frames = []
    for site in sites:
        stamps = pd.date_range(
            start, periods=int(pd.Timedelta(days=days) / pd.Timedelta(freq)), freq=freq, tz="UTC"
        )
        local_hour = stamps.tz_convert("America/Los_Angeles").hour
        phase = np.cos(2 * np.pi * (local_hour - 15.0) / 24.0)
        frames.append(
            pd.DataFrame(
                {
                    "site": site,
                    "datetime_UTC": stamps.strftime("%Y-%m-%dT%H:%M:%S"),
                    "deployment_depth_m": 7,
                    "temperature_C": 15.0 + phase,
                    "DO_percent_saturation": 100.0 + 15.0 * phase,
                    "DO_mgl": 8.0 + 1.2 * phase,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
