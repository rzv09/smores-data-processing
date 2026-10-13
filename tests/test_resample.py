"""The shared 2-minute binning."""

import numpy as np
import pandas as pd
from synthetic import schema_frame

from smores.data import resample, schema


def test_bins_are_labelled_by_their_start_and_hold_the_median():
    frame = schema_frame(n=8, freq="30s", do_sat=[1.0, 2.0, 3.0, 100.0, 5.0, 6.0, 7.0, 8.0])
    grid = resample.to_grid(frame)
    assert grid["timestamp"].tolist() == [
        pd.Timestamp("2026-01-01 00:00", tz="UTC"),
        pd.Timestamp("2026-01-01 00:02", tz="UTC"),
    ]
    # The 100.0 spike moves the median of its bin by half a step, not by 25.
    assert grid["do_sat"].tolist() == [2.5, 6.5]
    assert grid["n_obs"].tolist() == [4, 4]


def test_output_conforms_to_the_schema_after_dropping_n_obs():
    grid = resample.to_grid(schema_frame(n=10))
    schema.validate(grid.drop(columns="n_obs"))


def test_bins_never_straddle_deployments_or_sensors():
    a = schema_frame(n=4, sensor="S1", deployment="d1")
    b = schema_frame(n=4, sensor="S1", deployment="d2", do_sat=50.0)
    c = schema_frame(n=4, sensor="S2", deployment="d1", do_sat=70.0)
    grid = resample.to_grid(pd.concat([a, b, c], ignore_index=True))
    assert len(grid) == 3
    assert sorted(grid["do_sat"].tolist()) == [50.0, 70.0, 100.0]


def test_good_readings_win_over_flagged_ones_in_a_mixed_bin():
    frame = schema_frame(n=4, do_sat=[100.0, 100.0, 0.0, 0.0], qc_flag=[0, 0, 2, 2])
    grid = resample.to_grid(frame)
    assert grid["qc_flag"].tolist() == [0]
    assert grid["do_sat"].tolist() == [100.0]
    assert grid["n_obs"].tolist() == [2]


def test_a_bin_of_only_flagged_readings_stays_flagged():
    grid = resample.to_grid(schema_frame(n=4, qc_flag=2))
    assert grid["qc_flag"].tolist() == [2]


def test_min_obs_drops_underfilled_bins():
    frame = schema_frame(n=5, freq="30s")  # one full bin of 4, one bin of 1
    assert len(resample.to_grid(frame, min_obs=1)) == 2
    assert len(resample.to_grid(frame, min_obs=2)) == 1


def test_empty_bins_are_absent_not_interpolated():
    frame = schema_frame(n=3, freq="15min")
    grid = resample.to_grid(frame)
    assert len(grid) == 3
    assert grid["timestamp"].diff().dropna().unique().tolist() == [
        pd.Timedelta(minutes=14),
        pd.Timedelta(minutes=16),
    ]


def test_all_null_columns_stay_null():
    grid = resample.to_grid(schema_frame(n=4, temperature_c=np.nan))
    assert grid["temperature_c"].isna().all()
