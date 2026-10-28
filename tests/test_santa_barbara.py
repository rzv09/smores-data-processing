"""The Santa Barbara pipeline, on synthetic rows shaped like the SBC LTER CSV."""

import numpy as np
import pandas as pd
import pytest
import synthetic as syn

from smores import config
from smores.data import integrity, schema
from smores.data import santa_barbara as sb


def test_passthrough_columns_and_utc_timestamps():
    raw = syn.santa_barbara_raw(days=1)
    out = sb.to_schema(raw)
    assert str(out["timestamp"].dt.tz) == "UTC"
    assert out["timestamp"].iloc[0] == pd.Timestamp("2017-07-11 00:00", tz="UTC")
    np.testing.assert_allclose(out["do_sat"], raw["DO_percent_saturation"], rtol=1e-6)
    np.testing.assert_allclose(out["do_mgl"], raw["DO_mgl"], rtol=1e-6)
    assert out["depth_m"].unique().tolist() == [7.0]
    assert out["do_native_value"].isna().all()
    assert set(out["sensor_id"]) == {"SBH"}


def test_load_raw_filters_sites(tmp_path):
    path = tmp_path / "sb.csv"
    syn.santa_barbara_raw(days=1, sites=("SBH", "ALE")).to_csv(path, index=False)
    assert set(sb.load_raw(path, sites=["SBH"])["site"]) == {"SBH"}
    assert set(sb.load_raw(path)["site"]) == {"SBH", "ALE"}


def test_a_long_gap_starts_a_new_deployment():
    first = syn.santa_barbara_raw(days=1, start="2017-07-11")
    second = syn.santa_barbara_raw(days=1, start="2017-07-13")  # a 24 h gap
    out = sb.to_schema(pd.concat([first, second], ignore_index=True))
    assert sorted(out["deployment_id"].unique()) == ["SBH_7m_20170711", "SBH_7m_20170713"]


def test_a_depth_change_starts_a_new_deployment():
    raw = syn.santa_barbara_raw(days=1)
    raw.loc[raw.index >= 48, "deployment_depth_m"] = 11
    out = sb.to_schema(raw)
    assert out["deployment_id"].nunique() == 2


def test_a_short_gap_does_not_split():
    raw = syn.santa_barbara_raw(days=1).drop(index=range(10, 14))  # one missing hour
    assert sb.to_schema(raw)["deployment_id"].nunique() == 1


def test_rows_without_do_mgl_are_dropped():
    raw = syn.santa_barbara_raw(days=1)
    raw.loc[[3, 4], "DO_mgl"] = np.nan
    assert len(sb.to_schema(raw)) == len(raw) - 2


def test_rows_with_do_mgl_but_no_saturation_are_kept():
    raw = syn.santa_barbara_raw(days=1)
    raw.loc[[3, 4], "DO_percent_saturation"] = np.nan
    out = sb.to_schema(raw)
    assert len(out) == len(raw)
    assert out["do_sat"].isna().sum() == 2


def test_pipeline_grids_without_upsampling_and_warns(tmp_path):
    path = tmp_path / "sb.csv"
    syn.santa_barbara_raw(days=3).to_csv(path, index=False)
    grid, checks = sb.pipeline(path, verbose=False)
    assert len(grid) == 3 * 96  # one bin per 15-minute reading, none invented
    schema.validate(grid.drop(columns="n_obs"))

    report = integrity.build_report(grid, sb.SITE_ID, checks)
    assert report.ok, str(report)
    assert [c.name for c in report.warnings] == ["native_rate_vs_grid"]
    assert {c.name: c for c in report.checks}["diel_cycle"].passed


def test_sentinel_values_fail_integrity_loudly(tmp_path):
    path = tmp_path / "sb.csv"
    raw = syn.santa_barbara_raw(days=3)
    raw.loc[5, "DO_percent_saturation"] = -99999.0
    raw.to_csv(path, index=False)
    grid, checks = sb.pipeline(path, verbose=False)
    report = integrity.build_report(grid, sb.SITE_ID, checks)
    assert [c.name for c in report.failures] == ["range_do_sat"]


@pytest.mark.slow
@pytest.mark.skipif(not config.SANTA_BARBARA_CSV.is_file(), reason="SBC LTER CSV not present")
def test_real_file_passes_integrity():
    grid, checks = sb.pipeline(verbose=False)
    report = integrity.build_report(grid, sb.SITE_ID, checks)
    assert report.ok, str(report)
