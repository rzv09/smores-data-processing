"""Generic integrity checks and the report."""

import json

import numpy as np
import pandas as pd
import pytest
from synthetic import schema_frame

from smores.data import integrity, resample


def grid(**kwargs):
    return resample.to_grid(schema_frame(**kwargs))


def checks_by_name(frame):
    return {c.name: c for c in integrity.check_frame(frame)}


def test_a_clean_grid_passes_every_generic_check():
    report = integrity.build_report(grid(n=40), "florida_keys")
    assert report.ok, str(report)
    assert not report.warnings


def test_duplicate_keys_fail():
    frame = grid(n=8)
    doubled = pd.concat([frame, frame.iloc[[0]]]).sort_values("timestamp", ignore_index=True)
    assert not checks_by_name(doubled)["unique_keys"].passed


def test_off_grid_timestamps_fail():
    native = schema_frame(n=4, freq="30s")
    assert not checks_by_name(native)["on_grid"].passed


@pytest.mark.parametrize(
    ("column", "value"),
    [("do_sat", 1250.0), ("do_sat", -99999.0), ("temperature_c", 45.0), ("depth_m", -3.0)],
)
def test_out_of_range_values_fail(column, value):
    frame = grid(n=8)
    frame.loc[0, column] = value
    assert not checks_by_name(frame)[f"range_{column}"].passed


def test_overlapping_deployments_of_one_sensor_fail():
    a = schema_frame(n=8, deployment="d1")
    b = schema_frame(n=8, deployment="d2", start="2026-01-01 00:02")
    frame = resample.to_grid(pd.concat([a, b], ignore_index=True))
    assert not checks_by_name(frame)["deployments_disjoint"].passed


def test_coverage_counts_bins_and_gaps():
    frame = grid(n=20, freq="2min")
    frame = frame.drop(index=range(5, 12)).reset_index(drop=True)  # a 7-bin hole
    row = integrity.coverage(frame).iloc[0]
    assert row["n_bins"] == 13
    assert row["expected_bins"] == 20
    assert row["coverage_pct"] == pytest.approx(65.0)
    assert row["n_gaps"] == 1
    assert row["longest_gap"] == pd.Timedelta(minutes=14)


def test_warnings_do_not_fail_the_report():
    report = integrity.build_report(
        grid(n=8), "florida_keys", [integrity.Check("soft", False, "x", integrity.WARN)]
    )
    assert report.ok
    assert [c.name for c in report.warnings] == ["soft"]
    report.raise_for_failures()


def test_errors_fail_the_report_and_raise():
    report = integrity.build_report(
        grid(n=8), "florida_keys", [integrity.Check("hard", False, "x")]
    )
    assert not report.ok
    with pytest.raises(integrity.IntegrityError, match="hard"):
        report.raise_for_failures()


def test_report_round_trips_through_json(tmp_path):
    report = integrity.build_report(grid(n=8), "florida_keys")
    path = tmp_path / "report.json"
    report.write(path)
    loaded = json.loads(path.read_text())
    assert loaded["ok"] is True
    assert loaded["site_id"] == "florida_keys"
    assert {c["name"] for c in loaded["checks"]} >= {"unique_keys", "on_grid"}
    assert loaded["coverage"][0]["n_bins"] == 2  # 8 readings 30 s apart


def _diel_frame(peak_hour):
    stamps = pd.date_range("2026-07-01", "2026-07-03", freq="2min", tz="UTC", inclusive="left")
    hours = stamps.tz_convert("America/New_York").hour
    frame = schema_frame(n=len(stamps), freq="2min", start="2026-07-01")
    frame["do_mgl"] = (6.5 + 0.4 * np.cos(2 * np.pi * (hours - peak_hour) / 24)).astype("float32")
    frame["do_sat"] = np.float32(100.0)  # flat: the check must read the primary signal
    return frame


def test_diel_check_passes_an_afternoon_peak():
    check = integrity.diel_check(_diel_frame(16), "America/New_York")
    assert check.passed, check.detail


def test_diel_check_catches_a_timezone_shift():
    """A peak at 02:00 local is what a UTC/local mix-up produces."""
    check = integrity.diel_check(_diel_frame(2), "America/New_York")
    assert not check.passed
