"""The Florida Keys transform.

Everything here runs on small synthetic frames except the one test marked `slow`,
which touches the real 203 MB CSV. The synthetic frames mirror the real file's shape
exactly -- including the float noise in `t` -- so they exercise the parts of the
transform that are easy to get wrong without needing the data present.
"""

import numpy as np
import pandas as pd
import pytest

from smores import config
from smores.data import florida_keys as fk
from smores.data import schema

# The four real deployments: name, start hour, record duration in hours.
REAL_DEPLOYMENTS = [
    ("3oec_2017_7_11_12", 38, 18),
    ("3oec_2017_7_13_14", 11, 19),
    ("3oec_2017_7_15_16", 10, 20),
    ("3oec_2017_7_16_17", 16, 14),
]


def raw_frame(deployment="3oec_2017_7_11_12", start_hour=38, rows=4, o2=250.0,
              pressure=82.36, t_noise=0.0):
    """A frame shaped like the raw CSV.

    `t_noise` reproduces the real file's quirk: `t` carries ~1e-5 h of float noise
    while `t_increase` is exact. Timestamps must come out clean regardless.
    """
    t_increase = np.arange(rows) * fk.SAMPLE_INTERVAL_S
    return pd.DataFrame(
        {
            "deployment": deployment,
            "t": start_hour + t_increase / 3600.0 + t_noise,
            "t_increase": t_increase,
            "P": pressure,
            "O2_S1": o2,
            "O2_S2": o2,
            "O2_S3": o2,
        }
    )


def test_deployment_origin_anchors_on_first_named_day():
    assert fk.deployment_origin("3oec_2017_7_11_12") == pd.Timestamp(2017, 7, 11)
    assert fk.deployment_origin("3oec_2017_7_16_17") == pd.Timestamp(2017, 7, 16)


@pytest.mark.parametrize(("name", "hour", "_duration"), REAL_DEPLOYMENTS)
def test_start_hour_recovers_exact_integer(name, hour, _duration):
    frame = raw_frame(deployment=name, start_hour=hour, rows=100, t_noise=1e-5)
    assert fk.start_hour(frame) == hour


def test_start_hour_rejects_inconsistent_time_columns():
    frame = raw_frame()
    frame["t"] += 0.4  # t and t_increase now disagree by 24 minutes
    with pytest.raises(ValueError, match="not a whole number of hours"):
        fk.start_hour(frame)


def test_timestamps_are_utc_and_land_on_the_documented_start():
    """t=38 h from midnight Jul 11 is 14:00 Jul 12 EDT, i.e. 18:00 UTC.

    The source notes this deployment could not begin before 14:00 because of weather,
    which is what makes the as-recorded t=38 correct rather than a 24 h error.
    """
    stamps = fk.reconstruct_timestamps(raw_frame(start_hour=38, t_noise=1e-5))
    assert str(stamps.dt.tz) == "UTC"
    assert stamps.iloc[0] == pd.Timestamp("2017-07-12 18:00:00", tz="UTC")
    local = stamps.dt.tz_convert(fk.SOURCE_TZ)
    assert local.iloc[0].hour == 14
    assert local.iloc[0].date() == pd.Timestamp(2017, 7, 12).date()


def test_float_noise_in_t_does_not_leak_into_timestamps():
    """Timestamps come from t_increase, so noise in `t` must not perturb them."""
    clean = fk.reconstruct_timestamps(raw_frame(rows=50, t_noise=0.0))
    noisy = fk.reconstruct_timestamps(raw_frame(rows=50, t_noise=1.4e-5))
    pd.testing.assert_series_equal(clean, noisy)
    assert (clean.dt.nanosecond == 0).all()
    assert (clean.dt.microsecond % 1000 == 0).all()


def test_sample_spacing_is_exactly_125_milliseconds():
    stamps = fk.reconstruct_timestamps(raw_frame(rows=1000, t_noise=1e-5))
    spacing = stamps.diff().dropna().unique()
    assert list(spacing) == [pd.Timedelta(milliseconds=125)]


def test_reconstructed_spans_do_not_overlap():
    spans = []
    for name, hour, duration in REAL_DEPLOYMENTS:
        start = fk.deployment_origin(name) + pd.Timedelta(hours=hour)
        spans.append((start, start + pd.Timedelta(hours=duration)))
    spans.sort()
    for (_, earlier_end), (later_start, _) in zip(spans, spans[1:]):
        assert later_start >= earlier_end


def test_reconstructed_spans_sit_inside_the_july_campaign():
    for name, hour, duration in REAL_DEPLOYMENTS:
        start = fk.deployment_origin(name) + pd.Timedelta(hours=hour)
        end = start + pd.Timedelta(hours=duration)
        assert pd.Timestamp(2017, 7, 11) <= start
        assert end <= pd.Timestamp(2017, 7, 18)


def test_depth_comes_out_near_the_documented_site_depth():
    depth = fk.depth_from_pressure(pd.Series([82.36]))
    assert depth.iloc[0] == pytest.approx(8.232, abs=1e-3)
    # 9 m site depth less the 0.35 m sensor height, within the site's +/-1 m.
    assert abs(depth.iloc[0] - (fk.SITE_WATER_DEPTH_M - fk.HEIGHT_ABOVE_BED_M)) < 1.0


def test_do_sat_recovers_the_calibrated_percentage():
    """250 umol/L reported against freshwater solubility is ~105% air saturation."""
    out = fk.to_schema(raw_frame(o2=250.0))
    assert out["do_sat"].iloc[0] == pytest.approx(104.9, abs=0.1)
    # Sanity: it must not read as the physically impossible seawater-referenced value.
    assert out["do_sat"].max() < 115.0


def test_do_mgl_is_the_true_in_situ_concentration():
    out = fk.to_schema(raw_frame(o2=250.0))
    assert out["do_mgl"].iloc[0] == pytest.approx(6.57, abs=0.05)
    # Not the face-value conversion of the reported umol/L, which would be ~8.0.
    assert out["do_mgl"].iloc[0] < 7.5


def test_do_native_value_is_carried_through_untouched():
    """The audit column must stay byte-comparable to the published figures."""
    frame = raw_frame(o2=243.179754583)
    out = fk.to_schema(frame)
    assert out["do_native_value"].unique().tolist() == [
        pytest.approx(243.179754583, rel=1e-6)
    ]
    assert out["do_native_unit"].unique().tolist() == ["umol/L"]


def test_melt_triples_rows_and_labels_every_sensor():
    frame = raw_frame(rows=7)
    out = fk.to_schema(frame)
    assert len(out) == 3 * len(frame)
    assert sorted(out["sensor_id"].unique()) == ["FL1", "FL2", "FL3"]
    assert out.groupby("sensor_id", observed=True).size().unique().tolist() == [7]


def test_sensors_differ_per_row_and_stay_aligned():
    """A melt that mixed up columns would pass a row count but fail this."""
    frame = raw_frame(rows=2)
    frame[["O2_S1", "O2_S2", "O2_S3"]] = [[240.0, 250.0, 260.0]] * 2
    out = fk.to_schema(frame).set_index("sensor_id")
    assert out.loc["FL1", "do_native_value"].unique() == pytest.approx([240.0])
    assert out.loc["FL2", "do_native_value"].unique() == pytest.approx([250.0])
    assert out.loc["FL3", "do_native_value"].unique() == pytest.approx([260.0])


def test_output_conforms_to_the_schema():
    out = fk.to_schema(raw_frame(rows=5))
    schema.validate(out)
    assert list(out.columns) == list(schema.COLUMNS)
    assert out["site_id"].unique().tolist() == ["florida_keys"]
    assert out["data_source"].unique().tolist() == ["wcci_fl"]
    assert pd.api.types.is_string_dtype(out["deployment_id"])
    assert pd.api.types.is_string_dtype(out["sensor_id"])


def test_temperature_is_all_null_and_qc_is_all_good():
    out = fk.to_schema(raw_frame(rows=5))
    assert out["temperature_c"].isna().all()
    assert (out["qc_flag"] == 0).all()


def test_no_nulls_in_the_measured_columns():
    out = fk.to_schema(raw_frame(rows=5))
    assert not out[["timestamp", "do_sat", "do_mgl", "depth_m", "do_native_value"]].isna().any().any()


def test_output_is_sorted_by_timestamp():
    frame = pd.concat(
        [raw_frame("3oec_2017_7_15_16", 10, rows=3), raw_frame("3oec_2017_7_11_12", 38, rows=3)],
        ignore_index=True,
    )
    out = fk.to_schema(frame)
    assert out["timestamp"].is_monotonic_increasing
    assert out["timestamp"].iloc[0] == pd.Timestamp("2017-07-12 18:00:00", tz="UTC")


def test_build_without_writing_touches_no_disk(tmp_path):
    csv = tmp_path / "tiny.csv"
    raw_frame(rows=3).to_csv(csv, index=False)
    out = fk.build(write=False, path=csv)
    assert len(out) == 9
    assert not list(tmp_path.glob("*.parquet"))


def test_build_round_trips_through_parquet(tmp_path):
    csv = tmp_path / "tiny.csv"
    raw_frame(rows=3).to_csv(csv, index=False)
    destination = tmp_path / "out.parquet"
    written = fk.build(write=True, path=csv, out_path=destination)
    assert destination.is_file()
    schema.validate(pd.read_parquet(destination))
    pd.testing.assert_frame_equal(pd.read_parquet(destination), written)


@pytest.mark.slow
@pytest.mark.skipif(
    not config.FLORIDA_KEYS_CSV.is_file(), reason="Florida Keys raw CSV not present"
)
def test_full_dataset_transforms_to_expected_shape_and_ranges():
    out = fk.to_schema(fk.load_raw())

    assert len(out) == 6_134_415  # 2,044,805 rows x 3 sensors
    assert out["timestamp"].min() == pd.Timestamp("2017-07-12 18:00:00", tz="UTC")
    assert out["timestamp"].max() == pd.Timestamp("2017-07-17 10:00:00", tz="UTC")
    assert out["deployment_id"].nunique() == 4

    assert out["do_sat"].mean() == pytest.approx(104.9, abs=0.5)
    assert 90.0 < out["do_sat"].min() and out["do_sat"].max() < 120.0
    assert out["depth_m"].mean() == pytest.approx(8.23, abs=0.05)
    assert out["depth_m"].between(7.5, 9.0).all()
    assert out["do_native_value"].mean() == pytest.approx(249.99, abs=0.01)

    # Per-sensor, per-deployment spacing stays on the 8 Hz grid.
    one = out[(out["sensor_id"] == "FL1") & (out["deployment_id"] == "3oec_2017_7_15_16")]
    assert one["timestamp"].diff().dropna().unique().tolist() == [
        pd.Timedelta(milliseconds=125)
    ]


@pytest.mark.slow
@pytest.mark.skipif(
    not config.FLORIDA_KEYS_CSV.is_file(), reason="Florida Keys raw CSV not present"
)
def test_diel_cycle_survives_the_timezone_conversion():
    """The one check that would catch a timezone sign error.

    Every range and count assertion above passes just as happily with the hours
    shifted. Benthic metabolism does not: oxygen must trough before dawn and peak in
    the late afternoon, local time.
    """
    out = fk.to_schema(fk.load_raw())
    local_hour = out["timestamp"].dt.tz_convert(fk.SOURCE_TZ).dt.hour
    grouped = out.groupby(local_hour, observed=True)["do_sat"]

    # Deployment windows do not all cover the same hours, and the boundary hours can
    # hold as few as three rows -- deployment 7_11_12 contributes a single 08:00:00
    # instant. Those bins are noise, not signal, so drop anything under a minute.
    counts = grouped.size()
    by_hour = grouped.mean()[counts > 8 * 60]
    assert 8 not in by_hour.index, "the 3-row 08:00 bin should have been excluded"

    trough, peak = by_hour.idxmin(), by_hour.idxmax()
    assert trough in (5, 6, 7), f"trough at {trough}:00, expected pre-dawn"
    assert peak in (18, 19, 20, 21), f"peak at {peak}:00, expected late afternoon"
    assert by_hour.max() - by_hour.min() > 5.0  # a real diel swing, not noise

    # The claim stated as physics rather than as argmin indices.
    assert by_hour.loc[3:6].mean() < by_hour.loc[17:20].mean()


@pytest.mark.slow
@pytest.mark.skipif(
    not config.FLORIDA_KEYS_CSV.is_file(), reason="Florida Keys raw CSV not present"
)
def test_corrected_saturation_straddles_one_hundred_percent():
    """Independent corroboration of the freshwater-reference correction.

    A net-autotrophic benthic system sits a little above saturation and relaxes back
    toward it before dawn, so the series must cross 100%. Referenced against seawater
    solubility the same data would sit at 121-133% and never cross, which no real
    system does.
    """
    out = fk.to_schema(fk.load_raw())
    assert out["do_sat"].min() < 100.0 < out["do_sat"].max()
    assert out["do_sat"].mean() == pytest.approx(104.9, abs=0.5)
