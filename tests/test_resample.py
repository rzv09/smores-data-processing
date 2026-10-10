"""The fixed-grid resampler.

Everything here runs on small synthetic frames except the tests marked `slow`,
which touch the real processed parquet. The synthetic frames carry the legacy
column layout where that is what is under test, so the normalization path is
exercised without the 62 MB file being present.
"""

import numpy as np
import pandas as pd
import pytest

from smores import config
from smores.data import resample, schema

SENSORS = ("FL1", "FL2", "FL3")


def long_frame(start="2017-07-12T18:00:00Z", rows=960, freq="125ms",
               deployment="3oec_2017_7_11_12", sensors=SENSORS, do_sat=103.0,
               qc_flag=0):
    """A frame shaped like the current harmonized schema, one row per sensor-instant."""
    timestamp = pd.date_range(start, periods=rows, freq=freq, tz="UTC")
    pieces = []
    for sensor_id in sensors:
        pieces.append(
            pd.DataFrame(
                {
                    "timestamp": timestamp,
                    "site_id": "florida_keys",
                    "deployment_id": deployment,
                    "sensor_id": sensor_id,
                    "do_sat": np.full(rows, do_sat, dtype="float32"),
                    "temperature_c": np.nan,
                    "do_mgl": 6.4,
                    "do_native_value": 245.0,
                    "do_native_unit": "umol/L",
                    "qc_flag": qc_flag,
                    "data_source": "wcci_fl",
                    "depth_m": 8.3,
                }
            )
        )
    out = pd.concat(pieces, ignore_index=True).sort_values(
        "timestamp", kind="stable", ignore_index=True
    )
    return out.astype(
        {
            name: dtype
            for name, dtype in schema.DTYPES.items()
            if name not in ("deployment_id", "sensor_id")
        }
    )


def legacy_frame(**kwargs):
    """The same data in the older on-disk layout the processed parquet uses."""
    out = long_frame(**kwargs).rename(
        columns={"do_native_value": "raw_o2_umol", "deployment_id": "deployment"}
    )
    out = out.drop(columns="do_native_unit")
    out["height_above_bed_m"] = np.float32(0.35)
    for name in ("deployment", "sensor_id"):
        out[name] = out[name].astype("category")
    return out


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def test_normalize_passes_a_conforming_frame_through():
    frame = long_frame(rows=8)
    out = resample.normalize(frame)
    assert list(out.columns) == list(schema.COLUMNS)
    pd.testing.assert_frame_equal(out, frame[list(schema.COLUMNS)])


def test_normalize_translates_legacy_columns():
    out = resample.normalize(legacy_frame(rows=8))
    schema.validate(out)
    # The unit comes from the legacy column's own name, not from a default.
    assert set(out["do_native_unit"].unique()) == {"umol/L"}
    assert out["do_native_value"].eq(np.float32(245.0)).all()
    assert out["deployment_id"].eq("3oec_2017_7_11_12").all()
    assert "height_above_bed_m" not in out.columns


def test_normalize_rejects_a_frame_it_cannot_fix():
    frame = long_frame(rows=8).drop(columns="do_sat")
    with pytest.raises(schema.SchemaError):
        resample.normalize(frame)


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------

def test_bins_land_on_the_grid_and_count_their_reads():
    # 960 reads at 8 Hz is exactly two minutes, so one bin per sensor.
    out = resample.resample_long(long_frame(rows=960), freq="2min")
    assert len(out) == len(SENSORS)
    assert out["timestamp"].eq(pd.Timestamp("2017-07-12T18:00:00Z")).all()
    assert out["n_obs"].eq(960).all()
    assert list(out.columns) == list(resample.OUTPUT_COLUMNS)
    schema.validate(out[list(schema.COLUMNS)])


def test_grid_is_utc_anchored_not_relative_to_the_first_sample():
    """A deployment starting mid-bin still gets grid-aligned edges."""
    out = resample.resample_long(
        long_frame(start="2017-07-12T18:00:30Z", rows=960), freq="2min"
    )
    edges = out["timestamp"].unique()
    assert set(pd.DatetimeIndex(edges).minute) <= {0, 2}
    assert pd.DatetimeIndex(edges).second.tolist() == [0] * len(edges)


def test_aggregate_is_the_median_not_the_mean():
    """An optode spike must not move the bin; a mean would carry it through."""
    frame = long_frame(rows=9, freq="1s", sensors=("FL1",), do_sat=100.0)
    frame.loc[frame.index[-1], "do_sat"] = np.float32(1000.0)
    out = resample.resample_long(frame, freq="2min")
    assert len(out) == 1
    assert out["do_sat"].iloc[0] == pytest.approx(100.0)
    assert out["n_obs"].iloc[0] == 9


def test_empty_bins_are_dropped_rather_than_emitted_as_nulls():
    gap = pd.concat(
        [
            long_frame(start="2017-07-12T18:00:00Z", rows=8, freq="1s", sensors=("FL1",)),
            long_frame(start="2017-07-12T18:10:00Z", rows=8, freq="1s", sensors=("FL1",)),
        ],
        ignore_index=True,
    )
    out = resample.resample_long(gap, freq="2min")
    assert len(out) == 2  # not 6: the four empty bins between them are absent
    assert out["n_obs"].eq(8).all()
    assert not out["do_sat"].isna().any()


def test_bins_never_span_a_deployment_boundary():
    """Two deployments sharing a bin's worth of clock time stay separate rows."""
    overlapping = pd.concat(
        [
            long_frame(rows=8, freq="1s", sensors=("FL1",), deployment="dep_a", do_sat=100.0),
            long_frame(rows=8, freq="1s", sensors=("FL1",), deployment="dep_b", do_sat=200.0),
        ],
        ignore_index=True,
    )
    out = resample.resample_long(overlapping, freq="2min")
    assert len(out) == 2
    assert sorted(out["deployment_id"]) == ["dep_a", "dep_b"]
    # Neither bin is the 150.0 that averaging across the boundary would give.
    assert sorted(out["do_sat"].round(1)) == [100.0, 200.0]


def test_min_obs_drops_thin_bins():
    thin = pd.concat(
        [
            long_frame(start="2017-07-12T18:00:00Z", rows=100, freq="125ms", sensors=("FL1",)),
            long_frame(start="2017-07-12T18:02:00Z", rows=4, freq="125ms", sensors=("FL1",)),
        ],
        ignore_index=True,
    )
    assert len(resample.resample_long(thin, freq="2min")) == 2
    kept = resample.resample_long(thin, freq="2min", min_obs=10)
    assert len(kept) == 1
    assert kept["n_obs"].iloc[0] == 100


def test_min_obs_below_one_is_rejected():
    with pytest.raises(ValueError, match="min_obs"):
        resample.resample_long(long_frame(rows=8), min_obs=0)


# ---------------------------------------------------------------------------
# Native rate: the same code at different source frequencies
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("freq", "expected"),
    [("125ms", pd.Timedelta("125ms")), ("1s", pd.Timedelta("1s")),
     ("1min", pd.Timedelta("1min")), ("5min", pd.Timedelta("5min"))],
)
def test_native_interval_recovers_the_source_rate(freq, expected):
    assert resample.native_interval(long_frame(rows=20, freq=freq)) == expected


def test_native_interval_ignores_deployment_gaps_and_dropped_polls():
    """Measured within a series, by median: neither a long gap nor a dropout moves it."""
    frame = pd.concat(
        [
            long_frame(start="2017-07-12T18:00:00Z", rows=40, freq="1s", deployment="dep_a"),
            # dep_b starts 12 hours later, and skips a read 20 s in.
            long_frame(start="2017-07-13T06:00:00Z", rows=20, freq="1s", deployment="dep_b"),
            long_frame(start="2017-07-13T06:00:30Z", rows=20, freq="1s", deployment="dep_b"),
        ],
        ignore_index=True,
    ).sort_values("timestamp", kind="stable", ignore_index=True)
    assert resample.native_interval(frame) == pd.Timedelta("1s")


def test_native_interval_is_nat_without_two_reads_to_compare():
    assert pd.isna(resample.native_interval(long_frame(rows=1)))


@pytest.mark.parametrize(
    ("native", "freq", "expected"),
    [("125ms", "2min", 960), ("1s", "2min", 120), ("1min", "5min", 5), ("5min", "1h", 12)],
)
def test_obs_per_full_bin_scales_with_the_source_rate(native, freq, expected):
    assert resample.obs_per_full_bin(freq, pd.Timedelta(native)) == expected


def test_coverage_means_the_same_thing_at_any_source_rate():
    """A half-full bin reads 0.5 whether the source is 8 Hz or 1 Hz."""
    for native, rows in (("125ms", 480), ("1s", 60)):
        out = resample.resample_long(
            long_frame(rows=rows, freq=native, sensors=("FL1",)), freq="2min"
        )
        assert out["coverage"].iloc[0] == pytest.approx(0.5)
        assert out["n_obs"].iloc[0] == rows


def test_min_coverage_is_portable_where_min_obs_is_not():
    """One threshold, two source rates, the same bins dropped."""
    for native, full in (("125ms", 960), ("1s", 120)):
        frame = pd.concat(
            [
                long_frame(start="2017-07-12T18:00:00Z", rows=full,
                           freq=native, sensors=("FL1",)),
                long_frame(start="2017-07-12T18:02:00Z", rows=full // 4,
                           freq=native, sensors=("FL1",)),
            ],
            ignore_index=True,
        )
        kept = resample.resample_long(frame, freq="2min", min_coverage=0.5)
        assert len(kept) == 1
        assert kept["coverage"].iloc[0] == pytest.approx(1.0, abs=1e-3)


def test_freq_finer_than_the_native_interval_is_rejected():
    frame = long_frame(rows=20, freq="1min")
    with pytest.raises(ValueError, match="cannot upsample"):
        resample.resample_long(frame, freq="1s")


def test_interval_can_be_given_explicitly():
    """A nominal rate overrides what the timestamps show."""
    frame = long_frame(rows=60, freq="1s", sensors=("FL1",))
    out = resample.resample_long(frame, freq="2min", interval="2s")
    # 60 bins' worth of reads against a 2 s nominal rate = 60 of 60 expected.
    assert out["coverage"].iloc[0] == pytest.approx(1.0)


def test_coverage_is_null_when_no_rate_can_be_inferred():
    out = resample.resample_long(long_frame(rows=1, sensors=("FL1",)), freq="2min")
    assert len(out) == 1
    assert out["coverage"].isna().all()
    assert out["n_obs"].iloc[0] == 1


def test_min_coverage_without_an_inferable_rate_is_rejected():
    with pytest.raises(ValueError, match="min_coverage"):
        resample.resample_long(long_frame(rows=1, sensors=("FL1",)), min_coverage=0.5)


def test_min_coverage_outside_zero_to_one_is_rejected():
    with pytest.raises(ValueError, match="min_coverage"):
        resample.resample_long(long_frame(rows=8), min_coverage=1.5)


def test_calendar_offsets_resample_without_a_coverage_figure():
    """"1h" has a fixed width; "ME" does not, and must still work."""
    assert pd.isna(resample.bin_width("ME"))
    out = resample.resample_long(long_frame(rows=20, freq="1min"), freq="ME")
    assert len(out) == len(SENSORS)
    assert out["coverage"].isna().all()


# ---------------------------------------------------------------------------
# QC handling
# ---------------------------------------------------------------------------

def test_flagged_rows_are_excluded_from_the_aggregate():
    frame = pd.concat(
        [
            long_frame(rows=8, freq="1s", sensors=("FL1",), do_sat=100.0, qc_flag=0),
            long_frame(rows=8, freq="1s", sensors=("FL1",), do_sat=900.0, qc_flag=2),
        ],
        ignore_index=True,
    )
    out = resample.resample_long(frame, freq="2min")
    assert len(out) == 1
    assert out["do_sat"].iloc[0] == pytest.approx(100.0)
    assert out["n_obs"].iloc[0] == 8  # the flagged reads are not counted either


def test_a_bin_holding_an_interpolated_read_is_marked_interpolated():
    frame = pd.concat(
        [
            long_frame(rows=8, freq="1s", sensors=("FL1",), qc_flag=0),
            long_frame(rows=2, freq="1s", sensors=("FL1",), qc_flag=1),
        ],
        ignore_index=True,
    )
    out = resample.resample_long(frame, freq="2min")
    assert out["qc_flag"].tolist() == [1]
    assert out["qc_flag"].dtype == np.dtype("int8")


def test_all_rows_excluded_yields_a_typed_empty_frame():
    out = resample.resample_long(long_frame(rows=8, qc_flag=3), freq="2min")
    assert out.empty
    assert list(out.columns) == list(resample.OUTPUT_COLUMNS)


# ---------------------------------------------------------------------------
# Paths and I/O
# ---------------------------------------------------------------------------

def test_default_out_path_records_the_interval():
    assert resample.default_out_path("2min").name == "florida_keys_o2_2min.parquet"
    assert resample.default_out_path("5min").name == "florida_keys_o2_5min.parquet"
    assert resample.default_out_path("2min").parent == config.PROCESSED_DIR


def test_default_out_path_follows_an_arbitrary_source(tmp_path):
    out = resample.default_out_path("30s", tmp_path / "biscayne_o2.parquet")
    assert out == tmp_path / "biscayne_o2_30s.parquet"


def test_build_round_trips_through_parquet(tmp_path):
    source = tmp_path / "source.parquet"
    long_frame(rows=960).to_parquet(source, index=False)
    out_path = tmp_path / "binned.parquet"

    out = resample.build(freq="2min", source=source, out_path=out_path)
    assert out_path.is_file()
    reloaded = pd.read_parquet(out_path)
    schema.validate(reloaded[list(schema.COLUMNS)])
    # Parquet reads the id columns back as pandas string dtype rather than object;
    # the schema accepts either, so compare values and not that one dtype.
    for name in ("deployment_id", "sensor_id"):
        reloaded[name] = reloaded[name].astype(object)
    pd.testing.assert_frame_equal(reloaded, out)


def test_build_can_summarize_without_writing(tmp_path):
    source = tmp_path / "source.parquet"
    long_frame(rows=960).to_parquet(source, index=False)
    out = resample.build(freq="2min", write=False, source=source, out_path=tmp_path / "x.parquet")
    assert not (tmp_path / "x.parquet").exists()
    assert len(out) == len(SENSORS)


def test_cli_rejects_out_with_multiple_freqs():
    with pytest.raises(SystemExit):
        resample.parse_args(["--freq", "2min", "5min", "--out", "x.parquet"])


# ---------------------------------------------------------------------------
# The real file
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(
    not config.FLORIDA_KEYS_PARQUET.is_file(),
    reason="processed parquet not built yet (make data)",
)
def test_real_parquet_bins_to_the_expected_shape():
    out = resample.build(freq="2min", write=False)
    schema.validate(out[list(schema.COLUMNS)])
    assert sorted(out["sensor_id"].unique()) == list(SENSORS)
    assert out["deployment_id"].nunique() == 4
    # 8 Hz for two minutes; only the partial bins at deployment edges fall short.
    assert out["n_obs"].max() == 960
    assert out["n_obs"].mode().iloc[0] == 960
    assert out.groupby("deployment_id", observed=True)["timestamp"].is_monotonic_increasing.all()
