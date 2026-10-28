"""The Biscayne Bay pipeline.

Synthetic tests run on a small dump written by `synthetic.write_biscayne_dump`, shaped
like the real Harvard files. The `slow` tests read the real 13,476-file dump.
"""

import numpy as np
import pandas as pd
import pytest
import synthetic as syn

from smores import config
from smores.data import biscayne_bay as bb
from smores.data import integrity, schema

DEPLOYED_UTC = pd.Timestamp("2026-05-14 01:30", tz="UTC")  # 21:30 EDT on May 13


@pytest.fixture(scope="module")
def dump(tmp_path_factory):
    return syn.write_biscayne_dump(tmp_path_factory.mktemp("bb"))


@pytest.fixture(scope="module")
def built(dump):
    return bb.pipeline(dump, verbose=False)


@pytest.fixture(scope="module")
def grid(built):
    return built[0]


def by_sensor(grid, channel):
    return grid[grid["sensor_id"] == bb.sensor_id(channel)]


# --- Clock -----------------------------------------------------------------------


def test_files_are_keyed_by_prefix_and_chunk(dump):
    files = bb.discover(dump)
    assert set(files) == {(5, 1), (5, 2), (6, 1)}
    assert set(files[(5, 1)]) == {"RDO", "RTC", "External_PTS_Bar100"}


def test_rtc_fit_recovers_the_tick_rate(dump):
    fit = bb.fit_rtc(pd.read_csv(dump / "SENS_5_1_RTC.csv"))
    assert 1.0 / fit["slope"] == pytest.approx(syn.TICKS_PER_S, rel=1e-6)
    assert fit["rms"] < 0.01


def test_rtc_less_chunk_borrows_and_orphan_stays_unresolved(dump):
    files = bb.discover(dump)
    raw, spans = bb.load_rdo(files)
    clock = bb.resolve_missing_clocks(bb.clock_table(files), spans).set_index(["prefix", "chunk"])
    assert clock.loc[(5, 1), "clock_source"] == "own_rtc"
    assert clock.loc[(5, 2), "clock_source"] == "borrowed_prev"
    assert clock.loc[(6, 1), "clock_source"] == "unresolved"


def test_borrow_refuses_a_tick_counter_reset():
    donor = {"t0": 1e9, "t1": 2e9, "slope": 1e-8, "icpt": 0.0}
    assert bb._borrow(2.1e9, 3e9, donor) is not None
    assert bb._borrow(0.0, 5e8, donor) is not None  # just before the donor: still continuous
    far = 2e9 + 3 * 3600 * syn.TICKS_PER_S
    assert bb._borrow(far, far + 1e9, donor) is None  # > 2 h away


def test_unplaceable_reads_are_dropped(grid):
    assert bb.sensor_id(syn.ORPHAN_CHANNEL) not in set(grid["sensor_id"])


# --- Timestamps and deployments ---------------------------------------------------


def test_rtc_is_read_as_edt_and_output_as_utc(grid):
    assert str(grid["timestamp"].dt.tz) == "UTC"
    assert grid["timestamp"].min() == pd.Timestamp("2026-05-14 00:00", tz="UTC")  # 20:00 EDT


def test_bucket_test_is_its_own_deployment(grid):
    bucket = grid["deployment_id"] == bb.BUCKET_DEPLOYMENT_ID
    assert (grid.loc[bucket, "timestamp"] < DEPLOYED_UTC).all()
    assert (grid.loc[~bucket, "timestamp"] >= DEPLOYED_UTC).all()
    assert set(grid.loc[~bucket, "deployment_id"]) == {bb.DEPLOYMENT_ID}


# --- Oxygen and quality ------------------------------------------------------------


def test_do_sat_reads_near_saturation_in_the_bucket(grid):
    live = by_sensor(grid, syn.LIVE_CHANNEL)
    bucket = live[live["deployment_id"] == bb.BUCKET_DEPLOYMENT_ID]
    assert bucket["do_sat"].between(95, 105).all()


def test_native_value_is_po2_in_mmhg(grid):
    live = by_sensor(grid, syn.LIVE_CHANNEL)
    assert set(live["do_native_unit"]) == {"pO2_mmHg"}
    assert live["do_native_value"].between(119.0, 181.0).all()
    # ~0.65 % sat per mmHg at these temperatures.
    ratio = live["do_sat"] / live["do_native_value"]
    assert ratio.between(0.64, 0.67).all()


def test_do_mgl_is_seawater_referenced(grid):
    """At ~100% sat and 28 C, seawater (S 35) holds ~6.5 mg/L; fresh water would be ~7.9."""
    live = by_sensor(grid, syn.LIVE_CHANNEL)
    near_100 = live[live["do_sat"].between(99, 101)]
    assert near_100["do_mgl"].between(6.0, 7.0).all()


def test_stuck_channel_is_masked_not_deleted(grid):
    stuck = by_sensor(grid, syn.STUCK_CHANNEL)
    in_situ = stuck[stuck["deployment_id"] == bb.DEPLOYMENT_ID]
    assert len(in_situ) > 0
    assert (in_situ["qc_flag"] == 3).all()


def test_failing_quality_codes_flag_their_bins(grid):
    live = by_sensor(grid, syn.LIVE_CHANNEL)
    start, end = (t.tz_localize(bb.SOURCE_TZ).tz_convert("UTC") for t in syn.BB_FLAGGED_LOCAL)
    window = live["timestamp"].between(start, end - pd.Timedelta(minutes=2))
    assert (live.loc[window, "qc_flag"] == 2).all()
    assert (live.loc[~window & (live["deployment_id"] == bb.DEPLOYMENT_ID), "qc_flag"] == 0).all()


def test_depth_comes_from_bar100_and_is_zero_in_air(grid):
    live = by_sensor(grid, syn.LIVE_CHANNEL).set_index("timestamp")
    assert live.loc[: pd.Timestamp("2026-05-14 00:40", tz="UTC"), "depth_m"].eq(0).all()
    submerged = live.loc[DEPLOYED_UTC:, "depth_m"]
    assert submerged.dropna().between(0.89, 0.90).all()  # 0.09 bar gauge


def test_grid_is_schema_valid_and_passes_integrity(built):
    grid, checks = built
    schema.validate(grid.drop(columns="n_obs"))
    report = integrity.build_report(grid, bb.SITE_ID, checks)
    assert report.ok, str(report)
    names = {c.name: c for c in report.checks}
    assert names["known_params"].detail.startswith("1 read")
    assert names["saturation_matches_sensor"].passed
    assert names["diel_cycle_midwater"].passed


def test_liveness_ignores_the_bucket_to_burial_drop():
    """A channel that reads ~100% in the bucket and flatlines on burial is dead in situ."""
    local = pd.date_range("2026-05-13 18:00", "2026-05-14 06:00", freq="52s")
    po2 = np.where(local < bb.DEPLOYMENT_START_LOCAL, 150.0, 1.0)
    polls = pd.DataFrame(
        {
            "channel": 10,
            "local_time": local,
            "po2": po2,
            "do_mgl_reported": po2 / 20,
            "pctsat_reported": po2 / 1.5,
        }
    )
    assert set(bb.liveness(polls)["state"]) == {"stuck-constant"}


# --- Real data -----------------------------------------------------------------------

real = pytest.mark.skipif(
    not config.BISCAYNE_HARVARD_DIR.is_dir(), reason="Biscayne Bay raw dump not present"
)


@pytest.fixture(scope="module")
def real_built():
    return bb.pipeline(verbose=False)


@pytest.mark.slow
@real
def test_real_dump_passes_integrity(real_built):
    grid, checks = real_built
    report = integrity.build_report(grid, bb.SITE_ID, checks)
    assert report.ok, str(report)


@pytest.mark.slow
@real
def test_real_dump_live_channels_match_the_eda(real_built):
    """data/eda/REPORT.md: usable channels are ch2, 3, 25 (midwater), 7, 26 (upper), 8."""
    grid, _ = real_built
    good = grid[(grid["qc_flag"] == 0) & (grid["deployment_id"] == bb.DEPLOYMENT_ID)]
    assert set(good["sensor_id"]) == {"ch2", "ch3", "ch25", "ch7", "ch26", "ch8"}


@pytest.mark.slow
@real
def test_real_dump_spans_the_deployment(real_built):
    grid, _ = real_built
    assert grid["timestamp"].min() == pd.Timestamp("2026-05-13 14:10", tz="UTC")
    assert grid["timestamp"].max().floor("D") == pd.Timestamp("2026-06-22", tz="UTC")
