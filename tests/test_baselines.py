"""Tests for the forecast baselines.

The ones that matter are the leakage and deployment-boundary tests. A forecast harness
that quietly reads the future, or that lags across a 3-28 h gap between deployments,
produces numbers that look good and mean nothing -- and nothing else in the suite would
catch it.
"""

import numpy as np
import pandas as pd
import pytest

from smores.models import baselines


def wide_frame(start, hours, freq="30min", amplitude=10.0, level=250.0, phase_h=0.0):
    """A synthetic deployment: one diel sine, three identical sensors."""
    step_min = pd.Timedelta(freq).total_seconds() / 60
    index = pd.date_range(start, periods=int(hours * 60 / step_min), freq=freq, tz="UTC")
    hour = baselines.local_hour(index)
    values = level + amplitude * np.sin(2 * np.pi * (hour - phase_h) / 24)
    frame = pd.DataFrame({s: values for s in baselines.SENSORS}, index=index)
    frame[baselines.MEAN3] = values
    return frame


@pytest.fixture
def sine_frames():
    return {
        f"dep{i}": wide_frame(f"2017-07-{11 + 2 * i} 12:00", hours=36)
        for i in range(4)
    }


# ---------------------------------------------------------------------------
# Sanity anchors
# ---------------------------------------------------------------------------

def test_persistence_at_zero_horizon_is_exact(sine_frames):
    """A zero-step forecast is the observation itself. If this drifts, the shift
    alignment is off by one and every other number is quietly wrong."""
    frame = sine_frames["dep0"]
    profile = baselines.fit_hour_climatology([frame], baselines.MEAN3)
    y_true, y_pred = baselines.predict(
        "persistence", frame, 0, profile, baselines.MEAN3, baselines.MEAN3
    )
    assert len(y_true) > 0
    np.testing.assert_allclose(y_pred, y_true)


def test_persistence_error_grows_with_horizon(sine_frames):
    frame = sine_frames["dep0"]
    profile = baselines.fit_hour_climatology([frame], baselines.MEAN3)
    errors = []
    for steps in (1, 4, 12):
        y_true, y_pred = baselines.predict(
            "persistence", frame, steps, profile, baselines.MEAN3, baselines.MEAN3
        )
        errors.append(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    assert errors[0] < errors[1] < errors[2]


def test_climatology_beats_persistence_at_half_period(sine_frames):
    """On a pure diel sine, persistence at 12 h predicts the opposite phase -- the
    worst case -- while an hour-of-day model should be close to exact."""
    train = [f for name, f in sine_frames.items() if name != "dep3"]
    test = sine_frames["dep3"]
    steps = 24  # 12 h at 30-min resolution
    profile = baselines.fit_hour_climatology(train, baselines.MEAN3)

    scores = {}
    for model in ("persistence", "climatology"):
        y_true, y_pred = baselines.predict(
            model, test, steps, profile, baselines.MEAN3, baselines.MEAN3
        )
        scores[model] = np.sqrt(np.mean((y_pred - y_true) ** 2))
    assert scores["climatology"] < scores["persistence"]


# ---------------------------------------------------------------------------
# The two that actually guard correctness
# ---------------------------------------------------------------------------

def test_forecast_never_crosses_a_deployment_boundary():
    """Deployments are separated by 3-28 h gaps. A lag that straddles one is fabricated.

    Two deployments held at different constant levels: every prediction inside the
    second must come from the second's own level, never the first's.
    """
    first = wide_frame("2017-07-11 12:00", hours=18, amplitude=0.0, level=100.0)
    second = wide_frame("2017-07-14 12:00", hours=18, amplitude=0.0, level=200.0)
    frames = {"first": first, "second": second}

    profile = baselines.fit_hour_climatology([first], baselines.MEAN3)
    for model in ("persistence", "drift", "persist_clim"):
        _, y_pred = baselines.predict(
            model, frames["second"], 4, profile, baselines.MEAN3, baselines.MEAN3
        )
        assert len(y_pred) > 0
        assert np.all(np.abs(y_pred - 200.0) < 1e-6), f"{model} leaked across the gap"


def test_climatology_ignores_the_held_out_deployment(sine_frames):
    """The fitted profile must depend on training deployments only. Mutating the
    held-out frame beyond recognition must not move it at all."""
    train = [sine_frames["dep0"], sine_frames["dep1"]]
    before = baselines.fit_hour_climatology(train, baselines.MEAN3)

    poisoned = sine_frames["dep3"].copy()
    poisoned[baselines.MEAN3] = poisoned[baselines.MEAN3] * -50 + 9999
    after = baselines.fit_hour_climatology(train, baselines.MEAN3)

    pd.testing.assert_series_equal(before, after)


def test_hour_bins_below_min_count_get_no_anomaly():
    """A naive hour-of-day mean on this dataset is confounded by uneven coverage: an
    hour holding a handful of rows produced a spurious trough. Under-populated hours
    must fall back to zero rather than contribute a noisy estimate."""
    # 5-min bins so a well-covered hour clears min_count with room to spare.
    frame = wide_frame("2017-07-11 12:00", hours=36, freq="5min")
    # Keep every row except hour 3, which is thinned to two samples.
    hours = np.floor(baselines.local_hour(frame.index)).astype(int)
    keep = (hours != 3) | (np.cumsum(hours == 3) <= 2)
    sparse = frame[keep]

    profile = baselines.fit_hour_climatology(
        [sparse], baselines.MEAN3, min_count=10, min_deployments=1
    )
    assert profile[3] == 0.0
    assert profile.abs().sum() > 0  # well-populated hours still carry signal


def test_hour_covered_by_one_deployment_is_rejected():
    """Sample count alone is not enough. Hour-of-day coverage across the four Florida
    Keys deployments is uneven -- hours 7, 8 and 10 are each covered by exactly one --
    so one piece's 12 samples would otherwise clear the gate and let its local level
    masquerade as a diel feature."""
    full = wide_frame("2017-07-11 12:00", hours=36, freq="5min")
    # A second deployment starting 16:00 UTC, i.e. local hour 12, running four hours.
    partial = wide_frame("2017-07-14 16:00", hours=4, freq="5min", level=400.0)
    assert set(np.floor(baselines.local_hour(partial.index)).astype(int)) == {12, 13, 14, 15}

    profile = baselines.fit_hour_climatology(
        [full, partial], baselines.MEAN3, min_count=10, min_deployments=2
    )
    # Hour 2 is covered by `full` alone: plenty of samples, one deployment.
    assert profile[2] == 0.0
    # An hour both deployments cover survives.
    assert profile[13] != 0.0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def test_skill_score_sign_convention(sine_frames):
    pooled = baselines.pool(baselines.evaluate(sine_frames, "30min", horizons_min=(60,)))
    persistence = pooled[pooled["model"] == "persistence"]
    assert len(persistence) == 1
    # Persistence is the reference, so its skill is exactly zero by construction.
    assert persistence["skill_vs_persistence"].iloc[0] == pytest.approx(0.0, abs=1e-12)

    better = pooled[pooled["rmse"] < persistence["rmse"].iloc[0]]
    assert (better["skill_vs_persistence"] > 0).all()
    worse = pooled[pooled["rmse"] > persistence["rmse"].iloc[0]]
    assert (worse["skill_vs_persistence"] < 0).all()


def test_evaluate_uses_every_deployment_as_a_fold(sine_frames):
    per_fold = baselines.evaluate(sine_frames, "30min", horizons_min=(60,))
    assert set(per_fold["fold"]) == set(sine_frames)
    assert set(per_fold["model"]) == set(baselines.MODELS)


def test_optode_arms_share_one_target(sine_frames):
    """The comparison is only meaningful if both arms predict the same series.
    Scoring a single sensor against itself would hand the averaged arm a free win."""
    per_fold = baselines.optode_arms(sine_frames, "30min")
    assert set(per_fold["target"]) == {baselines.MEAN3}
    assert set(per_fold["input"]) == {baselines.MEAN3, baselines.SENSORS[0]}


# ---------------------------------------------------------------------------
# Against the real dataset
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_real_data_loads_as_separate_deployments():
    frames = baselines.load_wide(freq="5min")
    assert len(frames) == 4
    for frame in frames.values():
        assert baselines.MEAN3 in frame.columns
        assert frame.index.is_monotonic_increasing
        # 14-20 h pieces: nothing should span more than a day.
        assert (frame.index[-1] - frame.index[0]) < pd.Timedelta(hours=24)
