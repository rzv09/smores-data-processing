"""Oxygen solubility and the conversions built on it."""

import numpy as np
import pytest

from smores.data import units


@pytest.mark.parametrize(
    ("temperature_c", "salinity", "expected_umol_kg"),
    [
        # Garcia & Gordon (1992) / Benson & Krause reference points. Air-saturated
        # solubility falls with both temperature and salinity, and these anchor the
        # coefficient set at the corners of the range this project cares about.
        (0.0, 35.0, 347.90),
        (10.0, 35.0, 274.61),
        (20.0, 35.0, 225.54),
        (30.0, 35.0, 190.74),
        (20.0, 0.0, 284.65),
    ],
)
def test_o2sol_matches_published_values(temperature_c, salinity, expected_umol_kg):
    got = units.o2sol_umol_kg(temperature_c, salinity)
    assert got == pytest.approx(expected_umol_kg, rel=1e-3)


def test_o2sol_decreases_with_temperature_and_salinity():
    assert units.o2sol_umol_kg(31.0, 35.5) < units.o2sol_umol_kg(28.0, 35.5)
    assert units.o2sol_umol_kg(29.5, 35.5) < units.o2sol_umol_kg(29.5, 0.0)


def test_o2sol_accepts_arrays():
    got = units.o2sol_umol_kg(np.array([28.0, 29.5, 31.0]), 35.5)
    assert got.shape == (3,)
    assert np.all(np.diff(got) < 0)


def test_gravity_at_equator_and_pole():
    assert units.gravity_m_s2(0.0) == pytest.approx(9.7803, abs=1e-3)
    assert units.gravity_m_s2(90.0) == pytest.approx(9.8322, abs=1e-3)
    assert units.gravity_m_s2(24.7253) == pytest.approx(9.78937, abs=1e-4)


def test_seawater_is_about_ten_kpa_per_metre():
    """The coincidence worth knowing: ~1 m of seawater ~= 10 kPa."""
    assert units.kpa_per_metre(1022.0, 24.7253) == pytest.approx(10.0, abs=0.01)


def test_pressure_to_depth_is_linear_and_inverts():
    depth = units.pressure_kpa_to_depth_m(82.36, 1022.0, 24.7253)
    assert depth == pytest.approx(8.232, abs=1e-3)
    assert units.pressure_kpa_to_depth_m(0.0, 1022.0, 24.7253) == 0.0
    doubled = units.pressure_kpa_to_depth_m(164.72, 1022.0, 24.7253)
    assert doubled == pytest.approx(2 * depth)


def test_saturation_of_the_reference_itself_is_exactly_100_percent():
    o2sol = units.o2sol_umol_l(29.5, 35.5, 1022.0)
    assert units.saturation_percent(o2sol, o2sol) == pytest.approx(100.0)


def test_saturation_and_mgl_round_trip():
    """do_sat -> do_mgl inverts through the in-situ solubility."""
    o2sol = units.o2sol_umol_l(29.5, 35.5, 1022.0)
    do_sat = units.saturation_percent(206.0, o2sol)
    assert do_sat == pytest.approx(105.2, abs=0.1)

    mgl = units.mgl_from_saturation(do_sat, o2sol)
    recovered_umol = mgl / units.O2_MG_PER_UMOL
    assert recovered_umol == pytest.approx(206.0, rel=1e-9)


def test_fresh_reference_is_about_twenty_percent_higher_than_seawater():
    """The ratio underpinning the Florida Keys oxygen correction."""
    fresh = units.o2sol_umol_l(29.5, 0.0, units.DENSITY_FRESH_KG_M3)
    site = units.o2sol_umol_l(29.5, 35.5, 1022.0)
    assert site / fresh == pytest.approx(0.822, abs=0.005)


def test_fresh_to_seawater_ratio_is_insensitive_to_temperature():
    """Why the correction survives temperature never having been logged.

    Absolute saturation shifts with the assumed temperature, but the ratio used to
    convert freshwater-referenced output to true in-situ concentration does not.
    """
    ratios = [
        units.o2sol_umol_kg(t, 35.5) / units.o2sol_umol_kg(t, 0.0)
        for t in (28.0, 29.5, 31.0)
    ]
    assert max(ratios) - min(ratios) < 0.005
