"""Physical unit conversions shared across sites.

Oxygen solubility follows Garcia & Gordon (1992), "Oxygen solubility in seawater:
better fitting equations", Limnol. Oceanogr. 37(6), using the Benson & Krause
coefficient set. The equation returns umol/kg.

Converting umol/kg to umol/L needs seawater density, which depends on temperature,
salinity and pressure. Rather than pull in a full equation of state, callers pass a
constant density appropriate to their site. At the temperatures and salinities in this
project that is accurate to well under 1%, far inside the oxygen sensors' own
calibration uncertainty. If `gsw` is ever added as a dependency, replace the density
constants with `gsw.rho` and delete the approximation.
"""

import numpy as np

# Garcia & Gordon (1992), Benson & Krause coefficients; solubility in umol/kg.
_A = (5.80871, 3.20291, 4.17887, 5.10006, -9.86643e-2, 3.80369)
_B = (-7.01577e-3, -7.70028e-3, -1.13864e-2, -9.51519e-3)
_C0 = -2.75915e-7

# Molar mass of O2, mg per umol.
O2_MG_PER_UMOL = 0.031998

# Density of pure water near 29.5 C, kg/m3. Used as the reference for sensor output
# that was converted to umol/L without a salinity correction.
DENSITY_FRESH_KG_M3 = 996.0


def o2sol_umol_kg(temperature_c, salinity):
    """O2 solubility of air-saturated water at one atmosphere, in umol/kg.

    `temperature_c` is degrees Celsius (ITS-90), `salinity` is practical salinity.
    Scalars or arrays.
    """
    t = np.asarray(temperature_c, dtype="float64")
    s = np.asarray(salinity, dtype="float64")
    ts = np.log((298.15 - t) / (273.15 + t))
    ln_c = (
        _A[0]
        + _A[1] * ts
        + _A[2] * ts**2
        + _A[3] * ts**3
        + _A[4] * ts**4
        + _A[5] * ts**5
        + s * (_B[0] + _B[1] * ts + _B[2] * ts**2 + _B[3] * ts**3)
        + _C0 * s**2
    )
    return np.exp(ln_c)


def o2sol_umol_l(temperature_c, salinity, density_kg_m3):
    """O2 solubility in umol/L, using `density_kg_m3` for the umol/kg conversion."""
    return o2sol_umol_kg(temperature_c, salinity) * (density_kg_m3 / 1000.0)


def gravity_m_s2(latitude_deg):
    """Standard gravity at a latitude, from the 1967 Geodetic Reference System."""
    phi = np.radians(latitude_deg)
    return 9.780327 * (
        1 + 0.0053024 * np.sin(phi) ** 2 - 0.0000058 * np.sin(2 * phi) ** 2
    )


def kpa_per_metre(density_kg_m3, latitude_deg):
    """Hydrostatic pressure per metre of water, in kPa."""
    return density_kg_m3 * gravity_m_s2(latitude_deg) / 1000.0


def pressure_kpa_to_depth_m(pressure_kpa, density_kg_m3, latitude_deg):
    """Depth of water above a sensor, from its *gauge* pressure reading in kPa.

    Gauge, not absolute: the head of water is `p / (rho * g)` only when atmospheric
    pressure has already been removed. A reading below ~101 kPa at a site metres deep
    is itself the evidence the sensor reported gauge pressure.
    """
    return pressure_kpa / kpa_per_metre(density_kg_m3, latitude_deg)


def saturation_percent(o2_umol_l, o2sol_reference_umol_l):
    """Percent air saturation of an O2 concentration against a solubility reference.

    The reference must match the convention the *reported* concentration was derived
    under. Where a sensor's percent-saturation reading was converted to umol/L using
    freshwater solubility, dividing by that same freshwater solubility recovers the
    original calibrated percentage — see `smores.data.florida_keys`.
    """
    return 100.0 * o2_umol_l / o2sol_reference_umol_l


def mgl_from_saturation(do_sat, o2sol_insitu_umol_l):
    """mg/L implied by a percent air saturation and the true in-situ solubility."""
    return (do_sat / 100.0) * o2sol_insitu_umol_l * O2_MG_PER_UMOL
