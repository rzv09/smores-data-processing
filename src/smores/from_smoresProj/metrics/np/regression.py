"""Regression metrics, extended from rzv09/smores_proj.

`MARE` is by **Raman Zatsarenko** (github.com/rzv09), from
`src/metrics/np/regression.py` in rzv09/smores_proj -- where it is the whole module,
and the metric reported across 32 of that repo's files.

Local changes to his function: the epsilon is applied as `abs(truth) + eps` rather than
upstream's `abs(truth + eps)`, which is wrong-signed when truth is negative. The two
agree on the Florida Keys data (O2 ~250 umol/L throughout) but not on SMORES channels
sitting at a negative pO2 floor.

`assert_mare_safe`, `MAE` and `RMSE` are local additions with no upstream counterpart.
"""

import numpy as np

def MARE(prediction, truth):
    eps = 1e-8
    # np.abs(truth) + eps, not np.abs(truth + eps): the latter is wrong-signed
    # for negative truth (real below-detection pO2 on dead SMORES channels).
    return np.sum((np.abs(prediction - truth)) / (np.abs(truth) + eps)) / len(truth)

def assert_mare_safe(truth, floor: float = 1e-2):
    """MARE is only well-conditioned once truth is bounded away from zero.
    Call this before computing MARE on a new fold and fail loudly rather
    than silently reporting an unstable number (plan S6)."""
    min_abs = np.min(np.abs(truth))
    if min_abs < floor:
        raise ValueError(f"MARE precondition failed: min|truth|={min_abs} < floor={floor}")

def MAE(prediction, truth):
    return np.mean(np.abs(prediction - truth))

def RMSE(prediction, truth):
    return np.sqrt(np.mean((prediction - truth) ** 2))
