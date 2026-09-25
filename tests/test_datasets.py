"""Dataset availability.

Florida Keys must be present and readable. Biscayne Bay arrives later in the
project, so its absence must never break an import or a test run.
"""

import pandas as pd
import pytest

from smores import config

FLORIDA_KEYS_COLUMNS = [
    "deployment",
    "t",
    "t_increase",
    "Vx",
    "Vy",
    "Vz",
    "P",
    "O2_S1",
    "O2_S2",
    "O2_S3",
]


def test_florida_keys_csv_is_present():
    assert config.FLORIDA_KEYS_CSV.is_file(), f"missing dataset: {config.FLORIDA_KEYS_CSV}"


def test_florida_keys_csv_has_expected_columns():
    head = pd.read_csv(config.FLORIDA_KEYS_CSV, nrows=5)
    assert list(head.columns) == FLORIDA_KEYS_COLUMNS
    numeric = [c for c in FLORIDA_KEYS_COLUMNS if c != "deployment"]
    assert head[numeric].dtypes.map(str).eq("float64").all()


def test_biscayne_paths_resolve_without_existing():
    """Referencing the Biscayne constants is safe before the data arrives."""
    assert config.BISCAYNE_DIR.is_dir()
    assert config.BISCAYNE_HARVARD_DIR.parent == config.BISCAYNE_DIR
    assert config.BISCAYNE_HARVARD_DIR.name == "FULL data from Harvard team"


@pytest.mark.skipif(
    not config.BISCAYNE_HARVARD_DIR.exists(),
    reason="Biscayne Bay data not added yet",
)
def test_biscayne_harvard_dir_has_files():
    assert any(config.BISCAYNE_HARVARD_DIR.iterdir())
