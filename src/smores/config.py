"""Project paths and environment configuration.

Import paths from here rather than building relative paths, so code behaves the
same whether it runs from a notebook, a test, or the command line.
"""

from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FLORIDA_DIR = RAW_DIR / "florida_keys"
BISCAYNE_DIR = RAW_DIR / "biscayne_bay"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"

# Raw datasets. Point at these rather than retyping filenames — the Harvard
# directory name in particular has spaces in it.
FLORIDA_KEYS_CSV = FLORIDA_DIR / "florida_keys_data.csv"
BISCAYNE_HARVARD_DIR = BISCAYNE_DIR / "FULL data from Harvard team"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

load_dotenv(PROJECT_ROOT / ".env")
