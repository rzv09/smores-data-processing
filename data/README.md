# Data

Reference for everything under `data/` — layout, path constants, and what each raw
dataset actually contains.

## Layout

```
data/
├── README.md                               this file
├── raw/                                    immutable inputs — never edited in place
│   ├── florida_keys/
│   │   └── florida_keys_data.csv           203 MB · 2,044,805 × 10   [present]
│   └── biscayne_bay/
│       └── FULL data from Harvard team/    Harvard drop             [not added yet]
├── interim/                                intermediate outputs
├── processed/                              analysis-ready datasets
│   └── florida_keys_o2.parquet             62 MB · 6,134,415 × 12   [built]
└── external/                               third-party data
```

Dataset contents under `data/` are gitignored. Only the directory structure (via
`.gitkeep`) and markdown docs like this file are tracked, so a fresh clone gets the
skeleton and this reference, but no data.

## Path constants

Import from `smores.config`; never hardcode a filename or build a relative path. This
resolves identically from a notebook, a test, or the CLI.

| Constant | Points to |
| --- | --- |
| `FLORIDA_KEYS_CSV` | `data/raw/florida_keys/florida_keys_data.csv` |
| `FLORIDA_KEYS_PARQUET` | `data/processed/florida_keys_o2.parquet` |
| `BISCAYNE_HARVARD_DIR` | `data/raw/biscayne_bay/FULL data from Harvard team/` |
| `FLORIDA_DIR` | `data/raw/florida_keys/` |
| `BISCAYNE_DIR` | `data/raw/biscayne_bay/` |
| `RAW_DIR` | `data/raw/` |
| `INTERIM_DIR` | `data/interim/` |
| `PROCESSED_DIR` | `data/processed/` |
| `EXTERNAL_DIR` | `data/external/` |
| `DATA_DIR` | `data/` |

```python
from smores.config import FLORIDA_KEYS_PARQUET

df = pd.read_parquet(FLORIDA_KEYS_PARQUET)
```

`FLORIDA_KEYS_PARQUET` is the harmonized output; prefer it over re-deriving from the raw
CSV, since the raw oxygen values need a salinity correction first (see below).

## Florida Keys

`FLORIDA_KEYS_CSV` — 203 MB (212,737,786 bytes), 2,044,805 rows × 10 columns.

A full `pd.read_csv` takes ~2.5 s and ~198 MB in memory, so no `chunksize` or Parquet
conversion is needed at this size. Every column except `deployment` parses as `float64`
without dtype hints. **There are no nulls in any column.**

Units below are the submitter-supplied (BCO-DMO) units, not guesses from the names.

| Column | Type | Unit | Min | Max | Mean | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `deployment` | `str` | — | — | — | — | source filename; `YYYY_M_dayStart_dayEnd` |
| `t` | `float64` | **hours** | 10.00 | 56.00 | 27.57 | elapsed time — **not temperature** |
| `t_increase` | `float64` | seconds | 0 | 72,000 | 32,476 | elapsed time within a deployment |
| `Vx` | `float64` | cm s⁻¹ | −49.816 | 30.791 | −3.255 | velocity |
| `Vy` | `float64` | cm s⁻¹ | −49.840 | 26.650 | −3.157 | velocity |
| `Vz` | `float64` | cm s⁻¹ | −49.662 | 8.854 | −0.687 | velocity |
| `P` | `float64` | kPa | 77.00 | 87.00 | 82.36 | gauge pressure |
| `O2_S1` | `float64` | µmol L⁻¹ | 232.676 | 271.396 | 249.987 | oxygen sensor 1 |
| `O2_S2` | `float64` | µmol L⁻¹ | 230.045 | 270.708 | 249.987 | oxygen sensor 2 |
| `O2_S3` | `float64` | µmol L⁻¹ | 229.973 | 270.722 | 249.987 | oxygen sensor 3 |

**`t` is elapsed hours, not temperature.** Its 10–56 range and 27.6 mean read convincingly
as °C for a subtropical site, and the name invites it. It is time: `t` equals
`start_hour + t_increase/3600` exactly. There is **no temperature column in this dataset**
— only a documented 28–31 °C site range.

There is also no timestamp column; see "Processed outputs" below for how absolute time is
reconstructed, and why the oxygen values need a salinity correction before use.

### Instrumentation

An aquatic eddy-covariance deployment: a Nortek Vector ADV at 8 Hz with three co-located
Pyroscience oxygen sensors, ~9 km south of Long Key (24°43.52′N, 80°49.85′W), 9 ± 1 m
water depth on a flat carbonate platform of coral sand. The ADV measuring volume sat
~35 cm above the sediment–water interface. Site conditions: S 35–36, T 28–31 °C.

The oxygen sensors were two-point calibrated in air-saturated seawater (100% air
saturation) and sulfite-stripped seawater (0%), with accuracy specified in % air
saturation — which matters for interpreting the reported µmol/L (see below).

The three sensors are **replicates at one depth, not a depth ladder**: there is a single
ADV measuring volume, and inter-sensor correlations are 0.985–0.990.

### Deployments

| `deployment` | Rows |
| --- | --- |
| `3oec_2017_7_15_16` | 576,001 |
| `3oec_2017_7_13_14` | 547,202 |
| `3oec_2017_7_11_12` | 518,401 |
| `3oec_2017_7_16_17` | 403,201 |

The file stacks four separate deployments. Group by `deployment` before any time-series
operation — `t_increase` restarts at 0 for each one, so treating the file as a single
continuous series will silently produce wrong results.

Note that `3oec_2017_7_11_12` runs from 07-12 14:00 to 07-13 08:00 local, i.e. **outside
the day range its own name claims**. That is not a defect: the source records that
"measurements could not start before 14:00 due to bad weather preventing earlier
deployment", and 38 h after midnight on the 11th is exactly 14:00 on the 12th. The `t`
values are correct as recorded; the label is just misleading.

## Processed outputs

`FLORIDA_KEYS_PARQUET`, built by `make data` (`python -m smores.data.florida_keys`). Long format, one row per
sensor per instant: 6,134,415 rows (2,044,805 × 3 sensors), 62 MB, ~227 MB in memory.
The harmonized schema that all three sites conform to lives in `smores.data.schema`;
`smores.data.units` holds the oxygen solubility and pressure conversions.

| Column | dtype | Notes |
| --- | --- | --- |
| `timestamp` | `datetime64[ns, UTC]` | reconstructed; sort key |
| `site_id` | `category` | `florida_keys` |
| `depth_m` | `float32` | from gauge pressure; mean 8.232 m |
| `height_above_bed_m` | `float32` | 0.35 |
| `sensor_id` | `category` | `FL1`, `FL2`, `FL3` |
| `do_sat` | `float32` | % air saturation; mean 104.90 |
| `do_mgl` | `float32` | mg/L; mean 6.574 |
| `raw_o2_umol` | `float32` | reported value, untouched; mean 249.987 |
| `temperature_c` | `float32` | all null — never logged |
| `qc_flag` | `int8` | all 0; the source has no defects to flag |
| `data_source` | `category` | `wcci_fl` |
| `deployment` | `category` | retained beyond the shared schema, see below |

Three transforms are judgement calls rather than renames, so they are recorded here as
well as in the `smores.data.florida_keys` module docstring.

**Timestamps are reconstructed, and the logged clock is local EDT.** Each deployment's
clock starts at midnight on the first day named in its label; start hours recover as
`round(t - t_increase/3600)` giving 38, 11, 10, 16. Timestamps are built from
`t_increase`, not `t`, because `t` carries ~10 ms of float noise while `t_increase` is
exact multiples of 0.125 s. The timezone was determined empirically: binning oxygen by
reconstructed hour puts the trough at 06:00–07:00 and the peak at 19:00–20:00, against a
mid-July Keys sunrise/sunset of ~06:40/~20:20 EDT. Read as UTC the trough would fall at
02:00 local, which is not physical. Output is UTC.

**Oxygen carries a salinity-reference error, corrected on read.** The reported µmol/L
averages 250, which against seawater solubility at the documented S/T would be 127% air
saturation sustained for five days — not physical. Because the sensors were calibrated in
seawater with accuracy quoted in % air saturation, percent saturation is the native,
correctly referenced quantity, and the µmol/L figures were derived from it using
*freshwater* solubility: 105% × 238.3 µmol/L (fresh) = 250, where the truth is
105% × 195.9 = 206 µmol/L. Dividing by freshwater solubility therefore *recovers* the
original calibrated reading rather than inventing a correction. Reported concentrations
are high by a factor of ~1.22. The corrected series runs 96.5–113.9%, crossing 100%
daily, as a net-autotrophic carbonate platform should; the uncorrected one sits at
121–133% and never crosses. `raw_o2_umol` preserves the published value so the correction
stays auditable and reversible.

Caveat: with no temperature record, solubility uses the 29.5 °C midpoint of the
documented range. Across 28–31 °C that shifts *absolute* `do_sat` by ±2.7%, exceeding the
sensors' ±1% a.s. accuracy. The fresh→seawater *ratio* varies only 0.4%, so relative
structure — diel cycle, gradients, flux covariances — is unaffected.

**Depth comes from gauge pressure.** `P` must be gauge, since 82 kPa absolute would place
the sensor above the waterline. `depth_m = P / (ρ·g)` with ρ = 1022 kg/m³ and
g = 9.78937 m/s² at site latitude, i.e. 10.0047 kPa per metre. Mean 8.232 m against the
8.65 m expected from the 9 m site depth less the 0.35 m sensor height, with
per-deployment swings of 0.58–0.89 m — a plausible Keys tidal range.

**`Vx/Vy/Vz` are dropped** — the harmonized schema is oxygen-only. The benthic O2 flux
this deployment exists to measure is the ⟨w′C′⟩ covariance of those velocities with the
oxygen signal, so flux work must read the raw CSV, which stays immutable. `P` survives
only as `depth_m`. `deployment` is kept beyond the shared schema because segments are
separated by 3–28 h gaps that eddy-covariance windowing must not straddle.

When aggregating by hour of day, filter on sample count: deployment coverage is uneven and
the 08:00 bin holds only 3 rows, which is enough to produce a spurious trough.

## Biscayne Bay

Not added yet; arrives later in the project. `BISCAYNE_HARVARD_DIR` is already defined
and points at where it will live. A `Path` never touches the filesystem, so the missing
directory breaks no import and no test — when the data lands, drop it at that exact
path and the constant starts working with no code change.

Guard any code written ahead of the data:

```python
from smores.config import BISCAYNE_HARVARD_DIR

if BISCAYNE_HARVARD_DIR.exists():
    ...
```

`tests/test_datasets.py` has a contents check that is skipped until the directory
exists, then starts running on its own.

## Conventions

- **`raw/` is immutable.** Never write to it or edit a file in place. Derived data goes
  to `interim/` (intermediate) or `processed/` (analysis-ready).
- **Reads go through `smores.config`,** so paths work from any working directory.
- **Nothing in `data/` is committed.** Keep it reproducible from the source, or document
  where it came from here.
