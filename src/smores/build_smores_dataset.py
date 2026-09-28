"""
Builds the SMORES BB1..BBN folds from raw chunk CSVs and runs the
validation gates from SMORES_preprocessing_plan_v2.md section 10.

Usage:
    python run/build_smores_dataset.py <data_dir> [out_dir]

<data_dir> is a directory of SENS_<serial>_<chunk>_<Kind>.csv files (e.g.
the 17.8-hour sample, or the full deployment dump once available).
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import pandas as pd

from data_processing.smores import ingest, qc, folds

# Deployment completed 2130 EDT May 13, 2026 per the report; the RTC field
# is treated as already-local per plan S2 (open question, see plan S11.1).
DEPLOYMENT_START = pd.Timestamp("2026-05-13 21:30:00")
TIDAL_PERIOD_HOURS = 12.42
REPORT_QC_FAILURE_RATE = 0.0373


def run_validation_gates(df5, liveness, qc_rates, external_pts):
    print("\n--- Validation gates (plan S10) ---")

    # Gate 1: bucket window reads near 100% sat on live channels.
    majority = qc.channel_majority_state(liveness)
    live_channels = [c for c, s in majority.items() if s == "live"]
    for ch in live_channels:
        bucket = folds.get_bucket_calibration_window(df5, ch, DEPLOYMENT_START, value_col="pct_sat")
        if len(bucket):
            print(f"gate1 ch{ch}: bucket %sat mean={bucket['pct_sat'].mean():.1f}, "
                  f"range=[{bucket['pct_sat'].min():.1f}, {bucket['pct_sat'].max():.1f}]")

    # Gate 2: Bar100 submersion step vs logged 2130 EDT deployment.
    if len(external_pts):
        step = external_pts.set_index("datetime")["pressure_bar"].resample("15min").mean()
        window = step.loc[DEPLOYMENT_START - pd.Timedelta("2h"): DEPLOYMENT_START + pd.Timedelta("1h")]
        print(f"gate2 Bar100 pressure around deployment window:\n{window}")

    # Gate 5: live-channel QC failure rate vs report's 3.73%.
    live_qc = qc_rates[qc_rates["channel_id"].isin(live_channels)]
    if len(live_qc):
        rate = live_qc["qc_failure_rate"].mean()
        print(f"gate5 mean QC failure rate over live channels: {rate:.4%} "
              f"(report: {REPORT_QC_FAILURE_RATE:.2%})")

    print("gate3/gate4 require the optode-vs-Bar100 temperature comparison and a tidal FFT; "
          "not run here, see plan S10.")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    data_dir = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "./out/smores_dataset"

    result = ingest.ingest_directory(data_dir)

    df5 = ingest.resample_5min(result["rdo"])
    df5 = ingest.mark_long_gaps(df5)
    df5 = qc.add_pct_sat_column(df5, salinity_psu=35.0)

    liveness = qc.classify_liveness(result["rdo"])
    qc_rates = qc.qc_failure_rate(result["rdo_raw"])

    bb_folds, manifest = folds.build_bb_folds(df5, liveness, DEPLOYMENT_START)
    print("\nBB folds built:")
    print(manifest)

    ingest.persist_artifacts(out_dir, result["calibration"], liveness, qc_rates,
                              result["poll_exceptions"], manifest)
    for label, piece in bb_folds.items():
        piece.to_csv(os.path.join(out_dir, f"{label.lower()}.csv"))
    print(f"\nartifacts written to {out_dir}")

    run_validation_gates(df5, liveness, qc_rates, result["external_pts"])


if __name__ == "__main__":
    main()
