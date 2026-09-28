"""
SMORES fold construction (plan section 8), matching the `get_first_piece`
style interface of process_3oec.py / process_sb.py: each `get_bbN_piece`
returns a single-column, time-sliced DataFrame that
`data_processing.data_utils.standardize_piece` /
`create_sequences_single_step` can consume unchanged, so
`CrossRegionForgetting` / `CrossRegionFederatedForgetting` extend from the
existing FL1-4 / SB1-4 folds to FL1-4 / SB1-4 / BB1-BBN with a one-line
edit to `train_order`.

Depth-class assignment is read off Figure 1 of
SMORES_Report_Data_20260806.pdf (deployment diagram): blue addresses are
midwater/subtidal, stacked pairs are (upper=8.3cmbsf, lower=16.5cmbsf).
Addresses 20/21/22/27, clustered at the lander-frame bend in that figure,
have an ambiguous stacking order in the diagram and are left unmapped --
none of them are among the channels expected to survive liveness
classification (plan S5 / S11), so this doesn't block fold construction.
Address 9 is drawn in Figure 1 but never appears in any RDO file (plan
S11 open question 2).
"""

import pandas as pd

DEPTH_CLASS = {
    2: "midwater", 3: "midwater", 25: "midwater",
    9: "upper", 10: "lower",
    4: "upper", 16: "lower",
    14: "upper", 15: "lower",
    11: "upper", 13: "lower",
    17: "upper", 18: "lower",
    23: "upper", 24: "lower",
    7: "upper", 8: "lower",
    5: "upper", 6: "lower",
    26: "upper", 28: "lower",
}

# Channels the report itself plots as usable (Fig. 2 caption), ordered by
# depth class so BB-fold numbering groups midwater / upper / lower together.
REPORTED_LIVE_CHANNELS = [25, 2, 3, 7, 26, 8, 13]


def get_depth_class(channel_id: int):
    return DEPTH_CLASS.get(channel_id)


def get_channel_piece(
    df5min: pd.DataFrame,
    channel_id: int,
    start=None,
    end=None,
    value_col: str = "po2",
    channel_col: str = "channel_id",
    time_col: str = "datetime",
):
    """Single-channel, single-column piece on the 5-min grid, matching the
    3OEC/SB piece-getter shape. `start`/`end` exclude the pre-deployment
    bucket-test window (plan S8) when passed."""
    g = df5min[df5min[channel_col] == channel_id].set_index(time_col).sort_index()
    piece = g.loc[start:end, [value_col]]
    return piece


def build_bb_folds(
    df5min: pd.DataFrame,
    liveness_df: pd.DataFrame,
    deployment_start,
    channels=None,
    value_col: str = "po2",
):
    """Build one BBn fold per usable channel.

    A channel is usable if its majority per-day liveness state (plan S5)
    is "live" -- this drops flatlined-zero and stuck-constant channels
    per the S8 non-negotiable. `deployment_start` excludes the
    pre-deployment bucket-test window from every fold; retain that window
    separately as a labeled calibration segment (plan S8), not as a fold.

    Returns an ordered dict {"BB1": piece_df, ...}, plus a manifest
    DataFrame recording which channel/depth-class each label maps to.
    """
    from data_processing.smores.qc import channel_majority_state

    channels = channels if channels is not None else REPORTED_LIVE_CHANNELS
    majority_state = channel_majority_state(liveness_df)

    usable = [
        c for c in channels
        if majority_state.get(c) == "live"
    ]
    # group by depth class, preserving `channels` order within each class
    order = {"midwater": 0, "upper": 1, "lower": 2, None: 3}
    usable = sorted(usable, key=lambda c: (order.get(get_depth_class(c), 3), channels.index(c)))

    folds = {}
    manifest_rows = []
    for i, channel_id in enumerate(usable, start=1):
        label = f"BB{i}"
        folds[label] = get_channel_piece(df5min, channel_id, start=deployment_start, value_col=value_col)
        manifest_rows.append({
            "label": label,
            "channel_id": channel_id,
            "depth_class": get_depth_class(channel_id),
            "n_rows": len(folds[label]),
        })

    dropped = [c for c in channels if c not in usable]
    if dropped:
        print(f"[folds] excluded non-live channels from BB folds: {dropped}")

    return folds, pd.DataFrame(manifest_rows)


def build_pieces(data_dir, salinity_psu: float = 35.0, deployment_start=None, channels=None):
    """Ingest raw SMORES chunk data and build fold pieces ready to merge
    into an experiment's `pieces` dict alongside the existing FL1-4 / SB1-4
    (plan S9): `pieces.update(build_pieces(data_dir))`. Keys are lowercase
    ("bb1", "bb2", ...) to match the fl1/sb1 naming convention, so
    `dataset_info` ends up keyed uniformly and `train_order` just needs the
    new labels appended.

    `deployment_start` excludes the pre-deployment bucket-test window
    (plan S8); if omitted, the earliest available timestamp is used, i.e.
    no exclusion -- pass it explicitly once the bucket-test window is
    known for the data being ingested.
    """
    from data_processing.smores import ingest, qc

    result = ingest.ingest_directory(data_dir, verbose=False)
    df5 = ingest.resample_5min(result["rdo"])
    df5 = qc.add_pct_sat_column(df5, salinity_psu=salinity_psu)
    liveness = qc.classify_liveness(result["rdo"])

    if deployment_start is None:
        deployment_start = df5["datetime"].min()

    bb_folds, _ = build_bb_folds(df5, liveness, deployment_start, channels=channels)
    return {label.lower(): piece for label, piece in bb_folds.items()}


def get_bucket_calibration_window(df5min: pd.DataFrame, channel_id: int, deployment_start, value_col: str = "po2"):
    """The pre-deployment bucket-test segment for one channel, retained as
    a labeled calibration window rather than training data (plan S8):
    live channels should read 85-108% sat here (plan S10 gate 1)."""
    g = df5min[df5min["channel_id"] == channel_id].set_index("datetime").sort_index()
    return g.loc[:deployment_start, [value_col]]
