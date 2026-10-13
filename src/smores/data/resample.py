"""Bin native-rate schema frames onto the shared 2-minute grid.

Every site arrives at its own rate -- Florida Keys at 8 Hz, Biscayne Bay polls every
~52 s, Santa Barbara every 15 min -- and `to_grid` brings them all to
`schema.SAMPLE_INTERVAL` the same way:

* A bin is labelled by its start: `timestamp.floor(interval)`.
* Bins never straddle a sensor or a deployment; the group key is
  (site, deployment, sensor, bin).
* Values are the **median** of the bin's readings, as in the SMORES plan's 5-minute
  medians, so a single spike cannot move a bin.
* Best available quality wins: when a bin holds readings of mixed `qc_flag`, only the
  readings at the lowest (best) flag are aggregated and the bin carries that flag. A
  bin of good reads with one flagged read is a good bin; a bin of only flagged reads
  stays flagged.
* Nothing is interpolated or forward-filled. An empty bin is simply absent, so a site
  sampled more coarsely than the grid (Santa Barbara) comes out sparse, not upsampled.

`n_obs` (readings that went into each bin) is returned alongside the schema columns so
the caller can drop under-filled bins and the integrity report can show fill. It is
not a schema column; `pipeline` strips it before writing.
"""

from smores.data import schema

KEYS = ["site_id", "deployment_id", "sensor_id", "timestamp"]
VALUE_COLUMNS = ["do_sat", "temperature_c", "do_mgl", "do_native_value", "depth_m"]
CONSTANT_COLUMNS = ["do_native_unit", "data_source"]


def to_grid(df, interval=schema.SAMPLE_INTERVAL, min_obs=1):
    """Median-bin a schema frame onto `interval`, keeping bins with >= `min_obs` reads.

    Returns a frame with the schema's columns, in schema order and dtypes, plus an
    `n_obs` column. Sorted by timestamp, then sensor.
    """
    work = df[KEYS + VALUE_COLUMNS + CONSTANT_COLUMNS + ["qc_flag"]].copy()
    work["timestamp"] = work["timestamp"].dt.floor(interval)

    best = work.groupby(KEYS, observed=True, sort=False)["qc_flag"].transform("min")
    work = work[work["qc_flag"] == best]

    grouped = work.groupby(KEYS, observed=True, sort=False)
    out = grouped[VALUE_COLUMNS].median()
    out["qc_flag"] = grouped["qc_flag"].first()
    for name in CONSTANT_COLUMNS:
        out[name] = grouped[name].first()
    out["n_obs"] = grouped.size()
    out = out.reset_index()

    out = out[out["n_obs"] >= min_obs]
    out = out.sort_values(["timestamp", "sensor_id"], kind="stable", ignore_index=True)
    n_obs = out["n_obs"].astype("int32")
    out = schema.conform(out)
    out["n_obs"] = n_obs.to_numpy()
    return out
