"""The orchestrator: verify, integrity-check, and only then write."""

import json

import pandas as pd
import pytest
import synthetic as syn
from test_florida_keys import raw_frame

from smores import config
from smores.data import integrity, pipeline, schema


def test_every_site_is_registered():
    assert set(pipeline.SITES) == set(schema.SITE_IDS)
    for name, site in pipeline.SITES.items():
        assert site.site_id == name
        assert site.out_path.name == f"{name}_o2_2min.parquet"


def test_a_passing_run_writes_parquet_and_report(tmp_path):
    raw = tmp_path / "sb.csv"
    syn.santa_barbara_raw(days=3).to_csv(raw, index=False)
    out = tmp_path / "processed" / "santa_barbara_o2_2min.parquet"

    frame, report = pipeline.run("santa_barbara", raw_path=raw, out_path=out, verbose=False)

    assert report.ok
    written = pd.read_parquet(out)
    schema.validate(written)
    pd.testing.assert_frame_equal(written, frame)
    assert "n_obs" not in written

    saved = json.loads((out.parent / "santa_barbara_o2_2min_integrity.json").read_text())
    assert saved["ok"] is True
    assert saved["summary"]["rows"] == len(frame)


def test_biscayne_bay_also_writes_its_valid_sensors(tmp_path):
    dump = syn.write_biscayne_dump(tmp_path)
    out = tmp_path / "processed" / "biscayne_bay_o2_2min.parquet"

    frame, _ = pipeline.run("biscayne_bay", raw_path=dump, out_path=out, verbose=False)

    valid = pd.read_parquet(out.parent / "biscayne_bay_o2_2min_valid.parquet")
    schema.validate(valid)
    assert set(valid["sensor_id"]) == {"ch2"}  # ch10 is in the full file only
    assert "ch10" in set(frame["sensor_id"])
    expected = frame[frame["sensor_id"].isin(valid["sensor_id"].unique())]
    pd.testing.assert_frame_equal(valid, expected.reset_index(drop=True))


def test_valid_subset_path_matches_config():
    site = pipeline.SITES["biscayne_bay"]
    assert set(site.subsets["valid"]) == {"ch2", "ch3", "ch25", "ch7", "ch8", "ch26"}
    assert pipeline.subset_path(site.out_path, "valid") == config.BISCAYNE_BAY_2MIN_VALID_PARQUET


def test_a_failing_run_writes_nothing(tmp_path):
    """A few synthetic minutes cannot show a diel cycle, so the error check must fail."""
    raw = tmp_path / "fk.csv"
    raw_frame(rows=960 * 3).to_csv(raw, index=False)
    out = tmp_path / "processed" / "florida_keys_o2_2min.parquet"

    with pytest.raises(integrity.IntegrityError, match="diel_cycle"):
        pipeline.run("florida_keys", raw_path=raw, out_path=out, verbose=False)
    assert not out.parent.exists()


def test_no_write_mode_leaves_disk_untouched(tmp_path):
    raw = tmp_path / "sb.csv"
    syn.santa_barbara_raw(days=3).to_csv(raw, index=False)
    out = tmp_path / "processed" / "x.parquet"
    pipeline.run("santa_barbara", write=False, raw_path=raw, out_path=out, verbose=False)
    assert not out.parent.exists()


def test_cli_rejects_unknown_sites():
    with pytest.raises(SystemExit):
        pipeline.main(["atlantis"])
