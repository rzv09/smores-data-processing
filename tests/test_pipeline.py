"""The orchestrator: verify, integrity-check, and only then write."""

import json

import pandas as pd
import pytest
import synthetic as syn
from test_florida_keys import raw_frame

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
