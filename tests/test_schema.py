"""The schema's primary signal is do_mgl."""

import numpy as np
import pytest
from synthetic import schema_frame

from smores.data import schema


def test_primary_signal_is_do_mgl_and_required():
    assert schema.PRIMARY_SIGNAL == "do_mgl"
    assert "do_mgl" in schema.REQUIRED
    assert "do_sat" not in schema.REQUIRED


def test_null_do_mgl_fails_validation():
    frame = schema_frame(n=3)
    frame.loc[1, "do_mgl"] = np.nan
    with pytest.raises(schema.SchemaError, match="do_mgl contains nulls"):
        schema.validate(frame)


def test_null_do_sat_is_allowed():
    frame = schema_frame(n=3)
    frame["do_sat"] = np.float32(np.nan)
    schema.validate(frame)
