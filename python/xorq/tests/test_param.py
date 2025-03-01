from __future__ import annotations

import pytest

import xorq as xo
import xorq.expr.datatypes as dt
from xorq.tests.util import assert_series_equal, default_series_rename


@pytest.mark.parametrize(
    ("column", "raw_value"),
    [
        ("double_col", 0.0),
        ("double_col", 10.1),
        ("float_col", 1.1),
        ("float_col", 2.2),
    ],
)
def test_floating_scalar_parameter(alltypes, df, column, raw_value):
    value = xo.param(dt.double)
    expr = (alltypes[column] + value).name("tmp")
    expected = df[column] + raw_value
    result = expr.execute(params={value: raw_value})
    expected = default_series_rename(expected)
    assert_series_equal(result, expected, check_dtype=False)
