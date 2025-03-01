from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pytest import param

import xorq as xo
import xorq.common.exceptions as com
import xorq.expr.datatypes as dt
from xorq.tests.util import assert_frame_equal, reduction_tolerance
from xorq.vendor import ibis
from xorq.vendor.ibis import _
from xorq.vendor.ibis import literal as L


@ibis.udf.agg.builtin
def mean_udf(s: dt.double) -> dt.double:
    return s.mean()


aggregate_test_params = [
    param(lambda t: t.double_col.mean(), lambda t: t.double_col.mean(), id="mean"),
    param(lambda t: t.double_col.min(), lambda t: t.double_col.min(), id="min"),
    param(lambda t: t.double_col.max(), lambda t: t.double_col.max(), id="max"),
    param(
        lambda t: (t.double_col + 5).sum(),
        lambda t: (t.double_col + 5).sum(),
        id="complex_sum",
    ),
    param(
        lambda t: t.timestamp_col.max(),
        lambda t: t.timestamp_col.max(),
        id="timestamp_max",
    ),
]


@pytest.mark.parametrize(
    ("result_fn", "expected_fn"),
    aggregate_test_params,
)
def test_aggregate(alltypes, df, result_fn, expected_fn):
    expr = alltypes.aggregate(tmp=result_fn)
    result = expr.execute()

    # Create a single-row single-column dataframe with the Pandas `agg` result
    # (to match the output format of Ibis `aggregate`)
    expected = pd.DataFrame({"tmp": [expected_fn(df)]})

    assert_frame_equal(result, expected, check_dtype=False)


@pytest.mark.parametrize(
    ("result_fn", "expected_fn"),
    aggregate_test_params,
)
def test_aggregate_grouped(alltypes, df, result_fn, expected_fn):
    grouping_key_col = "bigint_col"

    # Two (equivalent) variations:
    #  1) `group_by` then `aggregate`
    #  2) `aggregate` with `by`
    expr1 = alltypes.group_by(grouping_key_col).aggregate(tmp=result_fn)
    expr2 = alltypes.aggregate(tmp=result_fn, by=grouping_key_col)
    result1 = expr1.execute()
    result2 = expr2.execute()

    # Note: Using `reset_index` to get the grouping key as a column
    expected = (
        df.groupby(grouping_key_col).apply(expected_fn).rename("tmp").reset_index()
    )

    # Row ordering may differ depending on backend, so sort on the
    # grouping key
    result1 = result1.sort_values(by=grouping_key_col).reset_index(drop=True)
    result2 = result2.sort_values(by=grouping_key_col).reset_index(drop=True)
    expected = expected.sort_values(by=grouping_key_col).reset_index(drop=True)

    assert_frame_equal(result1, expected, check_dtype=False)
    assert_frame_equal(result2, expected, check_dtype=False)


@pytest.mark.parametrize(
    ("result_fn", "expected_fn"),
    [
        param(
            lambda t, where: t.bool_col.count(where=where),
            lambda t, where: len(t.bool_col[where].dropna()),
            id="count",
        ),
        param(
            lambda t, where: t.bool_col.nunique(where=where),
            lambda t, where: t.bool_col[where].dropna().nunique(),
            id="nunique",
        ),
        param(
            lambda t, where: t.bool_col.any(where=where),
            lambda t, where: t.bool_col[where].any(),
            id="any",
        ),
        param(
            lambda t, where: t.bool_col.notany(where=where),
            lambda t, where: ~t.bool_col[where].any(),
            id="notany",
        ),
        param(
            lambda t, where: -t.bool_col.any(where=where),
            lambda t, where: ~t.bool_col[where].any(),
            id="any_negate",
        ),
        param(
            lambda t, where: t.bool_col.all(where=where),
            lambda t, where: t.bool_col[where].all(),
            id="all",
        ),
        param(
            lambda t, where: t.bool_col.notall(where=where),
            lambda t, where: ~t.bool_col[where].all(),
            id="notall",
        ),
        param(
            lambda t, where: -t.bool_col.all(where=where),
            lambda t, where: ~t.bool_col[where].all(),
            id="all_negate",
        ),
        param(
            lambda t, where: t.double_col.sum(where=where),
            lambda t, where: t.double_col[where].sum(),
            id="sum",
        ),
        param(
            lambda t, where: (t.int_col > 0).sum(where=where),
            lambda t, where: (t.int_col > 0)[where].sum(),
            id="bool_sum",
        ),
        param(
            lambda t, where: t.double_col.mean(where=where),
            lambda t, where: t.double_col[where].mean(),
            id="mean",
        ),
        param(
            lambda t, where: t.double_col.min(where=where),
            lambda t, where: t.double_col[where].min(),
            id="min",
        ),
        param(
            lambda t, where: t.double_col.max(where=where),
            lambda t, where: t.double_col[where].max(),
            id="max",
        ),
        param(
            lambda t, where: t.double_col.std(how="sample", where=where),
            lambda t, where: t.double_col[where].std(ddof=1),
            id="std",
        ),
        param(
            lambda t, where: t.double_col.var(how="sample", where=where),
            lambda t, where: t.double_col[where].var(ddof=1),
            id="var",
        ),
        param(
            lambda t, where: t.double_col.std(how="pop", where=where),
            lambda t, where: t.double_col[where].std(ddof=0),
            id="std_pop",
        ),
        param(
            lambda t, where: t.double_col.var(how="pop", where=where),
            lambda t, where: t.double_col[where].var(ddof=0),
            id="var_pop",
        ),
        param(
            lambda t, where: t.string_col.nunique(where=where),
            lambda t, where: t.string_col[where].nunique(),
            id="string_nunique",
        ),
        param(
            lambda t, where: t.double_col.first(where=where),
            lambda t, where: t.double_col[where].iloc[0],
            id="first",
        ),
        param(
            lambda t, where: t.double_col.last(where=where),
            lambda t, where: t.double_col[where].iloc[-1],
            id="last",
        ),
        param(
            lambda t, where: t.bigint_col.bit_and(where=where),
            lambda t, where: np.bitwise_and.reduce(t.bigint_col[where].values),
            id="bit_and",
        ),
        param(
            lambda t, where: t.bigint_col.bit_or(where=where),
            lambda t, where: np.bitwise_or.reduce(t.bigint_col[where].values),
            id="bit_or",
        ),
        param(
            lambda t, where: t.bigint_col.bit_xor(where=where),
            lambda t, where: np.bitwise_xor.reduce(t.bigint_col[where].values),
            id="bit_xor",
        ),
        param(
            lambda t, where: t.count(where=where),
            lambda t, where: len(t[where]),
            id="count_star",
        ),
    ],
)
@pytest.mark.parametrize(
    ("ibis_cond", "pandas_cond"),
    [
        param(lambda _: None, lambda _: slice(None), id="no_cond"),
        param(
            lambda t: t.string_col.isin(["1", "7"]),
            lambda t: t.string_col.isin(["1", "7"]),
            id="is_in",
        ),
        param(
            lambda _: xo._.string_col.isin(["1", "7"]),
            lambda t: t.string_col.isin(["1", "7"]),
            id="is_in_deferred",
        ),
    ],
)
def test_reduction_ops(
    alltypes,
    df,
    result_fn,
    expected_fn,
    ibis_cond,
    pandas_cond,
):
    expr = alltypes.agg(tmp=result_fn(alltypes, ibis_cond(alltypes))).tmp
    result = expr.execute().squeeze()
    expected = expected_fn(df, pandas_cond(df))

    try:
        np.testing.assert_allclose(result, expected, rtol=reduction_tolerance)
    except TypeError:  # assert_allclose only handles numerics
        # if we're not testing numerics, then the arrays should be exactly equal
        np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    ("result_fn", "expected_fn"),
    [
        param(
            lambda t, where: t.G.cov(t.RBI, where=where, how="pop"),
            lambda t, where: t.G[where].cov(t.RBI[where], ddof=0),
            id="covar_pop",
        ),
        param(
            lambda t, where: t.G.cov(t.RBI, where=where, how="sample"),
            lambda t, where: t.G[where].cov(t.RBI[where], ddof=1),
            id="covar_samp",
        ),
        param(
            lambda t, where: t.G.corr(t.RBI, where=where, how="pop"),
            lambda t, where: t.G[where].corr(t.RBI[where]),
            id="corr_pop",
        ),
        param(
            lambda t, where: t.G.corr(t.RBI, where=where, how="sample"),
            lambda t, where: t.G[where].corr(t.RBI[where]),
            id="corr_samp",
        ),
        param(
            lambda t, where: (t.G > 34.0).cov(
                t.G <= 34.0,
                where=where,
                how="pop",
            ),
            lambda t, where: (t.G[where] > 34.0).cov(t.G[where] <= 34.0, ddof=0),
            id="covar_pop_bool",
        ),
        param(
            lambda t, where: (t.G > 34.0).corr(
                t.G <= 34.0,
                where=where,
                how="pop",
            ),
            lambda t, where: (t.G[where] > 34.0).corr(t.G[where] <= 34.0),
            id="corr_pop_bool",
        ),
    ],
)
@pytest.mark.parametrize(
    ("ibis_cond", "pandas_cond"),
    [
        param(lambda _: None, lambda _: slice(None), id="no_cond"),
        param(
            lambda t: t.yearID.isin([2009, 2015]),
            lambda t: t.yearID.isin([2009, 2015]),
            id="cond",
        ),
    ],
)
def test_corr_cov(
    batting,
    batting_df,
    result_fn,
    expected_fn,
    ibis_cond,
    pandas_cond,
):
    expr = result_fn(batting, ibis_cond(batting)).name("tmp")
    result = expr.execute()

    expected = expected_fn(batting_df, pandas_cond(batting_df))

    assert pytest.approx(result) == expected


def test_approx_median(alltypes):
    expr = alltypes.double_col.approx_median()
    result = expr.execute()
    assert isinstance(result, float)


def test_median(alltypes, df):
    expr = alltypes.double_col.median()
    result = expr.execute()
    expected = df.double_col.median()
    assert result == expected


def test_topk_op(alltypes, df):
    # TopK expression will order rows by "count" but each backend
    # can have different result for that.
    # Note: Maybe would be good if TopK could order by "count"
    # and the field used by TopK
    t = alltypes.order_by(alltypes.string_col)
    df = df.sort_values("string_col")
    expr = t.string_col.topk(3)
    result = expr.execute()
    expected = df.groupby("string_col")["string_col"].count().head(3)
    assert all(result.iloc[:, 1].values == expected.values)


@pytest.mark.parametrize(
    ("result_fn", "expected_fn"),
    [
        param(
            lambda t: t.semi_join(t.string_col.topk(3), "string_col"),
            lambda t: t[
                t.string_col.isin(
                    t.groupby("string_col")["string_col"].count().head(3).index
                )
            ],
            id="string_col_filter_top3",
        )
    ],
)
def test_topk_filter_op(alltypes, df, result_fn, expected_fn):
    # TopK expression will order rows by "count" but each backend
    # can have different result for that.
    # Note: Maybe would be good if TopK could order by "count"
    # and the field used by TopK
    t = alltypes.order_by(alltypes.string_col)
    df = df.sort_values("string_col")
    expr = result_fn(t)
    result = expr.execute()
    expected = expected_fn(df)
    assert result.shape[0] == expected.shape[0]


def test_binds_are_cast(alltypes):
    expr = alltypes.aggregate(
        high_line_count=(
            alltypes.string_col.case().when("1-URGENT", 1).else_(0).end().sum()
        )
    )

    expr.execute()


def test_agg_sort(alltypes):
    query = alltypes.aggregate(count=alltypes.count())
    with pytest.raises(com.IntegrityError):
        query.order_by(alltypes.year)


def test_filter(alltypes, df):
    expr = (
        alltypes[_.string_col == "1"]
        .mutate(x=L(1, "int64"))
        .group_by(_.x)
        .aggregate(sum=_.double_col.sum())
    )

    result = expr.execute().astype({"x": "int64"})
    expected = (
        df.loc[df.string_col == "1", :]
        .assign(x=1)
        .groupby("x")
        .double_col.sum()
        .rename("sum")
        .reset_index()
    )
    assert_frame_equal(result, expected, check_like=True)


def test_agg_name_in_output_column(alltypes):
    query = alltypes.aggregate([alltypes.int_col.min(), alltypes.int_col.max()])
    df = query.execute()
    assert "min" in df.columns[0].lower()
    assert "max" in df.columns[1].lower()


def test_grouped_case(con):
    table = xo.memtable({"key": [1, 1, 2, 2], "value": [10, 30, 20, 40]})

    case_expr = xo.case().when(table.value < 25, table.value).else_(xo.null()).end()

    expr = table.group_by("key").aggregate(mx=case_expr.max()).order_by("key")
    result = con.execute(expr)
    expected = pd.DataFrame({"key": [1, 2], "mx": [10, 20]})
    assert_frame_equal(result, expected)


def test_value_counts_on_expr(alltypes, df):
    expr = alltypes.bigint_col.add(1).value_counts()
    columns = expr.columns
    expr = expr.order_by(columns)
    result = expr.execute().sort_values(columns).reset_index(drop=True)
    expected = df.bigint_col.add(1).value_counts().reset_index()
    expected.columns = columns
    expected = expected.sort_values(by=columns).reset_index(drop=True)
    assert_frame_equal(result, expected)
