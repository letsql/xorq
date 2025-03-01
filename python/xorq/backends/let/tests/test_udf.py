import operator
import pickle

import pandas as pd
import pyarrow as pa
import pytest
import toolz
import xgboost as xgb

import xorq as xo
import xorq.expr.datatypes as dt
from xorq.expr import udf
from xorq.expr.udf import make_pandas_expr_udf, make_pandas_udf
from xorq.tests.util import assert_frame_equal
from xorq.vendor.ibis import _
from xorq.vendor.ibis import literal as L
from xorq.vendor.ibis.selectors import of_type


pc = pytest.importorskip("pyarrow.compute")


@udf.agg.pyarrow
def my_mean(arr: dt.float64) -> dt.float64:
    return pc.mean(arr)


@udf.agg.pyarrow
def add_mean(a: dt.float64, b: dt.float64) -> dt.float64:
    return pc.mean(pc.add(a, b))


return_type = pa.struct(
    [
        ("a", pa.float64()),
        ("b", pa.float64()),
        ("c", pa.float64()),
    ]
)


@udf.agg.pyarrow
def centroid(a: dt.float64, b: dt.float64, c: dt.float64) -> return_type:
    return pa.scalar((pc.mean(a), pc.mean(b), pc.mean(c)), type=return_type)


@udf.agg.pyarrow
def centroid_list(a: dt.float64, b: dt.float64, c: dt.float64) -> pa.list_(
    pa.float64()
):
    return pa.scalar([pc.mean(a), pc.mean(b), pc.mean(c)], type=pa.list_(pa.float64()))


def test_udf_agg_pyarrow(ls_con, batting):
    batting = ls_con.register(xo.execute(batting), "pg-batting")
    result = my_mean(batting.G).execute()

    assert result == xo.execute(batting.G).mean()


def test_multiple_arguments_udf_agg_pyarrow(ls_con, batting):
    batting = ls_con.register(xo.execute(batting), "pg-batting")
    actual = add_mean(batting.G, batting.G).execute()
    expected = xo.execute(batting.G)
    expected = (expected + expected).mean()

    assert actual == expected


def test_multiple_arguments_struct_udf_agg_pyarrow(ls_con, batting):
    from math import isclose

    batting = ls_con.register(xo.execute(batting), "pg-batting")
    actual = centroid(batting.G, batting.G, batting.G).execute()
    expected = xo.execute(batting.G).mean()

    assert all(isclose(value, expected) for value in actual.values())


def test_multiple_arguments_list_udf_agg_pyarrow(ls_con, batting):
    from math import isclose

    batting = ls_con.register(xo.execute(batting), "pg-batting")
    actual = centroid_list(batting.G, batting.G, batting.G).execute()
    expected = xo.execute(batting.G).mean()

    assert all(isclose(value, expected) for value in actual)


@udf.agg.pyarrow
def my_sum(arr: dt.float64) -> dt.float64:
    return pc.sum(arr)


def test_group_by_udf_agg_pyarrow(ls_con, alltypes_df):
    alltypes = ls_con.register(alltypes_df, "pg-alltypes")

    expr = (
        alltypes[_.string_col == "1"]
        .mutate(x=L(1, "int64"))
        .group_by(_.x)
        .aggregate(sum=my_sum(_.double_col))
    )

    result = xo.execute(expr).astype({"x": "int64"})
    expected = (
        alltypes_df.loc[alltypes_df.string_col == "1", :]
        .assign(x=1)
        .groupby("x")
        .double_col.sum()
        .rename("sum")
        .reset_index()
    )

    assert_frame_equal(result, expected, check_like=True)


def test_udf_agg_pandas_df(ls_con, alltypes):
    name = "sum_sum"
    alltypes = ls_con.register(xo.execute(alltypes), "pg-alltypes")
    cols = (by, _) = ["year", "month"]
    expr = alltypes

    @udf.agg.pandas_df(
        schema=expr[cols].schema(),
        return_type=dt.int64(),
        name=name,
    )
    def sum_sum(df):
        return df.sum().sum()

    actual = (
        expr.group_by(by)
        .agg(sum_sum(*(expr[c] for c in cols)).name(name))
        .execute()
        .sort_values(by, ignore_index=True)
    )
    actual2 = (
        expr.group_by(by)
        .agg(sum_sum.on_expr(expr).name(name))
        .execute()
        .sort_values(by, ignore_index=True)
    )
    expected = (
        alltypes[cols]
        .execute()
        .groupby(by)
        .apply(sum_sum.fn)
        .rename(name)
        .reset_index()
        .sort_values(by, ignore_index=True)
    )
    assert actual.equals(expected)
    assert actual2.equals(expected)


def test_udf_pandas_df(ls_con, batting):
    typ = "int64"
    name = "my_least"
    batting = ls_con.register(xo.execute(batting), "pg-batting")
    my_least = operator.methodcaller("min", axis=1)
    schema = batting.select(of_type(typ)).schema()
    return_type = dt.dtype(typ)
    udf = make_pandas_udf(my_least, schema, return_type, name=name)
    order_by = ("playerID", "yearID", "stint")

    from_builtin = ls_con.execute(
        batting.mutate(xo.least(*(xo._[name] for name in schema)).name(name)).order_by(
            order_by
        )
    )
    from_udf = ls_con.execute(
        batting.mutate(udf.on_expr(batting).name(name)).order_by(order_by)
    )
    from_pd = (
        ls_con.execute(batting)
        .assign(**{name: lambda t: my_least(t[list(schema)])})
        .sort_values(list(order_by), ignore_index=True)
    )
    assert from_builtin.equals(from_udf)
    assert from_builtin.equals(from_pd)


def test_pandas_expr_udf():
    @toolz.curry
    def train_xgboost_model(df, features, target, seed=0):
        param = {"max_depth": 4, "eta": 1, "objective": "binary:logistic", "seed": seed}
        num_round = 10
        X = df[list(features)]
        y = df[target]
        dtrain = xgb.DMatrix(X, y)
        bst = xgb.train(param, dtrain, num_boost_round=num_round)
        return bst

    def predict_xgboost_model(model, df):
        return model.predict(xgb.DMatrix(df))

    features = (
        "emp_length",
        "dti",
        "annual_inc",
        "loan_amnt",
        "fico_range_high",
        "cr_age_days",
    )
    target = "event_occurred"
    train_fn = train_xgboost_model(features=features, target=target)
    name = "predicted"
    typ = "float64"

    con = xo.connect()
    t = con.read_parquet(xo.config.options.pins.get_path("lending-club"))

    # manual run
    df = xo.execute(t)
    model = train_fn(df)
    from_pd = df.assign(
        **{name: predict_xgboost_model(model, df[list(features)])}
    ).astype({name: typ})

    # using expr-scalar-udf
    model_udaf = udf.agg.pandas_df(
        fn=toolz.compose(pickle.dumps, train_fn),
        schema=t[features + (target,)].schema(),
        return_type=dt.binary,
        name="xgboost_model",
    )
    computed_kwargs_expr = model_udaf.on_expr(t)
    predict_expr_udf = make_pandas_expr_udf(
        computed_kwargs_expr,
        fn=predict_xgboost_model,
        schema=t[features].schema(),
        return_type=dt.dtype(typ),
        name=name,
    )
    expr = t.mutate(predict_expr_udf.on_expr(t).name(name))
    from_ls = con.execute(expr)

    pd._testing.assert_frame_equal(from_ls, from_pd)
