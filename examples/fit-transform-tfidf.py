import pathlib
import pickle

import dask
import pandas as pd
import pyarrow as pa
import toolz
from sklearn.feature_extraction.text import TfidfVectorizer

import xorq as xo
import xorq.expr.udf as udf
import xorq.vendor.ibis.expr.datatypes as dt
from xorq.common.caching import ParquetCacheStorage
from xorq.expr.udf import make_pandas_expr_udf


@toolz.curry
def deferred_fit_transform_series(
    expr, col, cls, return_type, name="predicted", storage=None
):
    def fit(fit_df, cls=cls):
        obj = cls()
        obj.fit(fit_df[col])
        return obj

    @toolz.curry
    def transform(model, df):
        return pa.array(
            model.transform(df[col]).toarray().tolist(),
            type=return_type.to_pyarrow(),
        )

    schema = xo.schema({col: expr.schema()[col]})
    model_udaf = udf.agg.pandas_df(
        fn=toolz.compose(pickle.dumps, fit),
        schema=schema,
        return_type=dt.binary,
        name="_" + dask.base.tokenize(fit).lower(),
    )
    computed_kwargs_expr = model_udaf.on_expr(expr)
    if storage:
        computed_kwargs_expr = computed_kwargs_expr.as_table().cache(storage=storage)

    predict_expr_udf = make_pandas_expr_udf(
        computed_kwargs_expr=computed_kwargs_expr,
        fn=transform,
        schema=schema,
        return_type=return_type,
        name=name,
    )

    # deferred_model, deferred_fit, deferred_transform
    return computed_kwargs_expr, model_udaf, predict_expr_udf


deferred_fit_transform_tfidf = deferred_fit_transform_series(
    cls=TfidfVectorizer, return_type=dt.Array(dt.float64)
)


if __name__ == "__main__":
    from xorq.common.utils.import_utils import import_path
    from xorq.expr.relations import flight_udxf

    m = import_path(
        pathlib.Path(__file__).parent.joinpath("hacker-news-udtf-example.py")
    )

    con = xo.connect()
    expr = flight_udxf(
        con.register(
            pd.DataFrame(({"maxitem": 43182839, "n": 1000},)),
            table_name="t",
        ),
        m.HackerNewsFetcher,
        name="outer",
        inner_name="inner",
    )
    # we must have two streams with different names: we can't just reuse the first expr
    expr2 = flight_udxf(
        con.register(
            pd.DataFrame(({"maxitem": 43182839, "n": 1000},)),
            table_name="t",
        ),
        m.HackerNewsFetcher,
        name="outer2",
        inner_name="inner2",
    )

    # # alternatively, we can execute, but this is cheating
    # con2 = xo.connect()
    # expr2 = con2.register(
    #     expr.execute(),
    #     table_name="t2",
    # )

    col = "title"
    (computed_kwargs_expr, model_udaf, predict_expr_udf) = deferred_fit_transform_tfidf(
        expr, col
    )
    model = computed_kwargs_expr.execute()
    # y = predict_expr_udf.on_expr(expr2).execute()
    expr3 = predict_expr_udf.on_expr(expr2)
    y = expr3.execute()

    # things work fine if we cache the model
    storage = ParquetCacheStorage(source=con)
    # do we have an issue of having to reregister the rbr?
    (computed_kwargs_expr, model_udaf, predict_expr_udf) = deferred_fit_transform_tfidf(
        expr, col, storage=storage
    )
    model2 = computed_kwargs_expr.execute()
    y2 = predict_expr_udf.on_expr(expr).execute()
