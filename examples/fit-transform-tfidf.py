import functools

import dask
import pandas as pd
import pyarrow as pa
import requests
import toolz
from sklearn.feature_extraction.text import TfidfVectorizer

import xorq as xo
import xorq.expr.udf as udf
import xorq.vendor.ibis.expr.datatypes as dt
from xorq.common.caching import ParquetCacheStorage
from xorq.expr.udf import make_pandas_expr_udf, wrap_model


@functools.cache
def hackernews_stories(n=1000):
    try:
        return pd.read_pickle("df.pkl")
    except Exception:
        # Get the max ID number from hacker news
        resp = requests.get("https://hacker-news.firebaseio.com/v0/maxitem.json")
        resp.raise_for_status()
        latest_item = resp.json()
        # Get items based on story ids from the HackerNews items endpoint
        results = []
        scope = range(latest_item - n, latest_item)
        for item_id in scope:
            resp = requests.get(
                f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json"
            )
            resp.raise_for_status()
            item = resp.json()
            results.append(item)
        # Store the results in a dataframe and filter on stories with valid titles
        df = pd.DataFrame(results)
        if len(df) > 0:
            df = df[df.type == "story"]
            df = df[~df.title.isna()]
        df.to_pickle("df.pkl")
    return df


@toolz.curry
def deferred_fit_transform_series(
    expr, col, cls, return_type, name="predicted", model_key="model", storage=None
):
    def fit(fit_df, cls=cls):
        obj = cls()
        obj.fit(fit_df[col])
        return obj

    def transform(df, **kwargs):
        (key, *rest) = tuple(kwargs)
        if key != model_key or rest:
            raise ValueError
        model = kwargs[model_key]
        return pa.array(
            model.transform(df[col]).toarray().tolist(),
            type=return_type.to_pyarrow(),
        )

    schema = xo.schema({col: expr.schema()[col]})
    model_udaf = udf.agg.pandas_df(
        fn=toolz.compose(wrap_model(model_key=model_key), fit),
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


# why is this still necessary?
xo.vendor.ibis.formats.pyarrow._from_pyarrow_types[pa.binary_view()] = dt.binary()


con = xo.connect()
t = con.register(hackernews_stories(), "t")
col = "title"


(computed_kwargs_expr, model_udaf, predict_expr_udf) = deferred_fit_transform_tfidf(
    t, col
)
x = computed_kwargs_expr.execute()
y = predict_expr_udf.on_expr(t).execute()


storage = ParquetCacheStorage(source=con)
(computed_kwargs_expr, model_udaf, predict_expr_udf) = deferred_fit_transform_tfidf(
    t, col, storage=storage
)
x = computed_kwargs_expr.execute()
y = predict_expr_udf.on_expr(t).execute()
