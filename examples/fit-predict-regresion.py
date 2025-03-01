import dask
import pandas as pd
import pyarrow as pa
import toolz
from sklearn.linear_model import LinearRegression

import xorq as xo
import xorq.expr.udf as udf
import xorq.vendor.ibis.expr.datatypes as dt
from xorq.common.caching import ParquetCacheStorage
from xorq.expr.udf import make_pandas_expr_udf, wrap_model


def make_data():
    import numpy as np

    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    # y = 1 * x_0 + 2 * x_1 + 3
    y = np.dot(X, np.array([1, 2])) + 3
    df = pd.DataFrame(np.hstack((X, y[:, np.newaxis]))).rename(
        columns=lambda x: chr(x + ord("a"))
    )
    (*features, target) = df.columns
    return (df, features, target)


@toolz.curry
def deferred_fit_predict(
    expr,
    target,
    features,
    cls,
    return_type,
    name="predicted",
    model_key="model",
    storage=None,
):
    def fit(fit_df, cls=cls):
        obj = cls()
        obj.fit(fit_df[features], fit_df[target])
        return obj

    def predict(df, **kwargs):
        (key, *rest) = tuple(kwargs)
        if key != model_key or rest:
            raise ValueError
        model = kwargs[model_key]
        return pa.array(
            model.predict(df[features]),
            type=return_type.to_pyarrow(),
        )

    features = features or tuple(expr.schema())
    fit_schema = xo.schema(
        {col: expr[col].type() for col in (*features, target) if col in expr}
    )
    predict_schema = xo.schema(
        {col: expr[col].type() for col in features if col in expr}
    )
    model_udaf = udf.agg.pandas_df(
        fn=toolz.compose(wrap_model(model_key=model_key), fit),
        schema=fit_schema,
        return_type=dt.binary,
        name="_" + dask.base.tokenize(fit).lower(),
    )
    computed_kwargs_expr = model_udaf.on_expr(expr)
    if storage:
        computed_kwargs_expr = computed_kwargs_expr.as_table().cache(storage=storage)

    predict_expr_udf = make_pandas_expr_udf(
        computed_kwargs_expr=computed_kwargs_expr,
        fn=predict,
        schema=predict_schema,
        return_type=return_type,
        name=name,
    )

    # deferred_model, deferred_fit, deferred_predict
    return computed_kwargs_expr, model_udaf, predict_expr_udf


(df, features, target) = make_data()
deferred_fit_predict = deferred_fit_predict(
    cls=LinearRegression, return_type=dt.float64
)


# why is this still necessary?
xo.vendor.ibis.formats.pyarrow._from_pyarrow_types[pa.binary_view()] = dt.binary()


con = xo.connect()
expr = con.register(df, "t")


(computed_kwargs_expr, model_udaf, predict_expr_udf) = deferred_fit_predict(
    expr, target, features
)
x = computed_kwargs_expr.execute()
y = predict_expr_udf.on_expr(expr).execute()


storage = ParquetCacheStorage(source=con)
(computed_kwargs_expr, model_udaf, predict_expr_udf) = deferred_fit_predict(
    expr, target, features, storage=storage
)
x = computed_kwargs_expr.execute()
y = predict_expr_udf.on_expr(expr).execute()
