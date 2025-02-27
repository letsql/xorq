import pickle

import pandas as pd
import toolz
import xgboost as xgb

import xorq as xo
import xorq.expr.udf as udf
import xorq.vendor.ibis.expr.datatypes as dt
from xorq.expr.udf import (
    make_pandas_expr_udf,
)


ROWNUM = "rownum"
features = (
    "emp_length",
    "dti",
    "annual_inc",
    "loan_amnt",
    "fico_range_high",
    "cr_age_days",
)
target = "event_occurred"
model_key = "model"
prediction_key = "predicted"
prediction_typ = "float32"


@toolz.curry
def train_xgboost_model(df, features=features, target=target, seed=0):
    param = {"max_depth": 4, "eta": 1, "objective": "binary:logistic", "seed": seed}
    num_round = 10
    if ROWNUM in df:
        # enforce order for reproducibility
        df = df.sort_values(ROWNUM, ignore_index=True)
    X = df[list(features)]
    y = df[target]
    dtrain = xgb.DMatrix(X, y)
    bst = xgb.train(param, dtrain, num_boost_round=num_round)
    return bst


@toolz.curry
def predict_xgboost_model(model, df, features=features):
    return model.predict(xgb.DMatrix(df[list(features)]))


t = xo.read_parquet(xo.config.options.pins.get_path("lending-club"))
(train, test) = xo.train_test_splits(
    t,
    unique_key=ROWNUM,
    test_sizes=0.7,
)

# manual run
model = train_xgboost_model(train.execute())
from_pd = test.execute().assign(
    **{prediction_key: predict_xgboost_model(model)},
)

# using expr-scalar-udf
model_udaf = udf.agg.pandas_df(
    fn=toolz.compose(pickle.dumps, train_xgboost_model),
    schema=t[features + (target,)].schema(),
    return_type=dt.binary,
    name=model_key,
)
predict_expr_udf = make_pandas_expr_udf(
    computed_kwargs_expr=model_udaf.on_expr(train),
    fn=predict_xgboost_model,
    schema=t[features].schema(),
    return_type=dt.dtype(prediction_typ),
    name=prediction_key,
)
from_ls = test.mutate(predict_expr_udf.on_expr(test).name(prediction_key)).execute()

pd._testing.assert_frame_equal(from_ls, from_pd)
