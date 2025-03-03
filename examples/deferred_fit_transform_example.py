import pathlib
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

import xorq as xo
import xorq.vendor.ibis.expr.datatypes as dt
from xorq.caching import ParquetStorage
from xorq.common.utils.import_utils import import_path
from xorq.expr.ml import deferred_fit_transform_series


m = import_path(pathlib.Path(__file__).parent.joinpath("hackernews_lib.py"))


deferred_fit_transform_tfidf = deferred_fit_transform_series(
    col="title", cls=TfidfVectorizer, return_type=dt.Array(dt.float64)
)


con = xo.connect()
t = xo.memtable(
    data=({"maxitem": 43182839, "n": 1000},),
    name="t",
).pipe(con.register, table_name="t")
# we must have two streams with different names
# and the inner name must be provided so we don't get a (uncacheable) non-deterministic name
train_expr = m.do_hackernews_fetcher_udxf(t, inner_name="inner-name", name="train")
transform_expr = m.do_hackernews_fetcher_udxf(
    t, inner_name="inner-name", name="transform"
)


# uncached run
(deferred_model, model_udaf, deferred_transform) = deferred_fit_transform_tfidf(
    train_expr,
)
model = deferred_model.execute()
predicted = transform_expr.mutate(deferred_transform.on_expr(transform_expr)).execute()


# cached run
storage = ParquetStorage(source=con)
(deferred_model, model_udaf, deferred_transform) = deferred_fit_transform_tfidf(
    train_expr, storage=storage
)
((cached_model,),) = deferred_model.execute().values
cached_predicted = transform_expr.mutate(
    deferred_transform.on_expr(transform_expr)
).execute()


assert predicted.equals(cached_predicted)
(x, y) = (pickle.loads(el) for el in (model, cached_model))
assert all(x.idf_ == y.idf_)
assert x.vocabulary_ == y.vocabulary_
