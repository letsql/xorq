import functools
import pickle
from typing import Any

import pyarrow as pa
import toolz

import xorq.expr.datatypes as dt
import xorq.vendor.ibis.expr.rules as rlz
import xorq.vendor.ibis.expr.types as ir
from xorq.vendor.ibis.common.annotations import Argument
from xorq.vendor.ibis.common.collections import FrozenDict
from xorq.vendor.ibis.expr.operations import Namespace
from xorq.vendor.ibis.expr.operations.udf import (
    _UDF,
    AggUDF,
    InputType,
    ScalarUDF,
    _make_udf_name,
    _wrap,
)


@toolz.curry
def wrap_model(value, model_key="model"):
    return pickle.dumps({model_key: value})


unwrap_model = pickle.loads


class ExprScalarUDF(ScalarUDF):
    @property
    def computed_kwargs_expr(self):
        # must push the expr into __config__ so that it doesn't get turned into a window function
        return self.__config__["computed_kwargs_expr"]

    @property
    def post_process_fn(self):
        return self.__config__["post_process_fn"]


@toolz.curry
def make_pandas_expr_udf(
    computed_kwargs_expr,
    fn,
    schema,
    return_type=dt.binary,
    database=None,
    catalog=None,
    name=None,
    *,
    post_process_fn=unwrap_model,
    **kwargs,
):
    import pandas as pd

    def fn_from_arrays(*arrays, computed_arg=None, **kwargs):
        if computed_arg is None:
            raise ValueError(
                "Caller must bind computed_arg to the output of computed_kwargs_expr"
            )
        df = pd.DataFrame(
            {name: array.to_pandas() for (name, array) in zip(schema, arrays)}
        )
        return pa.Array.from_pandas(
            fn(computed_arg, df, **kwargs), type=return_type.to_pyarrow()
        )

    name = name if name is not None else _make_udf_name(fn)
    bases = (ExprScalarUDF,)
    fields = {
        arg_name: Argument(pattern=rlz.ValueOf(typ), typehint=typ)
        for (arg_name, typ) in schema.items()
    }
    meta = {
        "dtype": return_type,
        "__input_type__": InputType.PYARROW,
        "__func__": property(
            fget=lambda _, fn_from_arrays=fn_from_arrays: fn_from_arrays
        ),
        # valid config keys: computed_kwargs_expr, post_process_fn, volatility
        "__config__": FrozenDict(
            computed_kwargs_expr=computed_kwargs_expr,
            post_process_fn=post_process_fn,
            **kwargs,
        ),
        "__udf_namespace__": Namespace(database=database, catalog=catalog),
        "__module__": fn.__module__,
        # FIXME: determine why this fails with case mismatch by default
        "__func_name__": name,
    }
    kwds = {
        **fields,
        **meta,
    }

    node = type(
        name,
        bases,
        kwds,
    )

    # FIXME: enable working with deferred like _wrap enables
    @functools.wraps(fn)
    def construct(*args: Any, **kwargs: Any) -> ir.Value:
        return node(*args, **kwargs).to_expr()

    def on_expr(e, **kwargs):
        return construct(*(e[c] for c in schema), **kwargs)

    construct.on_expr = on_expr
    construct.fn = fn
    return construct


@toolz.curry
def make_pandas_udf(
    fn, schema, return_type, database=None, catalog=None, name=None, **kwargs
):
    import pandas as pd

    from xorq.vendor.ibis.expr.operations.udf import ScalarUDF

    def fn_from_arrays(*arrays):
        df = pd.DataFrame(
            {name: array.to_pandas() for (name, array) in zip(schema, arrays)}
        )
        return pa.Array.from_pandas(fn(df), type=return_type.to_pyarrow())

    name = name if name is not None else _make_udf_name(fn)
    bases = (ScalarUDF,)
    fields = {
        arg_name: Argument(pattern=rlz.ValueOf(typ), typehint=typ)
        for (arg_name, typ) in schema.items()
    }
    meta = {
        "dtype": return_type,
        "__input_type__": InputType.PYARROW,
        "__func__": property(
            fget=lambda _, fn_from_arrays=fn_from_arrays: fn_from_arrays
        ),
        # valid config keys: volatility
        "__config__": FrozenDict(**kwargs),
        "__udf_namespace__": Namespace(database=database, catalog=catalog),
        "__module__": fn.__module__,
        # FIXME: determine why this fails with case mismatch by default
        "__func_name__": name,
    }
    kwds = {
        **fields,
        **meta,
    }

    node = type(
        name,
        bases,
        kwds,
    )

    # FIXME: enable working with deferred like _wrap enables
    @functools.wraps(fn)
    def construct(*args: Any, **kwargs: Any) -> ir.Value:
        return node(*args, **kwargs).to_expr()

    def on_expr(e, **kwargs):
        return construct(*(e[c] for c in schema), **kwargs)

    construct.on_expr = on_expr
    construct.fn = fn
    return construct


class agg(_UDF):
    __slots__ = ()

    _base = AggUDF

    @classmethod
    def pyarrow(
        cls,
        fn=None,
        name=None,
        signature=None,
        **kwargs,
    ):
        result = _wrap(
            cls._make_wrapper,
            InputType.PYARROW,
            fn,
            name=name,
            signature=signature,
            **kwargs,
        )
        return result

    @classmethod
    @toolz.curry
    def pandas_df(
        cls,
        fn,
        schema,
        return_type,
        database=None,
        catalog=None,
        name=None,
        **kwargs,
    ):
        import pandas as pd

        def fn_from_arrays(*arrays):
            df = pd.DataFrame(
                {name: array.to_pandas() for (name, array) in zip(schema, arrays)}
            )
            return fn(df)

        name = name if name is not None else _make_udf_name(fn)
        bases = (cls._base,)
        fields = {
            arg_name: Argument(pattern=rlz.ValueOf(typ), typehint=typ)
            for (arg_name, typ) in schema.items()
        }
        meta = {
            "dtype": return_type,
            "__input_type__": InputType.PYARROW,
            "__func__": property(
                fget=lambda _, fn_from_arrays=fn_from_arrays: fn_from_arrays
            ),
            # valid config keys: volatility
            "__config__": FrozenDict(**kwargs),
            "__udf_namespace__": Namespace(database=database, catalog=catalog),
            "__module__": fn.__module__,
            # FIXME: determine why this fails with case mismatch by default
            "__func_name__": name,
        }
        kwds = {
            **fields,
            **meta,
        }

        node = type(
            name,
            bases,
            kwds,
        )

        # FIXME: enable working with deferred like _wrap enables
        @functools.wraps(fn)
        def construct(*args: Any, **kwargs: Any) -> ir.Value:
            return node(*args, **kwargs).to_expr()

        def on_expr(e, **kwargs):
            return construct(*(e[c] for c in schema), **kwargs)

        construct.on_expr = on_expr
        construct.fn = fn
        return construct


def arbitrate_evaluate(
    uses_window_frame=False,
    supports_bounded_execution=False,
    include_rank=False,
    **config_kwargs,
):
    match (uses_window_frame, supports_bounded_execution, include_rank):
        case (False, False, False):
            return "evaluate_all"
        case (False, True, False):
            return "evaluate"
        case (False, _, True):
            return "evaluate_all_with_rank"
        case (True, _, _):
            return "evaluate"
        case _:
            raise RuntimeError


@toolz.curry
def pyarrow_udwf(
    fn,
    schema,
    return_type,
    name=None,
    namespace=Namespace(database=None, catalog=None),
    base=AggUDF,
    **config_kwargs,
):
    fields = {
        arg_name: Argument(pattern=rlz.ValueOf(typ), typehint=typ)
        for (arg_name, typ) in schema.items()
    }
    # which_evaluate = arbitrate_evaluate(**config_kwargs)
    name = name or fn.__name__
    meta = {
        "dtype": return_type,
        "__input_type__": InputType.PYARROW,
        "__func__": property(fget=toolz.functoolz.return_none),
        "__config__": FrozenDict(
            input_types=tuple(el.type for el in schema.to_pyarrow()),
            return_type=return_type.to_pyarrow(),
            name=name,
            **config_kwargs,
            # assert which_evaluate in ("evaluate", "evaluate_all", "evaluate_all_with_rank")
            # **{which_evaluate: fn},
            **{
                which_evaluate: fn
                for which_evaluate in (
                    "evaluate",
                    "evaluate_all",
                    "evaluate_all_with_rank",
                )
            },
        ),
        "__udf_namespace__": namespace,
        "__module__": __name__,
        "__func_name__": name,
    }
    node = type(
        name,
        (base,),
        {
            **fields,
            **meta,
        },
    )

    def construct(*args: Any, **kwargs: Any) -> ir.Value:
        return node(*args, **kwargs).to_expr()

    def on_expr(e, **kwargs):
        return construct(*(e[c] for c in schema), **kwargs)

    construct.on_expr = on_expr
    return construct
