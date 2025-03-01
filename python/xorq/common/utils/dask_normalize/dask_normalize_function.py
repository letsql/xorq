import functools
import types

import dask.base
import toolz

from xorq.common.utils.dask_normalize.dask_normalize_utils import (
    normalize_seq_with_caller,
)
from xorq.common.utils.inspect_utils import (
    get_partial_arguments,
)


CODE_ATTRS = (
    "co_argcount",
    "co_cellvars",
    "co_code",
    "co_consts",
    # co_flags | NESTED?
    "co_flags",
    "co_freevars",
    "co_kwonlyargcount",
    "co_name",
    "co_names",
    "co_nlocals",
    "co_stacksize",
    "co_varnames",
    # 'co_lnotab', 'co_filename', 'co_firstlineno',
)
FUNCTION_ATTRS = (
    "__class__",
    "__closure__",
    "__code__",
    "__defaults__",
    "__dict__",
    "__kwdefaults__",
    "__module__",
    "__name__",
    "__qualname__",
)


@toolz.curry
def normalize_by_attrs(attrs, obj):
    objs = tuple(getattr(obj, attr, None) for attr in attrs)
    return normalize_seq_with_caller(*objs)


@dask.base.normalize_token.register(
    (
        types.FunctionType,
        types.MethodType,
        functools._lru_cache_wrapper,
        # HAK: add classmethod:
        #      NOTE: we are *IN*-sensitive to class definition changes
        classmethod,
    )
)
def normalize_function(function):
    def unwrap(obj, attr_name):
        while hasattr(obj, attr_name):
            obj = getattr(obj, attr_name)
        return obj

    function = unwrap(function, "__wrapped__")
    normalized = normalize_by_attrs(FUNCTION_ATTRS, function)
    return normalized


normalize_code = normalize_by_attrs(CODE_ATTRS)
dask.base.normalize_token.register(types.CodeType, normalize_code)
dask.base.normalize_token.register(property, normalize_code)


@dask.base.normalize_token.register(toolz.functoolz.Compose)
def normalize_toolz_compose(composed):
    return normalize_seq_with_caller(
        toolz.functoolz.Compose,
        composed.first,
        composed.funcs,
    )


@dask.base.normalize_token.register(toolz.curry)
def normalize_toolz_curry(curried):
    partial_arguments = get_partial_arguments(
        curried.func, *curried.args, **curried.keywords
    )
    objs = sum(
        map(
            dask.base.normalize_token,
            (curried.func, partial_arguments),
        ),
        start=(),
    )
    return objs


def make_cell_typ():
    # https://stackoverflow.com/a/23830790
    def outer(x):
        def inner(y):
            return x * y

        return inner

    inner = outer(1)
    typ = type(inner.__closure__[0])
    return typ


@dask.base.normalize_token.register(make_cell_typ())
def normalize_cell(cell):
    return dask.base.normalize_token(cell.cell_contents)


@dask.base.normalize_token.register(toolz.functoolz.excepts)
def normalize_excepts(f):
    return normalize_seq_with_caller(
        f.exc,
        f.func,
        # FIXME: figure out how to include handler
        # f.handler,
    )
