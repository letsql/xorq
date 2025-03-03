import functools
import inspect
import operator
import os
import pickle
import types
import warnings
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING, Callable, Iterable, Iterator, List, Tuple, Union

import dask
import pandas as pd
import pyarrow as pa
import toolz

import xorq as xo
import xorq.expr.udf as udf
import xorq.vendor.ibis.expr.datatypes as dt
import xorq.vendor.ibis.expr.operations as ops
import xorq.vendor.ibis.expr.types as ir
from xorq.expr.udf import make_pandas_expr_udf
from xorq.vendor.ibis import literal
from xorq.vendor.ibis.common.annotations import Argument
from xorq.vendor.ibis.common.collections import FrozenDict
from xorq.vendor.ibis.common.patterns import pattern, replace
from xorq.vendor.ibis.expr.operations.udf import InputType, ScalarUDF
from xorq.vendor.ibis.expr.rules import ValueOf
from xorq.vendor.ibis.util import Namespace


if TYPE_CHECKING:
    from quickgrove import PyGradientBoostedDecisionTrees

SUPPORTED_TYPES = {
    "float": dt.float64,
    "i": dt.boolean,
    "int": dt.int64,
}

p = Namespace(pattern, module=ops)


def _calculate_bounds(
    test_sizes: Iterable[float] | float,
) -> Tuple[Tuple[float, float]]:
    """
    Calculates the cumulative sum of test_sizes and generates bounds for splitting data.

    Parameters
    ----------
    test_sizes : Iterable[float] | float
        An iterable of floats representing the desired proportions for data splits.
        Each value should be between 0 and 1, and their sum must equal 1. The
        order of test sizes determines the order of the generated subsets. If float is passed
        it assumes that the value is for the test size and that a tradition tain test split of (1-test_size, test_size) is returned.

    Returns
    -------
    Tuple[Tuple[float, float]]
        A Tuple of tuples, where each tuple contains two floats representing the
        lower and upper bounds for a split. These bounds are calculated based on
        the cumulative sum of the `test_sizes`.
    """
    # Convert to traditional train test split
    if isinstance(test_sizes, float):
        test_sizes = (1 - test_sizes, test_sizes)

    test_sizes = tuple(test_sizes)

    if not all(isinstance(test_size, float) for test_size in test_sizes):
        raise ValueError("Test size must be float.")

    if not all((0 < test_size < 1) for test_size in test_sizes):
        raise ValueError("test size should be a float between 0 and 1.")

    try:
        pd._testing.assert_almost_equal(sum(test_sizes), 1)
    except AssertionError:
        raise ValueError("Test sizes must sum to 1")

    cumulative_sizes = tuple(toolz.accumulate(operator.add, (0,) + test_sizes))
    bounds = tuple(zip(cumulative_sizes[:-1], cumulative_sizes[1:]))
    return bounds


def calc_split_conditions(
    table: ir.Table,
    unique_key: str | tuple[str] | list[str],
    test_sizes: Iterable[float] | float,
    num_buckets: int = 10000,
    random_seed: int | None = None,
) -> Iterator[ir.BooleanColumn]:
    """
    Parameters
    ----------
    table : ir.Table
        The input Ibis table to be split.
    unique_key : str | tuple[str] | list[str]
        The column name(s) that uniquely identify each row in the table. This
        unique_key is used to create a deterministic split of the dataset
        through a hashing process.
    test_sizes : Iterable[float] | float
        An iterable of floats representing the desired proportions for data splits.
        Each value should be between 0 and 1, and their sum must equal 1. The
        order of test sizes determines the order of the generated subsets. If float is passed
        it assumes that the value is for the test size and that a tradition tain test split of (1-test_size, test_size) is returned.
    num_buckets : int, optional
        The number of buckets into which the data can be binned after being
        hashed (default is 10000). It controls how finely the data is divided
        during the split process. Adjusting num_buckets can affect the
        granularity and efficiency of the splitting operation, balancing
        between accuracy and computational efficiency.
    random_seed : int | None, optional
        Seed for the random number generator. If provided, ensures
        reproducibility of the split (default is None).

    Returns
    -------
    conditions
        A generator of ir.BooleanColumn, each representing whether a row is included in the split

    Raises
    ------
    ValueError
        If `num_buckets` is not an integer greater than 1.

    Examples
    --------
    >>> import xorq as ls
    >>> unique_key = "key"
    >>> table = ls.memtable({unique_key: range(100), "value": range(100, 200)})
    >>> test_sizes = [0.2, 0.3, 0.5]
    >>> col = ls.expr.ml.calc_split_conditions(table, unique_key, test_sizes, num_buckets=10, random_seed=42)
    """

    if not (isinstance(num_buckets, int)):
        raise ValueError("num_buckets must be an integer.")

    if not (num_buckets > 1 and isinstance(num_buckets, int)):
        raise ValueError(
            "num_buckets = 1 places all data into training set. For any integer x  >=0 , x mod 1 = 0 . "
        )

    if isinstance(unique_key, str):
        unique_key = [unique_key]

    # Get cumulative bounds
    bounds = _calculate_bounds(test_sizes=test_sizes)

    # Set the random seed if set, & Generate a random 256-bit key
    random_str = str(Random(random_seed).getrandbits(256))

    comb_key = literal(",").join(table[col].cast("str") for col in unique_key)
    split_bucket = comb_key.concat(random_str).hash().abs().mod(num_buckets)
    conditions = (
        (literal(lower_bound).cast("decimal(38, 9)") * num_buckets <= split_bucket)
        & (
            split_bucket < literal(upper_bound).cast("decimal(38, 9)") * num_buckets
            if i != len(bounds)
            else literal(True)
        )
        for i, (lower_bound, upper_bound) in enumerate(bounds, start=1)
    )
    return conditions


def calc_split_column(
    table: ir.Table,
    unique_key: str | tuple[str] | list[str],
    test_sizes: Iterable[float] | float,
    num_buckets: int = 10000,
    random_seed: int | None = None,
    name: str = "split",
) -> ir.IntegerColumn:
    """
    Parameters
    ----------
    table : ir.Table
        The input Ibis table to be split.
    unique_key : str | tuple[str] | list[str]
        The column name(s) that uniquely identify each row in the table. This
        unique_key is used to create a deterministic split of the dataset
        through a hashing process.
    test_sizes : Iterable[float] | float
        An iterable of floats representing the desired proportions for data splits.
        Each value should be between 0 and 1, and their sum must equal 1. The
        order of test sizes determines the order of the generated subsets. If float is passed
        it assumes that the value is for the test size and that a tradition tain test split of (1-test_size, test_size) is returned.
    num_buckets : int, optional
        The number of buckets into which the data can be binned after being
        hashed (default is 10000). It controls how finely the data is divided
        during the split process. Adjusting num_buckets can affect the
        granularity and efficiency of the splitting operation, balancing
        between accuracy and computational efficiency.
    random_seed : int | None, optional
        Seed for the random number generator. If provided, ensures
        reproducibility of the split (default is None).
    name : str, optional
        Name for the returned IntegerColumn (default is "split").

    Returns
    -------
    ibis.IntergerColumn
        A column with split indices representing mutually exclusive subsets of the original table based on the specified test sizes.

    Raises
    ------
    ValueError
        If any value in `test_sizes` is not between 0 and 1.
        If `test_sizes` does not sum to 1.
        If `num_buckets` is not an integer greater than 1.

    Examples
    --------
    >>> import xorq as ls
    >>> unique_key = "key"
    >>> table = ls.memtable({unique_key: range(100), "value": range(100, 200)})
    >>> test_sizes = [0.2, 0.3, 0.5]
    >>> col = ls.expr.ml.calc_split_column(table, unique_key, test_sizes, num_buckets=10, random_seed=42, name="my-split")
    """

    conditions = calc_split_conditions(
        table=table,
        unique_key=unique_key,
        test_sizes=test_sizes,
        num_buckets=num_buckets,
        random_seed=random_seed,
    )
    col = xo.case()
    for i, condition in enumerate(conditions):
        col = col.when(condition, xo.literal(i, "int64"))
    col = col.end().name(name)
    return col


def train_test_splits(
    table: ir.Table,
    unique_key: str | tuple[str] | list[str],
    test_sizes: Iterable[float] | float,
    num_buckets: int = 10000,
    random_seed: int | None = None,
) -> Iterator[ir.Table]:
    """Generates multiple train/test splits of an Ibis table for different test sizes.

    This function splits an Ibis table into multiple subsets based on a unique key
    or combination of keys and a list of test sizes. It uses a hashing function to
    convert the unique key into an integer, then applies a modulo operation to split
    the data into buckets. Each subset of data is defined by a range of
    buckets determined by the cumulative sum of the test sizes.

    Parameters
    ----------
    table : ir.Table
        The input Ibis table to be split.
    unique_key : str | tuple[str] | list[str]
        The column name(s) that uniquely identify each row in the table. This
        unique_key is used to create a deterministic split of the dataset
        through a hashing process.
    test_sizes : Iterable[float] | float
        An iterable of floats representing the desired proportions for data splits.
        Each value should be between 0 and 1, and their sum must equal 1. The
        order of test sizes determines the order of the generated subsets. If float is passed
        it assumes that the value is for the test size and that a tradition tain test split of (1-test_size, test_size) is returned.
    num_buckets : int, optional
        The number of buckets into which the data can be binned after being
        hashed (default is 10000). It controls how finely the data is divided
        during the split process. Adjusting num_buckets can affect the
        granularity and efficiency of the splitting operation, balancing
        between accuracy and computational efficiency.
    random_seed : int | None, optional
        Seed for the random number generator. If provided, ensures
        reproducibility of the split (default is None).

    Returns
    -------
    Iterator[ir.Table]
        An iterator yielding Ibis table expressions, each representing a mutually exclusive
        subset of the original table based on the specified test sizes.

    Raises
    ------
    ValueError
        If any value in `test_sizes` is not between 0 and 1.
        If `test_sizes` does not sum to 1.
        If `num_buckets` is not an integer greater than 1.

    Examples
    --------
    >>> import xorq as ls
    >>> table = ls.memtable({"key": range(100), "value": range(100,200)})
    >>> unique_key = "key"
    >>> test_sizes = [0.2, 0.3, 0.5]
    >>> splits = ls.train_test_splits(table, unique_key, test_sizes, num_buckets=10, random_seed=42)
    >>> for i, split_table in enumerate(splits):
    ...     print(f"Split {i+1} size: {split_table.count().execute()}")
    ...     print(split_table.execute())
    Split 1 size: 20
    Split 2 size: 30
    Split 3 size: 50
    """
    conditions = calc_split_conditions(
        table=table,
        unique_key=unique_key,
        test_sizes=test_sizes,
        num_buckets=num_buckets,
        random_seed=random_seed,
    )
    return map(table.filter, conditions)


def fields_to_parameters(fields: dict) -> List[inspect.Parameter]:
    """Convert a field dict into `inspect.Parameter` objects for constructing function signatures."""

    parameters = []
    for name, arg in fields.items():
        param = inspect.Parameter(
            name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=arg.typehint,
        )
        parameters.append(param)
    return parameters


class UDFWrapper:
    """Wrapper to hold the constructed function, model, and path for nice representations."""

    def __init__(
        self,
        func: Callable,
        model: "PyGradientBoostedDecisionTrees",
        model_path: Union[str, Path, None],
        required_features: List[str] = None,
    ):
        self.func = func
        self.model = model
        self.model_path = model_path
        self.required_features = required_features or []
        functools.update_wrapper(self, func)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __repr__(self):
        path_str = f"\nModel path: {self.model_path}" if self.model_path else ""
        return f"{str(self.model)}{path_str}\nSignature: {str(self.func.__signature__)}"

    def on_expr(self, e: ir.Table, **kwargs) -> ir.Column:
        """
        Call the UDF by extracting the required columns from the table expression `e`.
        E.g., e['col1'], e['col2'], etc.
        """
        columns = [e[col] for col in self.required_features]
        return self(*columns, **kwargs)


def _load_quickgrove_model(
    model_or_path: Union[str, Path, "PyGradientBoostedDecisionTrees"],
) -> Tuple["PyGradientBoostedDecisionTrees", Union[str, Path, None]]:
    """Load the model if a path is provided; otherwise, pass through the model."""

    import quickgrove

    if isinstance(model_or_path, (str, Path)):
        model = quickgrove.json_load(model_or_path)
        return model, model_or_path
    else:
        return model_or_path, None


def _validate_model_features(
    model: "PyGradientBoostedDecisionTrees", supported_types: dict
) -> None:
    """Raise an error if the model has unsupported feature types."""
    schema = [model.feature_names[i] for i in sorted(model.required_features)]
    feature_types = [model.feature_types[i] for i in sorted(model.required_features)]

    unsupported = [
        f"{name}: {type_}"
        for name, type_ in zip(schema, feature_types)
        if type_ not in supported_types
    ]
    if unsupported:
        raise ValueError(f"Unsupported feature types: {', '.join(unsupported)}")


def _create_udf_node(
    model: "PyGradientBoostedDecisionTrees",
    fn_from_arrays: Callable,
    fields: dict,
    udf_name: str,
    extra_meta: dict = None,
) -> type:
    """Dynamically create a subclass of ScalarUDF for a quickgrove-based function."""

    meta = {
        "dtype": dt.float32,
        "__input_type__": InputType.PYARROW,
        "__func__": property(lambda self: fn_from_arrays),
        "__config__": FrozenDict(volatility="immutable"),
        "__udf_namespace__": p,
        "__module__": "xorq.expr.ml",
        "__func_name__": udf_name,
        "__fields__": fields,
        "model": model,
    }
    if extra_meta:
        meta.update(extra_meta)

    node = type(udf_name, (ScalarUDF,), {**fields, **meta})
    return node


def _all_predicates_are_features(filter_op: ops.Filter, model) -> bool:
    feature_set = set(model.feature_names)
    for pred in filter_op.predicates:
        if not (
            isinstance(pred, (ops.Less, ops.Greater, ops.GreaterEqual))
            and isinstance(pred.left, ops.Field)
            and pred.left.name in feature_set
        ):
            return False
    return True


def _extract_model_name(model_path):
    return os.path.splitext(os.path.basename(model_path))[0]


def _create_udf_function(node_cls: type) -> Callable:
    """
    Create a function that, when called, instantiates `node_cls` and returns the Ibis expression.
    Attach an `inspect.Signature` for introspection.
    """
    fn_from_arrays = node_cls.__func__.fget(None)

    @functools.wraps(fn_from_arrays)
    def construct(*args, **kwargs):
        return node_cls(*args, **kwargs).to_expr()

    signature = inspect.Signature(
        fields_to_parameters(node_cls.__fields__),
        return_annotation=dt.float32,
    )

    def actual_func(*args, **kwargs):
        return construct(*args, **kwargs)

    new_func = types.FunctionType(
        actual_func.__code__,
        actual_func.__globals__,
        name=node_cls.__func_name__,
        argdefs=actual_func.__defaults__,
        closure=actual_func.__closure__,
    )
    new_func.__signature__ = signature
    return new_func


def make_quickgrove_udf(
    model_or_path: Union[str, Path, "PyGradientBoostedDecisionTrees"], model_name=None
) -> UDFWrapper:
    """
    Create a UDF from a quickgrove (quickgrove) model.

    :param model_or_path: Either an already-loaded model or a path to the model.
    :param model_name: Alternative model name, if the name from path is not a valid Python identifier.
    :return:
    """

    model, model_path = _load_quickgrove_model(model_or_path)

    if not model_name:
        new_model_name = _extract_model_name(model_path)
        if not new_model_name.isidentifier():
            raise ValueError(
                f"The argument model_name was {model_name} and the name extracted from the path is not a valid Python identifier"
            )
        model_name = new_model_name
    elif not model_name.isidentifier():
        raise ValueError("The argument model_name is not a valid Python identifier")

    _validate_model_features(model, SUPPORTED_TYPES)

    required_features = sorted(model.required_features)
    schema = [model.feature_names[i] for i in required_features]
    feature_types = [model.feature_types[i] for i in required_features]

    fields = {
        name: Argument(
            pattern=ValueOf(SUPPORTED_TYPES[type_]),
            typehint=SUPPORTED_TYPES[type_],
        )
        for name, type_ in zip(schema, feature_types)
    }

    def fn_from_arrays(*arrays):
        return model.predict_arrays(list(arrays))

    udf_func = _create_udf_function(
        _create_udf_node(
            model=model,
            fn_from_arrays=fn_from_arrays,
            fields=fields,
            udf_name=model_name,
            extra_meta={"model_path": model_path},
        )
    )

    return UDFWrapper(
        udf_func,
        model,
        model_path,
        required_features=schema,
    )


def collect_predicates(filter_op: ops.Filter) -> List[dict]:
    """Extract pushdown predicates that compare a base column (ops.Field) to a literal.
    Ignore any predicates referencing a UDF output."""

    parent_op = filter_op.parent

    predicates = []
    for pred in filter_op.predicates:
        if isinstance(pred, (ops.Less, ops.Greater, ops.GreaterEqual)):
            if isinstance(pred.left, ops.Field) and isinstance(pred.right, ops.Literal):
                field_name = pred.left.name
                if isinstance(parent_op.values.get(field_name), ops.ScalarUDF):
                    raise ValueError(
                        f"Unsupported predicate on UDF column: {field_name}"
                    )

                predicates.append(
                    {
                        "column": field_name,
                        "op": pred.__class__.__name__,
                        "value": pred.right.value,
                    }
                )
    return predicates


def make_pruned_udf(
    original_udf: UDFWrapper,
    predicates: List[dict],
) -> UDFWrapper:
    """
    Create a new pruned UDF from an existing UDF and a list of predicates.
    `original_udf` must have a `.model` attribute.
    """

    from quickgrove import Feature

    model = original_udf.model
    pred_feature_names = {pred["column"] for pred in predicates}
    model_feature_names = set(model.feature_names)

    if not pred_feature_names.issubset(model_feature_names):
        warnings.warn(
            "Feature not found in predicates, skipping pruning...", UserWarning
        )
        return original_udf

    pruned_model = model.prune(
        [
            Feature(pred["column"]) < pred["value"]
            if pred["op"] == "Less"
            else Feature(pred["column"]) >= pred["value"]
            if pred["op"] == "Greater"
            else Feature(pred["column"]) >= pred["value"]
            for pred in predicates
        ]
    )

    _validate_model_features(pruned_model, SUPPORTED_TYPES)

    required_features = sorted(pruned_model.required_features)
    schema = [pruned_model.feature_names[i] for i in required_features]
    feature_types = [pruned_model.feature_types[i] for i in required_features]

    fields = {
        name: Argument(
            pattern=ValueOf(SUPPORTED_TYPES[type_]), typehint=SUPPORTED_TYPES[type_]
        )
        for name, type_ in zip(schema, feature_types)
    }

    def fn_from_arrays(*arrays):
        return pruned_model.predict_arrays(list(arrays))

    udf_func = _create_udf_function(
        _create_udf_node(
            model=pruned_model,
            fn_from_arrays=fn_from_arrays,
            fields=fields,
            udf_name=original_udf.__func_name__ + "_pruned",
        )
    )

    return UDFWrapper(
        udf_func,
        pruned_model,
        original_udf.model_path,
        required_features=schema,
    )


@replace(p.Filter(p.Project))
def prune_quickgrove_model(_, **kwargs):
    """Rewrite rule to prune quickgrove model based on filter predicates."""

    parent_op = _.parent
    predicates = collect_predicates(_)
    if not predicates:
        return _

    new_values = {}
    pruned_udf_wrapper = None

    for name, value in parent_op.values.items():
        if isinstance(value, ops.ScalarUDF) and hasattr(value, "model"):
            # return filter op if predicates are not in filter
            if not _all_predicates_are_features(_, value.model):
                return _

            pruned_udf_wrapper = make_pruned_udf(value, predicates)
            pruned_model = pruned_udf_wrapper.model
            required_features = [
                pruned_model.feature_names[i] for i in pruned_model.required_features
            ]
            required_features = [
                pruned_model.feature_names[i] for i in pruned_model.required_features
            ]
            udf_kwargs = {
                feat_name: parent_op.values[feat_name]
                for feat_name in required_features
            }
            if isinstance(pruned_udf_wrapper, Callable):
                new_values[name] = pruned_udf_wrapper(**udf_kwargs)
        else:
            new_values[name] = value

    if not pruned_udf_wrapper:
        return _

    new_project = ops.Project(parent_op.parent, new_values)

    subs = {
        ops.Field(parent_op, k): ops.Field(new_project, k) for k in parent_op.values
    }
    new_predicates = tuple(p.replace(subs, filter=ops.Value) for p in _.predicates)
    return ops.Filter(parent=new_project, predicates=new_predicates)


def rewrite_quickgrove_expr(expr: ir.Table) -> ir.Table:
    """Rewrite an Ibis expression by pruning quickgrove models based on filter conditions."""

    op = expr.op()
    new_op = op.replace(prune_quickgrove_model)
    return new_op.to_expr()


@toolz.curry
def deferred_fit_predict(
    expr,
    target,
    features,
    cls,
    return_type,
    name="predicted",
    storage=None,
):
    def fit(fit_df):
        obj = cls()
        obj.fit(fit_df[features], fit_df[target])
        return obj

    @toolz.curry
    def predict(model, df):
        return pa.array(
            model.predict(df[features]),
            type=return_type.to_pyarrow(),
        )

    features = features or list(expr.schema())
    predict_schema = xo.schema({col: expr[col].type() for col in features})
    fit_schema = predict_schema | xo.schema({target: expr[target].type()})
    model_udaf = udf.agg.pandas_df(
        fn=toolz.compose(pickle.dumps, fit),
        schema=fit_schema,
        return_type=dt.binary,
        name="_" + dask.base.tokenize(fit).lower(),
    )
    deferred_model = model_udaf.on_expr(expr)
    if storage:
        deferred_model = deferred_model.as_table().cache(storage=storage)

    deferred_predict = make_pandas_expr_udf(
        computed_kwargs_expr=deferred_model,
        fn=predict,
        schema=predict_schema,
        return_type=return_type,
        name=name,
    )

    return deferred_model, model_udaf, deferred_predict


@toolz.curry
def deferred_fit_transform_series(
    expr, col, cls, return_type, name="predicted", storage=None
):
    def fit(fit_df, cls=cls):
        model = cls()
        model.fit(fit_df[col])
        return model

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
    deferred_model = model_udaf.on_expr(expr)
    if storage:
        deferred_model = deferred_model.as_table().cache(storage=storage)

    deferred_transform = make_pandas_expr_udf(
        computed_kwargs_expr=deferred_model,
        fn=transform,
        schema=schema,
        return_type=return_type,
        name=name,
    )

    return deferred_model, model_udaf, deferred_transform
