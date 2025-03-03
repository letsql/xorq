from __future__ import annotations

import datetime
import decimal
import functools
import operator
from typing import Any

import xorq.vendor.ibis as ibis
import xorq.vendor.ibis.expr.datatypes as dt
import xorq.vendor.ibis.expr.operations as ops
import xorq.vendor.ibis.expr.operations.temporal as tm
import xorq.vendor.ibis.expr.types as ir
from xorq.expr.relations import CachedNode, Read, RemoteTable, into_backend
from xorq.ibis_yaml.common import (
    TranslationContext,
    _translate_type,
    register_from_yaml_handler,
    translate_from_yaml,
    translate_to_yaml,
)

# ruff: noqa: F401
from xorq.ibis_yaml.udf import _scalar_udf_from_yaml, _scalar_udf_to_yaml
from xorq.ibis_yaml.utils import (
    freeze,
    load_storage_from_yaml,
    translate_storage,
)
from xorq.vendor.ibis.expr.operations.relations import Namespace


def should_register_node(node_dict):
    if "parent" in node_dict and isinstance(node_dict["parent"], dict):
        return True
    return False


@_translate_type.register(dt.Timestamp)
def _translate_timestamp_type(dtype: dt.Timestamp) -> dict:
    base = {"name": "Timestamp", "nullable": dtype.nullable}
    if dtype.timezone is not None:
        base["timezone"] = dtype.timezone
    return freeze(base)


@_translate_type.register(dt.Decimal)
def _translate_decimal_type(dtype: dt.Decimal) -> dict:
    base = {"name": "Decimal", "nullable": dtype.nullable}
    if dtype.precision is not None:
        base["precision"] = dtype.precision
    if dtype.scale is not None:
        base["scale"] = dtype.scale
    return freeze(base)


@_translate_type.register(dt.Array)
def _translate_array_type(dtype: dt.Array) -> dict:
    return freeze(
        {
            "name": "Array",
            "value_type": _translate_type(dtype.value_type),
            "nullable": dtype.nullable,
        }
    )


@_translate_type.register(dt.Map)
def _translate_map_type(dtype: dt.Map) -> dict:
    return freeze(
        {
            "name": "Map",
            "key_type": _translate_type(dtype.key_type),
            "value_type": _translate_type(dtype.value_type),
            "nullable": dtype.nullable,
        }
    )


@_translate_type.register(dt.Interval)
def _tranlate_type_interval(dtype: dt.Interval) -> dict:
    return freeze(
        {
            "name": "Interval",
            "unit": _translate_temporal_unit(dtype.unit),
            "nullable": dtype.nullable,
        }
    )


@_translate_type.register(dt.Struct)
def _translate_struct_type(dtype: dt.Struct) -> dict:
    return freeze(
        {
            "name": "Struct",
            "fields": {
                name: _translate_type(field_type)
                for name, field_type in zip(dtype.names, dtype.types)
            },
            "nullable": dtype.nullable,
        }
    )


def _translate_temporal_unit(unit: tm.IntervalUnit) -> dict:
    if unit.is_date():
        unit_name = "DateUnit"
    elif unit.is_time():
        unit_name = "TimeUnit"
    else:
        unit_name = "IntervalUnit"
    return freeze({"name": unit_name, "value": unit.value})


def _translate_literal_value(value: Any, dtype: dt.DataType) -> Any:
    if value is None:
        return None
    elif isinstance(value, (bool, int, float, str)):
        return value
    elif isinstance(value, decimal.Decimal):
        return str(value)
    elif isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    elif isinstance(value, list):
        return [_translate_literal_value(v, dtype.value_type) for v in value]
    elif isinstance(value, dict):
        return {
            _translate_literal_value(k, dtype.key_type): _translate_literal_value(
                v, dtype.value_type
            )
            for k, v in value.items()
        }
    else:
        return value


@translate_to_yaml.register(ir.Expr)
def _expr_to_yaml(expr: ir.Expr, context: any) -> dict:
    return translate_to_yaml(expr.op(), context)


@translate_to_yaml.register(ops.WindowFunction)
def _window_function_to_yaml(
    op: ops.WindowFunction, context: TranslationContext
) -> dict:
    result = {
        "op": "WindowFunction",
        "args": [translate_to_yaml(op.func, context)],
        "type": _translate_type(op.dtype),
    }

    if op.group_by:
        result["group_by"] = [translate_to_yaml(expr, context) for expr in op.group_by]

    if op.order_by:
        result["order_by"] = [translate_to_yaml(expr, context) for expr in op.order_by]

    if op.start is not None:
        result["start"] = (
            translate_to_yaml(op.start.value, context)["value"]
            if isinstance(op.start, ops.WindowBoundary)
            else op.start
        )

    if op.end is not None:
        result["end"] = (
            translate_to_yaml(op.end.value, context)["value"]
            if isinstance(op.end, ops.WindowBoundary)
            else op.end
        )

    return freeze(result)


@register_from_yaml_handler("WindowFunction")
def _window_function_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    func = translate_from_yaml(yaml_dict["args"][0], context)
    group_by = [translate_from_yaml(g, context) for g in yaml_dict.get("group_by", [])]
    order_by = [translate_from_yaml(o, context) for o in yaml_dict.get("order_by", [])]
    start = ibis.literal(yaml_dict["start"]) if "start" in yaml_dict else None
    end = ibis.literal(yaml_dict["end"]) if "end" in yaml_dict else None
    window = ibis.window(
        group_by=group_by, order_by=order_by, preceding=start, following=end
    )
    return func.over(window)


@translate_to_yaml.register(ops.WindowBoundary)
def _window_boundary_to_yaml(
    op: ops.WindowBoundary, context: TranslationContext
) -> dict:
    return freeze(
        {
            "op": "WindowBoundary",
            "value": translate_to_yaml(op.value, context),
            "preceding": op.preceding,
            "type": _translate_type(op.dtype),
        }
    )


@register_from_yaml_handler("WindowBoundary")
def _window_boundary_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    value = translate_from_yaml(yaml_dict["value"], context)
    return ops.WindowBoundary(value, preceding=yaml_dict["preceding"])


@translate_to_yaml.register(ops.Node)
def _base_op_to_yaml(op: ops.Node, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": type(op).__name__,
            "args": [
                translate_to_yaml(arg, context)
                for arg in op.args
                if isinstance(arg, (ops.Value, ops.Node))
            ],
        }
    )


@translate_to_yaml.register(ops.UnboundTable)
def _unbound_table_to_yaml(op: ops.UnboundTable, context: TranslationContext) -> dict:
    schema_id = context.schema_registry.register_schema(op.schema)
    namespace_dict = freeze(
        {
            "catalog": op.namespace.catalog,
            "database": op.namespace.database,
        }
    )
    return freeze(
        {
            "op": "UnboundTable",
            "name": op.name,
            "schema_ref": schema_id,
            "namespace": namespace_dict,
        }
    )


@register_from_yaml_handler("UnboundTable")
def _unbound_table_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    table_name = yaml_dict["name"]

    schema_ref = yaml_dict["schema_ref"]
    try:
        schema_def = context.definitions["schemas"][schema_ref]
    except KeyError:
        raise ValueError(f"Schema {schema_ref} not found in definitions")
    namespace_dict = yaml_dict.get("namespace", {})
    catalog = namespace_dict.get("catalog")
    database = namespace_dict.get("database")
    schema = {
        name: _type_from_yaml(dtype_yaml) for name, dtype_yaml in schema_def.items()
    }
    # TODO: use UnboundTable node to construct instead of builder API
    return ibis.table(schema, name=table_name, catalog=catalog, database=database)


@translate_to_yaml.register(ops.DatabaseTable)
def _database_table_to_yaml(op: ops.DatabaseTable, context: TranslationContext) -> dict:
    profile_name = op.source._profile.hash_name
    schema_id = context.schema_registry.register_schema(op.schema)
    namespace_dict = freeze(
        {
            "catalog": op.namespace.catalog,
            "database": op.namespace.database,
        }
    )

    node_dict = freeze(
        {
            "op": "DatabaseTable",
            "table": op.name,
            "schema_ref": schema_id,
            "profile": profile_name,
            "namespace": namespace_dict,
        }
    )

    if should_register_node(node_dict) and hasattr(
        context.schema_registry, "register_node"
    ):
        node_hash = context.schema_registry.register_node(node_dict)
        return freeze({"node_ref": node_hash})

    return node_dict


@register_from_yaml_handler("DatabaseTable")
def database_table_from_yaml(yaml_dict: dict, context: TranslationContext) -> ibis.Expr:
    profile_name = yaml_dict.get("profile")
    table_name = yaml_dict.get("table")
    namespace_dict = yaml_dict.get("namespace", {})
    catalog = namespace_dict.get("catalog")
    database = namespace_dict.get("database")
    # we should validate that schema is the same
    schema_ref = yaml_dict.get("schema_ref")
    schema_def = context.definitions["schemas"][schema_ref]
    fields = []
    for name, dtype_yaml in schema_def.items():
        dtype = _type_from_yaml(dtype_yaml)
        fields.append((name, dtype))
    schema = ibis.Schema.from_tuples(fields)

    try:
        con = context.profiles[profile_name]
    except KeyError:
        raise ValueError(f"Profile {profile_name!r} not found in context.profiles")
    return ops.DatabaseTable(
        schema=schema,
        source=con,
        name=table_name,
        namespace=Namespace(catalog=catalog, database=database),
    ).to_expr()


@translate_to_yaml.register(CachedNode)
def _cached_node_to_yaml(op: CachedNode, context: any) -> dict:
    schema_id = context.schema_registry.register_schema(op.schema)
    # source should be called profile_name

    return freeze(
        {
            "op": "CachedNode",
            "schema_ref": schema_id,
            "parent": translate_to_yaml(op.parent, context),
            "source": op.source._profile.hash_name,
            "storage": translate_storage(op.storage, context),
        }
    )


@register_from_yaml_handler("CachedNode")
def _cached_node_from_yaml(yaml_dict: dict, context: any) -> ibis.Expr:
    schema_ref = yaml_dict["schema_ref"]
    try:
        schema_def = context.definitions["schemas"][schema_ref]
    except KeyError:
        raise ValueError(f"Schema {schema_ref} not found in definitions")

    schema = {
        name: _type_from_yaml(dtype_yaml) for name, dtype_yaml in schema_def.items()
    }

    parent_expr = translate_from_yaml(yaml_dict["parent"], context)
    profile_name = yaml_dict.get("source")
    try:
        source = context.profiles[profile_name]
    except KeyError:
        raise ValueError(f"Profile {profile_name!r} not found in context.profiles")
    storage = load_storage_from_yaml(yaml_dict["storage"], context)

    op = CachedNode(
        schema=schema,
        parent=parent_expr,
        source=source,
        storage=storage,
    )
    return op.to_expr()


@translate_to_yaml.register(RemoteTable)
def _remotetable_to_yaml(op: RemoteTable, context: any) -> dict:
    profile_name = op.source._profile.hash_name
    remote_expr_yaml = translate_to_yaml(op.remote_expr, context)
    schema_id = context.schema_registry.register_schema(op.schema)
    # TODO: change profile to profile_name
    return freeze(
        {
            "op": "RemoteTable",
            "table": op.name,
            "schema_ref": schema_id,
            "profile": profile_name,
            "remote_expr": remote_expr_yaml,
        }
    )


@register_from_yaml_handler("RemoteTable")
def _remotetable_from_yaml(yaml_dict: dict, context: any) -> ibis.Expr:
    profile_name = yaml_dict.get("profile")
    table_name = yaml_dict.get("table")
    remote_expr_yaml = yaml_dict.get("remote_expr")
    if profile_name is None:
        raise ValueError(
            "Missing keys in RemoteTable YAML; ensure 'profile_name' are present."
        )
    try:
        con = context.profiles[profile_name]
    except KeyError:
        raise ValueError(f"Profile {profile_name!r} not found in context.profiles")

    remote_expr = translate_from_yaml(remote_expr_yaml, context)

    remote_table_expr = into_backend(remote_expr, con, table_name)
    return remote_table_expr


@translate_to_yaml.register(Read)
def _read_to_yaml(op: Read, context: TranslationContext) -> dict:
    schema_id = context.schema_registry.register_schema(op.schema)
    profile_hash_name = (
        op.source._profile.hash_name if hasattr(op.source, "_profile") else None
    )

    return freeze(
        {
            "op": "Read",
            "method_name": op.method_name,
            "name": op.name,
            "schema_ref": schema_id,
            "profile": profile_hash_name,
            "read_kwargs": freeze(op.read_kwargs if op.read_kwargs else {}),
        }
    )


@register_from_yaml_handler("Read")
def _read_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    schema_ref = yaml_dict["schema_ref"]
    schema_def = context.definitions["schemas"][schema_ref]

    schema = {
        name: _type_from_yaml(dtype_yaml) for name, dtype_yaml in schema_def.items()
    }

    profile_hash_name = yaml_dict.get("profile")

    source = context.profiles[profile_hash_name]

    read_op = Read(
        method_name=yaml_dict["method_name"],
        name=yaml_dict["name"],
        schema=schema,
        source=source,
        read_kwargs=yaml_dict.get("read_kwargs", {}),
    )

    return read_op.to_expr()


@translate_to_yaml.register(ops.Literal)
def _literal_to_yaml(op: ops.Literal, context: TranslationContext) -> dict:
    value = _translate_literal_value(op.value, op.dtype)
    return freeze({"op": "Literal", "value": value, "type": _translate_type(op.dtype)})


@register_from_yaml_handler("Literal")
def _literal_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    value = yaml_dict["value"]
    dtype = _type_from_yaml(yaml_dict["type"])
    return ibis.literal(value, type=dtype)


@translate_to_yaml.register(ops.ValueOp)
def _value_op_to_yaml(op: ops.ValueOp, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": type(op).__name__,
            "type": _translate_type(op.dtype),
            "args": [
                translate_to_yaml(arg, context)
                for arg in op.args
                if isinstance(arg, (ops.Value, ops.Node))
            ],
        }
    )


@register_from_yaml_handler("ValueOp")
def _value_op_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    args = [translate_from_yaml(arg, context) for arg in yaml_dict["args"]]
    method_name = yaml_dict["op"].lower()
    method = getattr(args[0], method_name)
    return method(*args[1:])


@translate_to_yaml.register(ops.Lag)
def _lag_to_yaml(op: ops.Lag, context: TranslationContext) -> dict:
    result = {
        "op": "Lag",
        "arg": translate_to_yaml(op.arg, context),
        "type": _translate_type(op.dtype),
    }

    if op.offset is not None:
        result["offset"] = translate_to_yaml(op.offset, context)

    if op.default is not None:
        result["default"] = translate_to_yaml(op.default, context)

    node_dict = freeze(result)

    return node_dict


@register_from_yaml_handler("Lag")
def _lag_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    arg = translate_from_yaml(yaml_dict["arg"], context)

    offset = None
    if "offset" in yaml_dict:
        offset = translate_from_yaml(yaml_dict["offset"], context)

    default = None
    if "default" in yaml_dict:
        default = translate_from_yaml(yaml_dict["default"], context)

    return arg.lag(offset, default)


@translate_to_yaml.register(ops.StringUnary)
def _string_unary_to_yaml(op: ops.StringUnary, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": type(op).__name__,
            "args": [translate_to_yaml(op.arg, context)],
            "type": _translate_type(op.dtype),
        }
    )


@register_from_yaml_handler("StringUnary")
def _string_unary_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    arg = translate_from_yaml(yaml_dict["args"][0], context)
    method_name = yaml_dict["op"].lower()
    return getattr(arg, method_name)()


@translate_to_yaml.register(ops.Substring)
def _substring_to_yaml(op: ops.Substring, context: TranslationContext) -> dict:
    args = [
        translate_to_yaml(op.arg, context),
        translate_to_yaml(op.start, context),
    ]
    if op.length is not None:
        args.append(translate_to_yaml(op.length, context))
    return freeze({"op": "Substring", "args": args, "type": _translate_type(op.dtype)})


@register_from_yaml_handler("Substring")
def _substring_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    args = [translate_from_yaml(arg, context) for arg in yaml_dict["args"]]
    return args[0].substr(args[1], args[2] if len(args) > 2 else None)


@translate_to_yaml.register(ops.StringLength)
def _string_length_to_yaml(op: ops.StringLength, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": "StringLength",
            "args": [translate_to_yaml(op.arg, context)],
            "type": _translate_type(op.dtype),
        }
    )


@register_from_yaml_handler("StringLength")
def _string_length_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    arg = translate_from_yaml(yaml_dict["args"][0], context)
    return arg.length()


@translate_to_yaml.register(ops.StringConcat)
def _string_concat_to_yaml(op: ops.StringConcat, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": "StringConcat",
            "args": [translate_to_yaml(arg, context) for arg in op.arg],
            "type": _translate_type(op.dtype),
        }
    )


@register_from_yaml_handler("StringConcat")
def _string_concat_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    args = [translate_from_yaml(arg, context) for arg in yaml_dict["args"]]
    return functools.reduce(lambda x, y: x.concat(y), args)


@translate_to_yaml.register(ops.BinaryOp)
def _binary_op_to_yaml(op: ops.BinaryOp, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": type(op).__name__,
            "args": [
                translate_to_yaml(op.left, context),
                translate_to_yaml(op.right, context),
            ],
            "type": _translate_type(op.dtype),
        }
    )


@register_from_yaml_handler("BinaryOp")
def _binary_op_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    args = [translate_from_yaml(arg, context) for arg in yaml_dict["args"]]
    op_name = yaml_dict["op"].lower()
    return getattr(args[0], op_name)(args[1])


@translate_to_yaml.register(ops.Filter)
def _filter_to_yaml(op: ops.Filter, context: TranslationContext) -> dict:
    node_dict = freeze(
        {
            "op": "Filter",
            "parent": translate_to_yaml(op.parent, context),
            "predicates": [translate_to_yaml(pred, context) for pred in op.predicates],
        }
    )

    if should_register_node(node_dict) and hasattr(
        context.schema_registry, "register_node"
    ):
        node_hash = context.schema_registry.register_node(node_dict)
        return freeze({"node_ref": node_hash})

    return node_dict


@register_from_yaml_handler("Filter")
def _filter_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    parent = translate_from_yaml(yaml_dict["parent"], context)
    predicates = [
        translate_from_yaml(pred, context) for pred in yaml_dict["predicates"]
    ]
    filter_op = ops.Filter(parent, predicates)
    return filter_op.to_expr()


@translate_to_yaml.register(ops.Project)
def _project_to_yaml(op: ops.Project, context: TranslationContext) -> dict:
    node_dict = {
        "op": "Project",
        "parent": translate_to_yaml(op.parent, context),
        "values": {
            name: translate_to_yaml(val, context) for name, val in op.values.items()
        },
    }

    if should_register_node(node_dict):
        node_hash = context.schema_registry.register_node(freeze(node_dict))
        return freeze({"node_ref": node_hash})

    return freeze(node_dict)


@register_from_yaml_handler("Project")
def _project_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    parent = translate_from_yaml(yaml_dict["parent"], context)

    values_dict = yaml_dict.get("values", {})

    values = {
        name: translate_from_yaml(val, context) for name, val in values_dict.items()
    }

    projected = parent.projection(values)
    return projected


@translate_to_yaml.register(ops.Aggregate)
def _aggregate_to_yaml(op: ops.Aggregate, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": "Aggregate",
            "parent": translate_to_yaml(op.parent, context),
            "by": [translate_to_yaml(group, context) for group in op.groups.values()],
            "metrics": {
                name: translate_to_yaml(metric, context)
                for name, metric in op.metrics.items()
            },
        }
    )


@register_from_yaml_handler("Aggregate")
def _aggregate_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    parent = translate_from_yaml(yaml_dict["parent"], context)
    groups = tuple(
        translate_from_yaml(group, context) for group in yaml_dict.get("by", [])
    )

    metrics = {
        name: translate_from_yaml(metric, context)
        for name, metric in yaml_dict.get("metrics", {}).items()
    }

    result = (
        parent.group_by(list(groups)).aggregate(metrics)
        if groups
        else parent.aggregate(metrics)
    )
    return result


@translate_to_yaml.register(ops.JoinChain)
def _join_to_yaml(op: ops.JoinChain, context: TranslationContext) -> dict:
    node_dict = {
        "op": "JoinChain",
        "first": translate_to_yaml(op.first, context),
        "rest": [
            {
                "how": link.how,
                "table": translate_to_yaml(link.table, context),
                "predicates": [
                    translate_to_yaml(pred, context) for pred in link.predicates
                ],
            }
            for link in op.rest
        ],
    }

    node_dict["values"] = {
        name: translate_to_yaml(val, context) for name, val in op.values.items()
    }

    if should_register_node(node_dict):
        node_hash = context.schema_registry.register_node(freeze(node_dict))
        return freeze({"node_ref": node_hash})
    return freeze(node_dict)


@register_from_yaml_handler("JoinChain")
def _join_chain_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    first = translate_from_yaml(yaml_dict["first"], context)
    result = first

    for join in yaml_dict["rest"]:
        table = translate_from_yaml(join["table"], context)
        predicates = [translate_from_yaml(pred, context) for pred in join["predicates"]]
        result = result.join(table, predicates, how=join["how"])

    values = {
        name: translate_from_yaml(val, context)
        for name, val in yaml_dict["values"].items()
    }
    result = result.select(values)
    return result


@translate_to_yaml.register(ops.Sort)
def _sort_to_yaml(op: ops.Sort, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": "Sort",
            "parent": translate_to_yaml(op.parent, context),
            "keys": [translate_to_yaml(key, context) for key in op.keys],
        }
    )


@register_from_yaml_handler("Sort")
def _sort_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    parent = translate_from_yaml(yaml_dict["parent"], context)
    keys = tuple(translate_from_yaml(key, context) for key in yaml_dict["keys"])
    sort_op = ops.Sort(parent, keys=keys)
    return sort_op.to_expr()


@translate_to_yaml.register(ops.SortKey)
def _sort_key_to_yaml(op: ops.SortKey, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": "SortKey",
            "arg": translate_to_yaml(op.expr, context),
            "ascending": op.ascending,
            "nulls_first": op.nulls_first,
            "type": _translate_type(op.dtype),
        }
    )


@register_from_yaml_handler("SortKey")
def _sort_key_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    expr = translate_from_yaml(yaml_dict["arg"], context)
    ascending = yaml_dict.get("ascending", True)
    nulls_first = yaml_dict.get("nulls_first", False)
    return ops.SortKey(expr, ascending=ascending, nulls_first=nulls_first).to_expr()


@translate_to_yaml.register(ops.Limit)
def _limit_to_yaml(op: ops.Limit, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": "Limit",
            "parent": translate_to_yaml(op.parent, context),
            "n": op.n,
            "offset": op.offset,
        }
    )


@register_from_yaml_handler("Limit")
def _limit_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    parent = translate_from_yaml(yaml_dict["parent"], context)
    return parent.limit(yaml_dict["n"], offset=yaml_dict["offset"])


@translate_to_yaml.register(ops.ScalarSubquery)
def _scalar_subquery_to_yaml(
    op: ops.ScalarSubquery, context: TranslationContext
) -> dict:
    return freeze(
        {
            "op": "ScalarSubquery",
            "args": [translate_to_yaml(arg, context) for arg in op.args],
            "type": _translate_type(op.dtype),
        }
    )


@register_from_yaml_handler("ScalarSubquery")
def _scalar_subquery_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    subquery = translate_from_yaml(yaml_dict["args"][0], context)
    return ops.ScalarSubquery(subquery).to_expr()


@translate_to_yaml.register(ops.ExistsSubquery)
def _exists_subquery_to_yaml(
    op: ops.ExistsSubquery, context: TranslationContext
) -> dict:
    return freeze(
        {
            "op": "ExistsSubquery",
            "rel": translate_to_yaml(op.rel, context),
            "type": _translate_type(op.dtype),
        }
    )


@register_from_yaml_handler("ExistsSubquery")
def _exists_subquery_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    rel = translate_from_yaml(yaml_dict["rel"], context)
    return ops.ExistsSubquery(rel).to_expr()


@translate_to_yaml.register(ops.InSubquery)
def _in_subquery_to_yaml(op: ops.InSubquery, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": "InSubquery",
            "needle": translate_to_yaml(op.needle, context),
            "haystack": translate_to_yaml(op.rel, context),
            "type": _translate_type(op.dtype),
        }
    )


@register_from_yaml_handler("InSubquery")
def _in_subquery_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    needle = translate_from_yaml(yaml_dict["needle"], context)
    haystack = translate_from_yaml(yaml_dict["haystack"], context)
    return ops.InSubquery(haystack, needle).to_expr()


@translate_to_yaml.register(ops.Field)
def _field_to_yaml(op: ops.Field, context: TranslationContext) -> dict:
    result = {
        "op": "Field",
        "name": op.name,
        "relation": translate_to_yaml(op.rel, context),
        "type": _translate_type(op.dtype),
    }

    if op.args and len(op.args) >= 2 and isinstance(op.args[1], str):
        underlying_name = op.args[1]
        if underlying_name != op.name:
            result["original_name"] = underlying_name

    node_dict = freeze(result)

    if hasattr(context.schema_registry, "register_node"):
        node_hash = context.schema_registry.register_node(node_dict)
        return freeze({"node_ref": node_hash})

    return node_dict


@register_from_yaml_handler("Field")
def field_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    relation = translate_from_yaml(yaml_dict["relation"], context)
    target_name = yaml_dict["name"]
    source_name = yaml_dict.get("original_name", target_name)
    field = relation[source_name]
    if target_name != source_name:
        field = field.name(target_name)

    return freeze(field)


@translate_to_yaml.register(ops.InValues)
def _in_values_to_yaml(op: ops.InValues, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": "InValues",
            "args": [
                translate_to_yaml(op.value, context),
                *[translate_to_yaml(opt, context) for opt in op.options],
            ],
            "type": _translate_type(op.dtype),
        }
    )


@register_from_yaml_handler("InValues")
def _in_values_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    value = translate_from_yaml(yaml_dict["args"][0], context)
    options = tuple(translate_from_yaml(opt, context) for opt in yaml_dict["args"][1:])
    return ops.InValues(value, options).to_expr()


@translate_to_yaml.register(ops.SimpleCase)
def _simple_case_to_yaml(op: ops.SimpleCase, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": "SimpleCase",
            "base": translate_to_yaml(op.base, context),
            "cases": [translate_to_yaml(case, context) for case in op.cases],
            "results": [translate_to_yaml(result, context) for result in op.results],
            "default": translate_to_yaml(op.default, context),
            "type": _translate_type(op.dtype),
        }
    )


@register_from_yaml_handler("SimpleCase")
def _simple_case_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    base = translate_from_yaml(yaml_dict["base"], context)
    cases = tuple(translate_from_yaml(case, context) for case in yaml_dict["cases"])
    results = tuple(
        translate_from_yaml(result, context) for result in yaml_dict["results"]
    )
    default = translate_from_yaml(yaml_dict["default"], context)
    return ops.SimpleCase(base, cases, results, default).to_expr()


@translate_to_yaml.register(ops.IfElse)
def _if_else_to_yaml(op: ops.IfElse, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": "IfElse",
            "bool_expr": translate_to_yaml(op.bool_expr, context),
            "true_expr": translate_to_yaml(op.true_expr, context),
            "false_null_expr": translate_to_yaml(op.false_null_expr, context),
            "type": _translate_type(op.dtype),
        }
    )


@register_from_yaml_handler("IfElse")
def _if_else_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    bool_expr = translate_from_yaml(yaml_dict["bool_expr"], context)
    true_expr = translate_from_yaml(yaml_dict["true_expr"], context)
    false_null_expr = translate_from_yaml(yaml_dict["false_null_expr"], context)
    return ops.IfElse(bool_expr, true_expr, false_null_expr).to_expr()


@translate_to_yaml.register(ops.Coalesce)
def _coalesce_to_yaml(op: ops.Coalesce, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": "Coalesce",
            "args": [translate_to_yaml(arg, context) for arg in op.arg],
            "type": _translate_type(op.dtype),
        }
    )


@register_from_yaml_handler("Coalesce")
def _coalesce_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    args = [translate_from_yaml(arg, context) for arg in yaml_dict["args"]]
    return ibis.coalesce(*args)


@translate_to_yaml.register(ops.CountDistinct)
def _count_distinct_to_yaml(op: ops.CountDistinct, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": "CountDistinct",
            "args": [translate_to_yaml(op.arg, context)],
            "type": _translate_type(op.dtype),
        }
    )


@translate_to_yaml.register(ops.RankBase)
def _rank_base_to_yaml(op: ops.RankBase, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": type(op).__name__,
        }
    )


@register_from_yaml_handler("MinRank")
def _min_rank_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    return ibis.rank()


@register_from_yaml_handler("RowNumber")
def _row_number_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    return ibis.row_number()


@register_from_yaml_handler("CountDistinct")
def _count_distinct_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    arg = translate_from_yaml(yaml_dict["args"][0], context)
    return arg.nunique()


@translate_to_yaml.register(ops.SelfReference)
def _self_reference_to_yaml(op: ops.SelfReference, context: TranslationContext) -> dict:
    result = {"op": "SelfReference", "identifier": op.identifier}
    if op.args:
        result["args"] = [translate_to_yaml(op.args[0], context)]
    return freeze(result)


@register_from_yaml_handler("SelfReference")
def _self_reference_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    if "args" in yaml_dict and yaml_dict["args"]:
        underlying = translate_from_yaml(yaml_dict["args"][0], context)
    else:
        if underlying is None:
            raise NotImplementedError("No relation available for SelfReference")

    identifier = yaml_dict.get("identifier", 0)
    ref = ops.SelfReference(underlying, identifier=identifier)

    return ref.to_expr()


@translate_to_yaml.register(ops.DropColumns)
def _drop_columns_to_yaml(op: ops.DropColumns, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": "DropColumns",
            "parent": translate_to_yaml(op.parent, context),
            "columns_to_drop": list(op.columns_to_drop),
        }
    )


@register_from_yaml_handler("DropColumns")
def _drop_columns_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    parent = translate_from_yaml(yaml_dict["parent"], context)
    columns = frozenset(yaml_dict["columns_to_drop"])
    op = ops.DropColumns(parent, columns)
    return op.to_expr()


@translate_to_yaml.register(ops.SearchedCase)
def _searched_case_to_yaml(op: ops.SearchedCase, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": "SearchedCase",
            "cases": [translate_to_yaml(case, context) for case in op.cases],
            "results": [translate_to_yaml(result, context) for result in op.results],
            "default": translate_to_yaml(op.default, context),
            "dtype": _translate_type(op.dtype),
        }
    )


@register_from_yaml_handler("SearchedCase")
def _searched_case_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    cases = [translate_from_yaml(case, context) for case in yaml_dict["cases"]]
    results = [translate_from_yaml(result, context) for result in yaml_dict["results"]]
    default = translate_from_yaml(yaml_dict["default"], context)
    op = ops.SearchedCase(cases, results, default)
    return op.to_expr()


@register_from_yaml_handler("View")
def _view_from_yaml(yaml_dict: dict, context: any) -> ir.Expr:
    underlying = translate_from_yaml(yaml_dict["args"][0], context)
    alias = yaml_dict.get("name")
    if alias:
        return underlying.alias(alias)
    return underlying


@register_from_yaml_handler("Mean")
def _mean_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    args = [translate_from_yaml(arg, context) for arg in yaml_dict["args"]]
    return args[0].mean()


@register_from_yaml_handler("Add", "Subtract", "Multiply", "Divide")
def _binary_arithmetic_from_yaml(
    yaml_dict: dict, context: TranslationContext
) -> ir.Expr:
    left = translate_from_yaml(yaml_dict["args"][0], context)
    right = translate_from_yaml(yaml_dict["args"][1], context)
    op_map = {
        "Add": operator.add,
        "Subtract": operator.sub,
        "Multiply": operator.mul,
        "Divide": operator.truediv,
    }
    op_func = op_map.get(yaml_dict["op"])
    if op_func is None:
        raise ValueError(f"Unsupported arithmetic operation: {yaml_dict['op']}")
    return op_func(left, right)


@register_from_yaml_handler("Repeat")
def _repeat_from_yaml(
    yaml_dict: dict, context: TranslationContext
) -> ibis.expr.types.Expr:
    arg = translate_from_yaml(yaml_dict["args"][0], context)
    times = translate_from_yaml(yaml_dict["args"][1], context)
    return ops.Repeat(arg, times).to_expr()


@register_from_yaml_handler("Sum")
def _sum_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    args = [translate_from_yaml(arg, context) for arg in yaml_dict["args"]]
    return args[0].sum()


@register_from_yaml_handler("Min")
def _min_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    args = [translate_from_yaml(arg, context) for arg in yaml_dict["args"]]
    return args[0].min()


@register_from_yaml_handler("Max")
def _max_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    args = [translate_from_yaml(arg, context) for arg in yaml_dict["args"]]
    return args[0].max()


@register_from_yaml_handler("Abs")
def _abs_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    arg = translate_from_yaml(yaml_dict["args"][0], context)
    return arg.abs()


@register_from_yaml_handler("Count")
def _count_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    arg = translate_from_yaml(yaml_dict["args"][0], context)
    return arg.count()


@register_from_yaml_handler("JoinReference")
def _join_reference_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    table_yaml = yaml_dict["args"][0]
    return translate_from_yaml(table_yaml, context)


@register_from_yaml_handler(
    "Equals", "NotEquals", "GreaterThan", "GreaterEqual", "LessThan", "LessEqual"
)
def _binary_compare_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    left = translate_from_yaml(yaml_dict["args"][0], context)
    right = translate_from_yaml(yaml_dict["args"][1], context)

    op_map = {
        "Equals": operator.eq,
        "NotEquals": operator.ne,
        "GreaterThan": operator.gt,
        "GreaterEqual": operator.ge,
        "LessThan": operator.lt,
        "LessEqual": operator.le,
    }

    op_func = op_map.get(yaml_dict["op"])
    if op_func is None:
        raise ValueError(f"Unsupported comparison operation: {yaml_dict['op']}")
    return op_func(left, right)


@register_from_yaml_handler("Between")
def _between_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    args = [translate_from_yaml(arg, context) for arg in yaml_dict["args"]]
    return args[0].between(args[1], args[2])


@register_from_yaml_handler("Greater", "Less")
def _boolean_ops_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    args = [translate_from_yaml(arg, context) for arg in yaml_dict["args"]]
    op_name = yaml_dict["op"]
    op_map = {
        "Greater": operator.gt,
        "Less": operator.lt,
    }
    return op_map[op_name](*args)


@register_from_yaml_handler("And")
def _boolean_and_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    args = [translate_from_yaml(arg, context) for arg in yaml_dict.get("args", [])]
    if not args:
        raise ValueError("And operator requires at least one argument")
    return functools.reduce(operator.and_, args)


@register_from_yaml_handler("Or")
def _boolean_or_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    args = [translate_from_yaml(arg, context) for arg in yaml_dict.get("args", [])]
    if not args:
        raise ValueError("Or operator requires at least one argument")
    return functools.reduce(operator.or_, args)


@register_from_yaml_handler("Not")
def _not_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    arg = translate_from_yaml(yaml_dict["args"][0], context)
    return ~arg


@register_from_yaml_handler("IsNull")
def _is_null_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    arg = translate_from_yaml(yaml_dict["args"][0], context)
    return arg.isnull()


@register_from_yaml_handler(
    "ExtractYear",
    "ExtractMonth",
    "ExtractDay",
    "ExtractHour",
    "ExtractMinute",
    "ExtractSecond",
)
def _extract_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    arg = translate_from_yaml(yaml_dict["args"][0], context)
    op_map = {
        "ExtractYear": operator.methodcaller("year"),
        "ExtractMonth": operator.methodcaller("month"),
        "ExtractDay": operator.methodcaller("day"),
        "ExtractHour": operator.methodcaller("hour"),
        "ExtractMinute": operator.methodcaller("minute"),
        "ExtractSecond": operator.methodcaller("second"),
    }
    return op_map[yaml_dict["op"]](arg)


@register_from_yaml_handler("TimestampDiff")
def _timestamp_diff_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    left = translate_from_yaml(yaml_dict["args"][0], context)
    right = translate_from_yaml(yaml_dict["args"][1], context)
    return left - right


@register_from_yaml_handler("TimestampAdd", "TimestampSub")
def _timestamp_arithmetic_from_yaml(
    yaml_dict: dict, context: TranslationContext
) -> ir.Expr:
    timestamp = translate_from_yaml(yaml_dict["args"][0], context)
    interval = translate_from_yaml(yaml_dict["args"][1], context)
    if yaml_dict["op"] == "TimestampAdd":
        return timestamp + interval
    else:
        return timestamp - interval


@register_from_yaml_handler("Cast")
def _cast_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    arg = translate_from_yaml(yaml_dict["args"][0], context)
    target_dtype = _type_from_yaml(yaml_dict["type"])
    return arg.cast(target_dtype)


@register_from_yaml_handler("CountStar")
def _count_star_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    arg = translate_from_yaml(yaml_dict["args"][0], context)

    return ops.CountStar(arg).to_expr()


@register_from_yaml_handler("StringSQLLike")
def _string_sql_like_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    args = yaml_dict.get("args", [])
    if not args:
        raise ValueError("Missing arguments for StringSQLLike operator")

    col = translate_from_yaml(args[0], context)

    if len(args) >= 2:
        pattern_expr = translate_from_yaml(args[1], context)
    else:
        pattern_value = args[0].get("value")
        if pattern_value is None:
            pattern_value = yaml_dict.get("value")
        if pattern_value is None:
            raise ValueError("Missing pattern for StringSQLLike operator")
        pattern_expr = ibis.literal(pattern_value, type=dt.String())

    escape = yaml_dict.get("escape")

    return ops.StringSQLLike(col, pattern_expr, escape=escape).to_expr()


def _type_from_yaml(yaml_dict: dict) -> dt.DataType:
    type_name = yaml_dict["name"]
    base_type = REVERSE_TYPE_REGISTRY.get(type_name)
    if base_type is None:
        raise ValueError(f"Unknown type: {type_name}")
    if callable(base_type) and not isinstance(base_type, dt.DataType):
        base_type = base_type(yaml_dict)
    elif (
        "nullable" in yaml_dict
        and isinstance(base_type, dt.DataType)
        and not isinstance(base_type, (tm.IntervalUnit, dt.Timestamp))
    ):
        base_type = base_type.copy(nullable=yaml_dict["nullable"])
    return base_type


REVERSE_TYPE_REGISTRY = {
    "Int8": dt.Int8(),
    "Int16": dt.Int16(),
    "Int32": dt.Int32(),
    "Int64": dt.Int64(),
    "UInt8": dt.UInt8(),
    "UInt16": dt.UInt16(),
    "UInt32": dt.UInt32(),
    "UInt64": dt.UInt64(),
    "Float32": dt.Float32(),
    "Float64": dt.Float64(),
    "String": dt.String(),
    "Boolean": dt.Boolean(),
    "Date": dt.Date(),
    "Time": dt.Time(),
    "Binary": dt.Binary(),
    "JSON": dt.JSON(),
    "Null": dt.null,
    "Timestamp": lambda yaml_dict: dt.Timestamp(
        nullable=yaml_dict.get("nullable", True)
    ),
    "Decimal": lambda yaml_dict: dt.Decimal(
        precision=yaml_dict.get("precision"),
        scale=yaml_dict.get("scale"),
        nullable=yaml_dict.get("nullable", True),
    ),
    "IntervalUnit": lambda yaml_dict: tm.IntervalUnit(
        yaml_dict["value"] if isinstance(yaml_dict, dict) else yaml_dict
    ),
    "Interval": lambda yaml_dict: dt.Interval(
        unit=_type_from_yaml(yaml_dict["unit"]),
        nullable=yaml_dict.get("nullable", True),
    ),
    "DateUnit": lambda yaml_dict: tm.DateUnit(yaml_dict["value"]),
    "TimeUnit": lambda yaml_dict: tm.TimeUnit(yaml_dict["value"]),
    "TimestampUnit": lambda yaml_dict: tm.TimestampUnit(yaml_dict["value"]),
    "Array": lambda yaml_dict: dt.Array(
        _type_from_yaml(yaml_dict["value_type"]),
        nullable=yaml_dict.get("nullable", True),
    ),
    "Map": lambda yaml_dict: dt.Map(
        _type_from_yaml(yaml_dict["key_type"]),
        _type_from_yaml(yaml_dict["value_type"]),
        nullable=yaml_dict.get("nullable", True),
    ),
}
