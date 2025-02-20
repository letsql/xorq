from typing import Any, Dict, TypedDict

import letsql.vendor.ibis as ibis
import letsql.vendor.ibis.expr.operations as ops
import letsql.vendor.ibis.expr.types as ir
from letsql.expr.relations import Read, RemoteTable
from letsql.ibis_yaml.utils import find_all_backends, find_relations


class QueryInfo(TypedDict):
    engine: str
    profile_name: str
    sql: str


class SQLPlans(TypedDict):
    queries: Dict[str, QueryInfo]


def get_read_options(read_instance):
    read_kwargs_list = [{k: v} for k, v in read_instance.read_kwargs]

    return {
        "method_name": read_instance.method_name,
        "name": read_instance.name,
        "read_kwargs": read_kwargs_list,
    }


def find_remote_tables(op) -> Dict[str, Dict[str, Any]]:
    remote_tables = {}
    seen = set()

    def traverse(node):
        if node is None or id(node) in seen:
            return

        seen.add(id(node))

        if isinstance(node, ops.Node) and isinstance(node, RemoteTable):
            remote_expr = node.remote_expr
            original_backend = find_all_backends(remote_expr)[
                0
            ]  # this was _find_backend before

            engine_name = original_backend.name
            profile_name = original_backend._profile.hash_name
            remote_tables[node.name] = {
                "engine": engine_name,
                "profile_name": profile_name,
                "relations": find_relations(remote_expr),
                "sql": ibis.to_sql(remote_expr),
                "options": {},
            }
        if isinstance(node, Read):
            backend = node.source
            if backend is not None:
                engine_name = backend.name
                profile_name = backend._profile.hash_name
                remote_tables[node.make_unbound_dt().name] = {
                    "engine": engine_name,
                    "profile_name": profile_name,
                    "relations": [node.make_unbound_dt().name],
                    "sql": ibis.to_sql(node.make_unbound_dt().to_expr()),
                    "options": get_read_options(node),
                }

        if isinstance(node, ops.Node):
            for arg in node.args:
                if isinstance(arg, ops.Node):
                    traverse(arg)
                elif isinstance(arg, (list, tuple)):
                    for item in arg:
                        if isinstance(item, ops.Node):
                            traverse(item)
                elif isinstance(arg, dict):
                    for v in arg.values():
                        if isinstance(v, ops.Node):
                            traverse(v)

    traverse(op)
    return remote_tables


# TODO: rename to sqls
def generate_sql_plans(expr: ir.Expr) -> SQLPlans:
    remote_tables = find_remote_tables(expr.op())

    main_sql = ibis.to_sql(expr)
    backend = expr._find_backend()

    engine_name = backend.name
    profile_name = backend._profile.hash_name

    plans: SQLPlans = {
        "queries": {
            "main": {
                "engine": engine_name,
                "profile_name": profile_name,
                "relations": list(find_relations(expr)),
                "sql": main_sql.strip(),
                "options": {},
            }
        }
    }

    for table_name, info in remote_tables.items():
        plans["queries"][table_name] = {
            "engine": info["engine"],
            "relations": info["relations"],
            "profile_name": info["profile_name"],
            "sql": info["sql"].strip(),
            "options": info["options"],
        }

    return plans
