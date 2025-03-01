import operator

import toolz

import xorq as xo
import xorq.expr.relations as rel
import xorq.vendor.ibis.expr.operations as ops
from xorq.caching import SourceStorage
from xorq.common.utils.graph_utils import (
    find_all_sources,
    walk_nodes,
)


node_types = (
    ops.DatabaseTable,
    ops.SQLQueryResult,
    rel.CachedNode,
    rel.Read,
    rel.RemoteTable,
    # ExprScalarUDF has an expr we need to get to
    # FlightOperator has a dynamically generated connection: it should be passed a Profile instead
)


def make_expr():
    cons = (con0, con1, con2, con3) = (
        xo.connect(),
        xo.connect(),
        xo.duckdb.connect(),
        xo.connect(),
    )

    read_node0 = xo.examples.awards_players.fetch(con0)
    remote_node0 = read_node0.into_backend(con1)
    cached_node0 = remote_node0.cache(SourceStorage(source=con1))
    read_node1 = xo.examples.batting.fetch(con2)
    remote_node1 = read_node1.into_backend(con1)
    remote_node2 = cached_node0.join(
        remote_node1, predicates=["playerID", "yearID", "lgID"]
    ).into_backend(con3)
    cached_node1 = remote_node2.cache()
    expr = cached_node2 = cached_node1[lambda t: t.G == 1].cache()
    nodes = {
        rel.CachedNode: (
            cached_node0.op(),
            cached_node1.op(),
            cached_node2.op(),
        ),
        rel.Read: (
            read_node0.op(),
            read_node1.op(),
        ),
        rel.RemoteTable: (
            remote_node0.op(),
            remote_node1.op(),
            remote_node2.op(),
        ),
    }
    return (cons, nodes, expr)


def test_walk_nodes():
    (_, nodes, expr) = make_expr()
    node_types = tuple(nodes)
    walked_nodes = walk_nodes(node_types, expr)
    expected = sorted(
        ((k, set(v)) for k, v in nodes.items()),
        key=toolz.compose(operator.attrgetter("__name__"), operator.itemgetter(0)),
    )
    actual = sorted(
        ((k, set(v)) for k, v in toolz.groupby(type, walked_nodes).items()),
        key=toolz.compose(operator.attrgetter("__name__"), operator.itemgetter(0)),
    )
    assert actual == expected


def test_find_all_sources():
    (created_sources, _, expr) = make_expr()
    found_sources = find_all_sources(expr)
    actual = set(con._profile for con in created_sources)
    expected = set(con._profile for con in found_sources)
    assert actual == expected
