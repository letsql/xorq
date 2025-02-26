import functools
from collections import defaultdict
from typing import Any, Callable

import pyarrow as pa

import xorq as xo
from xorq.common.utils.rbr_utils import (
    copy_rbr_batches,
    instrument_reader,
)
from xorq.vendor import ibis
from xorq.vendor.ibis import Expr, Schema
from xorq.vendor.ibis.common.collections import FrozenDict
from xorq.vendor.ibis.common.graph import Graph
from xorq.vendor.ibis.expr import operations as ops
from xorq.vendor.ibis.expr.operations import Node, Relation


def replace_cache_table(node, kwargs):
    if kwargs:
        node = node.__recreate__(kwargs)

    if isinstance(node, CachedNode):
        return node.parent.op().replace(replace_cache_table)
    elif isinstance(node, RemoteTable):
        return node.remote_expr.op().replace(replace_cache_table)
    else:
        return node


def legacy_replace_cache_table(node, _, **kwargs):
    return replace_cache_table(node, (kwargs or dict(zip(node.argnames, node.args))))


# https://stackoverflow.com/questions/6703594/is-the-result-of-itertools-tee-thread-safe-python
class SafeTee(object):
    """tee object wrapped to make it thread-safe"""

    def __init__(self, teeobj, lock):
        self.teeobj = teeobj
        self.lock = lock

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.teeobj)

    def __copy__(self):
        return SafeTee(self.teeobj.__copy__(), self.lock)

    @classmethod
    def tee(cls, iterable, n=2):
        """tuple of n independent thread-safe iterators"""
        from itertools import tee
        from threading import Lock

        lock = Lock()
        return tuple(cls(teeobj, lock) for teeobj in tee(iterable, n))


def recursive_update(obj, replacements):
    if isinstance(obj, Node):
        if obj in replacements:
            return replacements[obj]
        else:
            return obj.__recreate__(
                {
                    name: recursive_update(arg, replacements)
                    for name, arg in zip(obj.argnames, obj.args)
                }
            )
    elif isinstance(obj, (tuple, list)):
        return tuple(recursive_update(o, replacements) for o in obj)
    elif isinstance(obj, dict):
        return {
            recursive_update(k, replacements): recursive_update(v, replacements)
            for k, v in obj.items()
        }
    else:
        return obj


def replace_source_factory(source: Any):
    def replace_source(node, _, **kwargs):
        if "source" in kwargs:
            kwargs["source"] = source
        return node.__recreate__(kwargs)

    return replace_source


def make_native_op(node):
    # FIXME: how to reference let.Backend.name?
    if node.source.name != "let":
        raise ValueError
    sources = node.source._sources
    native_source = sources.get_backend(node)
    if native_source.name == "let":
        raise ValueError

    def replace_table(_node, _kwargs):
        return sources.get_table_or_op(
            _node, _node.__recreate__(_kwargs) if _kwargs else _node
        )

    return node.replace(replace_table).to_expr()


class CachedNode(ops.Relation):
    schema: Schema
    parent: Any
    source: Any
    storage: Any
    values = FrozenDict()


gen_name = functools.partial(ibis.util.gen_name, "remote-expr-placeholder")


class RemoteTable(ops.DatabaseTable):
    remote_expr: Expr = None

    @classmethod
    def from_expr(cls, con, expr, name=None):
        name = name or gen_name()
        return cls(
            name=name,
            schema=expr.schema(),
            source=con,
            remote_expr=expr,
        )


def into_backend(expr, con, name=None):
    return RemoteTable.from_expr(con=con, expr=expr, name=name).to_expr()


class FlightExchange(ops.DatabaseTable):
    input_expr: Expr = None
    unbound_expr: Expr = None
    make_server: Callable = None
    make_connection: Callable = None
    do_instrument_reader: bool = False

    @classmethod
    def validate_schema(cls, input_expr, unbound_expr):
        (dt, *rest) = unbound_expr.op().find(ops.UnboundTable)
        if rest or not isinstance(dt, ops.UnboundTable):
            raise ValueError
        if dt.schema != input_expr.schema():
            raise ValueError

    @classmethod
    def from_exprs(
        cls,
        input_expr,
        unbound_expr,
        make_server=None,
        make_connection=None,
        name=None,
        **kwargs,
    ):
        from xorq.flight import FlightServer

        def roundtrip_cloudpickle(obj):
            import cloudpickle

            return cloudpickle.loads(cloudpickle.dumps(obj))

        cls.validate_schema(input_expr, unbound_expr)
        return cls(
            name=name or gen_name(),
            schema=unbound_expr.schema(),
            source=input_expr._find_backend(),
            input_expr=input_expr,
            unbound_expr=roundtrip_cloudpickle(unbound_expr),
            make_server=make_server or FlightServer,
            make_connection=make_connection or xo.connect,
            **kwargs,
        )

    def to_rbr(self, do_instrument_reader=None):
        from xorq.flight.action import AddExchangeAction
        from xorq.flight.exchanger import (
            UnboundExprExchanger,
        )

        if do_instrument_reader is None:
            do_instrument_reader = self.do_instrument_reader

        def inner(flight_exchange):
            rbr_in = flight_exchange.input_expr.to_pyarrow_batches()
            if do_instrument_reader:
                rbr_in = instrument_reader(rbr_in, "input: ")
            with flight_exchange.make_server() as server:
                client = server.client
                unbound_expr_exchanger = UnboundExprExchanger(
                    flight_exchange.unbound_expr
                )
                client.do_action(
                    AddExchangeAction.name,
                    unbound_expr_exchanger,
                    options=client._options,
                )
                (fut, rbr_out) = client.do_exchange(
                    unbound_expr_exchanger.command, rbr_in
                )
                if do_instrument_reader:
                    rbr_out = instrument_reader(rbr_out, "output: ")

                # HAK: account for https://github.com/apache/arrow-rs/issues/6471
                rbr_out = copy_rbr_batches(rbr_out)
                yield from rbr_out

        gen = inner(self)
        schema = self.schema.to_pyarrow()
        return pa.RecordBatchReader.from_batches(schema, gen)


def flight_operator(
    expr,
    unbound_expr,
    name=None,
    make_server=None,
    make_connection=None,
    inner_name=None,
    con=None,
    **kwargs,
):
    return (
        FlightExchange.from_exprs(
            expr,
            unbound_expr,
            make_server=make_server,
            make_connection=make_connection,
            name=inner_name,
            **kwargs,
        )
        .to_expr()
        .into_backend(con=con or expr._find_backend(), name=name)
    )


class FlightUDXF(ops.DatabaseTable):
    input_expr: Expr = None
    # FIXME: fix circular import issue so we can possibly pass an instance of AbstractExchanger
    udxf: type = None
    make_server: Callable = None
    make_connection: Callable = None
    do_instrument_reader: bool = False

    @classmethod
    def validate_schema(cls, input_expr, udxf):
        if not udxf.schema_in_condition(input_expr.schema().to_pyarrow()):
            raise ValueError
        schema_out = udxf.calc_schema_out(input_expr.schema().to_pyarrow())
        return schema_out

    @classmethod
    def from_expr(
        cls,
        input_expr,
        udxf,
        make_server=None,
        make_connection=None,
        name=None,
        **kwargs,
    ):
        from xorq.flight import FlightServer

        schema = cls.validate_schema(input_expr, udxf)
        return cls(
            name=name or gen_name(),
            schema=schema,
            source=input_expr._find_backend(),
            input_expr=input_expr,
            udxf=udxf,
            make_server=make_server or FlightServer,
            make_connection=make_connection or xo.connect,
            **kwargs,
        )

    def to_rbr(self, do_instrument_reader=None):
        from xorq.flight.action import AddExchangeAction

        if do_instrument_reader is None:
            do_instrument_reader = self.do_instrument_reader

        def inner(flight_udxf):
            rbr_in = flight_udxf.input_expr.to_pyarrow_batches()
            if do_instrument_reader:
                rbr_in = instrument_reader(rbr_in, "input: ")
            with flight_udxf.make_server() as server:
                client = server.client
                client.do_action(
                    AddExchangeAction.name,
                    self.udxf,
                    options=client._options,
                )
                (fut, rbr_out) = client.do_exchange(self.udxf.command, rbr_in)
                if do_instrument_reader:
                    rbr_out = instrument_reader(rbr_out, "output: ")
                # HAK: account for https://github.com/apache/arrow-rs/issues/6471
                rbr_out = copy_rbr_batches(rbr_out)
                yield from rbr_out

        gen = inner(self)
        schema = self.schema.to_pyarrow()
        return pa.RecordBatchReader.from_batches(schema, gen)


def flight_udxf(
    expr,
    udxf,
    name=None,
    make_server=None,
    make_connection=None,
    con=None,
    **kwargs,
):
    return (
        FlightUDXF.from_expr(
            input_expr=expr,
            udxf=udxf,
            make_server=make_server,
            make_connection=make_connection,
            name=name,
            **kwargs,
        )
        .to_expr()
        .into_backend(con=con or expr._find_backend(), name=name)
    )


class Read(ops.Relation):
    method_name: str
    name: str
    schema: Schema
    source: Any
    read_kwargs: Any
    values = FrozenDict()

    def make_dt(self):
        method = getattr(self.source, self.method_name)
        dt = method(**dict(self.read_kwargs)).op()
        return dt

    def make_unbound_dt(self):
        import dask

        name = f"{self.name}-{dask.base.tokenize(self)}"
        return ops.UnboundTable(
            name=name,
            schema=self.schema,
        )


def register_and_transform_remote_tables(expr):
    created = {}

    op = expr.op()
    graph, _ = Graph.from_bfs(op).toposort()
    counts = defaultdict(int)
    for node in graph:
        if isinstance(node, RemoteTable):
            counts[node] += 1

        if isinstance(node, Relation):
            for arg in node.__args__:
                if isinstance(arg, RemoteTable):
                    counts[arg] += 1

    batches_table = {}
    for arg, count in counts.items():
        ex = arg.remote_expr
        if not ex.op().find((RemoteTable, CachedNode, Read)):
            batches = ex.to_pyarrow_batches()  # execute in native backend
        else:
            batches = xo.to_pyarrow_batches(ex)
        schema = ex.as_table().schema().to_pyarrow()
        replicas = SafeTee.tee(batches, count)
        batches_table[arg] = (schema, list(replicas))

    def mark_remote_table(node):
        schema, batchess = batches_table[node]
        name = f"{node.name}_{len(batchess)}"
        reader = pa.RecordBatchReader.from_batches(schema, batchess.pop())
        result = node.source.read_record_batches(reader, table_name=name)
        created[name] = node.source
        return result.op()

    def replacer(node, kwargs):
        kwargs = kwargs or {}
        if isinstance(node, Relation):
            updated = {}
            for k, v in list(kwargs.items()):
                try:
                    if v in batches_table:
                        updated[v] = mark_remote_table(v)

                except TypeError:  # v may not be hashable
                    continue

            if len(updated) > 0:
                kwargs = {k: recursive_update(v, updated) for k, v in kwargs.items()}

        if kwargs:
            node = node.__recreate__(kwargs)
        if isinstance(node, RemoteTable):
            result = mark_remote_table(node)
            batches_table[result] = batches_table.pop(node)
            node = result

        return node

    expr = op.replace(replacer).to_expr()
    return expr, created
