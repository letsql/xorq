"""xorq expression API definitions."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import pyarrow as pa
import pyarrow.dataset as ds

import xorq.vendor.ibis.expr.types as ir
from xorq.common.utils.caching_utils import find_backend
from xorq.common.utils.defer_utils import rbr_wrapper
from xorq.expr.ml import (
    calc_split_column,
    train_test_splits,
)
from xorq.expr.relations import (
    CachedNode,
    register_and_transform_remote_tables,
)
from xorq.vendor.ibis.backends.sql.dialects import DataFusion
from xorq.vendor.ibis.expr import api
from xorq.vendor.ibis.expr.api import *  # noqa: F403
from xorq.vendor.ibis.expr.sql import SQLString
from xorq.vendor.ibis.expr.types import Table


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    import pandas as pd
    import pyarrow as pa

    from xorq.vendor.ibis.expr.schema import SchemaLike


__all__ = (
    "execute",
    "calc_split_column",
    "get_plans",
    "read_csv",
    "read_parquet",
    "read_postgres",
    "register",
    "train_test_splits",
    "to_parquet",
    "to_pyarrow",
    "to_pyarrow_batches",
    "to_sql",
    "get_plans",
    *api.__all__,
)


def memtable(
    data,
    *,
    columns: Iterable[str] | None = None,
    schema: SchemaLike | None = None,
    name: str | None = None,
) -> Table:
    """Construct an ibis table expression from in-memory data.

    Parameters
    ----------
    data
        A table-like object (`pandas.DataFrame`, `pyarrow.Table`, or
        `polars.DataFrame`), or any data accepted by the `pandas.DataFrame`
        constructor (e.g. a list of dicts).

        Note that ibis objects (e.g. `MapValue`) may not be passed in as part
        of `data` and will result in an error.

        Do not depend on the underlying storage type (e.g., pyarrow.Table),
        it's subject to change across non-major releases.
    columns
        Optional [](`typing.Iterable`) of [](`str`) column names. If provided,
        must match the number of columns in `data`.
    schema
        Optional [`Schema`](./schemas.qmd#ibis.expr.schema.Schema).
        The functions use `data` to infer a schema if not passed.
    name
        Optional name of the table.

    Returns
    -------
    Table
        A table expression backed by in-memory data.

    Examples
    --------
    >>> import xorq as ls
    >>> ls.options.interactive = False
    >>> t = ls.memtable([{"a": 1}, {"a": 2}])
    >>> t
    InMemoryTable
      data:
        PandasDataFrameProxy:
             a
          0  1
          1  2

    >>> t = ls.memtable([{"a": 1, "b": "foo"}, {"a": 2, "b": "baz"}])
    >>> t
    InMemoryTable
      data:
        PandasDataFrameProxy:
             a    b
          0  1  foo
          1  2  baz

    Create a table literal without column names embedded in the data and pass
    `columns`

    >>> t = ls.memtable([(1, "foo"), (2, "baz")], columns=["a", "b"])
    >>> t
    InMemoryTable
      data:
        PandasDataFrameProxy:
             a    b
          0  1  foo
          1  2  baz

    Create a table literal without column names embedded in the data. Ibis
    generates column names if none are provided.

    >>> t = ls.memtable([(1, "foo"), (2, "baz")])
    >>> t
    InMemoryTable
      data:
        PandasDataFrameProxy:
             col0 col1
          0     1  foo
          1     2  baz

    """

    if isinstance(data, ds.InMemoryDataset):
        data = data.to_table()

    if isinstance(data, pa.RecordBatch):
        data = data.to_pandas()

    return api.memtable(data, columns=columns, schema=schema, name=name)


def read_csv(
    sources: str | Path | Sequence[str | Path],
    table_name: str | None = None,
    **kwargs: Any,
) -> ir.Table:
    """Lazily load a CSV or set of CSVs.

    This function delegates to the `read_csv` method on the current default
    backend (DuckDB or `ibis.config.default_backend`).

    Parameters
    ----------
    sources
        A filesystem path or URL or list of same.  Supports CSV and TSV files.
    table_name
        A name to refer to the table.  If not provided, a name will be generated.
    kwargs
        Backend-specific keyword arguments for the file type. For the DuckDB
        backend used by default, please refer to:

        * CSV/TSV: https://duckdb.org/docs/data/csv/overview.html#parameters.

    Returns
    -------
    ir.Table
        Table expression representing a file

    Examples
    --------
    >>> import xorq
    >>> xorq.options.interactive = True
    >>> lines = '''a,b
    ... 1,d
    ... 2,
    ... ,f
    ... '''
    >>> with open("/tmp/lines.csv", mode="w") as f:
    ...     nbytes = f.write(lines)  # nbytes is unused
    >>> t = xorq.read_csv("/tmp/lines.csv")
    >>> t
    ┏━━━━━━━┳━━━━━━━━┓
    ┃ a     ┃ b      ┃
    ┡━━━━━━━╇━━━━━━━━┩
    │ int64 │ string │
    ├───────┼────────┤
    │     1 │ d      │
    │     2 │ NULL   │
    │  NULL │ f      │
    └───────┴────────┘

    """
    from xorq.config import _backend_init

    con = _backend_init()
    return con.read_csv(sources, table_name=table_name, **kwargs)


def read_parquet(
    sources: str | Path | Sequence[str | Path],
    table_name: str | None = None,
    **kwargs: Any,
) -> ir.Table:
    """Lazily load a parquet file or set of parquet files.

    This function delegates to the `read_parquet` method on the current default
    backend (DuckDB or `ibis.config.default_backend`).

    Parameters
    ----------
    sources
        A filesystem path or URL or list of same.
    table_name
        A name to refer to the table.  If not provided, a name will be generated.
    kwargs
        Backend-specific keyword arguments for the file type. For the DuckDB
        backend used by default, please refer to:

        * Parquet: https://duckdb.org/docs/data/parquet

    Returns
    -------
    ir.Table
        Table expression representing a file

    Examples
    --------
    >>> import xorq
    >>> import pandas as pd
    >>> xorq.options.interactive = True
    >>> df = pd.DataFrame({"a": [1, 2, 3], "b": list("ghi")})
    >>> df
       a  b
    0  1  g
    1  2  h
    2  3  i
    >>> df.to_parquet("/tmp/data.parquet")
    >>> t = xorq.read_parquet("/tmp/data.parquet")
    >>> t
    ┏━━━━━━━┳━━━━━━━━┓
    ┃ a     ┃ b      ┃
    ┡━━━━━━━╇━━━━━━━━┩
    │ int64 │ string │
    ├───────┼────────┤
    │     1 │ g      │
    │     2 │ h      │
    │     3 │ i      │
    └───────┴────────┘

    """
    from xorq.config import _backend_init

    con = _backend_init()
    return con.read_parquet(sources, table_name=table_name, **kwargs)


def register(
    source: str | Path | pa.Table | pa.RecordBatch | pa.Dataset | pd.DataFrame,
    table_name: str | None = None,
    **kwargs: Any,
):
    from xorq.config import _backend_init

    con = _backend_init()
    return con.register(source, table_name=table_name, **kwargs)


def read_postgres(
    uri: str,
    table_name: str | None = None,
    **kwargs: Any,
):
    from xorq.config import _backend_init

    con = _backend_init()
    return con.read_postgres(uri, table_name=table_name, **kwargs)


def read_sqlite(path: str | Path, *, table_name: str | None = None):
    from xorq.config import _backend_init

    con = _backend_init()
    return con.read_sqlite(path, table_name=table_name)


@functools.cache
def _cached_with_op(op, pretty):
    from xorq.config import _backend_init

    con = _backend_init()

    expr = op.to_expr()
    sg_expr = con.compiler.to_sqlglot(expr)
    sql = sg_expr.sql(dialect=DataFusion, pretty=pretty)
    return sql


def to_sql(expr: ir.Expr, pretty: bool = True) -> SQLString:
    """Return the formatted SQL string for an expression.

    Parameters
    ----------
    expr
        Ibis expression.
    pretty
        Whether to use pretty formatting.

    Returns
    -------
    str
        Formatted SQL string

    """

    return SQLString(_cached_with_op(expr.unbind().op(), pretty))


def _register_and_transform_cache_tables(expr):
    """This function will sequentially execute any cache node that is not already cached"""

    def fn(node, kwargs):
        if kwargs:
            node = node.__recreate__(kwargs)
        if isinstance(node, CachedNode):
            uncached, storage = node.parent, node.storage
            node = storage.set_default(uncached, uncached.op())
        return node

    op = expr.op()
    out = op.replace(fn)

    return out.to_expr()


def _transform_deferred_reads(expr):
    dt_to_read = {}

    def replace_read(node, _kwargs):
        from xorq.expr.relations import Read

        if isinstance(node, Read):
            if node.source.name == "pandas":
                # FIXME: pandas read is not lazy, leave it to the pandas executor to do
                node = dt_to_read[node] = node.make_dt()
            else:
                node = dt_to_read[node] = node.make_dt()
        else:
            if _kwargs:
                node = node.__recreate__(_kwargs)
        return node

    expr = expr.op().replace(replace_read).to_expr()
    return expr, dt_to_read


def execute(expr: ir.Expr, **kwargs: Any):
    batch_reader = to_pyarrow_batches(expr, **kwargs)
    return expr.__pandas_result__(batch_reader.read_pandas(timestamp_as_object=True))


def _transform_expr(expr):
    expr = _register_and_transform_cache_tables(expr)
    expr, created = register_and_transform_remote_tables(expr)
    expr, dt_to_read = _transform_deferred_reads(expr)
    return (expr, created)


def to_pyarrow_batches(
    expr: ir.Expr,
    *,
    chunk_size: int = 1_000_000,
    **kwargs: Any,
):
    from xorq.expr.relations import FlightExchange, FlightUDXF

    if isinstance(expr.op(), (FlightExchange, FlightUDXF)):
        return expr.op().to_rbr()
    (expr, created) = _transform_expr(expr)
    con, _ = find_backend(expr.op(), use_default=True)

    reader = con.to_pyarrow_batches(expr, chunk_size=chunk_size, **kwargs)

    def clean_up():
        for table_name, conn in created.items():
            try:
                conn.drop_table(table_name)
            except Exception:
                conn.drop_view(table_name)

    return rbr_wrapper(reader, clean_up)


def to_pyarrow(expr: ir.Expr, **kwargs: Any):
    batch_reader = to_pyarrow_batches(expr, **kwargs)
    arrow_table = batch_reader.read_all()
    return expr.__pyarrow_result__(arrow_table)


def to_parquet(
    expr: ir.Expr,
    path: str | Path,
    params: Mapping[ir.Scalar, Any] | None = None,
    **kwargs: Any,
):
    import pyarrow  # noqa: ICN001, F401
    import pyarrow.parquet as pq
    import pyarrow_hotfix  # noqa: F401

    with to_pyarrow_batches(expr, params=params) as batch_reader:
        with pq.ParquetWriter(path, batch_reader.schema, **kwargs) as writer:
            for batch in batch_reader:
                writer.write_batch(batch)


def get_plans(expr):
    _expr, _ = _transform_expr(expr)
    con, _ = find_backend(_expr.op())
    sql = f"EXPLAIN {to_sql(_expr)}"
    return con.con.sql(sql).to_pandas().set_index("plan_type")["plan"].to_dict()
