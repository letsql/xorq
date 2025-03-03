import pathlib
import re
import types

import dask
import sqlglot as sg

import xorq as xo
import xorq.vendor.ibis.expr.datatypes as dat
import xorq.vendor.ibis.expr.operations.relations as ir
from xorq.common.utils.dask_normalize.dask_normalize_utils import (
    normalize_seq_with_caller,
)
from xorq.common.utils.defer_utils import (
    Read,
)
from xorq.expr.relations import (
    RemoteTable,
    make_native_op,
)
from xorq.vendor import ibis
from xorq.vendor.ibis.expr.operations.udf import (
    AggUDF,
    ScalarUDF,
)


def expr_is_bound(expr):
    backends, _ = expr._find_backends()
    return bool(backends)


def unbound_expr_to_default_sql(expr):
    if expr_is_bound(expr):
        raise ValueError
    default_sql = xo.to_sql(expr)
    return str(default_sql)


def normalize_memory_databasetable(dt):
    if dt.source.name not in ("pandas", "let", "datafusion", "duckdb"):
        raise ValueError
    return normalize_seq_with_caller(
        # we are normalizing the data, we don't care about the connection
        # dt.source,
        dt.schema.to_pandas(),
        # in memory: so we can assume it's reasonable to hash the data
        tuple(
            dask.base.tokenize(el.serialize().to_pybytes())
            for el in xo.to_pyarrow_batches(dt.to_expr())
        ),
    )


def normalize_pandas_databasetable(dt):
    if dt.source.name != "pandas":
        raise ValueError
    return normalize_memory_databasetable(dt)


def normalize_datafusion_databasetable(dt):
    if dt.source.name not in ("datafusion", "let"):
        raise ValueError
    table = dt.source.con.table(dt.name)
    ep_str = str(table.execution_plan())
    if ep_str.startswith(("ParquetExec:", "CsvExec:")):
        return normalize_seq_with_caller(
            dt.schema.to_pandas(),
            # ep_str denotes the parquet files to be read
            # FIXME: md5sum on detected .parquet files?
            ep_str,
        )
    elif ep_str.startswith("MemoryExec:"):
        return normalize_memory_databasetable(dt)
    elif ep_str.startswith("PyRecordBatchProviderExec"):
        return normalize_seq_with_caller(
            dt.schema.to_pandas(),
            dt.name,
        )

    else:
        raise ValueError


def normalize_remote_databasetable(dt):
    return normalize_seq_with_caller(
        dt.name,
        dt.schema,
        dt.source,
        dt.namespace,
    )


def normalize_postgres_databasetable(dt):
    from xorq.common.utils.postgres_utils import get_postgres_n_reltuples

    if dt.source.name != "postgres":
        raise ValueError
    return normalize_seq_with_caller(
        dt.name,
        dt.schema,
        dt.source,
        dt.namespace,
        get_postgres_n_reltuples(dt),
    )


def normalize_snowflake_databasetable(dt):
    from xorq.common.utils.snowflake_utils import get_snowflake_last_modification_time

    if dt.source.name != "snowflake":
        raise ValueError
    return normalize_seq_with_caller(
        dt.name,
        dt.schema,
        dt.source,
        dt.namespace,
        get_snowflake_last_modification_time(dt).tobytes(),
    )


def normalize_bigquery_databasetable(dt):
    # https://stackoverflow.com/questions/44288261/get-the-last-modified-date-for-all-bigquery-tables-in-a-bigquery-project/44290543#44290543
    if dt.source.name != "bigquery":
        raise ValueError
    # https://stackoverflow.com/a/44290543
    query = f"""
    SELECT last_modified_time
    FROM `{dt.namespace.database}.__TABLES__` where table_id = '{dt.name}'
    """
    ((last_modified_time,),) = dt.source.raw_sql(query).to_dataframe()
    return normalize_seq_with_caller(
        dt.name,
        dt.schema,
        dt.source,
        dt.namespace,
        last_modified_time,
    )


def normalize_duckdb_databasetable(dt):
    if dt.source.name != "duckdb":
        raise ValueError
    name = sg.table(dt.name, quoted=dt.source.compiler.quoted).sql(
        dialect=dt.source.name
    )

    ((_, plan),) = dt.source.raw_sql(f"EXPLAIN SELECT * FROM {name}").fetchall()
    scan_line = plan.split("\n")[1]
    execution_plan_name = r"\s*│\s*(\w+)\s*│\s*"
    match re.match(execution_plan_name, scan_line).group(1):
        case "ARROW_SCAN":
            return normalize_memory_databasetable(dt)
        case "READ_PARQUET" | "READ_CSV" | "SEQ_SCAN":
            return normalize_duckdb_file_read(dt)
        case _:
            raise NotImplementedError(scan_line)


def normalize_duckdb_file_read(dt):
    name = sg.exp.convert(dt.name).sql(dialect=dt.source.name)
    (sql_ddl_statement,) = dt.source.con.sql(
        f"select sql from duckdb_views() where view_name = {name} UNION select sql from duckdb_tables() where table_name = {name}"
    ).fetchone()
    return normalize_seq_with_caller(
        dt.schema.to_pandas(),
        # sql_ddl_statement denotes the definition of the table, expressed as SQL DDL-statement.
        sql_ddl_statement,
    )


def normalize_letsql_databasetable(dt):
    from xorq.expr.relations import FlightExpr, FlightUDXF

    if dt.source.name != "let":
        raise ValueError
    if isinstance(dt, FlightExpr):
        return dask.base.normalize_token(
            (
                "normalize_letsql_databasetable",
                dt.input_expr,
                dt.unbound_expr,
                dt.make_connection,
            )
        )
    elif isinstance(dt, FlightUDXF):
        return dask.base.normalize_token(
            (
                "normalize_letsql_databasetable",
                dt.input_expr,
                dt.udxf.exchange_f,
                dt.make_connection,
            )
        )
    native_source = dt.source._sources.get_backend(dt)

    if native_source.name == "let":
        return normalize_datafusion_databasetable(dt)
    new_dt = make_native_op(dt)
    return dask.base.normalize_token(new_dt)


@dask.base.normalize_token.register(types.ModuleType)
def normalize_module(module):
    return normalize_seq_with_caller(
        module.__name__,
        module.__package__,
    )


@dask.base.normalize_token.register(dat.DataType)
def normalize_ibis_datatype(datatype):
    return normalize_seq_with_caller(datatype.name.lower(), *datatype.args)


@dask.base.normalize_token.register(Read)
def normalize_read(read):
    path = next(
        el
        for el in (
            dict(read.read_kwargs).get(name)
            for name in (
                "path",
                "source",
                "source_list",  # duckdb
            )
        )
        if el
    )
    if isinstance(path, (str, pathlib.Path)):
        path = str(path)
        if path.startswith("http") or path.startswith("https:"):
            import requests

            resp = requests.head(path)
            resp.raise_for_status()
            tpls = tuple(
                (k, resp.headers[k])
                for k in (
                    "Last-Modified",
                    "Content-Length",
                    "Content-Type",
                )
            )
        elif path.startswith("s3"):
            raise NotImplementedError
        elif (path := pathlib.Path(path)).exists():
            stat = path.stat()
            tpls = tuple(
                (attrname, getattr(stat, attrname))
                for attrname in (
                    "st_mtime",
                    "st_size",
                    # mtime, size <?-?> md5sum
                    "st_ino",
                )
            )
        else:
            raise NotImplementedError(f'Don\'t know how to deal with path "{path}"')
    elif isinstance(path, (list, tuple)) and all(isinstance(el, str) for el in path):
        raise NotImplementedError
    else:
        raise NotImplementedError
    return normalize_seq_with_caller(
        read.schema,
        tpls,
    )


@dask.base.normalize_token.register(ir.DatabaseTable)
def normalize_databasetable(dt):
    dct = {
        "pandas": normalize_pandas_databasetable,
        "datafusion": normalize_datafusion_databasetable,
        "postgres": normalize_postgres_databasetable,
        "snowflake": normalize_snowflake_databasetable,
        "let": normalize_letsql_databasetable,
        "duckdb": normalize_duckdb_databasetable,
        "trino": normalize_remote_databasetable,
        "bigquery": normalize_bigquery_databasetable,
    }
    f = dct[dt.source.name]
    return f(dt)


@dask.base.normalize_token.register(RemoteTable)
def normalize_remote_table(dt):
    if not isinstance(dt, RemoteTable):
        raise ValueError

    return normalize_seq_with_caller(
        ("schema", dt.schema),
        ("expr", dt.remote_expr),
        # only thing that matters is the type of source its going into
        ("source", dt.source.name),
    )


@dask.base.normalize_token.register(ibis.backends.BaseBackend)
def normalize_backend(con):
    name = con.name
    if name == "snowflake":
        con_details = con.con._host
    elif name == "postgres":
        con_dct = con.con.get_dsn_parameters()
        con_details = {k: con_dct[k] for k in ("host", "port", "dbname")}
    elif name == "pandas":
        con_details = id(con.dictionary)
    elif name in ("datafusion", "duckdb", "let"):
        con_details = id(con.con)
    elif name == "trino":
        con_details = con.con.host
    elif name == "bigquery":
        con_details = (
            con.project_id,
            con.dataset_id,
        )
    else:
        raise ValueError
    return normalize_seq_with_caller(
        name,
        con_details,
    )


@dask.base.normalize_token.register(ir.Schema)
def normalize_schema(schema):
    return normalize_seq_with_caller(
        schema.to_pandas(),
    )


@dask.base.normalize_token.register(ir.Namespace)
def normalize_namespace(ns):
    return normalize_seq_with_caller(
        ns.catalog,
        ns.database,
    )


@dask.base.normalize_token.register(ScalarUDF)
def normalize_scalar_udf(udf):
    typs = tuple(arg.dtype for arg in udf.args)
    return normalize_seq_with_caller(
        ScalarUDF,
        typs,
        udf.dtype,
        udf.__func__,
        #
        # ExprScalarUDF
        udf.__config__.get("computed_kwargs_expr"),
        # we are insensitive to these for now
        # udf.__udf_namespace__,
        # udf.__func_name__,
    )


@dask.base.normalize_token.register(AggUDF)
def normalize_agg_udf(udf):
    (*args, where) = udf.args
    if where is not None:
        # TODO: determine if sql string already contains
        #       the relevant information of `where`
        raise NotImplementedError
    typs = tuple(arg.dtype for arg in args)
    return normalize_seq_with_caller(
        AggUDF,
        typs,
        udf.dtype,
        udf.__func__,
        # we are insensitive to these for now
        # udf.__udf_namespace__,
        # udf.__func_name__,
    )


@dask.base.normalize_token.register(ibis.expr.types.Expr)
def normalize_expr(expr):
    # FIXME: replace bound table names with their hashes
    sql = unbound_expr_to_default_sql(expr.ls.uncached.unbind())

    if not (expr_is_bound(expr) or expr.op().find(Read)):
        return sql

    op = expr.op()
    if mem_dts := op.find(ir.InMemoryTable):
        # these should have been replaced by the time we get to them
        raise ValueError(f"{mem_dts}")
    reads = op.find(Read)
    dts = op.find(ir.DatabaseTable)
    udfs = op.find((AggUDF, ScalarUDF))
    token = normalize_seq_with_caller(
        sql,
        reads,
        dts,
        udfs,
    )
    return token
