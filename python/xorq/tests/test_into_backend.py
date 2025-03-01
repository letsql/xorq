from operator import methodcaller

import pandas as pd
import pyarrow as pa
import pytest

import xorq as xo
from xorq.caching import ParquetStorage, SourceStorage
from xorq.expr.relations import register_and_transform_remote_tables
from xorq.vendor import ibis
from xorq.vendor.ibis import _


expected_tables = (
    "array_types",
    "astronauts",
    "awards_players",
    "awards_players_special_types",
    "batting",
    "diamonds",
    "functional_alltypes",
    "geo",
    "geography_columns",
    "geometry_columns",
    "json_t",
    "map",
    "spatial_ref_sys",
    "topk",
    "tzone",
)


@pytest.fixture(scope="session")
def pg():
    conn = xo.postgres.connect_env()
    yield conn
    remove_unexpected_tables(conn)


@pytest.fixture(scope="session")
def trino_table():
    return (
        xo.trino.connect(database="tpch", schema="sf1")
        .table("orders")
        .cast({"orderdate": "date"})
    )


def make_merged(expr):
    agg = expr.group_by(["custkey", "orderdate"]).agg(
        _.totalprice.sum().name("totalprice")
    )
    w = ibis.window(group_by="custkey", order_by="orderdate")
    windowed = (
        agg.mutate(_.totalprice.cumsum().over(w).name("curev"))
        .mutate(_.curev.lag(1).over(w).name("curev@t-1"))
        .select(["custkey", "orderdate", "curev", "curev@t-1"])
    )
    merged = expr.asof_join(
        windowed,
        on="orderdate",
        predicates="custkey",
    ).select(
        [expr[c] for c in expr.columns]
        + [windowed[c] for c in windowed.columns if c not in expr.columns]
    )
    return merged


def remove_unexpected_tables(dirty):
    # drop tables
    for table in dirty.list_tables():
        if table not in expected_tables:
            dirty.drop_table(table, force=True)

    # drop view
    for table in dirty.list_tables():
        if table not in expected_tables:
            dirty.drop_view(table, force=True)

    if sorted(dirty.list_tables()) != sorted(expected_tables):
        raise ValueError


def test_multiple_record_batches(pg):
    con = xo.connect()

    table = pg.table("batting")
    left = con.register(table.to_pyarrow_batches(), "batting_0")
    right = con.register(table.to_pyarrow_batches(), "batting_1")

    expr = (
        left.join(right, "playerID")
        .limit(15)
        .select(player_id="playerID", year_id="yearID_right")
        .cache(SourceStorage(source=con))
    )

    res = expr.execute()
    assert isinstance(res, pd.DataFrame)
    assert 0 < len(res) <= 15


@pytest.mark.parametrize("method", [xo.to_pyarrow, xo.to_pyarrow_batches, xo.execute])
def test_into_backend_simple(pg, method):
    con = xo.connect()
    expr = pg.table("batting").into_backend(con, "ls_batting")
    res = method(expr)

    if isinstance(res, pa.RecordBatchReader):
        res = next(res)

    assert len(res) > 0


@pytest.mark.parametrize("method", ["to_pyarrow", "to_pyarrow_batches", "execute"])
def test_into_backend_complex(pg, method):
    con = xo.connect()

    t = pg.table("batting").into_backend(con, "ls_batting")

    expr = (
        t.join(t, "playerID")
        .limit(15)
        .select(player_id="playerID", year_id="yearID_right")
        .cache(SourceStorage(source=con))
    )

    assert xo.to_sql(expr).count("batting") == 2
    res = methodcaller(method)(expr)

    if isinstance(res, pa.RecordBatchReader):
        res = next(res)

    assert 0 < len(res) <= 15


def test_double_into_backend_batches(pg):
    con = xo.connect()
    ddb_con = xo.duckdb.connect()

    t = pg.table("batting").into_backend(con, "ls_batting")

    expr = (
        t.join(t, "playerID")
        .limit(15)
        .into_backend(ddb_con)
        .select(player_id="playerID", year_id="yearID_right")
        .cache(SourceStorage(source=con))
    )

    res = expr.to_pyarrow_batches()
    res = next(res)

    assert len(res) == 15


@pytest.mark.benchmark
def test_into_backend_cache(pg, tmp_path):
    con = xo.connect()
    ddb_con = xo.duckdb.connect()

    t = pg.table("batting").into_backend(con, "ls_batting")

    expr = (
        t.join(t, "playerID")
        .limit(15)
        .cache(SourceStorage(source=con))
        .into_backend(ddb_con)
        .select(player_id="playerID", year_id="yearID_right")
        .cache(ParquetStorage(source=ddb_con, path=tmp_path))
    )

    res = expr.execute()
    assert 0 < len(res) <= 15


def test_into_backend_duckdb(pg):
    ddb = xo.duckdb.connect()
    t = pg.table("batting").into_backend(ddb, "ls_batting")
    expr = (
        t.join(t, "playerID")
        .limit(15)
        .select(player_id="playerID", year_id="yearID_right")
    )

    expr, created = register_and_transform_remote_tables(expr)
    query = ibis.to_sql(expr, dialect="duckdb")

    res = ddb.con.sql(query).df()

    assert query.count("ls_batting") == 2
    assert 0 < len(res) <= 15
    assert len(created) == 3


def test_into_backend_duckdb_expr(pg):
    ddb = xo.duckdb.connect()
    t = pg.table("batting").into_backend(ddb, "ls_batting")
    expr = t.join(t, "playerID").limit(15).select(_.playerID * 2)

    expr, created = register_and_transform_remote_tables(expr)
    query = ibis.to_sql(expr, dialect="duckdb")

    res = ddb.con.sql(query).df()

    assert query.count("ls_batting") == 2
    assert 0 < len(res) <= 15
    assert len(created) == 3


def test_into_backend_duckdb_trino(trino_table):
    db_con = xo.duckdb.connect()
    expr = trino_table.head(10_000).into_backend(db_con).pipe(make_merged)

    expr, created = register_and_transform_remote_tables(expr)
    query = ibis.to_sql(expr, dialect="duckdb")

    df = db_con.con.sql(query).df()  # to bypass execute hotfix

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert len(created) == 3


def test_multiple_into_backend_duckdb_xorq(trino_table):
    db_con = xo.duckdb.connect()
    ls_con = xo.connect()

    expr = (
        trino_table.head(10_000)
        .into_backend(db_con)
        .pipe(make_merged)
        .into_backend(ls_con)[lambda t: t.orderstatus == "F"]
    )

    expr, created = register_and_transform_remote_tables(expr)

    df = expr.execute()

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert len(created) == 2


@pytest.mark.benchmark
def test_into_backend_duckdb_trino_cached(trino_table, tmp_path):
    db_con = xo.duckdb.connect()
    expr = (
        trino_table.head(10_000)
        .into_backend(db_con)
        .pipe(make_merged)
        .cache(ParquetStorage(path=tmp_path))
    )
    df = expr.execute()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_into_backend_to_pyarrow_batches(trino_table):
    db_con = xo.duckdb.connect()
    df = (
        trino_table.head(10_000)
        .into_backend(db_con)
        .pipe(make_merged)
        .to_pyarrow_batches()
        .read_pandas()
    )
    assert not df.empty


def test_to_pyarrow_batches_simple(pg):
    con = xo.duckdb.connect()

    t = pg.table("batting").into_backend(con, "ls_batting")

    expr = (
        t.join(t, "playerID")
        .limit(15)
        .select(player_id="playerID", year_id="yearID_right")
    )

    df = expr.to_pyarrow_batches().read_pandas()
    assert not df.empty
