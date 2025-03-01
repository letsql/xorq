from __future__ import annotations

import inspect
import pathlib
import time
import uuid

import dask
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pytest
import toolz

import xorq as xo
import xorq.vendor.ibis.expr.datatypes as dt
from xorq.backends.conftest import (
    get_storage_uncached,
)
from xorq.caching import (
    ParquetSnapshotStorage,
    ParquetStorage,
    SourceSnapshotStorage,
    SourceStorage,
)
from xorq.common.utils.inspect_utils import get_python_version_no_dot
from xorq.common.utils.postgres_utils import (
    do_analyze,
    get_postgres_n_scans,
)
from xorq.expr.relations import into_backend
from xorq.expr.udf import (
    agg,
)
from xorq.tests.util import (
    assert_frame_equal,
)
from xorq.vendor import ibis
from xorq.vendor.ibis import _


KEY_PREFIX = xo.config.options.cache.key_prefix


@pytest.fixture
def cached_two(ls_con, batting, tmp_path):
    parquet_storage = ParquetStorage(source=ls_con, path=tmp_path)
    return (
        batting[lambda t: t.yearID > 2014]
        .cache()[lambda t: t.stint == 1]
        .cache(storage=parquet_storage)
    )


@pytest.fixture(scope="function")
def pg_alltypes(pg):
    return pg.table("functional_alltypes")


@pytest.fixture(scope="session")
def csv_dir():
    root = pathlib.Path(__file__).absolute().parents[5]
    data_dir = root / "ci" / "ibis-testing-data" / "csv"
    return data_dir


@pytest.fixture(scope="session")
def parquet_dir():
    root = pathlib.Path(__file__).absolute().parents[5]
    data_dir = root / "ci" / "ibis-testing-data" / "parquet"
    return data_dir


def test_cache_simple(con, alltypes, alltypes_df):
    initial_tables = con.list_tables()

    expr = alltypes.select(
        alltypes.smallint_col, alltypes.int_col, alltypes.float_col
    ).filter(
        [
            alltypes.float_col > 0,
            alltypes.smallint_col == 9,
            alltypes.int_col < alltypes.float_col * 2,
        ]
    )
    cached = expr.cache(storage=SourceStorage(source=con))
    tables_after_caching = con.list_tables()

    expected = alltypes_df[
        (alltypes_df["float_col"] > 0)
        & (alltypes_df["smallint_col"] == 9)
        & (alltypes_df["int_col"] < alltypes_df["float_col"] * 2)
    ][["smallint_col", "int_col", "float_col"]]

    executed = cached.execute()
    tables_after_executing = con.list_tables()

    assert_frame_equal(executed, expected)
    assert not any(
        table_name.startswith(KEY_PREFIX)
        for table_name in set(tables_after_caching).difference(initial_tables)
    )
    assert any(
        table_name.startswith(KEY_PREFIX)
        for table_name in set(tables_after_executing).difference(initial_tables)
    )


def test_cache_multiple_times(con, alltypes, alltypes_df):
    expr = alltypes.select(
        alltypes.smallint_col, alltypes.int_col, alltypes.float_col
    ).filter(
        [
            alltypes.float_col > 0,
            alltypes.smallint_col == 9,
            alltypes.int_col < alltypes.float_col * 2,
        ]
    )
    cached = expr.cache()

    # reassign the expression
    expr = alltypes.select(
        alltypes.smallint_col, alltypes.int_col, alltypes.float_col
    ).filter(
        [
            alltypes.float_col > 0,
            alltypes.smallint_col == 9,
            alltypes.int_col < alltypes.float_col * 2,
        ]
    )

    re_cached = expr.cache()

    first = cached.execute()
    tables_after_first_caching = con.list_tables()

    second = re_cached.execute()
    tables_after_second_caching = con.list_tables()

    expected = alltypes_df[
        (alltypes_df["float_col"] > 0)
        & (alltypes_df["smallint_col"] == 9)
        & (alltypes_df["int_col"] < alltypes_df["float_col"] * 2)
    ][["smallint_col", "int_col", "float_col"]]

    assert_frame_equal(first, expected)
    assert_frame_equal(second, expected)

    first_tables = [t for t in tables_after_first_caching if t.startswith(KEY_PREFIX)]
    second_tables = [t for t in tables_after_second_caching if t.startswith(KEY_PREFIX)]

    assert sorted(first_tables) == sorted(second_tables)


def test_cache_to_sql(alltypes):
    expr = alltypes.select(
        alltypes.smallint_col, alltypes.int_col, alltypes.float_col
    ).filter(
        [
            alltypes.float_col > 0,
            alltypes.smallint_col == 9,
            alltypes.int_col < alltypes.float_col * 2,
        ]
    )
    cached = expr.cache()

    assert xo.to_sql(cached) == xo.to_sql(expr)


def test_op_after_cache(alltypes):
    expr = alltypes.select(alltypes.smallint_col, alltypes.int_col, alltypes.float_col)
    cached = expr.cache()
    cached = cached.filter(
        [
            _.float_col > 0,
            _.smallint_col == 9,
            _.int_col < _.float_col * 2,
        ]
    )

    full_expr = expr.filter(
        [
            alltypes.float_col > 0,
            alltypes.smallint_col == 9,
            alltypes.int_col < alltypes.float_col * 2,
        ]
    )

    actual = cached.execute()
    expected = full_expr.execute()

    assert_frame_equal(actual, expected)

    assert xo.to_sql(cached) == xo.to_sql(full_expr)


def test_cache_recreate(alltypes):
    def make_expr(alltypes):
        return alltypes.select(
            alltypes.smallint_col, alltypes.int_col, alltypes.float_col
        ).filter(
            [
                alltypes.float_col > 0,
                alltypes.smallint_col == 9,
                alltypes.int_col < alltypes.float_col * 2,
            ]
        )

    alltypes_df = alltypes.execute()
    cons = (con0, con1) = xo.connect(), xo.connect()
    ts = tuple(con.create_table("alltypes", alltypes_df) for con in cons)
    exprs = tuple(make_expr(t) for t in ts)

    for con, expr in zip(cons, exprs):
        # FIXME: execute one, simply check the other returns true for `expr.ls.exists()`
        expr.cache(storage=SourceStorage(source=con)).execute()

    (con_cached_tables0, con_cached_tables1) = (
        set(
            table_name
            for table_name in con.list_tables()
            if table_name.startswith(KEY_PREFIX)
        )
        for con in cons
    )

    assert con_cached_tables0
    assert con_cached_tables0 == con_cached_tables1
    for table_name in con_cached_tables1:
        assert_frame_equal(
            con0.table(table_name).to_pandas(), con1.table(table_name).to_pandas()
        )


def test_cache_execution(alltypes):
    cached = (
        alltypes.select(alltypes.smallint_col, alltypes.int_col, alltypes.float_col)
        .cache()
        .filter(
            [
                _.float_col > 0,
                _.smallint_col == 9,
                _.int_col < _.float_col * 2,
            ]
        )
        .select(_.int_col * 4)
        .cache()
    )

    actual = cached.execute()

    expected = (
        alltypes.select(alltypes.smallint_col, alltypes.int_col, alltypes.float_col)
        .filter(
            [
                alltypes.float_col > 0,
                alltypes.smallint_col == 9,
                alltypes.int_col < alltypes.float_col * 2,
            ]
        )
        .select(alltypes.int_col * 4)
        .execute()
    )

    assert_frame_equal(actual, expected)


def test_parquet_cache_storage(tmp_path, alltypes_df):
    tmp_path = pathlib.Path(tmp_path)
    path = tmp_path.joinpath("to-delete.parquet")

    con = xo.connect()
    alltypes_df.to_parquet(path)
    t = con.read_parquet(path, "t")
    cols = ["id", "bool_col", "float_col", "string_col"]
    expr = t[cols]
    expected = alltypes_df[cols]
    source = expr._find_backend()
    storage = ParquetStorage(
        source=source,
        path=tmp_path.joinpath("parquet-cache-storage"),
    )
    cached = expr.cache(storage=storage)
    actual = cached.execute()
    assert_frame_equal(actual, expected)

    # the file must exist and have the same schema
    alltypes_df.head(1).to_parquet(path)
    actual = cached.execute()
    assert_frame_equal(actual, expected)

    path.unlink()
    pattern = "".join(
        (
            "Object Store error: Object at location",
            ".*",
            "not found: No such file or directory",
        )
    )
    with pytest.raises(Exception, match=pattern):
        # if the file doesn't exist, we get a failure, even for cached
        cached.execute()


def test_parquet_remote_to_local(con, alltypes, tmp_path):
    tmp_path = pathlib.Path(tmp_path)

    expr = alltypes.select(
        alltypes.smallint_col, alltypes.int_col, alltypes.float_col
    ).filter(
        [
            alltypes.float_col > 0,
            alltypes.smallint_col == 9,
            alltypes.int_col < alltypes.float_col * 2,
        ]
    )
    storage = ParquetStorage(
        source=con,
        path=tmp_path.joinpath("parquet-cache-storage"),
    )
    cached = expr.cache(storage=storage)
    expected = expr.execute()
    actual = cached.execute()
    assert_frame_equal(actual, expected)


def test_postgres_cache_invalidation(pg, con):
    def modify_postgres_table(dt):
        (con, name) = (dt.source, dt.name)
        statement = f"""
        INSERT INTO "{name}"
        DEFAULT VALUES
        """
        con.raw_sql(statement)

    def assert_n_scans_changes(dt, n_scans_before):
        do_analyze(dt.source, dt.name)
        for _ in range(10):  # noqa: F402
            # give postgres some time to update its tables
            time.sleep(0.1)
            n_scans_after = get_postgres_n_scans(dt)
            if n_scans_before != n_scans_after:
                return n_scans_after
        else:
            raise

    (from_name, to_name) = ("batting", "batting_to_modify")
    if to_name in pg.tables:
        pg.drop_table(to_name)
    pg_t = pg.create_table(name=to_name, obj=pg.table(from_name))
    expr_cached = (
        pg_t.group_by("playerID")
        .size()
        .order_by("playerID")
        .cache(storage=SourceStorage(source=con))
    )
    dt = pg_t.op()
    (storage, uncached) = (expr_cached.ls.storage, expr_cached.ls.uncached_one)

    # assert initial state
    assert not storage.exists(uncached)
    n_scans_before = get_postgres_n_scans(dt)
    assert n_scans_before == 0

    # assert first execution state
    expr_cached.execute()
    n_scans_after = assert_n_scans_changes(dt, n_scans_before)
    # should we test that SourceStorage.get is called?
    assert n_scans_after == 1
    assert storage.exists(uncached)

    # assert no change after re-execution of cached expr
    expr_cached.execute()
    assert n_scans_after == get_postgres_n_scans(dt)

    # assert cache invalidation happens
    modify_postgres_table(dt)
    expr_cached.execute()
    assert_n_scans_changes(dt, n_scans_after)


def test_postgres_snapshot(pg, con):
    def modify_postgres_table(dt):
        (con, name) = (dt.source, dt.name)
        statement = f"""
        INSERT INTO "{name}"
        DEFAULT VALUES
        """
        con.raw_sql(statement)

    def assert_n_scans_changes(dt, n_scans_before):
        do_analyze(dt.source, dt.name)
        for _ in range(10):  # noqa: F402
            # give postgres some time to update its tables
            time.sleep(0.1)
            n_scans_after = get_postgres_n_scans(dt)
            if n_scans_before != n_scans_after:
                return n_scans_after
        else:
            raise

    (from_name, to_name) = ("batting", "batting_to_modify")
    if to_name in pg.tables:
        pg.drop_table(to_name)
    pg_t = pg.create_table(name=to_name, obj=pg.table(from_name))
    storage = SourceSnapshotStorage(source=con)
    expr_cached = (
        pg_t.group_by("playerID").size().order_by("playerID").cache(storage=storage)
    )
    dt = pg_t.op()
    (storage, uncached) = (expr_cached.ls.storage, expr_cached.ls.uncached_one)

    # assert initial state
    assert not storage.exists(uncached)
    n_scans_before = get_postgres_n_scans(dt)
    assert n_scans_before == 0

    # assert first execution state
    executed0 = expr_cached.execute()
    n_scans_after = assert_n_scans_changes(dt, n_scans_before)
    # should we test that SourceStorage.get is called?
    assert n_scans_after == 1
    assert storage.exists(uncached)

    # assert no change after re-execution of cached expr
    executed1 = expr_cached.execute()
    assert n_scans_after == get_postgres_n_scans(dt)
    assert executed0.equals(executed1)

    # assert NO cache invalidation
    modify_postgres_table(dt)
    executed2 = expr_cached.execute()
    assert executed0.equals(executed2)
    with pytest.raises(Exception):
        assert_n_scans_changes(dt, n_scans_after)

    executed3 = expr_cached.ls.uncached.execute()
    assert not executed0.equals(executed3)


def test_postgres_parquet_snapshot(pg, tmp_path):
    def modify_postgres_table(dt):
        (con, name) = (dt.source, dt.name)
        statement = f"""
        INSERT INTO "{name}"
        DEFAULT VALUES
        """
        con.raw_sql(statement)

    def assert_n_scans_changes(dt, n_scans_before):
        do_analyze(dt.source, dt.name)
        for _ in range(10):  # noqa: F402
            # give postgres some time to update its tables
            time.sleep(0.1)
            n_scans_after = get_postgres_n_scans(dt)
            if n_scans_before != n_scans_after:
                return n_scans_after
        else:
            raise

    (from_name, to_name) = ("batting", "batting_to_modify")
    if to_name in pg.tables:
        pg.drop_table(to_name)
    pg_t = pg.create_table(name=to_name, obj=pg.table(from_name))
    storage = ParquetSnapshotStorage(path=tmp_path.joinpath("parquet-snapshot-storage"))
    expr = pg_t.group_by("playerID").size().order_by("playerID")
    expr_cached = expr.cache(storage=storage)
    dt = pg_t.op()
    (storage, uncached) = (expr_cached.ls.storage, expr_cached.ls.uncached_one)

    # assert initial state
    assert not storage.exists(uncached)
    n_scans_before = get_postgres_n_scans(dt)
    assert n_scans_before == 0

    # assert first execution state
    executed0 = expr_cached.execute()
    n_scans_after = assert_n_scans_changes(dt, n_scans_before)
    # should we test that SourceStorage.get is called?
    assert n_scans_after == 1
    assert storage.exists(uncached)
    assert storage.exists(expr)

    # assert no change after re-execution of cached expr
    executed1 = expr_cached.execute()
    assert n_scans_after == get_postgres_n_scans(dt)
    assert executed0.equals(executed1)

    # assert NO cache invalidation
    modify_postgres_table(dt)
    executed2 = expr_cached.execute()
    assert executed0.equals(executed2)
    with pytest.raises(Exception):
        assert_n_scans_changes(dt, n_scans_after)

    executed3 = expr_cached.ls.uncached.execute()
    assert not executed0.equals(executed3)


def test_duckdb_cache_parquet(con, pg, tmp_path):
    name = "batting"
    parquet_path = tmp_path.joinpath(name).with_suffix(".parquet")
    pg.table(name).to_parquet(parquet_path)
    expr = (
        xo.duckdb.connect()
        .read_parquet(parquet_path)[lambda t: t.yearID > 2000]
        .cache(storage=ParquetStorage(source=con, path=tmp_path))
    )
    expr.execute()


def test_duckdb_cache_csv(con, pg, tmp_path):
    name = "batting"
    csv_path = tmp_path.joinpath(name).with_suffix(".csv")
    pg.table(name).to_csv(csv_path)
    expr = (
        xo.duckdb.connect()
        .read_csv(csv_path)[lambda t: t.yearID > 2000]
        .cache(storage=ParquetStorage(source=con, path=tmp_path))
    )
    expr.execute()


def test_duckdb_cache_arrow(con, pg, tmp_path):
    name = "batting"
    expr = (
        xo.duckdb.connect()
        .create_table(name, pg.table(name).to_pyarrow())[lambda t: t.yearID > 2000]
        .cache(storage=ParquetStorage(source=con, path=tmp_path))
    )
    expr.execute()


def test_cross_source_storage(pg):
    name = "batting"
    expr = (
        xo.duckdb.connect()
        .create_table(name, pg.table(name).to_pyarrow())[lambda t: t.yearID > 2000]
        .cache(storage=SourceStorage(source=pg))
    )
    expr.execute()


def test_caching_of_registered_arbitrary_expression(con, pg, tmp_path):
    table_name = "batting"
    t = pg.table(table_name)

    expr = t.filter(t.playerID == "allisar01")[
        ["playerID", "yearID", "stint", "teamID", "lgID"]
    ]
    expected = expr.execute()

    result = expr.cache(storage=ParquetStorage(source=con, path=tmp_path)).execute()

    assert result is not None
    assert_frame_equal(result, expected, check_like=True)


def test_read_parquet_and_cache(con, parquet_dir, tmp_path):
    batting_path = parquet_dir / "batting.parquet"
    t = con.read_parquet(batting_path, table_name=f"parquet_batting-{uuid.uuid4()}")
    expr = t.cache(storage=ParquetStorage(source=con, path=tmp_path))
    assert expr.execute() is not None


def test_read_parquet_compute_and_cache(con, parquet_dir, tmp_path):
    batting_path = parquet_dir / "batting.parquet"
    t = con.read_parquet(batting_path, table_name=f"parquet_batting-{uuid.uuid4()}")
    expr = (
        t[t.yearID == 2015]
        .cache(storage=ParquetStorage(source=con, path=tmp_path))
        .cache()
    )
    assert expr.execute() is not None


def test_read_csv_and_cache(ls_con, csv_dir, tmp_path):
    batting_path = csv_dir / "batting.csv"
    t = ls_con.read_csv(batting_path, table_name=f"csv_batting-{uuid.uuid4()}")
    expr = t.cache(storage=ParquetStorage(source=ls_con, path=tmp_path))
    assert expr.execute() is not None


def test_read_csv_compute_and_cache(ls_con, csv_dir, tmp_path):
    batting_path = csv_dir / "batting.csv"
    t = ls_con.read_csv(
        batting_path,
        table_name=f"csv_batting-{uuid.uuid4()}",
        schema_infer_max_records=50_000,
    )
    expr = (
        t[t.yearID == 2015]
        .cache(storage=ParquetStorage(source=ls_con, path=tmp_path))
        .cache()
    )
    assert expr.execute() is not None


@pytest.mark.parametrize("other_con", [xo.connect(), xo.duckdb.connect()])
def test_multi_engine_cache(pg, ls_con, tmp_path, other_con):
    table_name = "batting"
    pg_t = pg.table(table_name)[lambda t: t.yearID > 2014]
    db_t = other_con.create_table(
        f"db-{table_name}", pg.table(table_name).to_pyarrow()
    )[lambda t: t.stint == 1].pipe(into_backend, pg)

    expr = pg_t.join(
        db_t,
        db_t.columns,
    ).cache(
        storage=ParquetStorage(
            source=ls_con,
            path=tmp_path,
        )
    )

    assert expr.execute() is not None


def test_repeated_cache(pg, ls_con, tmp_path):
    storage = ParquetStorage(
        source=ls_con,
        path=tmp_path,
    )
    t = (
        pg.table("batting")[lambda t: t.yearID > 2014]
        .cache(storage=storage)[lambda t: t.stint == 1]
        .cache(storage=storage)
    )

    actual = t.execute()
    expected = pg.table("batting").filter([_.yearID > 2014, _.stint == 1]).execute()

    assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    "get_expr",
    [
        lambda t: t,
        lambda t: t.group_by("playerID").agg(t.stint.max().name("n-stints")),
    ],
)
def test_register_with_different_name_and_cache(csv_dir, get_expr):
    batting_path = csv_dir.joinpath("batting.csv")
    table_name = "batting"

    datafusion_con = xo.datafusion.connect()
    xorq_table_name = f"{datafusion_con.name}_{table_name}"
    t = datafusion_con.read_csv(
        batting_path, table_name=table_name, schema_infer_max_records=50_000
    )
    expr = t.pipe(get_expr).cache()

    assert table_name != xorq_table_name
    assert expr.execute() is not None


def test_cache_default_path_set(pg, ls_con, tmp_path):
    xo.options.cache.default_path = tmp_path

    storage = ParquetStorage(
        source=ls_con,
    )

    expr = (
        pg.table("batting")[lambda t: t.yearID > 2014].limit(1).cache(storage=storage)
    )

    result = expr.execute()

    cache_files = list(
        path
        for path in tmp_path.iterdir()
        if path.is_file()
        and path.name.startswith(KEY_PREFIX)
        and path.name.endswith(".parquet")
    )

    assert result is not None
    assert cache_files


def test_pandas_snapshot(ls_con, alltypes_df):
    group_by = "year"
    name = ibis.util.gen_name("tmp_table")

    # create a temp table we can mutate
    pd_con = xo.pandas.connect()
    table = pd_con.create_table(name, alltypes_df)

    cached_expr = (
        table.group_by(group_by)
        .agg({f"count_{col}": table[col].count() for col in table.columns})
        .pipe(into_backend, ls_con)
        .cache(storage=SourceSnapshotStorage(source=ls_con))
    )
    (storage, uncached) = get_storage_uncached(cached_expr)

    # test preconditions
    assert not storage.exists(uncached)

    # test cache creation
    executed0 = cached_expr.execute()

    with storage.normalization_context(uncached):
        normalized0 = dask.base.normalize_token(uncached)
    assert storage.exists(uncached)

    # test cache use
    executed1 = cached_expr.execute()
    assert executed0.equals(executed1)

    # test NO cache invalidation
    pd_con.reconnect()
    table2 = pd_con.create_table(name, pd.concat((alltypes_df, alltypes_df)))

    cached_expr = (
        table2.group_by(group_by)
        .agg({f"count_{col}": table2[col].count() for col in table2.columns})
        .pipe(into_backend, ls_con)
        .cache(storage)
    )
    (storage, uncached) = get_storage_uncached(cached_expr)
    with storage.normalization_context(uncached):
        normalized1 = dask.base.normalize_token(uncached)

    # the "stable rename" occurs inside of SnapshotStrategy.get_key
    assert normalized0[1][1][0] != normalized1[1][1][0]
    # everything else is stable, despite the different date
    assert normalized0[1][1][1:] == normalized1[1][1][1:]
    assert storage.exists(uncached)
    executed2 = cached_expr.ls.uncached.execute()
    assert not executed0.equals(executed2)


def test_duckdb_snapshot(ls_con, alltypes_df):
    group_by = "year"
    name = ibis.util.gen_name("tmp_table")

    # create a temp table we can mutate
    db_con = xo.duckdb.connect()
    table = db_con.create_table(name, alltypes_df)

    cached_expr = (
        table.group_by(group_by)
        .agg({f"count_{col}": table[col].count() for col in table.columns})
        .cache(storage=SourceSnapshotStorage(source=ls_con))
    )
    (storage, uncached) = get_storage_uncached(cached_expr)

    # test preconditions
    assert not storage.exists(uncached)

    # test cache creation
    executed0 = cached_expr.execute()
    assert storage.exists(uncached)

    # test cache use
    executed1 = cached_expr.execute()
    assert executed0.equals(executed1)

    # test NO cache invalidation
    db_con.insert(name, alltypes_df)
    executed2 = cached_expr.execute()
    executed3 = cached_expr.ls.uncached.execute()
    assert executed0.equals(executed2)
    assert not executed0.equals(executed3)


def test_datafusion_snapshot(ls_con, alltypes_df):
    group_by = "year"
    name = ibis.util.gen_name("tmp_table")

    # create a temp table we can mutate
    df_con = xo.datafusion.connect()
    table = df_con.create_table(name, alltypes_df)

    cached_expr = (
        table.group_by(group_by)
        .agg({f"count_{col}": table[col].count() for col in table.columns})
        .cache(storage=SourceSnapshotStorage(source=ls_con))
    )
    (storage, uncached) = get_storage_uncached(cached_expr)

    # test preconditions
    assert not storage.exists(uncached)

    # test cache creation
    executed0 = cached_expr.execute()
    assert storage.exists(uncached)

    # test cache use
    executed1 = cached_expr.execute()
    assert executed0.equals(executed1)

    # test NO cache invalidation
    df_con.insert(name, alltypes_df)
    executed2 = cached_expr.execute()
    executed3 = cached_expr.ls.uncached.execute()
    assert executed0.equals(executed2)
    assert not executed0.equals(executed3)


@pytest.mark.xfail
def test_udf_caching(ls_con, alltypes_df, snapshot):
    @ibis.udf.scalar.pyarrow
    def my_mul(tinyint_col: dt.int16, smallint_col: dt.int16) -> dt.int16:
        return pc.multiply(tinyint_col, smallint_col)

    @toolz.curry
    def wrapper(f, t):
        # here: map pandas into pyarrow
        # goal: map pyarrow into pandas: so users can define functions in pandas land
        inner = f.__wrapped__
        kwargs = {k: pa.array(v) for k, v in t.items()}
        return inner(**kwargs)

    cols = list(inspect.signature(my_mul).parameters)

    expr = (
        ls_con.create_table("alltypes", alltypes_df)[cols]
        .pipe(lambda t: t.mutate(mulled=my_mul(*(t[col] for col in cols))))
        .cache()
    )
    from_ls = expr.execute()
    from_pandas = alltypes_df[cols].assign(mulled=wrapper(my_mul))
    assert from_ls.equals(from_pandas)

    py_version = f"py{get_python_version_no_dot()}"
    snapshot.assert_match(expr.ls.get_key(), f"{py_version}_udf_caching.txt")


@pytest.mark.xfail
def test_udaf_caching(ls_con, alltypes_df, snapshot):
    def my_mul_sum(df):
        return df.sum().sum()

    cols = ["tinyint_col", "smallint_col"]
    ibis_output_type = dt.infer(alltypes_df[cols].sum().sum())
    by = "bool_col"
    name = "my_mul_sum"

    t = ls_con.create_table("alltypes", alltypes_df)
    agg_udf = agg.pandas_df(
        my_mul_sum,
        t[cols].schema(),
        ibis_output_type,
        name=name,
    )
    from_pandas = (
        alltypes_df.groupby(by)[cols]
        .apply(my_mul_sum)
        .rename(name)
        .reset_index()
        .sort_values(by)
    )
    expr = (
        t.group_by(by)
        .agg(**{name: agg_udf(*(t[col] for col in cols))})
        .order_by(by)
        .cache()
    )
    on_expr = t.group_by(by).agg(**{name: agg_udf.on_expr(t)}).order_by(by).cache()
    assert not expr.ls.exists()
    assert not on_expr.ls.exists()

    from_ls = expr.execute().sort_values(by="bool_col")
    assert_frame_equal(from_ls, from_pandas.sort_values(by="bool_col"))
    assert_frame_equal(from_ls, on_expr.execute().sort_values(by="bool_col"))
    assert expr.ls.exists()
    assert on_expr.ls.exists()

    py_version = f"py{get_python_version_no_dot()}"
    snapshot.assert_match(expr.ls.get_key(), f"{py_version}_test_udaf_caching.txt")
    snapshot.assert_match(on_expr.ls.get_key(), f"{py_version}_test_udaf_caching.txt")


def test_caching_pandas(csv_dir):
    diamonds_path = csv_dir / "diamonds.csv"
    pandas_con = xo.pandas.connect()
    cache = SourceStorage(source=pandas_con)
    t = pandas_con.read_csv(diamonds_path).cache(storage=cache)
    assert t.execute() is not None


@pytest.mark.parametrize("expr_con", [xo.datafusion.connect(), xo.duckdb.connect()])
def test_cross_source_snapshot(ls_con, alltypes_df, expr_con):
    group_by = "year"
    name = ibis.util.gen_name("tmp_table")

    # create a temp table we can mutate
    table = expr_con.create_table(name, alltypes_df)

    storage = ParquetSnapshotStorage(source=ls_con)

    expr = table.group_by(group_by).agg(
        {f"count_{col}": table[col].count() for col in table.columns}
    )

    cached_expr = expr.cache(storage=storage)

    # test preconditions
    assert not storage.exists(expr)  # the expr is not cached
    assert storage.source is not expr_con  # the cache is cross source

    # test cache creation
    df = cached_expr.execute()

    assert not df.empty
    assert storage.exists(expr)

    # test cache use
    executed1 = cached_expr.execute()
    assert df.equals(executed1)

    # test NO cache invalidation
    expr_con.insert(name, alltypes_df)
    executed2 = cached_expr.execute()
    executed3 = cached_expr.ls.uncached.execute()
    assert df.equals(executed2)
    assert not df.equals(executed3)


def test_storage_exists_doesnt_materialize(cached_two):
    storage = cached_two.ls.storage
    assert not storage.exists(cached_two)
    assert not storage.exists(cached_two)
    assert not cached_two.ls.exists()
    cached_two.count().execute()
    assert cached_two.ls.exists()
    assert storage.exists(cached_two)


def test_ls_exists_doesnt_materialize(cached_two):
    storage = cached_two.ls.storage
    assert not cached_two.ls.exists()
    assert not cached_two.ls.exists()
    assert not storage.exists(cached_two)
    cached_two.count().execute()
    assert cached_two.ls.exists()
    assert storage.exists(cached_two)


@pytest.mark.parametrize(
    "con", [xo.connect(), xo.datafusion.connect(), xo.duckdb.connect()]
)
@pytest.mark.parametrize(
    "cls",
    [ParquetSnapshotStorage, ParquetStorage, SourceSnapshotStorage, SourceStorage],
)
def test_cache_find_backend(con, cls):
    storage = cls(source=con)
    expr = con.read_parquet(xo.options.pins.get_path("astronauts")).cache(
        storage=storage
    )
    assert expr._find_backend().name == "let"
