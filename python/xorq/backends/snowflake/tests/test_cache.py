from __future__ import annotations

import re

import pandas as pd
import pytest

import xorq as xo
import xorq.vendor.ibis.expr.operations as ops
from xorq.backends.conftest import (
    get_storage_uncached,
)
from xorq.backends.snowflake.tests.conftest import (
    inside_temp_schema,
)
from xorq.caching import (
    ParquetStorage,
    SourceSnapshotStorage,
)
from xorq.common.utils.snowflake_utils import (
    get_session_query_df,
    get_snowflake_last_modification_time,
)
from xorq.vendor import ibis
from xorq.vendor.ibis.util import gen_name


@pytest.mark.snowflake
def test_snowflake_cache_with_name_multiplicity(sf_con):
    (catalog, db) = ("SNOWFLAKE_SAMPLE_DATA", "TPCH_SF1")
    assert sf_con.current_catalog == catalog
    assert sf_con.current_database == db
    table = "CUSTOMER"
    n_tables = (
        sf_con.table("TABLES", database=(catalog, "INFORMATION_SCHEMA"))[
            lambda t: t.TABLE_NAME == table
        ]
        .count()
        .execute()
    )
    assert n_tables > 1
    t = sf_con.table(table)
    (dt,) = t.op().find(ops.DatabaseTable)
    get_snowflake_last_modification_time(dt)


@pytest.mark.snowflake
def test_snowflake_cache_invalidation(sf_con, temp_catalog, temp_db, tmp_path):
    group_by = "key"
    df = pd.DataFrame({group_by: list("abc"), "value": [1, 2, 3]})
    name = gen_name("tmp_table")
    con = xo.connect()
    storage = ParquetStorage(source=con, path=tmp_path)

    # must explicitly invoke USE SCHEMA: use of temp_* DOESN'T impact internal create_table's CREATE TEMP STAGE
    with inside_temp_schema(sf_con, temp_catalog, temp_db):
        table = sf_con.create_table(
            name=name,
            obj=df,
        )
        uncached = table.group_by(group_by).agg(
            {f"min_{col}": table[col].min() for col in table.columns}
        )
        cached_expr = uncached.cache(storage)
        (storage, _) = get_storage_uncached(cached_expr)
        unbound_sql = re.sub(
            r"\s+",
            " ",
            ibis.to_sql(uncached, dialect=sf_con.name),
        )
        query_df = get_session_query_df(sf_con)

        # test preconditions
        assert not storage.exists(uncached)
        assert query_df.QUERY_TEXT.eq(unbound_sql).sum() == 0

        # test cache creation
        xo.execute(cached_expr)
        query_df = get_session_query_df(sf_con)
        assert storage.exists(uncached)
        assert query_df.QUERY_TEXT.eq(unbound_sql).sum() == 1

        # test cache use
        xo.execute(cached_expr)
        assert query_df.QUERY_TEXT.eq(unbound_sql).sum() == 1

        # test cache invalidation
        sf_con.insert(name, df, database=f"{temp_catalog}.{temp_db}")
        assert not storage.exists(uncached)


@pytest.mark.snowflake
def test_snowflake_simple_cache(sf_con, tmp_path):
    db_con = xo.duckdb.connect()
    with inside_temp_schema(sf_con, "SNOWFLAKE_SAMPLE_DATA", "TPCH_SF1"):
        table = sf_con.table("CUSTOMER")
        expr = table.limit(1).cache(ParquetStorage(source=db_con, path=tmp_path))
        xo.execute(expr)


@pytest.mark.snowflake
def test_snowflake_native_cache(sf_con, temp_catalog, temp_db, tmp_path):
    group_by = "key"
    df = pd.DataFrame({group_by: list("abc"), "value": [1, 2, 3]})
    name = gen_name("tmp_table")
    storage = ParquetStorage(source=sf_con, path=tmp_path)

    # must explicitly invoke USE SCHEMA: use of temp_* DOESN'T impact internal create_table's CREATE TEMP STAGE
    with inside_temp_schema(sf_con, temp_catalog, temp_db):
        # create a temp table we can mutate
        table = sf_con.create_table(
            name=name,
            obj=df,
        )
        cached_expr = (
            table.group_by(group_by)
            .agg({f"count_{col}": table[col].count() for col in table.columns})
            .cache(storage)
        )
        xo.execute(cached_expr)


@pytest.mark.snowflake
def test_snowflake_snapshot(sf_con, temp_catalog, temp_db):
    group_by = "key"
    df = pd.DataFrame({group_by: list("abc"), "value": [1, 2, 3]})
    name = gen_name("tmp_table")
    storage = SourceSnapshotStorage(source=xo.duckdb.connect())

    # must explicitly invoke USE SCHEMA: use of temp_* DOESN'T impact internal create_table's CREATE TEMP STAGE
    with inside_temp_schema(sf_con, temp_catalog, temp_db):
        # create a temp table we can mutate
        table = sf_con.create_table(
            name=name,
            obj=df,
        )
        uncached = table.group_by(group_by).agg(
            {f"count_{col}": table[col].count() for col in table.columns}
        )
        cached_expr = uncached.cache(storage)
        (storage, _) = get_storage_uncached(cached_expr)
        unbound_sql = re.sub(
            r"\s+",
            " ",
            ibis.to_sql(uncached, dialect=sf_con.name),
        )
        query_df = get_session_query_df(sf_con)

        # test preconditions
        assert not storage.exists(uncached)
        assert query_df.QUERY_TEXT.eq(unbound_sql).sum() == 0

        # test cache creation
        executed0 = xo.execute(cached_expr)
        query_df = get_session_query_df(sf_con)
        assert storage.exists(uncached)
        assert query_df.QUERY_TEXT.eq(unbound_sql).sum() == 1

        # test cache use
        executed1 = xo.execute(cached_expr)
        assert query_df.QUERY_TEXT.eq(unbound_sql).sum() == 1
        assert executed0.equals(executed1)

        # test NO cache invalidation
        sf_con.insert(name, df, database=f"{temp_catalog}.{temp_db}")
        (storage, uncached) = get_storage_uncached(cached_expr)
        assert storage.exists(uncached)
        executed2 = xo.execute(cached_expr.ls.uncached)
        assert not executed0.equals(executed2)
