from __future__ import annotations

import re

import ibis
import pandas as pd
import pytest
from ibis.util import gen_name

import letsql as ls
from letsql.backends.conftest import (
    get_storage_uncached,
)
from letsql.backends.snowflake.tests.conftest import (
    inside_temp_schema,
)
from letsql.common.utils.snowflake_utils import (
    get_session_query_df,
)


@pytest.mark.snowflake
def test_snowflake_cache_invalidation(sf_con, temp_catalog, temp_db, tmp_path):
    group_by = "key"
    df = pd.DataFrame({group_by: list("abc"), "value": [1, 2, 3]})
    name = gen_name("tmp_table")
    con = ls.connect()
    storage = ls.common.caching.ParquetCacheStorage(tmp_path, source=con)

    # must explicitly invoke USE SCHEMA: use of temp_* DOESN'T impact internal create_table's CREATE TEMP STAGE
    with inside_temp_schema(sf_con, temp_catalog, temp_db):
        table = sf_con.create_table(
            name=name,
            obj=df,
        )
        t = con.register(table, f"let_{table.op().name}")
        cached_expr = (
            t.group_by(group_by)
            .agg({f"min_{col}": t[col].min() for col in t.columns})
            .cache(storage)
        )
        (storage, uncached) = get_storage_uncached(con, cached_expr)
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
        cached_expr.execute()
        query_df = get_session_query_df(sf_con)
        assert storage.exists(uncached)
        assert query_df.QUERY_TEXT.eq(unbound_sql).sum() == 1

        # test cache use
        cached_expr.execute()
        assert query_df.QUERY_TEXT.eq(unbound_sql).sum() == 1

        # test cache invalidation
        sf_con.insert(name, df, database=f"{temp_catalog}.{temp_db}")
        assert not storage.exists(uncached)
