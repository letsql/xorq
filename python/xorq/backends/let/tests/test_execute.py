import itertools
import random
from operator import methodcaller
from pathlib import Path, PosixPath

import numpy as np
import pandas as pd
import pytest
from pytest import param

import xorq as xo
from xorq.caching import SourceStorage
from xorq.tests.util import (
    assert_frame_equal,
)
from xorq.vendor import ibis


KEY_PREFIX = xo.config.options.cache.key_prefix


def _pandas_semi_join(left, right, on, **_):
    assert len(on) == 1, str(on)
    inner = pd.merge(left, right, how="inner", on=on)
    filt = left.loc[:, on[0]].isin(inner.loc[:, on[0]])
    return left.loc[filt, :]


def _pandas_anti_join(left, right, on, **_):
    inner = pd.merge(left, right, how="left", indicator=True, on=on)
    return inner[inner["_merge"] == "left_only"]


IMPLS = {
    "semi": _pandas_semi_join,
    "anti": _pandas_anti_join,
}


def check_eq(left, right, how, **kwargs):
    impl = IMPLS.get(how, pd.merge)
    return impl(left, right, how=how, **kwargs)


@pytest.fixture
def union_subsets(alltypes, alltypes_df):
    cols_a, cols_b, cols_c = (alltypes.columns.copy() for _ in range(3))

    random.seed(89)
    random.shuffle(cols_a)
    random.shuffle(cols_b)
    random.shuffle(cols_c)
    assert cols_a != cols_b != cols_c

    cols_a = [ca for ca in cols_a if ca != "timestamp_col"]
    cols_b = [cb for cb in cols_b if cb != "timestamp_col"]
    cols_c = [cc for cc in cols_c if cc != "timestamp_col"]

    a = alltypes.filter((alltypes.id >= 5200) & (alltypes.id <= 5210))[cols_a]
    b = alltypes.filter((alltypes.id >= 5205) & (alltypes.id <= 5215))[cols_b]
    c = alltypes.filter((alltypes.id >= 5213) & (alltypes.id <= 5220))[cols_c]

    da = alltypes_df[(alltypes_df.id >= 5200) & (alltypes_df.id <= 5210)][cols_a]
    db = alltypes_df[(alltypes_df.id >= 5205) & (alltypes_df.id <= 5215)][cols_b]
    dc = alltypes_df[(alltypes_df.id >= 5213) & (alltypes_df.id <= 5220)][cols_c]

    return (a, b, c), (da, db, dc)


@pytest.fixture(scope="session")
def csv_dir():
    root = Path(__file__).absolute().parents[5]
    data_dir = root / "ci" / "ibis-testing-data" / "csv"
    return data_dir


@pytest.fixture(scope="session")
def parquet_dir():
    root = Path(__file__).absolute().parents[5]
    data_dir = root / "ci" / "ibis-testing-data" / "parquet"
    return data_dir


@pytest.fixture(scope="session")
def dirty_duckdb_con(csv_dir):
    con = xo.duckdb.connect()
    con.read_csv(csv_dir / "awards_players.csv", table_name="ddb_players")
    con.read_csv(csv_dir / "batting.csv", table_name="batting")
    return con


@pytest.fixture(scope="function")
def duckdb_con(dirty_duckdb_con):
    from duckdb import CatalogException

    expected_tables = ("ddb_players", "batting")
    for table in dirty_duckdb_con.list_tables():
        if table not in expected_tables:
            try:
                dirty_duckdb_con.drop_view(table, force=True)
            except CatalogException:
                dirty_duckdb_con.drop_table(table, force=True)
    yield dirty_duckdb_con


@pytest.fixture(scope="function")
def pg_batting(pg):
    return pg.table("batting")


@pytest.fixture(scope="function")
def parquet_batting(parquet_dir):
    return parquet_dir / "batting.parquet"


@pytest.fixture(scope="function")
def ls_batting(parquet_batting):
    return xo.connect().read_parquet(parquet_batting)


@pytest.fixture(scope="function")
def ddb_batting(duckdb_con):
    return duckdb_con.create_table(
        "db-batting",
        duckdb_con.table("batting").to_pyarrow(),
    )


def test_join(ls_con, alltypes, alltypes_df):
    first_10 = alltypes_df.head(10)
    in_memory = ls_con.create_table("in_memory", first_10).into_backend(
        alltypes.op().source
    )
    expr = alltypes.join(in_memory, predicates=[alltypes.id == in_memory.id])
    actual = expr.execute().sort_values("id")
    expected = pd.merge(
        alltypes_df, first_10, how="inner", on="id", suffixes=("", "_right")
    ).sort_values("id")

    assert_frame_equal(actual, expected)


@pytest.mark.parametrize("how", ["semi", "anti"])
def test_filtering_join(batting, awards_players, how):
    left = batting[batting.yearID == 2015]
    right = awards_players[awards_players.lgID == "NL"].drop("yearID", "lgID")

    left_df = left.execute()
    right_df = right.execute()
    predicate = ["playerID"]
    result_order = ["playerID", "yearID", "lgID", "stint"]

    expr = left.join(right, predicate, how=how)
    result = (
        expr.execute()
        .fillna(np.nan)
        .sort_values(result_order)[left.columns]
        .reset_index(drop=True)
    )

    expected = check_eq(
        left_df,
        right_df,
        how=how,
        on=predicate,
        suffixes=("", "_y"),
    ).sort_values(result_order)[list(left.columns)]

    assert_frame_equal(result, expected, check_like=True)


@pytest.mark.parametrize("distinct", [False, True], ids=["all", "distinct"])
def test_union(ls_con, union_subsets, distinct):
    (a, _, _), (da, db, dc) = union_subsets

    b = ls_con.create_table("b", db)
    expr = ibis.union(a.into_backend(ls_con), b, distinct=distinct).order_by("id")
    result = expr.execute()

    expected = pd.concat([da, db], axis=0).sort_values("id").reset_index(drop=True)

    if distinct:
        expected = expected.drop_duplicates("id")

    assert_frame_equal(result, expected)


def test_union_mixed_distinct(ls_con, union_subsets):
    (a, _, _), (da, db, dc) = union_subsets

    a = a.into_backend(ls_con)
    b = ls_con.create_table("b", db)
    c = ls_con.create_table("c", dc)

    expr = a.union(b, distinct=True).union(c, distinct=False).order_by("id")
    result = expr.execute()
    expected = pd.concat(
        [pd.concat([da, db], axis=0).drop_duplicates("id"), dc], axis=0
    ).sort_values("id")

    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "left_filter",
    [
        lambda t: t.yearID == 2015,
        lambda t: t.yearID != 2015,
        lambda t: t.yearID >= 2015,
        lambda t: t.yearID >= 2014.5,
        lambda t: t.yearID <= 2015,
        lambda t: t.yearID > 2015,
        lambda t: t.yearID < 2015,
        lambda t: ~(t.yearID < 2015),
        lambda t: t.yearID.notnull(),
        lambda t: t.yearID.isnull(),
        lambda t: t.playerID == "wilsobo02",
        lambda t: t.yearID.isin([2015, 2014]),
        lambda t: t.yearID.notin([2015, 2014]),
        lambda t: t.yearID.between(2013, 2016),
    ],
)
def test_join_non_trivial_filters(pg, duckdb_con, left_filter):
    awards_players = duckdb_con.table("ddb_players")
    batting = pg.table("batting").into_backend(duckdb_con)

    left = batting[left_filter]
    right = awards_players[awards_players.lgID == "NL"].drop("yearID", "lgID")
    right_df = right.execute()
    left_df = left.execute()
    predicate = "playerID"
    result_order = ["playerID", "yearID", "lgID", "stint"]

    expr = left.join(right, predicate, how="inner")
    result = (
        expr.execute()
        .fillna(np.nan)[left.columns]
        .sort_values(result_order)
        .reset_index(drop=True)
    )

    expected = (
        check_eq(
            left_df,
            right_df,
            how="inner",
            on=predicate,
            suffixes=("_x", "_y"),
        )[left.columns]
        .sort_values(result_order)
        .reset_index(drop=True)
    )

    assert_frame_equal(result, expected, check_like=True)


@pytest.mark.parametrize(
    ("predicate", "pandas_value"),
    [
        # True
        param(True, True, id="true"),
        param(ibis.literal(True), True, id="true-literal"),
        param([True], True, id="true-list"),
        param([ibis.literal(True)], True, id="true-literal-list"),
        # only True
        param([True, True], True, id="true-true-list"),
        param(
            [ibis.literal(True), ibis.literal(True)], True, id="true-true-literal-list"
        ),
        param([True, ibis.literal(True)], True, id="true-true-const-expr-list"),
        param([ibis.literal(True), True], True, id="true-true-expr-const-list"),
        # False
        param(False, False, id="false"),
        param(ibis.literal(False), False, id="false-literal"),
        param([False], False, id="false-list"),
        param([ibis.literal(False)], False, id="false-literal-list"),
        # only False
        param([False, False], False, id="false-false-list"),
        param(
            [ibis.literal(False), ibis.literal(False)],
            False,
            id="false-false-literal-list",
        ),
        param([False, ibis.literal(False)], False, id="false-false-const-expr-list"),
        param([ibis.literal(False), False], False, id="false-false-expr-const-list"),
    ],
)
@pytest.mark.parametrize(
    "how",
    [
        "inner",
        "left",
        "right",
        "outer",
    ],
)
def test_join_with_trivial_predicate(
    duckdb_con, awards_players, predicate, how, pandas_value
):
    ddb_players = duckdb_con.table("ddb_players")

    n = 5

    base = awards_players.into_backend(duckdb_con).limit(n)
    ddb_base = ddb_players.limit(n)

    left = base.select(left_key="playerID")
    right = ddb_base.select(right_key="playerID")

    left_df = pd.DataFrame({"key": [True] * n})
    right_df = pd.DataFrame({"key": [pandas_value] * n})

    expected = pd.merge(left_df, right_df, on="key", how=how)

    expr = left.join(right, predicate, how=how)
    result = expr.execute()

    assert len(result) == len(expected)


def test_sql_execution(ls_con, duckdb_con, awards_players, batting):
    def make_right(t):
        return t[t.lgID == "NL"].drop("yearID", "lgID")

    ddb_players = ls_con.register(
        duckdb_con.table("ddb_players"), table_name="ddb_players"
    )

    left = batting[batting.yearID == 2015]
    right_df = make_right(awards_players).execute()
    left_df = xo.execute(left)
    predicate = ["playerID"]
    result_order = ["playerID", "yearID", "lgID", "stint"]

    expr = ls_con.register(left, "batting").join(
        make_right(ls_con.register(ddb_players, "players")),
        predicate,
        how="inner",
    )
    query = xo.to_sql(expr)

    result = (
        ls_con.sql(query)
        .execute()
        .fillna(np.nan)[left.columns]
        .sort_values(result_order)
        .reset_index(drop=True)
    )

    expected = (
        check_eq(
            left_df,
            right_df,
            how="inner",
            on=predicate,
            suffixes=("_x", "_y"),
        )[left.columns]
        .sort_values(result_order)
        .reset_index(drop=True)
    )

    assert_frame_equal(result, expected, check_like=True)


def test_multiple_execution_letsql_register_table(ls_con, csv_dir):
    table_name = "csv_players"
    t = ls_con.read_csv(csv_dir / "awards_players.csv", table_name=table_name)
    ls_con.register(t, table_name=f"{ls_con.name}_{table_name}")

    assert (first := t.execute()) is not None
    assert (second := t.execute()) is not None
    assert_frame_equal(first, second)


@pytest.mark.parametrize(
    "other_con",
    [
        xo.connect(),
        xo.datafusion.connect(),
        xo.duckdb.connect(),
        xo.postgres.connect(
            host="localhost",
            port=5432,
            user="postgres",
            password="postgres",
            database="ibis_testing",
        ),
    ],
)
def test_expr_over_same_table_multiple_times(ls_con, parquet_dir, other_con):
    batting_path = parquet_dir.joinpath("batting.parquet")
    table_name = "batting"
    col = "playerID"

    batting_name = f"{ls_con.name}_{table_name}"
    batting = ls_con.register(batting_path, table_name=batting_name)

    if other_con.name == "postgres":
        t = other_con.table(table_name)
    else:
        t = other_con.read_parquet(batting_path, table_name=table_name)

    ls_table_name = f"{other_con.name}_{table_name}"
    ls_con.register(t, ls_table_name)
    other = ls_con.table(ls_table_name)

    expr = batting[[col]].distinct().join(other[[col]].distinct(), col)

    assert (first := expr.execute()) is not None
    assert (second := expr.execute()) is not None
    assert_frame_equal(first.sort_values(col), second.sort_values(col))


def test_register_arbitrary_expression(ls_con, duckdb_con):
    batting_table_name = "batting"
    t = duckdb_con.table(batting_table_name)

    expr = t.filter(t.playerID == "allisar01")[
        ["playerID", "yearID", "stint", "teamID", "lgID"]
    ]
    expected = expr.execute()

    ddb_batting_table_name = f"{duckdb_con.name}_{batting_table_name}"
    table = ls_con.register(expr, table_name=ddb_batting_table_name)
    result = table.execute()

    assert result is not None
    assert_frame_equal(result, expected, check_like=True)


def test_arbitrary_expression_multiple_tables(duckdb_con):
    batting_table_name = "batting"
    batting_table = duckdb_con.table(batting_table_name)

    players_table_name = "ddb_players"
    awards_players_table = duckdb_con.table(players_table_name)

    left = batting_table[batting_table.yearID == 2015]
    right = awards_players_table[awards_players_table.lgID == "NL"].drop(
        "yearID", "lgID"
    )

    left_df = left.execute()
    right_df = right.execute()
    predicate = ["playerID"]
    result_order = ["playerID", "yearID", "lgID", "stint"]

    expr = left.join(right, predicate, how="inner")

    result = (
        expr.execute()
        .fillna(np.nan)
        .sort_values(result_order)[left.columns]
        .reset_index(drop=True)
    )

    expected = check_eq(
        left_df,
        right_df,
        how="inner",
        on=predicate,
        suffixes=("", "_y"),
    ).sort_values(result_order)[list(left.columns)]

    assert_frame_equal(result, expected, check_like=True)


@pytest.mark.parametrize(
    "new_con",
    [
        xo.connect(),
        xo.duckdb.connect(),
    ],
)
def test_multiple_pipes(pg, new_con):
    """This test address the issue reported on bug #69
    link: https://github.com/letsql/letsql/issues/69

    NOTE
    The previous tests didn't catch it because the con object registered the table batting.
    In this test (and the rest) ls_con is a clean (no tables) letsql connection
    """

    table_name = "batting"
    pg_t = pg.table(table_name)[lambda t: t.yearID == 2015]
    db_t = new_con.create_table(f"db-{table_name}", pg_t.to_pyarrow())[
        lambda t: t.yearID == 2014
    ]

    expr = pg_t.join(
        db_t.into_backend(pg),
        "playerID",
    )

    assert expr.execute() is not None


@pytest.mark.parametrize(
    "function",
    ["to_pyarrow", "execute", "to_pyarrow_batches"],
)
@pytest.mark.parametrize("remote", [True, False])
def test_duckdb_datafusion_roundtrip(ls_con, pg, duckdb_con, function, remote):
    source = pg if remote else ls_con
    storage = SourceStorage(source=source)

    table_name = "batting"
    pg_t = pg.table(table_name)[lambda t: t.yearID == 2015].cache(storage)

    db_t = duckdb_con.create_table(f"ls-{table_name}", xo.to_pyarrow(pg_t))[
        lambda t: t.yearID == 2014
    ]

    expr = pg_t.join(
        db_t.into_backend(pg if remote else ls_con),
        "playerID",
    )

    assert methodcaller(function)(expr) is not None
    assert any(table_name.startswith(KEY_PREFIX) for table_name in source.list_tables())


def test_to_pyarrow_native_execution(pg, mocker):
    table_name = "batting"
    spy = mocker.spy(pg, "to_pyarrow_batches")

    pg_t = pg.table(table_name)[lambda t: t.yearID == 2015]
    db_t = pg.table(table_name)[lambda t: t.yearID == 2014]

    expr = pg_t.join(
        db_t,
        "playerID",
    )

    assert expr.to_pyarrow() is not None
    assert spy.call_count == 1


@pytest.mark.parametrize(
    "tables",
    [
        param(pair, id="-".join(pair))
        for pair in itertools.combinations_with_replacement(
            ["batting", "pg_batting", "parquet_batting", "ls_batting", "ddb_batting"],
            r=2,
        )
    ],
)
def test_execution_expr_multiple_tables(ls_con, tables, request, mocker):
    left, right = map(request.getfixturevalue, tables)

    left_t = (left if not isinstance(left, PosixPath) else ls_con.read_parquet(left))[
        lambda t: t.yearID == 2015
    ]
    left_backend = left_t._find_backend(use_default=False)

    right_t = (
        right if not isinstance(right, PosixPath) else ls_con.read_parquet(right)
    )[lambda t: t.yearID == 2014]
    right_backend = right_t._find_backend(use_default=False)

    # FIXME there seems to be an issue when doing into_backend from duckdb into duckdb
    expr = left_t.join(
        right_t.into_backend(left_backend)
        if right_backend is not left_backend
        else right_t,
        "playerID",
    )

    native_backend = left_backend is right_backend and left_backend.name != "let"
    spy = mocker.spy(left_backend, "to_pyarrow_batches") if native_backend else None

    assert expr.execute() is not None
    assert getattr(spy, "call_count", 0) == int(native_backend)


@pytest.mark.parametrize(
    "tables",
    [
        param(
            pair,
            id="-".join(pair),
        )
        for pair in itertools.combinations(
            ["pg_batting", "ls_batting", "ddb_batting"],
            r=2,
        )
    ],
)
def test_execution_expr_multiple_tables_cached(ls_con, tables, request):
    from xorq.caching import SourceStorage

    table_name = "batting"
    left, right = map(request.getfixturevalue, tables)
    source = right.op().source

    left_storage = SourceStorage(source=left.op().source)
    right_storage = SourceStorage(source=right.op().source)

    left_t = ls_con.register(left, table_name=f"left-{table_name}")[
        lambda t: t.yearID == 2015
    ].cache(right_storage)

    right_t = ls_con.register(right, table_name=f"right-{table_name}")[
        lambda t: t.yearID == 2014
    ].cache(left_storage)

    actual = (
        left_t.join(
            right_t.into_backend(source),
            "playerID",
        )
        .cache(left_storage)
        .execute()
    )

    expected = (
        ls_con.table(f"left-{table_name}")[lambda t: t.yearID == 2015]
        .join(
            ls_con.table(f"right-{table_name}")[lambda t: t.yearID == 2014], "playerID"
        )
        .execute()
        .sort_values(by="playerID")
    )

    columns = list(actual.columns)
    assert_frame_equal(actual.sort_values(columns), expected.sort_values(columns))


def test_no_registration_same_table_name(ls_con, pg_batting):
    ddb_con = xo.duckdb.connect()
    table = pg_batting[["playerID", "yearID"]].to_pyarrow()
    ddb_batting = ddb_con.create_table("batting", table)
    ls_batting = ls_con.register(table, "batting")

    expr = ddb_batting.join(
        ls_batting.into_backend(ddb_con),
        "playerID",
    )

    assert expr.execute() is not None
