import functools
import itertools
import pathlib

import duckdb
import pandas as pd
import pytest
from adbc_driver_manager import ProgrammingError
from attr import (
    field,
    frozen,
)
from attr.validators import (
    in_,
    optional,
)

import xorq as xo
from xorq.caching import (
    ParquetStorage,
)
from xorq.common.utils.defer_utils import (
    deferred_read_csv,
    deferred_read_parquet,
)
from xorq.common.utils.inspect_utils import (
    get_partial_arguments,
)


@frozen
class PinsResource:
    name = field(validator=in_(xo.options.pins.get_board().pin_list()))
    suffix = field(validator=optional(in_((".csv", ".parquet"))), default=None)

    def __attrs_post_init__(self):
        if self.suffix is None:
            object.__setattr__(self, "suffix", self.path.suffix)
        if self.path.suffix != self.suffix:
            raise ValueError

    @property
    def table_name(self):
        return f"test-{self.name}"

    @property
    @functools.cache
    def path(self):
        return pathlib.Path(xo.options.pins.get_path(self.name))

    def get_underlying_method(self, con):
        return getattr(con, self.deferred_reader.method_name)

    @property
    def deferred_reader(self):
        match self.suffix:
            case ".parquet":
                return deferred_read_parquet
            case ".csv":
                return deferred_read_csv
            case _:
                raise ValueError

    @property
    def immediate_reader(self):
        match self.suffix:
            case ".parquet":
                return pd.read_parquet
            case ".csv":
                return pd.read_csv
            case _:
                raise ValueError

    @property
    @functools.cache
    def df(self):
        return self.immediate_reader(self.path)


@pytest.fixture(scope="session")
def iris_csv():
    return PinsResource(name="iris", suffix=".csv")


@pytest.fixture(scope="session")
def astronauts_parquet():
    return PinsResource(name="astronauts", suffix=".parquet")


def filter_sepal_length(t):
    return t.sepal_length > 5


def filter_field21(t):
    return t.field21 > 2


def ensure_tmp_csv(csv_name, tmp_path):
    source_path = pathlib.Path(xo.options.pins.get_path(csv_name))
    target_path = tmp_path.joinpath(source_path.name)
    if not target_path.exists():
        target_path.write_text(source_path.read_text())
    return target_path


def mutate_csv(path, line=None):
    if line is None:
        line = path.read_text().strip().rsplit("\n", 1)[-1]
    with path.open("at") as fh:
        fh.writelines([line])


@pytest.mark.parametrize("pins_resource", ("iris_csv", "astronauts_parquet"))
def test_deferred_read_cache_key_check(con, tmp_path, pins_resource, request):
    # check that we don't invoke read when we calc key
    pins_resource = request.getfixturevalue(pins_resource)
    storage = ParquetStorage(source=xo.connect(), path=tmp_path)

    assert pins_resource.table_name not in con.tables
    t = pins_resource.deferred_reader(con, pins_resource.path, pins_resource.table_name)
    storage.get_key(t)
    assert pins_resource.table_name not in con.tables


@pytest.mark.parametrize("pins_resource", ("iris_csv", "astronauts_parquet"))
def test_deferred_read_to_sql(con, pins_resource, request):
    # check that we don't invoke read when we convert to sql
    pins_resource = request.getfixturevalue(pins_resource)
    assert pins_resource.table_name not in con.tables
    t = pins_resource.deferred_reader(con, pins_resource.path, pins_resource.table_name)
    xo.to_sql(t)
    assert pins_resource.table_name not in con.tables


@pytest.mark.parametrize(
    "con,pins_resource",
    itertools.product(
        (xo.pandas.connect(), xo.postgres.connect_env()),
        ("iris_csv", "astronauts_parquet"),
    ),
)
def test_deferred_read(con, pins_resource, request):
    pins_resource = request.getfixturevalue(pins_resource)
    assert pins_resource.table_name not in con.tables
    t = pins_resource.deferred_reader(con, pins_resource.path, pins_resource.table_name)
    assert xo.execute(t).equals(pins_resource.df)
    assert pins_resource.table_name in con.tables
    # is this a test of mode for postgres?
    if con.name != "pandas":
        # verify that we can't execute again (pandas happily clobbers)
        with pytest.raises(
            ProgrammingError,
            match=f'relation "{pins_resource.table_name}" already exists',
        ):
            assert xo.execute(t).equals(pins_resource.df)
    con.drop_table(pins_resource.table_name, force=True)
    assert pins_resource.table_name not in tuple(con.tables)


@pytest.mark.parametrize(
    "con,pins_resource",
    itertools.product(
        (xo.postgres.connect_env(),),
        ("iris_csv", "astronauts_parquet"),
    ),
)
def test_deferred_read_temporary(con, pins_resource, request):
    pins_resource = request.getfixturevalue(pins_resource)
    t = pins_resource.deferred_reader(con, pins_resource.path, None, temporary=True)
    table_name = t.op().name
    assert xo.execute(t).equals(pins_resource.df)
    assert table_name in con.tables
    con.drop_table(table_name)
    assert table_name not in con.tables


@pytest.mark.parametrize(
    "con,pins_resource,filter_",
    (
        (con, pins_resource, filter_)
        for con in (xo.pandas.connect(), xo.postgres.connect_env(), xo.duckdb.connect())
        for (pins_resource, filter_) in (
            ("iris_csv", filter_sepal_length),
            ("astronauts_parquet", filter_field21),
        )
    ),
)
def test_cached_deferred_read(con, pins_resource, filter_, request, tmp_path):
    pins_resource = request.getfixturevalue(pins_resource)
    storage = ParquetStorage(source=xo.connect(), path=tmp_path)

    df = pins_resource.df[filter_].reset_index(drop=True)
    t = pins_resource.deferred_reader(con, pins_resource.path, pins_resource.table_name)
    expr = t[filter_].cache(storage=storage)

    # no work is done yet
    assert pins_resource.table_name not in con.tables
    assert not storage.exists(expr)

    # something exists in both con and storage
    assert xo.execute(expr).equals(df)
    assert pins_resource.table_name in con.tables
    assert storage.exists(expr)

    # we read from cache even if the table disappears
    try:
        con.drop_table(t.op().name, force=True)
    except duckdb.duckdb.CatalogException:
        con.drop_view(t.op().name)

    assert xo.execute(expr).equals(df)
    assert pins_resource.table_name not in con.tables

    # we repopulate the cache
    storage.drop(expr)
    assert xo.execute(expr).equals(df)
    assert pins_resource.table_name in con.tables
    assert storage.exists(expr)

    if con.name == "postgres":
        # we are mode="create" by default, which means losing cache creates collision
        mode = get_partial_arguments(pins_resource.get_underlying_method(con))["mode"]
        assert mode == "create"
        storage.drop(expr)
        with pytest.raises(
            ProgrammingError,
            match=f'relation "{pins_resource.table_name}" already exists',
        ):
            xo.execute(expr)

        # with mode="replace" we can clobber
        t = pins_resource.deferred_reader(
            con, pins_resource.path, pins_resource.table_name, mode="replace"
        )
        expr = t[filter_].cache(storage=storage)
        assert xo.execute(expr).equals(df)
        assert storage.exists(expr)
        assert pins_resource.table_name in con.tables
        # this fails above, but works here because of mode="replace"
        storage.drop(expr)
        assert xo.execute(expr).equals(df)


@pytest.mark.parametrize(
    "con",
    (xo.pandas.connect(), xo.postgres.connect_env()),
)
def test_cached_csv_mutate(con, iris_csv, tmp_path):
    target_path = ensure_tmp_csv(iris_csv.name, tmp_path)
    storage = ParquetStorage(source=xo.connect(), path=tmp_path)
    # make sure the con is "clean"
    if iris_csv.table_name in con.tables:
        con.drop_table(iris_csv.table_name, force=True)

    df = iris_csv.df
    kwargs = {"mode": "replace"} if con.name == "postgres" else {}
    t = iris_csv.deferred_reader(con, target_path, iris_csv.table_name, **kwargs)
    expr = t.cache(storage=storage)

    # nothing exists yet
    assert iris_csv.table_name not in con.tables
    assert not storage.exists(expr)

    # initial cache population
    assert xo.execute(expr).equals(df)
    assert iris_csv.table_name in con.tables
    assert storage.exists(expr)

    # mutate
    mutate_csv(target_path)
    df = iris_csv.immediate_reader(target_path)
    assert not storage.exists(expr)
    assert xo.execute(expr).equals(df)
    assert storage.exists(expr)
