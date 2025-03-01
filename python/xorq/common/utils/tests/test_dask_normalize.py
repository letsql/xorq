import hashlib
import pathlib
import re

import dask
import pytest

import xorq as xo
import xorq.common.utils.dask_normalize  # noqa: F401
from xorq.caching import (
    SourceSnapshotStorage,
)
from xorq.common.utils.dask_normalize.dask_normalize_utils import (
    patch_normalize_token,
)


def test_ensure_deterministic():
    assert dask.config.get("tokenize.ensure-deterministic")


def test_unregistered_raises():
    class Unregistered:
        pass

    with pytest.raises(ValueError, match="cannot be deterministically hashed"):
        dask.base.tokenize(Unregistered())


def test_tokenize_datafusion_memory_expr(alltypes_df, snapshot):
    con = xo.datafusion.connect()
    typ = type(con)
    t = con.register(alltypes_df, "t")
    with patch_normalize_token(type(con)) as mocks:
        actual = dask.base.tokenize(t)
    mocks[typ].assert_not_called()
    snapshot.assert_match(actual, "datafusion_memory_key.txt")


def test_tokenize_datafusion_parquet_expr(alltypes_df, tmp_path, snapshot):
    path = pathlib.Path(tmp_path).joinpath("data.parquet")
    alltypes_df.to_parquet(path)
    con = xo.datafusion.connect()
    t = con.register(path, "t")
    # work around tmp_path variation
    (prefix, suffix) = (
        re.escape(part)
        for part in (
            r"file_groups={1 group: [[",
            r"]]",
        )
    )
    to_hash = re.sub(
        prefix + f".*?/{path.name}" + suffix,
        prefix + f"/{path.name}" + suffix,
        str(tuple(dask.base.normalize_token(t))),
    )
    actual = hashlib.md5(to_hash.encode(), usedforsecurity=False).hexdigest()
    snapshot.assert_match(actual, "datafusion_key.txt")


def test_tokenize_pandas_expr(alltypes_df, snapshot):
    con = xo.pandas.connect()
    typ = type(con)
    t = con.create_table("t", alltypes_df)
    with patch_normalize_token(type(t.op().source)) as mocks:
        actual = dask.base.tokenize(t)
    mocks[typ].assert_not_called()
    snapshot.assert_match(actual, "pandas_key.txt")


def test_tokenize_duckdb_expr(batting, snapshot):
    con = xo.duckdb.connect()
    typ = type(con)
    t = con.register(batting.to_pyarrow(), "dashed-name")
    with patch_normalize_token(type(con)) as mocks:
        actual = dask.base.tokenize(t)
    mocks[typ].assert_not_called()

    snapshot.assert_match(actual, "duckdb_key.txt")


def test_pandas_snapshot_key(alltypes_df, snapshot):
    con = xo.pandas.connect()
    t = con.create_table("t", alltypes_df)
    storage = SourceSnapshotStorage(source=con)
    actual = storage.get_key(t)
    snapshot.assert_match(actual, "pandas_snapshot_key.txt")


def test_duckdb_snapshot_key(batting, snapshot):
    con = xo.duckdb.connect()
    t = con.register(batting.to_pyarrow(), "dashed-name")
    storage = SourceSnapshotStorage(source=con)
    actual = storage.get_key(t)
    snapshot.assert_match(actual, "duckdb_snapshot_key.txt")
