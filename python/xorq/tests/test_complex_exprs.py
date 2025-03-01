import operator

import pytest
import toolz

import xorq as xo
from xorq.caching import (
    ParquetSnapshotStorage,
    ParquetStorage,
)
from xorq.common.utils.defer_utils import deferred_read_parquet
from xorq.expr.relations import into_backend


rate_to_rate_str = toolz.compose(operator.methodcaller("replace", ".", "p"), str)


def asof_join_flight_data(con, tail, flight, airborne_only=True):
    """Create an expression for a particular flight's data"""

    def rate_to_parquet(tail, flight, rate):
        base_url = (
            f"https://nasa-avionics-data-ml.s3.us-east-2.amazonaws.com/{tail}_parquet"
        )
        filename = f"{flight}.{rate_to_rate_str(rate)}.parquet"
        return f"{base_url}/{filename}"

    rates = (0.25, 1.0, 2.0, 4.0, 8.0, 16.0)
    ts = (
        deferred_read_parquet(con, parquet_path, rate_to_rate_str(rate)).mutate(
            flight=xo.literal(flight)
        )
        for rate, parquet_path in (
            (rate, rate_to_parquet(tail, flight, rate))
            for rate in sorted(rates, reverse=True)
        )
    )
    db_con = xo.duckdb.connect()
    (expr, *others) = (
        into_backend(t, db_con, name=f"flight-{flight}-{t.op().parent.name}")
        for t in ts
    )
    for other in others:
        expr = expr.asof_join(other, on="time").drop(["time_right", "flight_right"])
    if airborne_only:
        expr = expr[lambda t: t.GS != 0]
    return expr


@pytest.mark.parametrize("cls", [ParquetSnapshotStorage, ParquetStorage])
@pytest.mark.parametrize("cross_source_caching", [True, False])
def test_complex_storage(cls, cross_source_caching, tmp_path):
    tail = "Tail_652_1"
    flight = "652200101120916"

    con = xo.connect()
    storage_con = xo.connect() if cross_source_caching else con
    storage = cls(source=storage_con, path=tmp_path)

    expr = asof_join_flight_data(con, tail, flight)
    cached = expr.cache(storage=storage)
    assert not storage.cache.exists(expr)
    out = cached.count().execute()
    assert out == 44260
    assert cached.ls.exists()
    assert storage.exists(cached)
    # ParquetCacheStorage has an issue with this, regardless of cross_source_caching
    assert storage.cache.exists(expr)
