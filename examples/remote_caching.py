import xorq as xo
from xorq import _
from xorq.caching import SourceStorage


con = xo.connect()
ddb = xo.duckdb.connect()
pg = xo.postgres.connect_env()

name = "batting"

right = (
    xo.examples.get_table_from_name(name, backend=ddb)
    .filter(_.yearID == 2014)
    .into_backend(con)
)
left = pg.table(name).filter(_.yearID == 2015).into_backend(con)

expr = left.join(
    right,
    "playerID",
).cache(SourceStorage(source=pg))

res = expr.execute()
print(res)
