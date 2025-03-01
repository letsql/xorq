import xorq as xo
from xorq import _
from xorq.caching import ParquetStorage


pg = xo.postgres.connect_examples()
con = xo.connect()

for table_name in pg.list_tables():
    if table_name.startswith(xo.config.options.cache.key_prefix):
        pg.drop_table(table_name)

cache = ParquetStorage(source=con)

t = (
    pg.table("batting")
    .mutate(row_number=xo.row_number().over(group_by=[_.playerID], order_by=[_.yearID]))
    .filter(_.row_number == 1)
    .cache(storage=cache)
)
print(f"{t.ls.get_key()} exists?: {t.ls.exists()}")
res = xo.execute(t)
print(res)
print(f"{t.ls.get_key()} exists?: {t.ls.exists()}")
