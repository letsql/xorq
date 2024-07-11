import letsql as ls
from letsql import _
from letsql.common.caching import SourceStorage, KEY_PREFIX
from letsql.common.utils.postgres_utils import (
    make_connection,
)

pg = make_connection(
    host="localhost",
    port=5432,
    user="postgres",
    password="postgres",
    database="ibis_testing",
)

for table_name in pg.list_tables():
    if table_name.startswith(KEY_PREFIX):
        pg.drop_table(table_name)

cache = SourceStorage(source=pg)

t = (
    pg.table("batting")
    .mutate(row_number=ls.row_number().over(group_by=[_.playerID], order_by=[_.yearID]))
    .filter(_.row_number == 1)
    .cache(storage=cache)
)

res = t.execute()
print(res)
