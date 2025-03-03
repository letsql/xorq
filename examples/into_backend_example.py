import xorq as xo
from xorq.caching import SourceStorage


con = xo.connect()
pg = xo.postgres.connect_env()

batting = pg.table("batting")
t = batting.filter(batting.yearID == 2015).into_backend(con, "ls_batting")

expr = (
    t.join(t, "playerID").limit(15).select(player_id="playerID", year_id="yearID_right")
    # .cache(SourceStorage(source=con))
)

print(expr)

# print(expr.execute())
# print(con.list_tables())
