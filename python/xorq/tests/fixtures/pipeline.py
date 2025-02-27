import xorq as xo
from xorq.common.utils.defer_utils import deferred_read_parquet
from xorq.expr.relations import into_backend


pg = xo.postgres.connect_env()
db = xo.duckdb.connect()

batting = pg.table("batting")

integer = 1

backend = xo.duckdb.connect()
awards_players = deferred_read_parquet(
    backend,
    xo.config.options.pins.get_path("awards_players"),
    table_name="award_players",
)
left = batting.filter(batting.yearID == 2015)
right = awards_players.filter(awards_players.lgID == "NL").drop("yearID", "lgID")
expr = left.join(
    into_backend(right, pg, "pg-filtered-table"), ["playerID"], how="semi"
)[["yearID", "stint"]]
