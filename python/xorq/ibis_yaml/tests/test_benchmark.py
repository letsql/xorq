import pytest

import xorq as xo
from xorq import _
from xorq.ibis_yaml.compiler import BuildManager


@pytest.mark.benchmark
def test_baseball_stats_compilation(build_dir):
    pg = xo.postgres.connect_examples()

    batting_old = (
        pg.table("batting")
        .filter(_.yearID < 2000)
        .filter(_.yearID >= 1990)
        .mutate(batting_avg=_.H / _.AB, era=_.yearID.cast("int32"))
    )

    batting_recent = (
        pg.table("batting")
        .filter(_.yearID >= 2000)
        .mutate(batting_avg=_.H / _.AB, era=_.yearID.cast("int32"))
    )

    all_batting = batting_old.union(batting_recent)

    league_averages = all_batting.group_by(["yearID", "lgID"]).aggregate(
        league_avg=_.batting_avg.mean(), league_HR_avg=_.HR.mean()
    )

    normalized_stats = all_batting.join(league_averages, ["yearID", "lgID"]).mutate(
        avg_vs_league=_.batting_avg - _.league_avg, HR_vs_league=_.HR - _.league_HR_avg
    )

    prev_year_stats = (
        normalized_stats.order_by(["playerID", "yearID"])
        .group_by("playerID")
        .mutate(
            prev_year_avg=_.batting_avg.lag(1).over(order_by=_.yearID),
            prev_year_HR=_.HR.lag(1).over(order_by=_.yearID),
            three_year_avg=_.batting_avg.mean().over(
                order_by=_.yearID, window=xo.window(preceding=2, following=0)
            ),
        )
    )

    build_manager = BuildManager(build_dir)

    expr_hash = build_manager.compile_expr(prev_year_stats)

    assert isinstance(expr_hash, str)
    assert len(expr_hash) > 0
