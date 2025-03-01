import pandas as pd
import pytest


@pytest.mark.parametrize("how", ["semi", "anti"])
def test_expr_build(tmp_path, batting, awards_players, how):
    left = batting[batting.yearID == 2015]
    right = awards_players[awards_players.lgID == "NL"].drop("yearID", "lgID")

    expr = left.join(right, ["playerID"], how=how)

    # Create a subdirectory for the test
    build_dir = tmp_path / "builds"

    result = expr.build(build_dir).execute()
    assert build_dir.exists()
    assert isinstance(result, pd.DataFrame)
