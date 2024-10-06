import ibis
import pandas as pd
import pytest

from letsql.pipe.ibis import select, join, limit, mutate, sql, head, _


@pytest.fixture
def left():
    left = pd.DataFrame({"a": list(range(10)), "b": [f"b_{i}" for i in range(10, 20)]})
    return ibis.memtable(left, name="left")


@pytest.fixture
def right():
    right = pd.DataFrame({"a": list(range(10)), "c": [f"c_{i}" for i in range(10, 20)]})
    return ibis.memtable(right, name="right")


@pytest.fixture
def penguins():
    return ibis.examples.penguins.fetch(table_name="penguins")


def test_select_limit(left):
    pipeline = left | select("a") | limit(5)
    assert pipeline.execute() is not None


def test_select_mutate_and_join(left, right):
    pipeline = left | join(right, "a").select("a", bx="b", cx="c").mutate(
        a2=_.a * 2, bcx=_.bx + _.cx
    ).limit(5)
    assert pipeline.execute() is not None


def test_select_mutate_and_join_with_pipes(left, right):
    pipeline = (
        left
        | join(right, "a")
        | select("a", bx="b", cx="c")
        | mutate(a2=_.a * 2, bcx=_.bx + _.cx)
        | limit(5)
    )
    assert pipeline.execute() is not None


def test_filter(left):
    pipeline = left | select("a").filter(_.a < 5)
    assert pipeline.execute() is not None


def test_head(left):
    pipeline = left | select("a").head()
    assert pipeline.execute() is not None


def test_limit(left):
    pipeline = left | select("a").limit(5)
    assert pipeline.execute() is not None


def test_join(left, right):
    pipeline = left | select("a").join(right, "a")
    assert pipeline.execute() is not None


def test_mutate(left):
    pipeline = left | mutate(a2=_.a * 2)
    assert pipeline.execute() is not None


def test_mutate_select(left):
    pipeline = left | mutate(a2=_.a * 2).select("a", "a2")
    assert pipeline.execute() is not None


def test_select_mutate(right):
    pipeline = right | select("a").mutate(a2=_.a * 2)
    assert pipeline.execute() is not None


def test_sql(penguins):
    pipeline = (
        penguins
        | sql(
            """
        SELECT island, mean(bill_length_mm) AS avg_bill_length
        FROM penguins
        GROUP BY 1
        ORDER BY 2 DESC
        """
        )
        | head(4)
    )

    assert pipeline.execute() is not None
