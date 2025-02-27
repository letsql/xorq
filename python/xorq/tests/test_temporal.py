from __future__ import annotations

import datetime
import operator
import warnings
from operator import methodcaller

import numpy as np
import pandas as pd
import pytest
from pytest import param

import xorq as xo
import xorq.expr.datatypes as dt
from xorq.tests.util import (
    assert_frame_equal,
    assert_series_equal,
    default_series_rename,
)


@pytest.mark.parametrize("attr", ["year", "month", "day"])
@pytest.mark.parametrize(
    "expr_fn",
    [
        param(lambda c: c.date(), id="date"),
        param(
            lambda c: c.cast("date"),
            id="cast",
        ),
    ],
)
def test_date_extract(alltypes, df, attr, expr_fn):
    expr = getattr(expr_fn(alltypes.timestamp_col), attr)()
    expected = getattr(df.timestamp_col.dt, attr).astype("int32")

    result = expr.name(attr).execute()

    assert_series_equal(result, expected.rename(attr))


@pytest.mark.parametrize(
    "attr",
    [
        "year",
        "month",
        "day",
        "day_of_year",
        "quarter",
        "hour",
        "minute",
        "second",
    ],
)
def test_timestamp_extract(alltypes, df, attr):
    method = getattr(alltypes.timestamp_col, attr)
    expr = method().name(attr)
    result = xo.execute(expr)
    expected = (
        getattr(df.timestamp_col.dt, attr.replace("_", "")).astype("int32").rename(attr)
    )
    assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("func", "expected"),
    [
        param(
            methodcaller("year"),
            2015,
            id="year",
        ),
        param(
            methodcaller("month"),
            9,
            id="month",
        ),
        param(
            methodcaller("day"),
            1,
            id="day",
        ),
        param(
            methodcaller("hour"),
            14,
            id="hour",
        ),
        param(
            methodcaller("minute"),
            48,
            id="minute",
        ),
        param(
            methodcaller("second"),
            5,
            id="second",
        ),
        param(
            methodcaller("millisecond"),
            359,
            id="millisecond",
        ),
        param(
            lambda x: x.day_of_week.index(),
            1,
            id="day_of_week_index",
        ),
        param(
            lambda x: x.day_of_week.full_name(),
            "Tuesday",
            id="day_of_week_full_name",
        ),
    ],
)
def test_timestamp_extract_literal(con, func, expected):
    value = xo.timestamp("2015-09-01 14:48:05.359")
    assert con.execute(func(value).name("tmp")) == expected


def test_timestamp_extract_microseconds(alltypes, df):
    expr = alltypes.timestamp_col.microsecond().name("microsecond")
    result = expr.execute()
    expected = df.timestamp_col.dt.microsecond.astype("int32").rename("microsecond")
    assert_series_equal(result, expected)


def test_timestamp_extract_milliseconds(alltypes, df):
    expr = alltypes.timestamp_col.millisecond().name("millisecond")
    result = expr.execute()
    expected = (
        (df.timestamp_col.dt.microsecond // 1_000).astype("int32").rename("millisecond")
    )
    assert_series_equal(result, expected)


def test_timestamp_extract_epoch_seconds(alltypes, df):
    expr = alltypes.timestamp_col.epoch_seconds().name("tmp")
    result = expr.execute()

    expected = df.timestamp_col.astype("datetime64[s]").astype("int64").astype("int32")
    assert_series_equal(result, expected)


def test_timestamp_extract_week_of_year(alltypes, df):
    expr = alltypes.timestamp_col.week_of_year().name("tmp")
    result = expr.execute()
    expected = df.timestamp_col.dt.isocalendar().week.astype("int32")
    assert_series_equal(result, expected)


PANDAS_UNITS = {
    "m": "Min",
    "ms": "L",
}

TIMESTAMP_PANDAS_UNITS = {
    "m": "Min",
    "ms": "L",
    "M": "ME",
    "Y": "YE",
}


@pytest.mark.parametrize(
    "unit",
    [
        param(
            "Y",
        ),
        param(
            "M",
        ),
        param(
            "D",
        ),
        param(
            "W",
        ),
        param(
            "h",
        ),
        param(
            "m",
        ),
        param(
            "s",
        ),
    ],
)
def test_timestamp_truncate(alltypes, df, unit):
    expr = alltypes.timestamp_col.truncate(unit).name("tmp")

    unit = PANDAS_UNITS.get(unit, unit)

    try:
        ts_unit = TIMESTAMP_PANDAS_UNITS.get(unit, unit)
        expected = df.timestamp_col.dt.floor(ts_unit)
    except ValueError:
        unit = PANDAS_UNITS.get(unit, unit)
        expected = df.timestamp_col.dt.to_period(unit).dt.to_timestamp()

    result = expr.execute()
    assert_series_equal(result, expected)


date_value = pd.Timestamp("2017-12-31")
timestamp_value = pd.Timestamp("2018-01-01 18:18:18")


@pytest.mark.parametrize(
    "func_name",
    [
        "gt",
        "ge",
        "lt",
        "le",
        "eq",
        "ne",
    ],
)
def test_timestamp_comparison_filter(con, alltypes, df, func_name):
    ts = pd.Timestamp("20100302", tz="UTC").to_pydatetime()

    comparison_fn = getattr(operator, func_name)
    expr = alltypes.filter(
        comparison_fn(alltypes.timestamp_col.cast("timestamp('UTC')"), ts)
    )

    col = df.timestamp_col.dt.tz_localize("UTC")
    expected = df[comparison_fn(col, ts)]
    result = con.execute(expr)

    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "func_name",
    [
        "gt",
        "ge",
        "lt",
        "le",
        "eq",
        "ne",
    ],
)
def test_timestamp_comparison_filter_numpy(con, alltypes, df, func_name):
    ts = np.datetime64("2010-03-02 00:00:00.000123")

    comparison_fn = getattr(operator, func_name)
    expr = alltypes.filter(
        comparison_fn(alltypes.timestamp_col.cast("timestamp('UTC')"), ts)
    )

    ts = pd.Timestamp(ts.item(), tz="UTC")

    col = df.timestamp_col.dt.tz_localize("UTC")
    expected = df[comparison_fn(col, ts)]
    result = con.execute(expr)

    assert_frame_equal(result, expected)


def test_interval_add_cast_scalar(alltypes):
    timestamp_date = alltypes.timestamp_col.date()
    delta = xo.literal(10).cast("interval('D')")
    expr = (timestamp_date + delta).name("result")
    result = expr.execute()
    expected = timestamp_date.name("result").execute() + pd.Timedelta(10, unit="D")
    assert_series_equal(result, expected.astype(result.dtype))


def test_interval_add_cast_column(alltypes, df):
    timestamp_date = alltypes.timestamp_col.date()
    delta = alltypes.bigint_col.cast("interval('D')")
    expr = alltypes["id", (timestamp_date + delta).name("tmp")]
    result = expr.execute().sort_values("id").reset_index().tmp
    df = df.sort_values("id").reset_index(drop=True)
    expected = (
        df["timestamp_col"]
        .dt.normalize()
        .add(df.bigint_col.astype("timedelta64[D]"))
        .rename("tmp")
        .dt.date
    )
    assert_series_equal(result, expected.astype(result.dtype))


unit_factors = {"s": 10**9, "ms": 10**6, "us": 10**3, "ns": 1}


@pytest.mark.parametrize(
    ("date", "expected_index", "expected_day"),
    [
        param("2017-01-01", 6, "Sunday", id="sunday"),
        param("2017-01-02", 0, "Monday", id="monday"),
        param("2017-01-03", 1, "Tuesday", id="tuesday"),
        param("2017-01-04", 2, "Wednesday", id="wednesday"),
        param("2017-01-05", 3, "Thursday", id="thursday"),
        param("2017-01-06", 4, "Friday", id="friday"),
        param("2017-01-07", 5, "Saturday", id="saturday"),
    ],
)
def test_day_of_week_scalar(con, date, expected_index, expected_day):
    expr = xo.literal(date).cast(dt.date)
    result_index = con.execute(expr.day_of_week.index().name("tmp"))
    assert result_index == expected_index

    result_day = con.execute(expr.day_of_week.full_name().name("tmp"))
    assert result_day.lower() == expected_day.lower()


def test_day_of_week_column(alltypes, df):
    expr = alltypes.timestamp_col.day_of_week

    result_index = expr.index().name("tmp").execute()
    expected_index = df.timestamp_col.dt.dayofweek.astype("int16")

    assert_series_equal(result_index, expected_index, check_names=False)

    result_day = expr.full_name().name("tmp").execute()
    expected_day = df.timestamp_col.dt.day_name()

    assert_series_equal(result_day, expected_day, check_names=False)


@pytest.mark.parametrize(
    ("day_of_week_expr", "day_of_week_pandas"),
    [
        param(
            lambda t: t.timestamp_col.day_of_week.index().count(),
            lambda s: s.dt.dayofweek.count(),
            id="day_of_week_index",
        ),
        param(
            lambda t: t.timestamp_col.day_of_week.full_name().length().sum(),
            lambda s: s.dt.day_name().str.len().sum(),
            id="day_of_week_full_name",
        ),
    ],
)
def test_day_of_week_column_group_by(
    alltypes, df, day_of_week_expr, day_of_week_pandas
):
    expr = alltypes.group_by("string_col").aggregate(
        day_of_week_result=day_of_week_expr
    )
    schema = expr.schema()
    assert schema["day_of_week_result"] == dt.int64

    result = expr.execute().sort_values("string_col")
    expected = (
        df.groupby("string_col")
        .timestamp_col.apply(day_of_week_pandas)
        .reset_index()
        .rename(columns={"timestamp_col": "day_of_week_result"})
    )

    assert_frame_equal(result, expected, check_dtype=False)


def test_date_scalar_from_iso(con):
    expr = xo.literal("2022-02-24")
    expr2 = xo.date(expr)

    result = con.execute(expr2)
    assert result.strftime("%Y-%m-%d") == "2022-02-24"


def test_date_column_from_iso(con, alltypes, df):
    expr = (
        alltypes.year.cast("string")
        + "-"
        + alltypes.month.cast("string").lpad(2, "0")
        + "-13"
    )
    expr = xo.date(expr)

    result = con.execute(expr.name("tmp"))
    golden = df.year.astype(str) + "-" + df.month.astype(str).str.rjust(2, "0") + "-13"
    actual = result.map(datetime.date.isoformat)
    assert_series_equal(golden.rename("tmp"), actual.rename("tmp"))


def test_timestamp_extract_milliseconds_with_big_value(con):
    timestamp = xo.timestamp("2021-01-01 01:30:59.333456")
    millis = timestamp.millisecond()
    result = con.execute(millis.name("tmp"))
    assert result == 333


def test_big_timestamp(con):
    value = xo.timestamp("2419-10-11 10:10:25")
    result = con.execute(value.name("tmp"))
    expected = datetime.datetime(2419, 10, 11, 10, 10, 25)
    assert result == expected


DATE = datetime.date(2010, 11, 1)


def build_date_col(t):
    return (
        t.year.cast("string")
        + "-"
        + t.month.cast("string").lpad(2, "0")
        + "-"
        + (t.int_col + 1).cast("string").lpad(2, "0")
    ).cast("date")


@pytest.mark.parametrize(
    ("left_fn", "right_fn"),
    [
        param(build_date_col, lambda _: DATE, id="column_date"),
        param(lambda _: DATE, build_date_col, id="date_column"),
    ],
)
def test_timestamp_date_comparison(alltypes, df, left_fn, right_fn):
    left = left_fn(alltypes)
    right = right_fn(alltypes)
    expr = left == right
    result = expr.name("result").execute()
    expected = (
        pd.to_datetime(
            (
                df.year.astype(str)
                .add("-")
                .add(df.month.astype(str).str.rjust(2, "0"))
                .add("-")
                .add(df.int_col.add(1).astype(str).str.rjust(2, "0"))
            ),
            format="%Y-%m-%d",
            exact=True,
        )
        .eq(pd.Timestamp(DATE))
        .rename("result")
    )
    assert_series_equal(result, expected)


def test_large_timestamp(con):
    huge_timestamp = datetime.datetime(year=4567, month=1, day=1)
    expr = xo.timestamp("4567-01-01 00:00:00")
    result = con.execute(expr)
    assert result.replace(tzinfo=None) == huge_timestamp


@pytest.mark.parametrize(
    ("ts", "scale", "unit"),
    [
        param(
            "2023-01-07 13:20:05.561",
            3,
            "ms",
            id="ms",
        ),
        param(
            "2023-01-07 13:20:05.561021",
            6,
            "us",
            id="us",
        ),
        param(
            "2023-01-07 13:20:05.561000231",
            9,
            "ns",
            id="ns",
        ),
    ],
)
def test_timestamp_precision_output(con, ts, scale, unit):
    dtype = dt.Timestamp(scale=scale)
    expr = xo.literal(ts).cast(dtype)
    result = con.execute(expr)
    expected = pd.Timestamp(ts).floor(unit)
    assert result == expected


@pytest.mark.parametrize(
    ("func", "expected"),
    [
        param(
            methodcaller("hour"),
            14,
            id="hour",
        ),
        param(
            methodcaller("minute"),
            48,
            id="minute",
        ),
        param(
            methodcaller("second"),
            5,
            id="second",
        ),
        param(
            methodcaller("millisecond"),
            359,
            id="millisecond",
        ),
    ],
)
def test_time_extract_literal(con, func, expected):
    value = xo.time("14:48:05.359")
    assert con.execute(func(value).name("tmp")) == expected


def test_now(con):
    expr = xo.now()
    result = con.execute(expr.name("tmp"))
    assert isinstance(result, datetime.datetime)


def test_now_from_projection(alltypes):
    n = 2
    expr = alltypes.select(now=xo.now()).limit(n)
    result = expr.execute()
    ts = result.now
    assert len(result) == n
    assert ts.nunique() == 1
    assert not pd.isna(ts.iat[0])


@pytest.mark.parametrize(
    "unit",
    [
        "Y",
        "M",
        "W",
        "D",
    ],
)
def test_integer_to_interval_date(con, alltypes, df, unit):
    interval = alltypes.int_col.as_interval(unit=unit)
    array = alltypes.date_string_col.split("/")
    month, day, year = array[0], array[1], array[2]
    date_col = xo.literal("-").join(["20" + year, month, day]).cast("date")
    expr = (date_col + interval).name("tmp")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        result = con.execute(expr)

    def convert_to_offset(x):
        resolution = f"{interval.type().resolution}s"
        return pd.offsets.DateOffset(**{resolution: x})

    offset = df.int_col.apply(convert_to_offset)
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", category=(UserWarning, pd.errors.PerformanceWarning)
        )
        expected = (
            pd.to_datetime(df.date_string_col)
            .add(offset)
            .map(lambda ts: ts.normalize().date(), na_action="ignore")
        )

    expected = default_series_rename(expected)
    assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("unit", "displacement_type"),
    [
        param(
            "Y",
            pd.offsets.DateOffset,
        ),
        param(
            "M",
            pd.offsets.DateOffset,
        ),
        param(
            "W",
            pd.offsets.DateOffset,
        ),
        param("D", pd.offsets.DateOffset),
        param("h", pd.Timedelta),
        param("m", pd.Timedelta),
        param("s", pd.Timedelta),
        param(
            "ms",
            pd.Timedelta,
        ),
        param(
            "us",
            pd.Timedelta,
        ),
    ],
)
def test_integer_to_interval_timestamp(con, alltypes, df, unit, displacement_type):
    interval = alltypes.int_col.as_interval(unit=unit)
    expr = (alltypes.timestamp_col + interval).name("tmp")

    def convert_to_offset(offset, displacement_type=displacement_type):
        resolution = f"{interval.op().dtype.resolution}s"
        return displacement_type(**{resolution: offset})

    with warnings.catch_warnings():
        # both the implementation and test code raises pandas
        # PerformanceWarning, because We use DateOffset addition
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        result = con.execute(expr)
        offset = df.int_col.apply(convert_to_offset)
        expected = df.timestamp_col + offset

    expected = default_series_rename(expected)
    assert_series_equal(result, expected.astype(result.dtype))


def test_string_to_timestamp(alltypes):
    fmt = "%m/%d/%y"
    result = alltypes.mutate(date=alltypes.date_string_col.as_timestamp(fmt)).execute()

    for i, val in enumerate(result["date"]):
        assert val.strftime(fmt) == result["date_string_col"][i]


def test_string_to_date(alltypes):
    fmt = "%m/%d/%y"
    result = alltypes.mutate(date=alltypes.date_string_col.as_date(fmt)).execute()

    for i, val in enumerate(result["date"]):
        assert val.strftime(fmt) == result["date_string_col"][i]


@pytest.mark.parametrize(
    "kws, pd_freq",
    [
        param(
            {"seconds": 2},
            "2s",
            id="seconds",
        ),
        param(
            {"minutes": 5},
            "300s",
            id="minutes",
        ),
        param(
            {"hours": 2},
            "2h",
            id="hours",
        ),
        param(
            {"days": 2},
            "2D",
            id="days",
        ),
    ],
)
def test_timestamp_bucket(alltypes, kws: dict, pd_freq):
    ts = alltypes.timestamp_col.execute().rename("ts")
    res = alltypes.timestamp_col.bucket(**kws).execute().rename("ts")
    sol = ts.dt.floor(pd_freq)
    assert_series_equal(res, sol)


@pytest.mark.parametrize("offset_in_minutes", [2, -2], ids=["pos", "neg"])
def test_timestamp_bucket_offset(alltypes, offset_in_minutes):
    ts = alltypes.timestamp_col
    expr = ts.bucket(minutes=5, offset=xo.interval(minutes=offset_in_minutes))
    res = expr.execute().astype("datetime64[ns]").rename("ts")
    td = pd.Timedelta(minutes=offset_in_minutes)
    sol = ((ts.execute().rename("ts") - td).dt.floor("300s") + td).astype(
        "datetime64[ns]"
    )
    assert_series_equal(res, sol)


@pytest.mark.parametrize("offset_in_hours", [2, -2], ids=["pos", "neg"])
def test_timestamp_bucket_offset_in_hours(alltypes, offset_in_hours):
    ts = alltypes.timestamp_col
    expr = ts.bucket(minutes=5, offset=xo.interval(hours=offset_in_hours))
    res = expr.execute().astype("datetime64[ns]").rename("ts")
    td = pd.Timedelta(hours=offset_in_hours)
    sol = ((ts.execute().rename("ts") - td).dt.floor("300s") + td).astype(
        "datetime64[ns]"
    )
    assert_series_equal(res, sol)
