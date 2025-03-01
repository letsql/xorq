from __future__ import annotations

import pandas as pd
import pytest
from pytest import param

import xorq as xo
import xorq.expr.datatypes as dt
from xorq.tests.util import assert_frame_equal, assert_series_equal


@pytest.mark.parametrize(
    "text_value",
    [
        param(
            "STRING",
            id="string",
        ),
        param(
            "STRI'NG",
            id="string-quote1",
        ),
        param(
            'STRI"NG',
            id="string-quote2",
        ),
    ],
)
def test_string_literal(con, text_value):
    expr = xo.literal(text_value)
    result = con.execute(expr)
    assert result == text_value


def is_text_type(x):
    return isinstance(x, str)


def test_string_col_is_unicode(alltypes, df):
    dtype = alltypes.string_col.type()
    assert dtype == dt.String(nullable=dtype.nullable)
    assert df.string_col.map(is_text_type).all()
    result = alltypes.string_col.execute()
    assert result.map(is_text_type).all()


@pytest.mark.parametrize(
    ("result_func", "expected_func"),
    [
        param(
            lambda t: t.string_col.contains("6"),
            lambda t: t.string_col.str.contains("6"),
            id="contains",
        ),
        param(
            lambda t: t.string_col.rlike("|".join(map(str, range(10)))),
            lambda t: t.string_col == t.string_col,
            id="rlike",
        ),
        param(
            lambda t: ("a" + t.string_col + "a").re_search(r"\d+"),
            lambda t: t.string_col.str.contains(r"\d+"),
            id="re_search_substring",
        ),
        param(
            lambda t: t.string_col.re_search(r"\d+"),
            lambda t: t.string_col.str.contains(r"\d+"),
            id="re_search",
        ),
        param(
            lambda t: t.string_col.re_search(r"[[:digit:]]+"),
            lambda t: t.string_col.str.contains(r"\d+"),
            id="re_search_posix",
        ),
        param(
            lambda t: t.string_col.re_extract(r"([[:digit:]]+)", 1),
            lambda t: t.string_col.str.extract(r"(\d+)", expand=False),
            id="re_extract_posix",
        ),
        param(
            lambda t: (t.string_col + "1").re_extract(r"\d(\d+)", 0),
            lambda t: (t.string_col + "1").str.extract(r"(\d+)", expand=False),
            id="re_extract_whole_group",
        ),
        param(
            lambda t: t.date_string_col.re_extract(r"(\d+)\D(\d+)\D(\d+)", 1),
            lambda t: t.date_string_col.str.extract(
                r"(\d+)\D(\d+)\D(\d+)", expand=False
            ).iloc[:, 0],
            id="re_extract_group_1",
        ),
        param(
            lambda t: t.date_string_col.re_extract(r"(\d+)\D(\d+)\D(\d+)", 2),
            lambda t: t.date_string_col.str.extract(
                r"(\d+)\D(\d+)\D(\d+)", expand=False
            ).iloc[:, 1],
            id="re_extract_group_2",
        ),
        param(
            lambda t: t.date_string_col.re_extract(r"(\d+)\D(\d+)\D(\d+)", 3),
            lambda t: t.date_string_col.str.extract(
                r"(\d+)\D(\d+)\D(\d+)", expand=False
            ).iloc[:, 2],
            id="re_extract_group_3",
        ),
        param(
            lambda t: t.date_string_col.re_extract(r"^(\d+)", 1),
            lambda t: t.date_string_col.str.extract(r"^(\d+)", expand=False),
            id="re_extract_group_at_beginning",
        ),
        param(
            lambda t: t.date_string_col.re_extract(r"(\d+)$", 1),
            lambda t: t.date_string_col.str.extract(r"(\d+)$", expand=False),
            id="re_extract_group_at_end",
        ),
        param(
            lambda t: t.string_col.re_replace(r"[[:digit:]]+", "a"),
            lambda t: t.string_col.str.replace(r"\d+", "a", regex=True),
            id="re_replace_posix",
        ),
        param(
            lambda t: t.string_col.re_replace(r"\d+", "a"),
            lambda t: t.string_col.str.replace(r"\d+", "a", regex=True),
            id="re_replace",
        ),
        param(
            lambda t: t.string_col.repeat(2),
            lambda t: t.string_col * 2,
            id="repeat_method",
        ),
        param(
            lambda t: 2 * t.string_col,
            lambda t: 2 * t.string_col,
            id="repeat_left",
        ),
        param(
            lambda t: t.string_col * 2,
            lambda t: t.string_col * 2,
            id="repeat_right",
        ),
        param(
            lambda t: t.string_col.translate("01", "ab"),
            lambda t: t.string_col.str.translate(str.maketrans("01", "ab")),
            id="translate",
        ),
        param(
            lambda t: t.string_col.find("a"),
            lambda t: t.string_col.str.find("a"),
            id="find",
        ),
        param(
            lambda t: t.date_string_col.find("13", 3),
            lambda t: t.date_string_col.str.find("13", 3),
            id="find_start",
        ),
        param(
            lambda t: t.string_col.lpad(10, "a"),
            lambda t: t.string_col.str.pad(10, fillchar="a", side="left"),
            id="lpad",
        ),
        param(
            lambda t: t.string_col.rpad(10, "a"),
            lambda t: t.string_col.str.pad(10, fillchar="a", side="right"),
            id="rpad",
        ),
        param(
            lambda t: t.string_col.lower(),
            lambda t: t.string_col.str.lower(),
            id="lower",
        ),
        param(
            lambda t: t.string_col.upper(),
            lambda t: t.string_col.str.upper(),
            id="upper",
        ),
        param(
            lambda t: t.string_col.reverse(),
            lambda t: t.string_col.str[::-1],
            id="reverse",
        ),
        param(
            lambda t: t.string_col.ascii_str(),
            lambda t: t.string_col.map(ord).astype("int32"),
            id="ascii_str",
        ),
        param(
            lambda t: t.string_col.length(),
            lambda t: t.string_col.str.len().astype("int32"),
            id="length",
        ),
        param(
            lambda t: t.int_col.cases([(1, "abcd"), (2, "ABCD")], "dabc").startswith(
                "abc"
            ),
            lambda t: t.int_col == 1,
            id="startswith",
        ),
        param(
            lambda t: t.date_string_col.startswith("2010-01"),
            lambda t: t.date_string_col.str.startswith("2010-01"),
            id="startswith-simple",
        ),
        param(
            lambda t: t.string_col.strip(),
            lambda t: t.string_col.str.strip(),
            id="strip",
        ),
        param(
            lambda t: t.string_col.lstrip(),
            lambda t: t.string_col.str.lstrip(),
            id="lstrip",
        ),
        param(
            lambda t: t.string_col.rstrip(),
            lambda t: t.string_col.str.rstrip(),
            id="rstrip",
        ),
        param(
            lambda t: t.string_col.capitalize(),
            lambda t: t.string_col.str.capitalize(),
            id="capitalize",
        ),
        param(
            lambda t: t.date_string_col.substr(2, 3),
            lambda t: t.date_string_col.str[2:5],
            id="substr",
        ),
        param(
            lambda t: t.date_string_col.substr(2),
            lambda t: t.date_string_col.str[2:],
            id="substr-start-only",
        ),
        param(
            lambda t: t.date_string_col.left(2),
            lambda t: t.date_string_col.str[:2],
            id="left",
        ),
        param(
            lambda t: t.date_string_col.right(2),
            lambda t: t.date_string_col.str[-2:],
            id="right",
        ),
        param(
            lambda t: t.date_string_col[1:3],
            lambda t: t.date_string_col.str[1:3],
            id="slice",
        ),
        param(
            lambda t: t.date_string_col[-2],
            lambda t: t.date_string_col.str[-2],
            id="negative-index",
        ),
        param(
            lambda t: t.date_string_col[: t.date_string_col.length()],
            lambda t: t.date_string_col,
            id="expr_slice_end",
        ),
        param(
            lambda t: t.date_string_col[:],
            lambda t: t.date_string_col,
            id="expr_empty_slice",
        ),
        param(
            lambda t: t.date_string_col[
                t.date_string_col.length() - 2 : t.date_string_col.length() - 1
            ],
            lambda t: t.date_string_col.str[-2:-1],
            id="expr_slice_begin_end",
        ),
        param(
            lambda t: xo.literal("-").join(["a", t.string_col, "c"]),
            lambda t: "a-" + t.string_col + "-c",
            id="join",
        ),
        param(
            lambda t: t.string_col + t.date_string_col,
            lambda t: t.string_col + t.date_string_col,
            id="concat_columns",
        ),
        param(
            lambda t: t.string_col + "a",
            lambda t: t.string_col + "a",
            id="concat_column_scalar",
        ),
        param(
            lambda t: "a" + t.string_col,
            lambda t: "a" + t.string_col,
            id="concat_scalar_column",
        ),
        param(
            lambda t: t.string_col.replace("1", "42"),
            lambda t: t.string_col.str.replace("1", "42"),
            id="replace",
        ),
    ],
)
def test_string(alltypes, df, result_func, expected_func):
    expr = result_func(alltypes).name("tmp")
    result = expr.execute()

    expected = expected_func(df)
    assert_series_equal(result, expected)


def test_re_replace_global(con):
    expr = xo.literal("aba").re_replace("a", "c")
    result = con.execute(expr)
    assert result == "cbc"


def test_substr_with_null_values(alltypes, df):
    table = alltypes.mutate(
        substr_col_null=xo.case()
        .when(alltypes["bool_col"], alltypes["string_col"])
        .else_(None)
        .end()
        .substr(0, 2)
    )
    result = table.execute()

    expected = df.copy()
    mask = ~expected["bool_col"]
    expected["substr_col_null"] = expected["string_col"]
    expected.loc[mask, "substr_col_null"] = None
    expected["substr_col_null"] = expected["substr_col_null"].str.slice(0, 2)

    assert_frame_equal(result.fillna(pd.NA), expected.fillna(pd.NA))


@pytest.mark.parametrize(
    ("result_func", "expected"),
    [
        param(lambda d: d.protocol(), "http", id="protocol"),
        param(
            lambda d: d.authority(),
            "user:pass@example.com:80",
            id="authority",
        ),
        param(
            lambda d: d.userinfo(),
            "user:pass",
            id="userinfo",
        ),
        param(
            lambda d: d.host(),
            "example.com",
            id="host",
        ),
        param(lambda d: d.path(), "/docs/books/tutorial/index.html", id="path"),
        param(lambda d: d.query(), "name=networking", id="query"),
        param(lambda d: d.query("name"), "networking", id="query-key"),
        param(
            lambda d: d.query(xo.literal("na") + xo.literal("me")),
            "networking",
            id="query-dynamic-key",
        ),
        param(lambda d: d.fragment(), "DOWNLOADING", id="ref"),
    ],
)
def test_parse_url(con, result_func, expected):
    url = "http://user:pass@example.com:80/docs/books/tutorial/index.html?name=networking#DOWNLOADING"
    expr = result_func(xo.literal(url))
    result = con.execute(expr)
    assert result == expected


def test_capitalize(con):
    s = xo.literal("aBc")
    expected = "Abc"
    expr = s.capitalize()
    assert con.execute(expr) == expected


def test_subs_with_re_replace(con):
    expr = xo.literal("hi").re_replace("i", "a").substitute({"d": "b"}, else_="k")
    result = con.execute(expr)
    assert result == "k"


def test_multiple_subs(con):
    m = {"foo": "FOO", "bar": "BAR"}
    expr = xo.literal("foo").substitute(m)
    result = con.execute(expr)
    assert result == "FOO"


@pytest.mark.parametrize(
    "expr",
    [
        param(xo.case().when(True, "%").end(), id="case"),
        param(xo.ifelse(True, "%", xo.null()), id="ifelse"),
    ],
)
def test_no_conditional_percent_escape(con, expr):
    assert con.execute(expr) == "%"


def test_string_length(con):
    t = xo.memtable({"s": ["aaa", "a", "aa"]})
    assert con.execute(t.s.length()).gt(0).all()


def test_hash(con):
    s = xo.literal("aBc")
    expected = 5357040098349975229
    expr = s.hash()
    assert con.execute(expr) == expected


def test_hash_column(alltypes):
    res = alltypes.string_col.hash().name("tmp").execute()
    assert res is not None
