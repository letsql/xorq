import functools
import json
import pathlib

import pandas as pd
import requests
import toolz

import xorq as xo
from xorq.expr.relations import flight_udxf
from xorq.flight.exchanger import make_udxf


base_api_url = "https://hacker-news.firebaseio.com/v0"


@toolz.curry
def simple_disk_cache(f, cache_dir, serde):
    cache_dir.mkdir(parents=True, exist_ok=True)

    def wrapped(**kwargs):
        name = ",".join(f"{key}={value}" for key, value in kwargs.items())
        path = cache_dir.joinpath(name)
        if path.exists():
            return serde.loads(path.read_text())
        else:
            value = f(**kwargs)
            path.write_text(serde.dumps(value))
            return value

    return wrapped


@simple_disk_cache(cache_dir=pathlib.Path("./hackernews-items"), serde=json)
def get_hackernews_item(*, item_id):
    resp = requests.get(f"{base_api_url}/item/{item_id}.json")
    resp.raise_for_status()
    item = resp.json()
    return item


@functools.cache
def get_hackernews_maxitem():
    resp = requests.get(f"{base_api_url}/maxitem.json")
    resp.raise_for_status()
    maxitem = resp.json()
    return maxitem


def get_hackernews_stories(maxitem, n):
    df = pd.DataFrame(
        get_hackernews_item(item_id=item_id) for item_id in range(maxitem - n, maxitem)
    )
    df = df.reindex(columns=schema_out)[df.type.eq("story") & df.title.notnull()]
    return df


def get_hackernews_stories_batch(df):
    series = df.apply(lambda row: get_hackernews_stories(**row), axis=1)
    return pd.concat(series.values, ignore_index=True)


schema_in = xo.schema({"maxitem": int, "n": int})
schema_out = xo.schema(
    {
        "by": "string",
        "id": "int64",
        "parent": "float64",
        "text": "string",
        "time": "int64",
        "type": "string",
        "kids": "array<int64>",
        # "deleted": "bool",
        "descendants": "float64",
        "score": "float64",
        "title": "string",
        "url": "string",
        # "dead": "bool",
    }
)


HackerNewsFetcher = make_udxf(
    name="HackerNewsFetcher",
    process_df=get_hackernews_stories_batch,
    maybe_schema_in=schema_in.to_pyarrow(),
    maybe_schema_out=schema_out.to_pyarrow(),
)


if __name__ == "__main__":
    import pandas as pd

    import xorq as xo

    t = xo.memtable(
        pd.DataFrame(({"maxitem": 43182839, "n": 1000},)),
        name="t",
    )
    expr = flight_udxf(
        t,
        HackerNewsFetcher,
    )
    out = expr.count().execute()
