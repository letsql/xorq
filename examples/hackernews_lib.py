import functools
import json
import pathlib

import pandas as pd
import requests
import toolz

import xorq as xo


base_api_url = "https://hacker-news.firebaseio.com/v0"


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


def get_json(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()


@simple_disk_cache(cache_dir=pathlib.Path("./hackernews-items"), serde=json)
def get_hackernews_item(*, item_id):
    return get_json(f"{base_api_url}/item/{item_id}.json")


@functools.cache
def get_hackernews_maxitem():
    return get_json(f"{base_api_url}/maxitem.json")


def get_hackernews_stories(maxitem, n):
    gen = (
        get_hackernews_item(item_id=item_id) for item_id in range(maxitem - n, maxitem)
    )
    df = pd.DataFrame(gen).reindex(columns=schema_out)[
        lambda t: t.type.eq("story") & t.title.notnull()
    ]
    return df


def get_hackernews_stories_batch(df):
    series = df.apply(lambda row: get_hackernews_stories(**row), axis=1)
    return pd.concat(series.values, ignore_index=True)


do_hackernews_fetcher_udxf = xo.expr.relations.flight_udxf(
    process_df=get_hackernews_stories_batch,
    maybe_schema_in=schema_in.to_pyarrow(),
    maybe_schema_out=schema_out.to_pyarrow(),
    name="HackerNewsFetcher",
)
