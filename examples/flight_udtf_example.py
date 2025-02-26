import functools
import json
import pathlib

import pandas as pd
import requests
import toolz

import xorq as xo
from xorq.common.utils.rbr_utils import instrument_reader
from xorq.flight import FlightServer, make_con
from xorq.flight.action import AddExchangeAction
from xorq.flight.exchanger import make_udxf


base_api_url = "https://hacker-news.firebaseio.com/v0"


@toolz.curry
def simple_disk_cache(f, cache_dir, serde):
    cache_dir.mkdir(parents=True, exist_ok=True)

    def wrapped(**kwargs):
        name = ",".join(f"{key}={value}" for key, value in kwargs.items())
        path = cache_dir.joinpath(name)
        if path.exists():
            value = serde.loads(path.read_text())
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


def get_hackernews_stories_batch(df):
    series = df.apply(lambda row: get_hackernews_stories(**row), axis=1)
    return pd.concat(series.values, ignore_index=True)


HackerNewsFetcher = make_udxf(
    name="HackerNewsFetcher",
    process_df=get_hackernews_stories_batch,
    maybe_schema_in=schema_in.to_pyarrow(),
    maybe_schema_out=schema_out.to_pyarrow(),
)


con = xo.connect()
df = pd.DataFrame(({"maxitem": 43182839, "n": 1000},))
t = con.register(df, "t")
rbr_in = instrument_reader(xo.to_pyarrow_batches(t), prefix="input ::")
with FlightServer() as server:
    client = make_con(server).con
    client.do_action(AddExchangeAction.name, HackerNewsFetcher, options=client._options)
    (fut, rbr_out) = client.do_exchange(HackerNewsFetcher.command, rbr_in)
    df_out = instrument_reader(rbr_out, prefix="output ::").read_pandas()
    print(fut.result())
