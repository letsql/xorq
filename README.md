# xorq: do-anything, run-anywhere pandas-style pipelines

[![Downloads](https://static.pepy.tech/badge/letsql)](https://pepy.tech/project/letsql)
![PyPI - Version](https://img.shields.io/pypi/v/letsql)
![GitHub License](https://img.shields.io/github/license/letsql/letsql)
![PyPI - Status](https://img.shields.io/pypi/status/letsql)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/letsql/letsql/ci-test.yml)
![Codecov](https://img.shields.io/codecov/c/github/letsql/letsql)

xorq is a deferred computation toolchain that brings the replicability and
performance of declarative pipelines to the Python ML ecosystem. It enables us
to write pandas-style transformations that never run out of memory,
automatically cache intermediate results, and seamlessly move between SQL
engines and Python UDFs—all while maintaining replicability. xorq is built on
top of Ibis and DataFusion.

I'll add more detailed descriptions for each of xorq's features:

| Feature | Description |
|---------|-------------|
| **Declarative expressions** | Powered by Ibis, xorq lets you define transformations in a Pythonic, declarative style without being tied to a specific execution engine. This abstraction allows you to write expressions once and execute them across various backends (DuckDB, DataFusion, Spark, Snowflake) without rewriting code. The `.into_backend()` method enables seamless transitions between engines within a single pipeline. |
| **[Built-in caching](https://docs.xorq.dev/core_concepts#caching-system)** | xorq automatically tracks the computational graph of your pipeline and intelligently caches intermediate results, minimizing repeated work. The cache system hashes expressions with metadata, enabling smart invalidation when code or parameters change. Supports multiple storage backends (in-memory, disk, etc.) and can materialize results as Arrow RecordBatches for optimal performance. |
| **[Multi-engine](https://docs.xorq.dev/core_concepts#multi-engine-system)** | Create unified ML workflows that leverage the strengths of different data engines in a single pipeline. xorq orchestrates data movement between engines (e.g., Snowflake for initial extraction, DuckDB for transformations, and Python for ML model training), handling all the complexity of cross-engine compatibility and data serialization behind the scenes. |
| **Serializable pipelines** | All pipeline definitions, including UDFs, are serialized to YAML format, enabling robust version control, reproducibility, and CI/CD integration. This serialization captures the complete execution graph, ensuring consistent results across environments and making it easy to track changes over time. |
| **Portable UDFs** | User-defined functions in xorq can be registered, serialized, and reused across different environments. These UDFs maintain consistent behavior regardless of execution context, supporting variants like aggregates, window functions, and transformations. The embedded engine provides a portable runtime for UDF execution, enhancing reproducibility. |
| **Arrow-native architecture** | Built on Apache Arrow's columnar memory format and Arrow Flight transport layer, xorq achieves high-performance data transfer without cumbersome serialization overhead. This design enables efficient data movement between services, supports both ephemeral processing for partial expressions and long-lived services for production deployments. |


## Getting Started
xorq functions as both an interactive library for building expressions and a
command-line interface. This dual nature enables seamless transition
from exploratory research to production-ready artifacts. The steps below will
guide through using both the CLI and library components to get started.

> [!CAUTION] 
> This library does not currently have a stable release. Both the
> API and implementation are subject to change, and future updates may not be
> backward compatible.
### Installation

xorq is available as [`xorq`](https://pypi.org/project/xorq/) on PyPI:

```shell
pip install xorq
```

> [!NOTE]
> We are changing the name from LETSQL to xorq.

### Usage

```python
import xorq as xo


pg = xo.postgres.connect_env()
db = xo.duckdb.connect()

batting = pg.table("batting")
awards_players = xo.examples.awards_players.fetch(backend=db)

left = batting.filter(batting.yearID == 2015)

right = (awards_players.filter(awards_players.lgID == "NL")
                       .drop("yearID", "lgID")
                       .into_backend(pg, "filtered"))

expr = (left.join(right, ["playerID"], how="semi")
            .cache()
            .select(["yearID", "stint"]))

# expr.build().execute()
result = expr.execute()
```

for more examples on how to use letsql, check the
[examples](https://github.com/letsql/xorq/tree/main/examples) directory, note
that in order to run some of the scripts in there, you need to install the
library with `examples` extra:

```shell
pip install 'xorq[examples]'
```

## Contributing

Contributions are welcome and highly appreciated. To get started, check out the [contributing guidelines](https://github.com/letsql/xorq/blob/main/CONTRIBUTING.md).

## Support

If you have any issues with this repository, please don't hesitate to [raise them](https://github.com/letsql/xorq/issues/new).
It is actively maintained, and we will do our best to help you.

## Acknowledgements

This project heavily relies on [Ibis](https://github.com/ibis-project/ibis) and [DataFusion](https://github.com/apache/datafusion).   

## Liked the work?

If you've found this repository helpful, why not give it a star? It's an easy way to show your appreciation and support for the project.
Plus, it helps others discover it too!

## License

This repository is licensed under the [Apache License](https://github.com/letsql/letsql/blob/main/LICENSE)
