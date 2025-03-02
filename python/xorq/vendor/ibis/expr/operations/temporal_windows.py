"""Temporal window operations."""

from __future__ import annotations

from typing import Literal, Optional

from public import public

import xorq.vendor.ibis.expr.datatypes as dt
from xorq.vendor.ibis.common.annotations import attribute
from xorq.vendor.ibis.common.collections import FrozenOrderedDict
from xorq.vendor.ibis.expr.operations.core import Column, Scalar  # noqa: TCH001
from xorq.vendor.ibis.expr.operations.relations import Relation, Unaliased
from xorq.vendor.ibis.expr.schema import Schema


@public
class WindowAggregate(Relation):
    parent: Relation
    window_type: Literal["tumble", "hop"]
    time_col: Unaliased[Column]
    groups: FrozenOrderedDict[str, Unaliased[Column]]
    metrics: FrozenOrderedDict[str, Unaliased[Scalar]]
    window_size: Scalar[dt.Interval]
    window_slide: Optional[Scalar[dt.Interval]] = None
    window_offset: Optional[Scalar[dt.Interval]] = None

    @attribute
    def values(self):
        return FrozenOrderedDict({**self.groups, **self.metrics})

    @attribute
    def schema(self):
        field_pairs = {
            "window_start": dt.timestamp,
            "window_end": dt.timestamp,
            **{k: v.dtype for k, v in self.groups.items()},
            **{k: v.dtype for k, v in self.metrics.items()},
        }
        return Schema(field_pairs)
