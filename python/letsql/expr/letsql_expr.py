import functools
from typing import Any, Generic, TypeVar

import ibis.common.exceptions as ibis_exc
import ibis.expr.types as ir

from letsql.common.caching import maybe_prevent_cross_source_caching
from letsql.common.utils.caching_utils import find_backend
from letsql.config import _backend_init
from letsql.expr.relations import CachedNode, RemoteTable, into_backend


T = TypeVar("T", bound=ir.Expr)


def wrap_ibis_function(func):
    """Decorator to wrap an Ibis function so it raises `LetSQLError`."""

    @functools.wraps(func)
    def _wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ibis_exc.IbisError as e:
            raise LetSQLError(f"Error in {func.__name__}: {e}") from e

    return _wrapped


def wrap_with_bridge_expr(fun):
    @functools.wraps(fun)
    def wrapped(*args, **kwargs):
        try:
            return BridgeExpr(fun(*args, **kwargs))
        except ibis_exc.IbisError as e:
            raise LetSQLError(f"Error in {fun.__name__}: {e}") from e

    return wrapped


class LetSQLError(Exception):
    """All user-facing errors for LetSQL."""

    pass


class BridgeExpr(Generic[T]):
    """Expr bridge around a raw Ibis expression."""

    __slots__ = ("_ibis_expr",)

    def __init__(self, ibis_expr: T):
        self._ibis_expr = ibis_expr

    def op(self):
        return self._ibis_expr.op()

    def schema(self):
        return self._ibis_expr.schema()

    def execute(self, **kwargs: Any):
        # avoid circular import
        from letsql.expr.api import execute

        return execute(self._ibis_expr, **kwargs)

    def into_backend(self, backend, name):
        new_ibis_expr = into_backend(self._ibis_expr, backend, name)
        return BridgeExpr(new_ibis_expr)

    def cache(self, storage=None) -> "BridgeExpr":
        try:
            current_backend, _ = find_backend(self._ibis_expr.op(), use_default=True)
        except ibis_exc.IbisError as e:
            if "Multiple backends found" in str(e):
                current_backend = _backend_init()
            else:
                raise

        if storage is None:
            from letsql.common.caching import SourceStorage

            storage = SourceStorage(source=current_backend)

        new_expr = maybe_prevent_cross_source_caching(self._ibis_expr, storage)

        op = CachedNode(
            schema=new_expr.schema(),
            parent=new_expr,
            source=current_backend,
            storage=storage,
        )
        cached_expr = op.to_expr()

        return BridgeExpr(cached_expr)

    def __repr__(self):
        lines = []
        lines.append("┌──── LetSQL Expression Plan ─────┐")
        lines.extend(self._ascii_plan_lines())
        lines.append("└─────────────────────────────────┘")
        return "\n".join(lines)

    def _ascii_plan_lines(self):
        op = self._ibis_expr.op()
        lines = []

        MAX_STEPS = 20
        visited = set()
        step = 0

        def arrow_line(i):
            if i == 0:
                return ""
            return "   ↓"

        while op not in visited and step < MAX_STEPS:
            visited.add(op)
            step += 1

            if isinstance(op, CachedNode):
                lines.append(arrow_line(step - 1))
                emoji = "🗃️"
                lines.append(f"  [CachedNode {emoji}]")
                lines.append(f"   source: {op.source}")
                lines.append(f"   storage: {op.storage}")
                op = op.parent.op()

            elif isinstance(op, RemoteTable):
                lines.append(arrow_line(step - 1))
                emoji = "🚚"
                lines.append(f"  [RemoteTable {emoji}]")
                lines.append(f"   name: {op.name}, source: {op.source}")
                if op.remote_expr is not None:
                    op = op.remote_expr.op()
                else:
                    break

            elif getattr(op, "__class__", None).__name__ == "InMemoryTable":
                emoji = "📦"
                lines.append(arrow_line(step - 1))
                lines.append(f"  [InMemoryTable {emoji}]")
                break

            elif getattr(op, "__class__", None).__name__ == "UnboundTable":
                emoji = "🗒️"
                lines.append(arrow_line(step - 1))
                lines.append(f"  [UnboundTable {emoji}] name={op.name}")
                break

            else:
                node_name = type(op).__name__
                lines.append(arrow_line(step - 1))
                lines.append(f"  [{node_name}] 🤷")
                break

        if step == MAX_STEPS:
            lines.append("   ... (truncated)")

        return lines
