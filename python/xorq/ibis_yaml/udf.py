import base64
from typing import Any

import cloudpickle

import xorq.expr.datatypes as dt
import xorq.vendor.ibis.expr.operations as ops
import xorq.vendor.ibis.expr.rules as rlz
from xorq.ibis_yaml.common import (
    _translate_type,
    register_from_yaml_handler,
    translate_from_yaml,
    translate_to_yaml,
)
from xorq.ibis_yaml.utils import freeze
from xorq.vendor.ibis.common.annotations import Argument


def serialize_udf_function(fn: callable) -> str:
    pickled = cloudpickle.dumps(fn)
    encoded = base64.b64encode(pickled).decode("ascii")
    return encoded


def deserialize_udf_function(encoded_fn: str) -> callable:
    pickled = base64.b64decode(encoded_fn)
    return cloudpickle.loads(pickled)


@translate_to_yaml.register(ops.ScalarUDF)
def _scalar_udf_to_yaml(op: ops.ScalarUDF, compiler: Any) -> dict:
    if getattr(op.__class__, "__input_type__", None) != ops.udf.InputType.BUILTIN:
        raise NotImplementedError(
            f"Translation of UDFs with input type {getattr(op.__class__, '__input_type__', None)} is not supported"
        )
    arg_names = [
        name
        for name in dir(op)
        if not name.startswith("__") and name not in op.__class__.__slots__
    ]

    return freeze(
        {
            "op": "ScalarUDF",
            "func_name": op.__func_name__,
            "input_type": "builtin",
            "args": [translate_to_yaml(arg, compiler) for arg in op.args],
            "type": _translate_type(op.dtype),
            "pickle": serialize_udf_function(op.__func__),
            "module": op.__module__,
            "class_name": op.__class__.__name__,
            "arg_names": arg_names,
        }
    )


@register_from_yaml_handler("ScalarUDF")
def _scalar_udf_from_yaml(yaml_dict: dict, compiler: any) -> any:
    encoded_fn = yaml_dict.get("pickle")
    if not encoded_fn:
        raise ValueError("Missing pickle data for ScalarUDF")
    fn = deserialize_udf_function(encoded_fn)

    args = tuple(
        translate_from_yaml(arg, compiler) for arg in yaml_dict.get("args", [])
    )
    if not args:
        raise ValueError("ScalarUDF requires at least one argument")

    arg_names = yaml_dict.get("arg_names", [f"arg{i}" for i in range(len(args))])

    fields = {
        name: Argument(pattern=rlz.ValueOf(arg.type()), typehint=arg.type())
        for name, arg in zip(arg_names, args)
    }

    bases = (ops.ScalarUDF,)
    meta = {
        "dtype": dt.dtype(yaml_dict["type"]["name"]),
        "__input_type__": ops.udf.InputType.BUILTIN,
        "__func__": property(fget=lambda _, f=fn: f),
        "__config__": {"volatility": "immutable"},
        "__udf_namespace__": None,
        "__module__": yaml_dict.get("module", "__main__"),
        "__func_name__": yaml_dict["func_name"],
    }

    kwds = {**fields, **meta}
    class_name = yaml_dict.get("class_name", yaml_dict["func_name"])

    node = type(
        class_name,
        bases,
        kwds,
    )

    return node(*args).to_expr()
