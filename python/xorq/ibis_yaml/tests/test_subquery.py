import xorq.vendor.ibis as ibis
import xorq.vendor.ibis.expr.operations as ops


def test_scalar_subquery(compiler, t):
    expr = ops.ScalarSubquery(t.c.mean().as_table()).to_expr()
    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]

    assert expression["op"] == "ScalarSubquery"
    assert expression["args"][0]["op"] == "Aggregate"

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_exists_subquery(compiler):
    t1 = ibis.table(dict(a="int", b="string"), name="t1")
    t2 = ibis.table(dict(a="int", c="float"), name="t2")

    filtered = t2.filter(t2.a == t1.a)
    expr = ops.ExistsSubquery(filtered).to_expr()
    yaml_dict = compiler.to_yaml(expr)

    expression = yaml_dict["expression"]

    assert expression["op"] == "ExistsSubquery"
    node_ref = expression["rel"]["node_ref"]
    assert yaml_dict["definitions"]["nodes"][node_ref]["op"] == "Filter"

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_in_subquery(compiler):
    t1 = ibis.table(dict(a="int", b="string"), name="t1")
    t2 = ibis.table(dict(a="int", c="float"), name="t2")

    expr = ops.InSubquery(t1.select("a"), t2.a).to_expr()
    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]

    assert expression["op"] == "InSubquery"
    assert expression["type"]["name"] == "Boolean"

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)
