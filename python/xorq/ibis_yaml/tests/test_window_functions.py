import xorq.vendor.ibis as ibis


def test_window_function_roundtrip(compiler, t):
    expr = t.select(
        [
            t.c.mean()
            .over(ibis.window(preceding=5, following=0, group_by=t.a))
            .name("mean_c")
        ]
    )

    yaml_dict = compiler.to_yaml(expr)

    reconstructed_expr = compiler.from_yaml(yaml_dict)

    assert expr.equals(reconstructed_expr)


def test_aggregation_window(compiler, t):
    cases = [
        (None, None),
        (5, 0),
        (0, 5),
        (5, 5),
    ]

    for preceding, following in cases:
        expr = t.select(
            [
                t.c.mean()
                .over(
                    ibis.window(preceding=preceding, following=following, group_by=t.a)
                )
                .name("mean_c")
            ]
        )

        yaml_dict = compiler.to_yaml(expr)
        node_ref = yaml_dict["expression"]["node_ref"]

        expression = yaml_dict["definitions"]["nodes"][node_ref]
        assert expression["op"] == "Project"
        window_func = expression["values"]["mean_c"]
        assert window_func["op"] == "WindowFunction"
        assert window_func["args"][0]["op"] == "Mean"

        if preceding is None:
            assert "start" not in window_func
        else:
            assert window_func["start"] == preceding

        if following is None:
            assert "end" not in window_func
        else:
            assert window_func["end"] == following


def test_row_number_simple_roundtrip(compiler, t):
    expr = t.select([ibis.row_number().name("row_num")])
    yaml_dict = compiler.to_yaml(expr)
    reconstructed_expr = compiler.from_yaml(yaml_dict)
    assert expr.equals(reconstructed_expr)


def test_row_number_window_roundtrip(compiler, t):
    expr = t.select(
        [
            ibis.row_number()
            .over(
                ibis.window(
                    group_by=[t.a, t.b],
                    order_by=[t.c.desc(), t.d],
                    preceding=5,
                    following=0,
                )
            )
            .name("row_num")
        ]
    )
    yaml_dict = compiler.to_yaml(expr)
    reconstructed_expr = compiler.from_yaml(yaml_dict)
    assert expr.equals(reconstructed_expr)


def test_multiple_rank_expressions_roundtrip(compiler, t):
    expr = t.select(
        [
            ibis.row_number().over(ibis.window(group_by=t.a)).name("simple_row_num"),
            ibis.row_number()
            .over(ibis.window(group_by=[t.a, t.b], order_by=t.c.desc()))
            .name("ordered_row_num"),
            t.c.mean()
            .over(ibis.window(preceding=3, following=0, group_by=t.a))
            .name("mean_c"),
        ]
    )
    yaml_dict = compiler.to_yaml(expr)
    reconstructed_expr = compiler.from_yaml(yaml_dict)
    assert expr.equals(reconstructed_expr)
