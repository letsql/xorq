import csv
import json
import re
import sys
from io import StringIO
from pathlib import Path

import pytest

from xorq.cli import build_command, main


def test_build_command(monkeypatch, tmp_path, capsys):
    target_dir = tmp_path / "build"
    script_path = Path(__file__).absolute().parent / "fixtures" / "pipeline.py"

    test_args = [
        "xorq",
        "build",
        str(script_path),
        "-e",
        "expr",
        "--builds-dir",
        str(target_dir),
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    # Run the CLI (with try/except to prevent SystemExit)
    try:
        main()
    except SystemExit:
        pass

    # Check output
    captured = capsys.readouterr()
    assert "Building expr" in captured.out

    assert target_dir.exists()


def test_build_command_on_notebook(monkeypatch, tmp_path, capsys):
    target_dir = tmp_path / "build"
    script_path = Path(__file__).absolute().parent / "fixtures" / "pipeline.ipynb"

    test_args = [
        "xorq",
        "build",
        str(script_path),
        "-e",
        "expr",
        "--builds-dir",
        str(target_dir),
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    # Run the CLI (with try/except to prevent SystemExit)
    try:
        main()
    except SystemExit:
        pass

    # Check output
    captured = capsys.readouterr()
    assert "Building expr" in captured.out

    assert target_dir.exists()


@pytest.mark.parametrize("output_format", ["csv", "json"])
def test_run_command(monkeypatch, tmp_path, capsys, output_format):
    target_dir = tmp_path / "build"
    script_path = Path(__file__).absolute().parent / "fixtures" / "pipeline.py"

    build_command(str(script_path), ["expr"], builds_dir=str(target_dir))
    capture = capsys.readouterr()

    if match := re.search(f"{target_dir}/([0-9a-f]+)", str(capture.out)):
        expression_hash = match.group(1)
        test_args = [
            "xorq",
            "run",
            expression_hash,
            "--builds-dir",
            str(target_dir),
            "--format",
            output_format,
        ]
        monkeypatch.setattr(sys, "argv", test_args)

        try:
            main()
        except SystemExit:
            pass

        capture = capsys.readouterr()
        run_capture = str(capture.out)

        match output_format:
            case "csv":
                reader = csv.DictReader(StringIO(run_capture))
                assert list(reader)
            case "json":
                assert json.loads(run_capture)
    else:
        raise AssertionError("No expression hash")


def test_build_command_not_implemented(monkeypatch, capsys):
    script_path = Path(__file__).absolute().parent / "fixtures" / "pipeline.py"

    test_args = [
        "xorq",
        "build",
        str(script_path),
        "--builds-dir",
        str(Path.cwd()),
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    # Run the CLI (with try/except to prevent SystemExit)
    try:
        main()
    except SystemExit:
        pass

    # Check output
    captured = capsys.readouterr()
    assert "Expected one, and only one expression" in captured.err


@pytest.mark.parametrize(
    "expression,message",
    [
        ("integer", "The object integer must be an instance of"),
        ("missing", "Expression missing not found"),
    ],
)
def test_build_command_bad_expr_name(
    monkeypatch, tmp_path, capsys, expression, message
):
    target_dir = tmp_path / "build"
    script_path = Path(__file__).absolute().parent / "fixtures" / "pipeline.py"

    test_args = [
        "xorq",
        "build",
        str(script_path),
        "-e",
        expression,
        "--builds-dir",
        str(target_dir),
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    # Run the CLI (with try/except to prevent SystemExit)
    try:
        main()
    except SystemExit:
        pass

    # Check output
    captured = capsys.readouterr()
    assert message in captured.err
