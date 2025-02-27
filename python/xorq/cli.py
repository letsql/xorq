import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

from xorq.ibis_yaml.compiler import BuildManager
from xorq.vendor.ibis import Expr


def import_from_path(path):
    """
    Import a Python script or Jupyter notebook as a module.

    Args:
        path (Path or str): pathlib.Path object or string pointing to a Python script (.py) or
                           Jupyter notebook (.ipynb)

    Returns:
        module: The imported module

    Raises:
        ImportError: If the file cannot be imported
        ValueError: If the file type is not supported
    """
    # Convert to Path object if it's a string
    if isinstance(path, str):
        path = Path(path)

    # Check if path exists
    if not path.exists():
        raise ImportError(f"File not found: {path}")

    # Create a unique module name based on the file path
    module_name = f"dynamically_imported_{path.stem}"

    # Handle based on file extension
    if path.suffix == ".py":
        return _import_python_file(path, module_name)
    elif path.suffix == ".ipynb":
        return _import_jupyter_notebook(path, module_name)
    else:
        raise ValueError(
            f"Unsupported file type: {path.suffix}. Only .py and .ipynb files are supported."
        )


def _import_python_file(path, module_name):
    """Import a Python file as a module."""
    file_path = str(path.resolve())

    # Create the spec
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Could not load script: {file_path}")

    # Create the module
    module = importlib.util.module_from_spec(spec)

    # Add the module to sys.modules
    sys.modules[module_name] = module

    # Execute the module in its own namespace
    spec.loader.exec_module(module)

    return module


def _import_jupyter_notebook(path, module_name):
    """Import a Jupyter notebook as a module."""
    file_path = str(path.resolve())

    # Create a new module
    module = type(sys)(module_name)
    module.__file__ = file_path

    # Add the module to sys.modules
    sys.modules[module_name] = module

    # Parse the notebook JSON
    with open(file_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    # Extract code from code cells
    code_cells = [
        cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"
    ]

    # Combine all code cells, skipping cells marked with "# skip-import" comment
    code = ""
    for cell in code_cells:
        cell_source = "".join(cell.get("source", []))
        if "# skip-import" not in cell_source:
            code += cell_source + "\n\n"

    # Execute the code in the module's namespace
    try:
        exec(code, module.__dict__)
    except Exception as e:
        # Clean up if execution fails
        if module_name in sys.modules:
            del sys.modules[module_name]
        raise ImportError(f"Error executing notebook code: {e}")

    return module


def build_command(script_path, expression, target_dir="builds"):
    """
    Generate artifacts from an expression in a given Python script

    Parameters
    ----------
    script_path : Path to the Python script
    expression : The name of the expression to build
    target_dir : Directory where artifacts will be generated

    Returns
    -------

    """

    if len(expression) > 1 or len(expression) == 0:
        print("Expected one, and only one expression", file=sys.stderr)
        sys.exit(1)

    (expression,) = expression

    if not os.path.exists(script_path):
        print(f"Error: Script not found at {script_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Building {expression} from {script_path}")

    build_manager = BuildManager(target_dir)

    vars_module = import_from_path(script_path)

    if not hasattr(vars_module, expression):
        print(f"Expression {expression} not found", file=sys.stderr)
        sys.exit(1)

    expr = getattr(vars_module, expression)

    if not isinstance(expr, Expr):
        print(
            f"The object {expression} must be an instance of {type(expr)}",
            file=sys.stderr,
        )
        sys.exit(1)

    expr_hash = build_manager.compile_expr(expr)
    print(
        f"Written '{expression}' to {build_manager.artifact_store.get_path(expr_hash)}"
    )


def run_command(builds_dir, hash_id):
    """
    Run a build, by recreating the expression from the build

    Parameters
    ----------
    builds_dir : Path to the builds directory
    hash_id : The hash identifier of the build to run

    Returns
    -------

    """
    try:
        build_manager = BuildManager(builds_dir)
        expr = build_manager.load_expr(hash_id)
        print(f"Executing expression with {hash_id}")
        expr.execute()

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point for the xorq CLI."""
    parser = argparse.ArgumentParser(description="xorq - build and run expressions")

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    subparsers.required = True

    # Create parser for the "build" command
    build_parser = subparsers.add_parser(
        "build", help="Generate artifacts from an expression"
    )
    build_parser.add_argument("script_path", help="Path to the Python script")
    build_parser.add_argument(
        "-e",
        "--expressions",
        nargs="?",
        help="Name of the expression variable in the Python script",
    )
    build_parser.add_argument(
        "--target-dir", default="build", help="Directory for all generated artifacts"
    )

    # Create parser for the "run" command
    run_parser = subparsers.add_parser(
        "run", help="Run a build from a builds directory"
    )
    run_parser.add_argument("builds_dir", help="Path to the builds directory")
    run_parser.add_argument(
        "expression_hash", help="Hash identifier of the build to run"
    )

    args = parser.parse_args()

    match args.command:
        case "build":
            expressions = [args.expressions] if args.expressions else []
            build_command(args.script_path, expressions, args.target_dir)
        case "run":
            run_command(args.builds_dir, args.expression_hash)


if __name__ == "__main__":
    main()
