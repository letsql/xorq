import argparse
import importlib.util
import os
import sys

from xorq.ibis_yaml.compiler import BuildManager
from xorq.vendor.ibis import Expr


def load_variables_from_script(script_path):
    """Load variables from a Python script."""
    # Get the module name from the file path
    module_name = script_path.replace("/", ".").rstrip(".py")

    # Load the module specification
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None:
        raise ImportError(f"Could not load script: {script_path}")

    # Create the module
    module = importlib.util.module_from_spec(spec)

    # Execute the module
    spec.loader.exec_module(module)

    # Return the module (containing all variables)
    return module


def build_command(script_path, expression, target_dir="build"):
    """

        Implementation of the 'build' command.

    Args:
        script_path (str): Path to the Python script
        name (str): Name for the build
        target_dir (str): Directory where artifacts will be generated

    Parameters
    ----------
    script_path : Path to the Python script
    expression : The name of the expression to build
    target_dir : Directory where artifacts will be generated

    Returns
    -------

    """

    if len(expression) > 1 or len(expression) == 0:
        print("Expected one, and only one expression")
        sys.exit(1)

    (expression,) = expression

    if not os.path.exists(script_path):
        print(f"Error: Script not found at {script_path}")
        sys.exit(1)

    print(f"Building {expression} from {script_path}")

    # Create the target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    build_manager = BuildManager(target_dir)

    script_vars = load_variables_from_script(script_path)

    if not hasattr(script_vars, expression):
        print(f"Expression {expression} not found")
        sys.exit(1)

    expr = getattr(script_vars, expression)

    if not isinstance(expr, Expr):
        print(f"The object {expression} must be an instance of {type(expr)}")
        sys.exit(1)

    build_manager.compile_expr(expr)


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

    # Parse the arguments
    args = parser.parse_args()

    # Execute the appropriate command
    if args.command == "build":
        expressions = [args.expressions] if args.expressions else []
        build_command(args.script_path, expressions, args.target_dir)


if __name__ == "__main__":
    main()
