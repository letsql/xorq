import importlib
import pathlib


def import_path(path, name=None):
    path = pathlib.Path(path)
    return importlib.machinery.SourceFileLoader(
        name or path.stem, str(path)
    ).load_module()
