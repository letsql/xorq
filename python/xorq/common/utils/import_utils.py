import importlib


def import_path(path, name=None):
    return importlib.machinery.SourceFileLoader(
        name or path.stem, str(path)
    ).load_module()
