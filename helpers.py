import sys, importlib.util, importlib

def load_config(config_path: str | None = None,
                config_module: str | None = None):
    """
    Load a pyloco_config from either:
      - a filesystem path to a Python file, or
      - a Python module path, e.g. 'Examples.ESRF.pyloco_config'.

    Registers it under the canonical name 'pyloco_config' so that
    all existing imports in the repo keep working unchanged.
    """
    if config_module:
        mod = importlib.import_module(config_module)
    elif config_path:
        spec = importlib.util.spec_from_file_location("pyloco_config_user", config_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module("pyloco_config")

    sys.modules["pyloco_config"] = mod
    return mod
