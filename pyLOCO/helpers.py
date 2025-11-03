import sys, importlib.util, importlib
from typing import Optional

def load_config(config_path: Optional[str] = None,
                config_module: Optional[str] = None):

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
        # Register directly under the canonical name
        spec = importlib.util.spec_from_file_location("pyloco_config", config_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module("pyloco_config")

    # Ensure all other imports use the same module
    sys.modules["pyloco_config"] = mod
    return mod

