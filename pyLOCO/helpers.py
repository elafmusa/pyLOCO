"""Configuration loading helpers for pyLOCO.

Historically pyLOCO imported a top-level ``pyloco_config`` module.  The
canonical configuration now lives in :mod:`pyLOCO.config`; this helper keeps
example/user config files working by copying their public configuration symbols
onto the internal module and by publishing the same module under the legacy
``pyloco_config`` name.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from types import ModuleType
from typing import Optional


_CONFIG_EXPORTS = (
    "LOCOOptions",
    "loco_options",
    "RMConfig",
    "FitInitConfig",
    "FitResumeConfig",
    "LOCOAPI",
    "BACKEND",
    "get_mcf",
    "FixedParameters",
    "fixed_parameters",
    "_cfg_get",
    "BLOCK_ORDER",
    "DEFAULT_INIT_POLICY",
)


def load_config(config_path: Optional[str] = None, config_module: Optional[str] = None) -> ModuleType:
    """Load a configuration module into :mod:`pyLOCO.config`.

    Parameters are kept for backwards compatibility with existing examples.  If
    neither is supplied, the built-in internal configuration is returned unless a
    legacy ``pyloco_config`` module is importable.
    """

    if config_module:
        mod = importlib.import_module(config_module)
    elif config_path:
        spec = importlib.util.spec_from_file_location("pyloco_config", config_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load pyLOCO config from {config_path!r}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        try:
            mod = importlib.import_module("pyloco_config")
        except ModuleNotFoundError:
            import pyLOCO.config as internal_config

            sys.modules["pyloco_config"] = internal_config
            return internal_config

    return install_config(mod)


def install_config(mod: ModuleType) -> ModuleType:
    """Install ``mod`` as the active internal pyLOCO configuration."""

    import pyLOCO.config as internal_config

    for name in _CONFIG_EXPORTS:
        if hasattr(mod, name):
            setattr(internal_config, name, getattr(mod, name))
    sys.modules["pyloco_config"] = internal_config
    return internal_config
