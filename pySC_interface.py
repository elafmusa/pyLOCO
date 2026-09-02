"""Compatibility import for integrations that used the supplied root module."""
from pyLOCO.control_system.pysc_server import pySCServerOrbitInterface

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 13131

__all__ = ["DEFAULT_HOST", "DEFAULT_PORT", "pySCServerOrbitInterface"]
