"""PySide6 GUI package for pyLOCO.

The GUI manages LOCO projects, imports lattice and measurement files, and
orchestrates execution through the existing numerical backend from a responsive
Qt worker thread. Numerical algorithms remain in the non-GUI pyLOCO modules.
"""

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

try:
    # A source checkout must report its own canonical build metadata, not an
    # unrelated older editable installation present in the interpreter.
    _pyproject = Path(__file__).parents[2] / "pyproject.toml"
    __version__ = tomllib.loads(_pyproject.read_text())["project"]["version"]
except (OSError, KeyError, ValueError):
    try:
        __version__ = version("pyLOCO")
    except PackageNotFoundError:
        __version__ = "unknown"
