"""PySide6 GUI package for pyLOCO.

The GUI manages LOCO projects, imports lattice and measurement files, and
orchestrates execution through the existing numerical backend from a responsive
Qt worker thread. Numerical algorithms remain in the non-GUI pyLOCO modules.
"""

__all__ = ["__version__"]

__version__ = "0.3.0"
