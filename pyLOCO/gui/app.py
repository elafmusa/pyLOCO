"""Application entry point for the pyLOCO GUI shell.

This module owns QApplication creation and intentionally keeps all GUI
startup code separate from the numerical pyLOCO backend. Milestone 1 is
an offline application shell only; it does not execute LOCO fits.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QStyleFactory

from .main_window import MainWindow


APPLICATION_NAME = "pyLOCO GUI"
ORGANIZATION_NAME = "pyLOCO"


def build_application(argv: Sequence[str] | None = None) -> QApplication:
    """Create and configure the Qt application instance.

    Parameters
    ----------
    argv:
        Optional command-line arguments. If omitted, ``sys.argv`` is used.

    Returns
    -------
    QApplication
        Configured application object ready to show windows.
    """

    app = QApplication(list(sys.argv if argv is None else argv))
    app.setApplicationName(APPLICATION_NAME)
    app.setOrganizationName(ORGANIZATION_NAME)
    app.setStyle(QStyleFactory.create("Fusion"))
    app.setFont(QFont("Segoe UI", 10))
    return app


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pyLOCO GUI shell."""

    app = build_application(argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover - manual GUI entry point
    raise SystemExit(main())
