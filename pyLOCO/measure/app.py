"""Application entry point for pyLOCO Measure."""
from __future__ import annotations

import sys
import argparse
from pathlib import Path
from collections.abc import Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from pyLOCO.gui.branding import application_icon
from .main_window import MeasureMainWindow


def build_application(argv: Sequence[str] | None = None) -> QApplication:
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication.instance() or QApplication(list(sys.argv if argv is None else argv))
    app.setApplicationName("pyLOCO Measure")
    app.setOrganizationName("pyLOCO")
    app.setWindowIcon(application_icon())
    return app


def main(argv: Sequence[str] | None = None) -> int:
    values=list(sys.argv[1:] if argv is None else argv); parser=argparse.ArgumentParser(prog="pyloco-measure"); parser.add_argument("project",nargs="?"); args=parser.parse_args(values)
    app = build_application(["pyloco-measure"])
    window = MeasureMainWindow()
    if args.project:
        from .project import load_measure_project
        window.project=load_measure_project(Path(args.project)); window.project_path=Path(args.project).expanduser().resolve(); window._load_project_widgets()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
