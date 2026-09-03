"""Application entry point for pyLOCO Correct (offline dry-run only)."""
from __future__ import annotations
import sys
import argparse
from collections.abc import Sequence
from PySide6.QtCore import Qt,QTimer
from PySide6.QtWidgets import QApplication
from pyLOCO.gui.branding import application_icon
from pyLOCO.gui.appearance import ensure_suite_appearance
from .main_window import CorrectMainWindow

def build_application(argv: Sequence[str] | None = None) -> QApplication:
    app=QApplication.instance()
    if app is None:
        QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
        app=QApplication(list(sys.argv if argv is None else argv)); app.setApplicationName("pyLOCO Correct"); app.setOrganizationName("pyLOCO"); app.setWindowIcon(application_icon())
    ensure_suite_appearance(app)
    return app

def main(argv: Sequence[str] | None = None) -> int:
    values=list(sys.argv[1:] if argv is None else argv); parser=argparse.ArgumentParser(prog="pyloco-correct"); parser.add_argument("source",nargs="?"); parser.add_argument("--results"); parser.add_argument("--iteration",type=int); args=parser.parse_args(values); source=args.results or args.source
    app=build_application(["pyloco-correct"]); window=CorrectMainWindow(); window.show(); window.raise_(); window.activateWindow()
    if source:
        window.statusBar().showMessage(f"Loading pyLOCO Results: {source}")
        QTimer.singleShot(0,lambda:window._load(source,iteration=args.iteration))
    return app.exec()

if __name__=="__main__": raise SystemExit(main())
