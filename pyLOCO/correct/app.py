"""Application entry point for pyLOCO Correct (offline dry-run only)."""
from __future__ import annotations
import sys
import argparse
import json
from pathlib import Path
from collections.abc import Sequence
from PySide6.QtCore import Qt,QTimer
from PySide6.QtWidgets import QApplication
from pyLOCO.gui.branding import application_icon
from pyLOCO.gui.appearance import ensure_suite_appearance
from .main_window import CorrectMainWindow
from pyLOCO.machine_mapping import is_document as is_machine_mapping

def build_application(argv: Sequence[str] | None = None) -> QApplication:
    app=QApplication.instance()
    if app is None:
        QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
        app=QApplication(list(sys.argv if argv is None else argv)); app.setApplicationName("pyLOCO Correct"); app.setApplicationDisplayName("pyLOCO Correct"); app.setOrganizationName("pyLOCO"); app.setWindowIcon(application_icon("correct"))
    else:
        app.setApplicationName("pyLOCO Correct"); app.setApplicationDisplayName("pyLOCO Correct"); app.setWindowIcon(application_icon("correct"))
    ensure_suite_appearance(app)
    return app

def main(argv: Sequence[str] | None = None) -> int:
    values=list(sys.argv[1:] if argv is None else argv); parser=argparse.ArgumentParser(prog="pyloco-correct"); parser.add_argument("source",nargs="?"); parser.add_argument("--results"); parser.add_argument("--iteration",type=int); parser.add_argument("--mapping-handoff"); args=parser.parse_args(values); source=args.results or args.source
    transferred_mapping = None
    if args.mapping_handoff:
        handoff = Path(args.mapping_handoff)
        try:
            candidate = json.loads(handoff.read_text(encoding="utf-8"))
            if not is_machine_mapping(candidate):
                raise ValueError("FIT mapping handoff is not a pyLOCO Machine Mapping document.")
            transferred_mapping = candidate
        finally:
            handoff.unlink(missing_ok=True)
    app=build_application(["pyloco-correct"]); window=CorrectMainWindow(); window.show(); window.raise_(); window.activateWindow()
    if source:
        source_path=Path(source)
        try:
            source_document=json.loads(source_path.read_text(encoding="utf-8")) if source_path.is_file() else {}
        except Exception:
            source_document={}
        if isinstance(source_document,dict) and source_document.get("schema")=="pyloco.correct_project":
            window.statusBar().showMessage(f"Loading pyLOCO Correct project: {source}")
            QTimer.singleShot(0,lambda:window.load_project_path(source))
        else:
            window.statusBar().showMessage(f"Loading pyLOCO Results: {source}")
            QTimer.singleShot(0,lambda:window.load_source_responsively(
                source, iteration=args.iteration, machine_mapping=transferred_mapping
            ))
    return app.exec()

if __name__=="__main__": raise SystemExit(main())
