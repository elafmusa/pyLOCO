#!/usr/bin/env python3
"""Native Qt click-path check for the Monday Measure demo (not offscreen)."""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from pyLOCO.measure.main_window import MeasureMainWindow


SHOT = Path(__file__).resolve().parent / "screenshots" / "monday-controls-demo" / "10-native-measurement-start-visible.png"


def close_information_dialog():
    dialog=QApplication.activeModalWidget()
    if isinstance(dialog,QMessageBox):dialog.accept()


def main() -> int:
    app=QApplication(["pyloco-native-monday-check"]); window=MeasureMainWindow(); window.resize(1200,800); window.show(); app.processEvents()
    window.adapter_combo.setCurrentIndex(window.adapter_combo.findData("pysc")); app.processEvents()
    QTimer.singleShot(50,close_information_dialog); QTest.mouseClick(window.test_connection_button,Qt.LeftButton); app.processEvents()
    if not window.connection_verified:raise RuntimeError("Machine → Test connection did not reach CONNECTED")
    window.tabs.setCurrentIndex(1); window.selection_method.setCurrentIndex(window.selection_method.findData("manual")); window.manual_input.setText("0, 1, 2"); window.refresh_preview()
    window.tabs.setCurrentIndex(2); app.processEvents()
    if not window.run_group.isVisible() or not window.start_button.isVisible():raise RuntimeError("Acquisition controls are not visible on Measurement")
    if not window.start_button.isEnabled():raise RuntimeError(window.start_block_reason.text())
    button_center=window.start_button.mapTo(window,window.start_button.rect().center())
    if not window.rect().contains(button_center):raise RuntimeError("Start button lies outside the normal 1200×800 window")
    SHOT.parent.mkdir(parents=True,exist_ok=True); window.grab().save(str(SHOT)); print(f"PASS native click path; Start visible and enabled: {SHOT}")
    window.output_directory.setText(str(Path(__file__).resolve().parent/"monday-validation-output")); window.readings.setValue(2); window.delay.setValue(0.01); QTest.mouseClick(window.start_button,Qt.LeftButton)
    def finish_when_done():
        if window.thread is not None:QTimer.singleShot(50,finish_when_done); return
        if window.result is None:raise RuntimeError("Start click did not complete BPM-noise acquisition")
        print(f"PASS native Start click; saved: {window.saved_measurement_path}"); window.close(); app.quit()
    QTimer.singleShot(50,finish_when_done); return app.exec()


if __name__=="__main__":raise SystemExit(main())
