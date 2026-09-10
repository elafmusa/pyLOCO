from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM","offscreen")

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication,QDialog

from pyLOCO.gui.app import build_application
from pyLOCO.gui.main_window import MainWindow
from pyLOCO.measure.main_window import MeasureMainWindow
from pyLOCO.correct.main_window import CorrectMainWindow


@pytest.mark.parametrize("kind",("fit","measure","correct"))
def test_logo_about_dialog_preserves_originating_window_page_and_reuses_dialog(kind):
    app=QApplication.instance() or build_application(["suite-about-test"])
    if kind=="fit":
        window=MainWindow(); window._workspace.setCurrentIndex(2); click=lambda:window.header_brand.click()
    elif kind=="measure":
        window=MeasureMainWindow(); window.tabs.setCurrentIndex(2); click=lambda:window.logo_button.clicked.emit()
    else:
        window=CorrectMainWindow(); window.tabs.setCurrentIndex(1); click=lambda:window.logo_button.clicked.emit()
    tabs=window._workspace if kind=="fit" else window.tabs
    original_index=tabs.currentIndex(); window.resize(1000,700); window.show(); app.processEvents()

    click(); app.processEvents(); dialog=window._about_dialog
    assert isinstance(dialog,QDialog) and dialog.parent() is window and dialog.isVisible()
    assert dialog.windowModality()==Qt.WindowModal
    assert window.isVisible() and tabs.currentIndex()==original_index
    assert dialog.width()<=720 and dialog.height()<=640

    dialog.reject(); app.processEvents()
    assert window.isVisible() and tabs.currentIndex()==original_index and not dialog.isVisible()
    click(); app.processEvents()
    assert window._about_dialog is dialog and dialog.isVisible() and tabs.currentIndex()==original_index
    dialog.reject(); window.close()
