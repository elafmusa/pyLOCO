from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication, QMessageBox

from pyLOCO.correct.main_window import CorrectMainWindow
from pyLOCO.gui.main_window import MainWindow
from pyLOCO.measure.main_window import MeasureMainWindow


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _result(controller, label):
    return next(result for result in controller.search(label) if result.label == label)


def test_fit_search_finds_documented_examples_and_navigates_without_activation(
    app, monkeypatch,
):
    window = MainWindow()
    window.show()
    app.processEvents()
    search = window.control_search

    assert any(result.label == "Save Jacobian" for result in search.search("Jacobian"))
    assert any("Normal Quadrupoles" in result.label for result in search.search("Quadrupole"))
    assert any(result.label == "Save Screenshot" for result in search.search("Screenshot"))
    assert any(result.label == "Find Machine Mapping" for result in search.search("Mapping"))

    fixed_path = _result(search, "fixedpathlength")
    assert fixed_path.widget is window.rm_fixedpath
    assert fixed_path.requires_advanced
    original = window.rm_fixedpath.isChecked()
    monkeypatch.setattr(QMessageBox, "question", lambda *_args, **_kwargs: QMessageBox.Yes)
    assert search.navigate_to(fixed_path)
    app.processEvents()
    assert window.project.mode == "Advanced"
    assert window._workspace.tabText(window._workspace.currentIndex()) == "Fit"
    assert window.rm_fixedpath.isChecked() is original

    search.shortcut.activated.emit()
    assert search.search_box.hasFocus()
    window.project.modified = False
    window.close()


def test_search_result_rows_show_control_and_location(app):
    window = MainWindow()
    window.show()
    window.control_search.search_box.setText("Mapping")
    app.processEvents()
    assert window.control_search.result_list.count() > 0
    first = window.control_search.result_list.item(0).text()
    assert "—" in first
    assert "Machine Components" in "\n".join(
        window.control_search.result_list.item(row).text()
        for row in range(window.control_search.result_list.count())
    )
    window.project.modified = False
    window.close()


def test_measure_search_explains_disabled_control_without_clicking(app, monkeypatch):
    window = MeasureMainWindow()
    window.show()
    app.processEvents()
    search = window.control_search
    assert any(result.label == "Find Machine Mapping" for result in search.search("Mapping"))
    assert any(result.label == "Save Screenshot" for result in search.search("Screenshot"))

    target = _result(search, "Open in FIT")
    spy = QSignalSpy(window.session_toolbar_buttons["Open in FIT"].clicked)
    messages = []
    monkeypatch.setattr(
        QMessageBox, "information",
        lambda _parent, title, message, *_args, **_kwargs: messages.append((title, message)),
    )
    assert not target.widget.isEnabled()
    assert search.navigate_to(target)
    app.processEvents()
    assert spy.count() == 0
    assert not target.widget.isEnabled()
    assert messages and "disabled" in messages[-1][1]
    window._session_dirty = False
    window.close()


def test_correct_search_opens_mapping_tab_without_activating_action(app, monkeypatch):
    window = CorrectMainWindow()
    window.show()
    app.processEvents()
    search = window.control_search
    mapping = _result(search, "Load Machine Mapping")
    spy = QSignalSpy(mapping.widget.clicked)
    monkeypatch.setattr(QMessageBox, "information", lambda *_args, **_kwargs: None)
    assert search.navigate_to(mapping)
    app.processEvents()
    assert window.tabs.currentIndex() == 1
    assert spy.count() == 0
    assert any(result.label == "Save Screenshot" for result in search.search("Screenshot"))
    window.close()
