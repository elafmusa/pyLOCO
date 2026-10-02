import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from pyLOCO.correct.main_window import CorrectMainWindow
from pyLOCO.correct import pyaml_widget
from test_correct_pyaml import setup


class Settings:
    values = {}

    def __init__(self, *args):
        pass

    def value(self, key, default=None):
        return self.values.get(key, default)

    def setValue(self, key, value):
        self.values[key] = value


@pytest.fixture
def window(monkeypatch):
    app = QApplication.instance() or QApplication([])
    Settings.values = {}
    monkeypatch.setattr(pyaml_widget, "QSettings", Settings)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: QMessageBox.Ok)
    owner = CorrectMainWindow()
    owner.backend_combo.setCurrentIndex(owner.backend_combo.findData("pyaml"))
    yield owner
    workspace = owner.pyaml_workspace
    workspace.timer.stop()
    if workspace.pending():
        workspace.transaction.restore()
    owner.close()
    app.processEvents()


def test_defaults_browser_cancel_and_connection_confirmation(window, setup, monkeypatch):
    widget = window.pyaml_workspace
    assert widget.mode.currentText() == "live"
    assert widget.fields["normal_array"].text() == "Quads"
    assert widget.fields["skew_array"].text() == "SkewQuads"
    assert widget.writes.isChecked()
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(setup.config), ""))
    widget.browse_button.click()
    assert widget.fields["config_path"].text() == str(setup.config)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: ("", ""))
    widget.browse_button.click()
    assert widget.fields["config_path"].text() == str(setup.config)
    calls = []
    def connect(**options):
        calls.append(options)
        return setup.connection
    monkeypatch.setattr(pyaml_widget, "PyAMLCorrectionConnection", connect)
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.No)
    widget.connect_machine()
    assert not calls
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.Yes)
    widget.connect_machine()
    assert len(calls) == 1
    assert Settings.values["correct/pyaml/normal_array"] == "Quads"
    assert all(not magnet.strength.writes for magnet in setup.magnets.values())
    widget.fields["skew_array"].setText("Other")
    assert widget.connection is None


def connect_workspace(window, setup):
    widget = window.pyaml_workspace
    window.review = setup.review
    widget.connection = setup.connection
    widget.transaction = setup.transaction
    return widget


def test_close_keeps_pyaml_strengths_and_recovery_journal(window, setup, monkeypatch):
    from PySide6.QtGui import QCloseEvent
    from pyLOCO.correct.pyaml_transaction import PyAMLCorrectionTransaction
    widget = connect_workspace(window, setup)
    widget.preview()
    setup.transaction.apply(setup.review, confirmed=True)
    strengths = {name: magnet.strength.value for name, magnet in setup.magnets.items()}
    writes = {name: list(magnet.strength.writes) for name, magnet in setup.magnets.items()}
    journal = setup.transaction.path.read_bytes()
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args))
    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted()
    assert not warnings
    assert {name: magnet.strength.value for name, magnet in setup.magnets.items()} == strengths
    assert {name: magnet.strength.writes for name, magnet in setup.magnets.items()} == writes
    assert setup.transaction.path.read_bytes() == journal
    recovered = PyAMLCorrectionTransaction(setup.connection, setup.transaction.directory)
    recovered.recover(setup.transaction.path)
    assert recovered.pending


def test_close_blocked_during_pyaml_operation(window, monkeypatch):
    from PySide6.QtGui import QCloseEvent
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args))
    window.pyaml_workspace.thread = object()
    try:
        event = QCloseEvent()
        window.closeEvent(event)
        assert not event.isAccepted()
        assert warnings[0][1] == "Operation in progress"
    finally:
        window.pyaml_workspace.thread = None


def test_apply_mouse_click_survives_refresh(window, setup, monkeypatch):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    widget = connect_workspace(window, setup)
    widget.timer.stop()
    widget.preview(); widget.refresh()
    window.tabs.setCurrentIndex(3)
    window.show()
    QApplication.processEvents()
    confirmations = []
    def confirm(*args):
        confirmations.append(args)
        return QMessageBox.No
    monkeypatch.setattr(QMessageBox, "question", confirm)
    QTest.mousePress(window.apply_button, Qt.LeftButton)
    assert window.apply_button.isDown()
    widget.refresh()
    QTest.mouseRelease(window.apply_button, Qt.LeftButton)
    assert len(confirmations) == 1
    assert all(not magnet.strength.writes for magnet in setup.magnets.values())


def test_shared_workflow_and_backend_switch(window, setup):
    widget = connect_workspace(window, setup)
    widget.refresh()
    assert window.tabs.count() == 4
    assert not window.pyaml_connection_group.isHidden()
    assert window.pysc_connection_group.isHidden()
    assert widget.table is window.table
    assert widget.apply_button is window.apply_button
    assert widget.preview_button is window.preview_apply_button
    assert widget.restore_button is window.restore_button
    assert window.table.item(0, 5).text() == "mapped (identity)"
    window.backend_combo.setCurrentIndex(window.backend_combo.findData("mock"))
    assert window.pyaml_connection_group.isHidden()
    assert not window.pysc_connection_group.isHidden()
    assert window.preview_apply_button.isEnabled()
    assert all(item.control_name is None for item in setup.review.items)


def test_shared_mapping_import_export(window, setup, tmp_path, monkeypatch):
    from pyLOCO.correct.petra_readonly import load_mapping
    widget = connect_workspace(window, setup)
    widget.refresh()
    path = tmp_path / "mapping.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(path), ""))
    window.generate_mapping_template()
    mappings = load_mapping(path)
    assert [entry.control_name for entry in mappings] == ["M1", "M2", "S1"]
    assert [entry.component for entry in mappings] == ["B2L", "B2L", "A2L"]
    widget.clear_mapping()
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(path), ""))
    window.load_petra_mapping()
    assert widget.mapping_path == path
    assert all(item.metadata["match_source"] == "explicit" for item in setup.review.items)
    assert all(not magnet.strength.writes for magnet in setup.magnets.values())


def test_mapping_table_preview_invalidates_and_pending_locks(window, setup):
    widget = connect_workspace(window, setup)
    widget.preview(); widget.refresh()
    assert widget.table.rowCount() == 3
    assert widget.table.item(0, 4).text() == "M1"
    assert widget.apply_button.isEnabled()
    setup.review.items[0].individual_scale = .5
    widget.refresh()
    assert not widget.apply_button.isEnabled()
    widget.preview()
    setup.transaction.apply(setup.review, confirmed=True)
    widget.refresh()
    assert not widget.settings_box.isEnabled()
    assert not widget.mapping_box.isEnabled()
    assert not window.backend_combo.isEnabled()
    assert widget.restore_button.isEnabled()
    assert window.tabs.isTabEnabled(3)
    original = window.review
    window._accept_loaded_review(SimpleNamespace(items=[]), setup.source)
    assert window.review is original
    setup.transaction.restore(); widget.refresh()
    assert window.backend_combo.isEnabled()


@pytest.mark.parametrize("fraction", [.1, .25, 1.0])
def test_background_apply_and_restore(window, setup, monkeypatch, fraction):
    import time
    widget = connect_workspace(window, setup)
    setup.review.set_global_scale(fraction)
    widget.preview(); widget.refresh()
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.Yes)
    window.apply_button.click()
    deadline = time.monotonic() + 5
    while widget.thread and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(.01)
    assert widget.thread is None
    assert setup.transaction.record["status"] == "applied"
    assert setup.transaction.record["fraction"] == fraction
    window.restore_button.click()
    deadline = time.monotonic() + 5
    while widget.thread and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(.01)
    assert widget.thread is None
    assert not setup.transaction.pending


def test_failed_connection_discards_old_connection(window, setup, monkeypatch):
    widget = connect_workspace(window, setup)
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.Yes)
    def fail(**options):
        raise RuntimeError("unavailable")
    monkeypatch.setattr(pyaml_widget, "PyAMLCorrectionConnection", fail)
    with pytest.raises(RuntimeError, match="unavailable"):
        widget.connect_machine()
    assert widget.connection is None
    assert widget.transaction is None


def test_recovery_displays_frozen_mapping_without_loaded_review(window, setup, monkeypatch):
    from pyLOCO.correct.pyaml_transaction import PyAMLCorrectionTransaction
    tx = setup.transaction
    tx.preview(setup.review); tx.apply(setup.review, confirmed=True)
    writes_before = sum(len(magnet.strength.writes) for magnet in setup.magnets.values())
    widget = window.pyaml_workspace
    widget.connection = setup.connection
    widget.transaction = PyAMLCorrectionTransaction(setup.connection, tx.directory)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(tx.path), ""))
    widget.recover(); widget.refresh()
    assert sum(len(magnet.strength.writes) for magnet in setup.magnets.values()) == writes_before
    assert window.machine_changes_table.rowCount() == 3
    assert window.machine_changes_table.item(0, 0).text() == "M1"
    assert widget.restore_button.isEnabled()
    assert not widget.apply_button.isEnabled()
