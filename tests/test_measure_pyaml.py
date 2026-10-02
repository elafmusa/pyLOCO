import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QMessageBox, QFileDialog

from pyLOCO.control_system import InterfaceRegistry
from pyLOCO.measure.main_window import MeasureMainWindow
from test_pyaml_adapter import make_adapter, writable_adapter


def test_configuration_file_browser_and_default_mode(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    window = MeasureMainWindow()
    selected = str(tmp_path / "machine.yaml")
    calls = []
    def browse(*args, **kwargs):
        calls.append((args, kwargs))
        return selected, ""
    monkeypatch.setattr(QFileDialog, "getOpenFileName", browse)
    try:
        window.adapter_combo.setCurrentIndex(window.adapter_combo.findData("pyaml"))
        assert window.pyaml_fields["mode"].text() == "live"
        window.pyaml_browse_button.click()
        assert window.pyaml_fields["config_path"].text() == selected
        assert not window.connection_verified
        assert "*.yaml" in calls[0][0][3]
        window.pyaml_browse_button.click()
        assert calls[-1][0][2] == selected
        monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: ("", ""))
        window.pyaml_browse_button.click()
        assert window.pyaml_fields["config_path"].text() == selected
        window.project.metadata.pop("pyaml_options", None)
        window._load_project_widgets()
        assert window.pyaml_fields["mode"].text() == "live"
        window.project.metadata["pyaml_options"] = {"mode": "design"}
        window._load_project_widgets()
        assert window.pyaml_fields["mode"].text() == "design"
    finally:
        window.close()
        app.processEvents()


def test_connect_invalidate_and_project_settings(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = MeasureMainWindow()
    window.pyaml_magnet_writes.setChecked(False)
    window.pyaml_rf_writes.setChecked(False)
    try:
        window.adapter_combo.setCurrentIndex(window.adapter_combo.findData("pyaml"))
        assert not window.devices
        assert not window.start_button.isEnabled()
        window.pyaml_fields["config_path"].setText("machine.yaml")
        adapter, bpms = make_adapter()
        monkeypatch.setattr(InterfaceRegistry, "create", lambda self, key: SimpleNamespace(adapter=adapter))
        window._test_connection()
        assert window.connection_verified
        assert [device.name for device in window.devices] == ["B2", "B1"]
        assert window.start_button.isEnabled()
        window._refresh_corrector_preview("hcor")
        assert window.corrector_selection_widgets["hcor"]["table"].item(0, 5).text() == "2"
        project = window._collect_project()
        assert project.metadata["pyaml_options"]["config_path"] == "machine.yaml"
        window.pyaml_fields["mode"].setText("design")
        assert not window.connection_verified
        assert not window.devices
        assert not window.start_button.isEnabled()
        monkeypatch.setattr(QMessageBox, "warning", lambda *args: None)
        bpms.values = [[1, 2]]
        window._test_connection()
        assert not window.connection_verified
        assert not window.devices
        window.project = project
        window._load_project_widgets()
        assert window.pyaml_fields["mode"].text() == "live"
        assert not window.connection_verified
    finally:
        window.close()
        app.processEvents()


def test_write_opt_in_live_confirmation_and_project_reset(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = MeasureMainWindow()
    calls = []
    def create(registry, key):
        calls.append(registry.pyaml_options)
        adapter, _ = writable_adapter(allow_writes=registry.pyaml_options["allow_writes"],
                                      allow_rf_writes=registry.pyaml_options["allow_rf_writes"])
        return SimpleNamespace(adapter=adapter)
    monkeypatch.setattr(InterfaceRegistry, "create", create)
    try:
        window.adapter_combo.setCurrentIndex(window.adapter_combo.findData("pyaml"))
        assert window.pyaml_fields["mode"].text() == "live"
        assert window.pyaml_fields["rf_name"].text() == "RF"
        assert window.pyaml_magnet_writes.isChecked()
        assert window.pyaml_rf_writes.isChecked()
        monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.No)
        window._test_connection()
        assert not calls
        assert not window.connection_verified
        monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.Yes)
        window._test_connection()
        assert window.connection_verified
        assert window.rf_control_mode.currentData() == "automatic"
        assert window.measurement_type.model().item(window.measurement_type.findData("orm")).isEnabled()
        assert "WRITES ENABLED" in window.status_badge.text()
        window.project = window._collect_project()
        assert "allow_writes" not in window.project.metadata["pyaml_options"]
        window._load_project_widgets()
        assert window.pyaml_magnet_writes.isChecked()
        assert window.pyaml_rf_writes.isChecked()
        assert not window.connection_verified
    finally:
        window.close()
        app.processEvents()
