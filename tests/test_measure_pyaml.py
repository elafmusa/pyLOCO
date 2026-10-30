import os
import time
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QMessageBox, QFileDialog

from pyLOCO.control_system import InterfaceRegistry
from pyLOCO.measure.acquisition import BpmDevice, CorrectorDevice
from pyLOCO.measure.main_window import MeasureMainWindow
from test_pyaml_adapter import make_adapter, writable_adapter


def _wait_for_connection_test(app, window, timeout=5.0):
    deadline = time.monotonic() + timeout
    while getattr(window, "connection_thread", None) is not None and time.monotonic() < deadline:
        app.processEvents()
    assert getattr(window, "connection_thread", None) is None


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
        _wait_for_connection_test(app, window)
        assert window.connection_verified
        assert window.adapter_combo.currentData() == "pyaml"
        assert window.measurement_label.text() == "pyLOCO BPM noise — pyAML"
        assert window.header_mode.text() == "Mode · Live"
        assert window.header_capabilities.text() == "Available · Magnet + RF"
        assert window.header_access.text() == "Authorized · Read Only"
        assert window.header_configuration.toolTip() == "machine.yaml"
        assert [device.name for device in window.devices] == ["B2", "B1"]
        assert window.start_button.isEnabled()
        window._refresh_corrector_preview("hcor")
        diagnostics=window.corrector_selection_widgets["hcor"]["table"]
        assert diagnostics.item(0, 4).text() == "2 mrad"
        assert diagnostics.item(0, 5).text() == "Not supported"
        project = window._collect_project()
        assert project.metadata["pyaml_options"]["config_path"] == "machine.yaml"
        window.pyaml_fields["mode"].setText("design")
        assert not window.connection_verified
        assert not window.devices
        assert not window.start_button.isEnabled()
        monkeypatch.setattr(QMessageBox, "warning", lambda *args: None)
        bpms.values = [[1, 2]]
        window._test_connection()
        _wait_for_connection_test(app, window)
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
        assert window.header_capabilities.text() == "Available · Magnet + RF"
        assert window.header_access.text() == "Authorized · Read Only"
        assert window.pyaml_fields["rf_name"].text() == "RF"
        assert window.pyaml_magnet_writes.isChecked()
        assert window.pyaml_rf_writes.isChecked()
        monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.No)
        window._test_connection()
        assert not calls
        assert not window.connection_verified
        monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.Yes)
        window._test_connection()
        _wait_for_connection_test(app, window)
        assert window.connection_verified
        assert window.rf_control_mode.currentData() == "automatic"
        assert window.measurement_type.model().item(window.measurement_type.findData("orm")).isEnabled()
        assert "WRITES ENABLED" in window.status_badge.text()
        assert window.header_capabilities.text() == "Available · Magnet + RF"
        assert window.header_access.text() == "Authorized · Magnet + RF"
        window.refresh_plan()
        assert window.plan_values["measurement_i/o"].text() == "Read only — orbit reads only"
        assert window.plan_values["noise_definition"].text() == "Population σ per BPM (ddof=0)"
        window.project = window._collect_project()
        assert "allow_writes" not in window.project.metadata["pyaml_options"]
        window._load_project_widgets()
        assert not window.pyaml_magnet_writes.isChecked()
        assert not window.pyaml_rf_writes.isChecked()
        assert not window.connection_verified
    finally:
        window.close()
        app.processEvents()


def test_simulator_restoration_evidence_is_reported_from_runtime_identity():
    app=QApplication.instance() or QApplication([]); window=MeasureMainWindow()
    try:
        unattested,_=writable_adapter(allow_writes=True,corrector_values_in_radians=True)
        window._set_pyaml_simulator_identity(unattested)
        assert "strength.get() is available" in window.pyaml_simulator_identity_status.text()
        assert "does not authorize real-machine operation" in window.pyaml_simulator_identity_status.text()
        attested,_=writable_adapter(allow_writes=True,corrector_values_in_radians=True,allow_get_only_restoration=True)
        window._set_pyaml_simulator_identity(attested)
        assert "pytest-local-pysc" in window.pyaml_simulator_identity_status.text()
        assert "Adapter value matched" in window.pyaml_simulator_identity_status.text()
        window.pyaml_fields["config_path"].setText("different-target.yaml")
        assert "Test connection" in window.pyaml_simulator_identity_status.text()
    finally:
        window.close(); app.processEvents()


def test_verified_simulator_needs_only_one_connected_adapter():
    app=QApplication.instance() or QApplication([]); window=MeasureMainWindow()
    try:
        window.adapter_combo.setCurrentIndex(window.adapter_combo.findData("pyaml"))
        first,_=writable_adapter(allow_writes=True,corrector_values_in_radians=True)
        first.accelerator.simulator_identity=lambda:{"verified":True,"backend":"pySC","instance_id":"runtime-1","configuration_sha256":"4"*64,"evidence":"test attestation"}
        first.simulator_identity=first._verified_simulator_identity(first.accelerator,"live","unused.yaml")
        first.allow_get_only_restoration=True
        window.adapter=first; window._set_pyaml_simulator_identity(first)
        assert first.corrector_restoration_evidence("H1")["level"]=="adapter_value"
        assert "Adapter value matched" in window.pyaml_simulator_identity_status.text()
    finally:
        window.close(); app.processEvents()


def test_verified_simulator_connection_does_not_request_second_authorization(monkeypatch):
    app=QApplication.instance() or QApplication([]); window=MeasureMainWindow()
    try:
        window.adapter_combo.setCurrentIndex(window.adapter_combo.findData("pyaml"))
        adapter,_=writable_adapter(allow_writes=True,corrector_values_in_radians=True)
        adapter.accelerator.simulator_identity=lambda:{"verified":True,"backend":"pySC","instance_id":"guided-runtime","configuration_sha256":"5"*64,"evidence":"runtime attestation"}
        adapter.simulator_identity=adapter._verified_simulator_identity(adapter.accelerator,"live","unused.yaml")
        answers=[]
        monkeypatch.setattr(QMessageBox,"question",lambda *args:(answers.append(args),QMessageBox.Yes)[1])
        adapter.allow_get_only_restoration=True
        options={"config_path":"verified.yaml","mode":"live","bpm_group":"BPM","hcor_group":"HCorr","vcor_group":"VCorr","rf_name":"RF","allow_writes":True,"allow_rf_writes":False,"corrector_values_in_radians":True}
        payload={"key":"pyaml","session":SimpleNamespace(adapter=adapter),"result":adapter.test_connection(),
            "bpms":tuple(BpmDevice(**item) for item in adapter.list_devices("bpm")),
            "hcor":tuple(CorrectorDevice(**item) for item in adapter.list_devices("hcor")),
            "vcor":tuple(CorrectorDevice(**item) for item in adapter.list_devices("vcor")),"options":options}
        window._connection_test_completed(payload)
        assert not answers
        assert window.connection_verified
        assert adapter.corrector_restoration_evidence("H1")["level"]=="adapter_value"
    finally:
        window.close(); app.processEvents()
