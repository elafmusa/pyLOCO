import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QFileDialog, QGroupBox, QLabel, QMessageBox

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
    assert not widget.writes.isChecked()
    assert widget.authorization_result.text() == "READ ONLY — writes not authorized"
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
    widget.writes.setChecked(True)
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.No)
    widget.connect_machine()
    assert not calls
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.Yes)
    widget.connect_machine()
    assert len(calls) == 1
    assert widget.connection_result.text() == "Connected — writes authorized"
    assert "Inventory verified — 3 quadrupole devices" in widget.inventory_result.text()
    assert widget.authorization_result.text() == "WRITE AUTHORIZED"
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


def test_transaction_recovery_controls_are_separate_and_explain_possible_writes(window):
    widget=window.pyaml_workspace
    assert widget.recover_button.text()=="Restore after interrupted application…"
    assert "Loading performs no writes" in widget.recover_button.toolTip()
    assert widget.cancel_button.text()=="Stop application and restore originals"
    assert "not guaranteed" in widget.cancel_button.toolTip()
    recovery=next(group for group in window.findChildren(QGroupBox) if group.title()=="Application safety & recovery")
    assert recovery.isAncestorOf(widget.extra_actions)
    assert not window.plot_tabs.isAncestorOf(widget.extra_actions)
    assert not recovery.isChecked() and not window.transaction_recovery_body.isVisible()
    assert "No correction application is currently running" in window.recovery_activity_status.text()
    assert "No recovery record is currently loaded" in window.recovery_availability_status.text()


def test_recovery_state_feedback_is_truthful_for_active_cancelled_and_failed_states(window,setup):
    widget=connect_workspace(window,setup); widget.timer.stop()
    widget.refresh()
    assert "No recovery record is currently loaded" in window.recovery_availability_status.text()
    widget.thread=object(); widget.cancel.clear(); widget.refresh()
    assert window.transaction_recovery_group.isChecked()
    assert "currently running" in window.recovery_activity_status.text()
    widget.request_cancel()
    assert "Cancellation requested" in window.recovery_activity_status.text()
    widget.thread=None
    record={
        "status":"restore_failed","restore_required":True,"attempted":["M1","M2"],"fraction":.1,
        "identity":{"connection":{"options":{"config_path":"ebs.yaml","mode":"live"}}},
        "changes":[
            {"name":"M1","original":1.0,"unit":"m^-1","status":"restored"},
            {"name":"M2","original":2.0,"unit":"m^-1","status":"restore_failed"},
        ],"restoration_errors":["M2: readback mismatch"],
    }
    transaction=SimpleNamespace(pending=True,record=record)
    transaction.restore=lambda:setattr(transaction,"pending",False)
    widget.transaction=transaction
    widget.refresh()
    assert "2 magnet(s) may have been modified" in window.recovery_availability_status.text()
    assert "Manual intervention required: M2" in window.recovery_feedback.text()
    details=window.recovery_record_details.toPlainText()
    assert "M1: 1.0 m^-1" in details and "M2: 2.0 m^-1" in details
    assert "readback mismatch" in details
    record["status"]="restoring"; record["changes"][1]["status"]="restoring"
    widget._update_recovery_display(True,True,False,record)
    assert "not yet been readback-verified" in window.recovery_feedback.text()
    record["status"]="restored"; record["restore_required"]=False
    for row in record["changes"]:row["status"]="restored"
    widget._update_recovery_display(True,False,False,record)
    assert "Successfully restored and readback-verified" in window.recovery_feedback.text()


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
    assert window.table.item(0, 9).text() == "mapped (identity)"
    window.backend_combo.setCurrentIndex(window.backend_combo.findData("mock"))
    assert window.pyaml_connection_group.isHidden()
    assert window.pysc_connection_group.isHidden()
    assert not window.offline_connection_group.isHidden()
    assert window.preview_apply_button.isEnabled()
    assert all(item.control_name is None for item in setup.review.items)


def test_shared_mapping_import_export(window, setup, tmp_path, monkeypatch):
    from pyLOCO.correct.pyaml_connection import load_mapping
    widget = connect_workspace(window, setup)
    widget.refresh()
    path = tmp_path / "mapping.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(path), ""))
    window.generate_mapping_template()
    mappings = load_mapping(path)
    assert [entry["control_name"] for entry in mappings] == ["M1", "M2", "S1"]
    assert [entry["component"] for entry in mappings] == [None, None, None]
    widget.clear_mapping()
    monkeypatch.setattr(pyaml_widget, "find_reference_lattice", lambda *_: setup.source)
    def validate(review, connection, selected, lattice):
        entries = pyaml_widget.load_mapping(selected)
        for entry,item in zip(entries,setup.review.items):
            entry["lattice_name"]=item.name
        rows = pyaml_widget.resolve_mapping(review, connection.inventory, entries)
        return entries, rows, "authoritative legacy mapping — revalidated", {"sha256": "test", "element_count": 3}
    monkeypatch.setattr(pyaml_widget, "validate_mapping_bundle", validate)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(path), ""))
    window.load_petra_mapping()
    assert widget.mapping_path == path
    assert all(item.metadata["match_source"] == "explicit" for item in setup.review.items)
    assert all(not magnet.strength.writes for magnet in setup.magnets.values())


def test_template_prefills_exact_connected_targets_but_is_not_loaded_or_trusted(window, setup, tmp_path, monkeypatch):
    widget = connect_workspace(window, setup)
    for item in setup.review.items:
        item.control_name = None
        item.metadata["mapping_status"] = "unmapped"
    path = tmp_path / "mapping-template.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(path), ""))
    window.generate_mapping_template()
    data = __import__("json").loads(path.read_text())
    assert [row["pyaml_device"] for row in data["mappings"]] == ["M1", "M2", "S1"]
    assert [row["role"] for row in data["mappings"]] == ["normal_quadrupole","normal_quadrupole","skew_quadrupole"]
    assert data["schema"] == "pyloco.machine_mapping"
    assert all(set(row)=={"pyaml_device","role","lattice_ordinal"} for row in data["mappings"])
    assert set(data)=={"schema","schema_version","template","mappings"}
    assert widget.mapping_path is None
    assert widget.explicit == []
    assert "not loaded" in window.mapping_file_status.text().lower()
    assert not window.apply_button.isEnabled()
    assert all(not magnet.strength.writes for magnet in setup.magnets.values())


def test_automatic_mapping_reuses_authoritative_profile_mapping(window, setup, tmp_path, monkeypatch):
    widget = connect_workspace(window, setup)
    for row in setup.connection.inventory:
        row["lattice_name"] = "unmatched"
        row["lattice_names"] = ("unmatched",)
    path = tmp_path / "approved.mapping.json"
    path.write_text("{}")
    entries = [
        {"lattice_name": item.name, "lattice_ordinal": item.lattice_ordinal,
         "control_name": name, "component": "B2L" if index < 2 else "A2L", "unit": "m^-1"}
        for index, (item, name) in enumerate(zip(setup.review.items, ("M1", "M2", "S1")))
    ]
    rows = pyaml_widget.resolve_mapping(setup.review, setup.connection.inventory, entries)
    monkeypatch.setattr(pyaml_widget, "approved_mapping_candidates", lambda *_: [path])
    monkeypatch.setattr(pyaml_widget, "find_reference_lattice", lambda *_: setup.source)
    monkeypatch.setattr(
        pyaml_widget, "validate_mapping_bundle",
        lambda *_: (entries, rows, "authoritative legacy mapping — revalidated", {}),
    )
    widget.clear_mapping()
    assert len(widget.explicit) == 3
    assert "3 validated, 0 unresolved" in window.automatic_mapping_feedback.text()
    assert all(not magnet.strength.writes for magnet in setup.magnets.values())


def test_find_mapping_reuses_embedded_complete_mapping_by_ordinal(window, setup):
    from pyLOCO.machine_mapping import entry

    widget = connect_workspace(window, setup)
    rows = [
        entry(
            role=item.correction_type,
            pyaml_device=target,
            lattice_ordinal=item.lattice_ordinal,
            lattice_fam_name=f"AT-{item.name}",
        )
        for item, target in zip(setup.review.items, ("M1", "M2", "S1"))
    ]
    setup.review.machine_mapping = {
        "schema": "pyloco.machine_mapping",
        "schema_version": 3,
        "mappings": rows,
    }
    # Neither FIT display names nor backend lattice aliases match. The saved
    # AT ordinals and exact connected control identities are sufficient.
    for index, item in enumerate(setup.review.items):
        item.name = f"FIT-ALIAS-{index}"
    for row in setup.connection.inventory:
        row["lattice_name"] = "unmatched"
        row["lattice_names"] = ("unmatched",)

    widget.clear_mapping()
    widget.refresh()

    assert len(widget.explicit) == 3
    assert [item.control_name for item in setup.review.items] == ["M1", "M2", "S1"]
    assert all(item.metadata["mapping_status"] == "mapped"
               for item in setup.review.items)
    assert "matched by AT ordinal" in window.automatic_mapping_feedback.text()
    assert all(not magnet.strength.writes for magnet in setup.magnets.values())


def test_mapping_help_and_details_are_self_explanatory(window, setup):
    widget = connect_workspace(window, setup)
    widget.refresh()
    help_text = " ".join(label.text() for label in window.mapping_help_body.findChildren(QLabel))
    for term in ("lattice_name", "lattice_ordinal", "control_name", "Derived metadata", "mapping status", "QD3J"):
        assert term in help_text
    assert "Fill in only the machine device names. pyLOCO validates the mapping automatically when loaded." in help_text
    mapping_page = window.tabs.widget(1).widget()
    assert mapping_page.layout().itemAt(0).widget().title() == "1. Machine connection"
    assert mapping_page.layout().itemAt(1).widget().title() == "ⓘ Mapping workflow and template format"
    assert widget.clear_mapping_button.text() == "Find machine mappings"
    assert window.generate_mapping_button.text() == "Create Machine Mapping template…"
    assert window.mapping_button.text() == "Load Machine Mapping…"
    assert window.mapping_details_table.columnCount() == 11
    assert window.mapping_details_table.horizontalHeaderItem(0).text() == "Fitted magnet"
    assert window.mapping_details_table.horizontalHeaderItem(1).text() == "AT FamName"
    assert window.mapping_details_table.horizontalHeaderItem(2).text() == "AT Device"
    assert window.mapping_details_table.horizontalHeaderItem(3).text() == "AT CommonName"
    assert window.mapping_details_table.horizontalHeaderItem(5).text() == "pyAML target / control"
    assert window.mapping_details_table.rowCount() == 3
    window.mapping_filter.setText("M2")
    assert window.mapping_details_table.rowCount() == 1
    assert window.mapping_details_table.item(0, 5).text() == "M2"


def test_mapping_status_filter_and_readonly_strength_journey(window, setup):
    widget = connect_workspace(window, setup)
    widget.refresh()
    assert window.mapping_summary.text().startswith("3 validated")
    window.mapping_attention_filter.setCurrentIndex(window.mapping_attention_filter.findData("attention"))
    assert window.mapping_details_table.rowCount() == 0
    window.mapping_attention_filter.setCurrentIndex(window.mapping_attention_filter.findData("all"))

    widget.read_current()
    assert "Read 3 integrated KL value(s)" in window.current_k_status.text()
    assert "no writes" in window.current_k_status.text()
    assert "calibration remains independently unverified" in window.current_k_status.text()
    assert all(not magnet.strength.writes for magnet in setup.magnets.values())


def test_backend_neutral_disconnected_and_zero_match_states(window, setup):
    widget = window.pyaml_workspace
    assert widget.connection_result.text() == "Disconnected"
    assert widget.inventory_result.text() == "Inventory not discovered"
    assert not widget.writes.isChecked()
    window.backend_combo.setCurrentIndex(window.backend_combo.findData("mock"))
    assert window.mapping_group.title() == "2. Machine Mapping"
    assert window.mapping_details_table.horizontalHeaderItem(5).text() == "pyAML target / control"
    assert window.automatic_mapping_group.isHidden()
    assert not window.offline_connection_group.isHidden()
    assert window.pysc_connection_group.isHidden()
    assert not window.read_current_k_button.isEnabled()
    assert "MOCK" in window.badge.text() and "READ ONLY" in window.badge.text()

    window.backend_combo.setCurrentIndex(window.backend_combo.findData("pyaml"))
    widget = connect_workspace(window, setup)
    for index,item in enumerate(setup.review.items):
        item.name = f"NO_MATCH_{index}"
    widget.clear_mapping(); widget.refresh()
    assert window.mapping_summary.text().startswith("0 validated")
    feedback=window.automatic_mapping_feedback.text()
    assert "Matching completed — 0 validated, 3 unresolved, 0 ambiguous" in feedback
    assert "create a template" in feedback
    assert all(not magnet.strength.writes for magnet in setup.magnets.values())

    for item,name in zip(setup.review.items,("Q1","Q2","SQ1")):
        item.name=name
    setup.review.items[0].name = "NO_EXACT_IDENTITY"
    widget.clear_mapping(); widget.refresh()
    assert window.mapping_summary.text().startswith("2 validated")
    assert "2 validated, 1 unresolved" in window.automatic_mapping_feedback.text()
    window.mapping_attention_filter.setCurrentIndex(window.mapping_attention_filter.findData("attention"))
    assert window.mapping_details_table.rowCount() == 1
    with pytest.raises(ValueError, match="Valid mappings are required"):
        widget.read_current()
    assert all(not magnet.strength.writes for magnet in setup.magnets.values())


def test_embedded_fit_mapping_is_visible_offline_then_revalidated_on_connection(window, setup, monkeypatch):
    from pyLOCO.machine_mapping import entry
    rows=[]
    for item,target in zip(setup.review.items,("M1","M2","S1")):
        rows.append(entry(role=item.correction_type,pyaml_device=target,
            lattice_ordinal=item.lattice_ordinal,lattice_fam_name=item.name,
            lattice_device=f"device/{item.name}",component="B2L" if item.correction_type=="normal_quadrupole" else "A2L",unit="m^-1",
            status="mapped",identity_match="exact_device"))
    setup.review.machine_mapping={"schema":"pyloco.machine_mapping","schema_version":3,
                                  "machine_profile":{"backend":"pyaml","configuration_path":str(setup.config),
                                                     "mode":"live","normal_array":"Quads","skew_array":"SkewQuads"},
                                  "mapping_provenance":"Measure → FIT validated mapping","mappings":rows}
    window.backend_combo.setCurrentIndex(window.backend_combo.findData("mock"))
    window._accept_loaded_review(setup.review,setup.source)
    assert window.backend_combo.currentData()=="pyaml"
    assert window.tabs.currentIndex()==1
    assert not window.pyaml_connection_group.isHidden()
    assert window.pyaml_connection_group.isAncestorOf(window.pyaml_workspace.connect_button)
    assert not window.pyaml_workspace.connect_button.isHidden()
    assert window.pyaml_workspace.fields["config_path"].text()==str(setup.config.resolve())
    assert window.pyaml_workspace.connection is None
    assert not window.pyaml_workspace.writes.isChecked()
    assert window.pyaml_workspace.connection_result.text()=="Disconnected"
    assert "saved mapping remains unverified" in window.connection_guidance.text()
    assert all(item.metadata["mapping_status"]=="saved" for item in window.review.items)
    assert "saved pending revalidation" in window.mapping_summary.text()
    assert window.mapping_details_table.item(0,9).text()=="Saved mapping — revalidation required"
    widget=window.pyaml_workspace
    window.backend_combo.setCurrentIndex(window.backend_combo.findData("pyaml"))
    monkeypatch.setattr(pyaml_widget,"PyAMLCorrectionConnection",lambda **_options:setup.connection)
    monkeypatch.setattr(pyaml_widget,"find_reference_lattice",lambda *_args:setup.source)
    selected=[dict(correction_type=item.correction_type,lattice_name=item.name,
                   lattice_ordinal=item.lattice_ordinal,control_name=target,
                   component="B2L" if item.correction_type=="normal_quadrupole" else "A2L",unit="m^-1")
              for item,target in zip(setup.review.items,("M1","M2","S1"))]
    monkeypatch.setattr(pyaml_widget,"validate_mapping_document",
                        lambda *_args,**_kwargs:(selected,[],"revalidated saved mapping",{}))
    setup.connection.allow_writes=False
    widget.connect_machine(); widget.refresh()
    assert widget.connection_result.text()=="Connected — read only"
    assert widget.inventory_result.text()=="Inventory verified — 3 quadrupole devices"
    assert "Mapping verified for 3 fitted magnet(s)" in window.connection_guidance.text()
    assert all(item.metadata["mapping_status"]=="mapped" for item in window.review.items)
    assert all(item.metadata["mapping_validation_level"]=="connected_inventory" for item in window.review.items)
    assert window.mapping_details_table.item(0,9).text()=="Validated against current inventory"


def test_backend_switch_does_not_erase_saved_fit_mapping(window, setup):
    from pyLOCO.machine_mapping import entry
    rows=[]
    for item,target in zip(setup.review.items,("M1","M2","S1")):
        rows.append(entry(role=item.correction_type,pyaml_device=target,
            lattice_ordinal=item.lattice_ordinal,lattice_fam_name=item.name,
            component="B2L" if item.correction_type=="normal_quadrupole" else "A2L",
            unit="m^-1",status="mapped",identity_match="exact_device"))
    setup.review.machine_mapping={
        "schema":"pyloco.machine_mapping","schema_version":3,
        "machine_profile":{"backend":"pyaml","configuration_path":str(setup.config)},
        "mappings":rows,
    }
    window._accept_loaded_review(setup.review,setup.source)
    expected=[item.control_name for item in window.review.items]
    window.backend_combo.setCurrentIndex(window.backend_combo.findData("mock"))
    assert [item.control_name for item in window.review.items]==expected
    assert all(item.metadata["mapping_status"]=="saved" for item in window.review.items)
    window.backend_combo.setCurrentIndex(window.backend_combo.findData("pyaml"))
    assert [item.control_name for item in window.review.items]==expected
    assert not window.pyaml_workspace.writes.isChecked()


def test_pyaml_representation_lock_explains_k_to_kl_conversion(window, setup):
    widget = connect_workspace(window, setup)
    widget.refresh(); window.refresh_all()
    assert not window.correction_representation.isEnabled()
    assert window.correction_representation.currentData() == "integrated"
    explanation = window.representation_explanation.text()
    assert "locked" in explanation
    assert "Δ(KL) = ΔK × L" in explanation
    assert "m⁻²" in explanation and "m⁻¹" in explanation
    assert "integrated B2L/A2L" in window.correction_representation.toolTip()
    assert window.table.horizontalHeaderItem(12).text() == "Current K [m⁻²]"
    assert window.table.horizontalHeaderItem(20).text() == "Final Δ(KL) [m⁻¹]"
    assert window.table.horizontalHeaderItem(32).text() == "Current KL [m⁻¹]"
    assert window.table.horizontalHeaderItem(33).text() == "Target KL [m⁻¹]"


def test_mapping_table_preview_invalidates_and_pending_locks(window, setup):
    widget = connect_workspace(window, setup)
    widget.preview(); widget.refresh()
    assert widget.table.rowCount() == 3
    assert widget.table.item(0, 6).text() == "M1"
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
    assert "validated recovery record is loaded" in window.recovery_availability_status.text()
    assert "writes required for restoration" in window.recovery_feedback.text()
    assert "Original strengths to restore" in window.recovery_record_details.toPlainText()


def test_missing_or_invalid_recovery_selection_is_safe_and_actionable(window,setup,tmp_path,monkeypatch):
    widget=connect_workspace(window,setup); widget.timer.stop(); writes_before=sum(len(m.strength.writes) for m in setup.magnets.values())
    monkeypatch.setattr(QFileDialog,"getOpenFileName",lambda *a,**k:("",""))
    widget.recover()
    assert "No recovery record is currently loaded" in window.recovery_availability_status.text()
    invalid=tmp_path/"invalid-recovery.json"; invalid.write_text('{"schema":"wrong"}')
    monkeypatch.setattr(QFileDialog,"getOpenFileName",lambda *a,**k:(str(invalid),""))
    warnings=[]; monkeypatch.setattr(QMessageBox,"warning",lambda *a,**k:warnings.append(a))
    widget.run(widget.recover)
    assert warnings and "Not a pending" in window.apply_status.text()
    assert sum(len(m.strength.writes) for m in setup.magnets.values())==writes_before
