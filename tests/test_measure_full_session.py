from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from pyLOCO.data_schema import MeasurementSession, SessionFile, load_session, save_session, write_bpm_noise, write_dispersion, write_orm
from pyLOCO.measure.full_session import (file_sha256, open_full_session_bundle,
                                         save_full_session_bundle, save_full_session_state)
from pyLOCO.measure.project import MeasureProject, load_measure_project, save_measure_project
from pyLOCO.measure.main_window import MeasureMainWindow
from pyLOCO.measure.acquisition import BpmDevice, BpmNoiseResult, DispersionResult, DispersionStateResult, ORMResult
from pyLOCO.gui.measurement_mapping import mapping_counts
from pyLOCO.gui.suite import inspect_measurement_session


def measurement_files(root):
    names=["B1","B2"]
    noise=root/"noise.h5"; write_bpm_noise(noise,noise_x_m=[1e-9,2e-9],noise_y_m=[2e-9,3e-9],bpm_names=names,raw_orbits_x_m=np.zeros((2,2)),raw_orbits_y_m=np.zeros((2,2)))
    dispersion=root/"dispersion.h5"; write_dispersion(dispersion,measured_eta_x=[1e-5,2e-5],measured_eta_y=[3e-5,4e-5],bpm_names=names,rf_frequency_hz=[499_999_900,500_000_100],rf_setpoint_hz=[499_999_900,500_000_100],rf_readback_hz=[499_999_900,500_000_100],raw_orbits_x_m=np.zeros((2,2)),raw_orbits_y_m=np.zeros((2,2)),rf_step_hz=200,bidirectional=True)
    orm=root/"orm.h5"; write_orm(orm,response_matrix=np.zeros((4,2)),bpm_names=names,horizontal_corrector_names=["H1"],vertical_corrector_names=["V1"],requested_kick_h_rad=[1e-4],requested_kick_v_rad=[1e-4],actual_kick_h_rad=[1e-4],actual_kick_v_rad=[1e-4],orbit_plus_m=np.zeros((2,4)),orbit_minus_m=np.zeros((2,4)))
    return {"bpm_noise":noise,"dispersion":dispersion,"orm":orm}


def source_session(tmp_path, roles=("bpm_noise","dispersion","orm")):
    files=measurement_files(tmp_path)
    manifest=tmp_path/"source.pyloco-session.json"
    save_session(manifest,MeasurementSession("test-session",tuple(SessionFile(role,files[role].name) for role in roles)))
    return manifest


def project(tmp_path):
    config=tmp_path/"machine.yaml"; config.write_text("approved: simulator\n")
    selection=MeasureProject().measurement_selections
    selection["orm"]["bpm"]["resolved_names"]=["B1","B2"]
    selection["orm"]["hcor"]["resolved_names"]=["H1"]
    selection["orm"]["vcor"]["resolved_names"]=["V1"]
    return MeasureProject(adapter="pyAML",output_directory=str((tmp_path/"absolute-output").resolve()),measurement_selections=selection,metadata={"pyaml_options":{"config_path":str(config),"mode":"live"}}),config


def test_full_session_roundtrip_is_portable_and_fit_compatible(tmp_path):
    manifest=source_session(tmp_path); lattice=tmp_path/"betamodel.mat"; lattice.write_bytes(b"4434-element-test-lattice")
    configured,config=project(tmp_path)
    bundle=save_full_session_bundle(tmp_path/"bundle.pyloco-measure-session",project=configured,session_manifest=manifest,reference_lattice=lattice,reference_mat_variable="betamodel",reference_element_count=4434,mapping={"schema":"test"})
    assert bundle.session.is_complete_for_loco
    assert bundle.project.adapter=="pyAML"
    assert bundle.project.output_directory=="measurements"
    assert bundle.project.reference_model_mat_variable=="betamodel"
    assert bundle.mapping=={"schema":"test"}
    assert "full_measure_session" in bundle.session.metadata
    assert not (bundle.root/"configuration"/config.name).exists()
    moved=tmp_path/"moved"; shutil.move(str(bundle.root),moved)
    restored=open_full_session_bundle(moved/"measurement-session.pyloco-session.json")
    assert restored.session.is_complete_for_loco
    assert all(path.is_file() for path in restored.measurement_paths.values())
    assert restored.project.reference_model_path==str((moved/"lattice"/lattice.name).resolve())
    handoff=inspect_measurement_session(restored.manifest_path)
    assert handoff.available_roles==("orm","bpm_noise","dispersion")
    assert set(handoff.provenance["measurement_files"])=={"orm","bpm_noise","dispersion"}


def test_full_session_can_explicitly_include_approved_pyaml_configuration(tmp_path):
    configured,config=project(tmp_path); lattice=tmp_path/"ring.mat"; lattice.write_bytes(b"ring")
    bundle=save_full_session_bundle(tmp_path/"included.pyloco-measure-session",project=configured,session_manifest=source_session(tmp_path,("bpm_noise",)),reference_lattice=lattice,include_pyaml_configuration=True)
    restored=open_full_session_bundle(bundle.manifest_path)
    copied=restored.root/"configuration"/config.name
    assert copied.is_file()
    assert restored.project.metadata["pyaml_options"]["config_path"]==str(copied)
    assert restored.session.missing_roles==("orm","dispersion")


def test_full_session_detects_modified_measurement_and_does_not_overwrite(tmp_path):
    configured,_=project(tmp_path); lattice=tmp_path/"ring.mat"; lattice.write_bytes(b"ring")
    destination=tmp_path/"bundle.pyloco-measure-session"
    bundle=save_full_session_bundle(destination,project=configured,session_manifest=source_session(tmp_path),reference_lattice=lattice)
    bundle.measurement_paths["orm"].write_bytes(b"modified")
    with pytest.raises(ValueError,match="hash mismatch"):open_full_session_bundle(bundle.manifest_path)
    with pytest.raises(FileExistsError):save_full_session_bundle(destination,project=configured,session_manifest=source_session(tmp_path),reference_lattice=lattice)


def test_measure_project_accepts_pyaml_and_absolute_local_output(tmp_path):
    configured,_=project(tmp_path); path=tmp_path/"project.pyloco-measure.json"
    save_measure_project(path,configured)
    restored=load_measure_project(path)
    assert restored.adapter=="pyAML"
    assert restored.output_directory==configured.output_directory


def test_save_session_twice_updates_state_without_copying_or_modifying_hdf5(tmp_path):
    manifest=source_session(tmp_path); configured,_=project(tmp_path); lattice=tmp_path/"ring.mat"; lattice.write_bytes(b"ring")
    original=load_session(manifest).resolve(manifest)
    before={role:(path,file_sha256(path),path.stat().st_mtime_ns) for role,path in original.items()}
    first=save_full_session_state(manifest,project=configured,reference_lattice=lattice,reference_mat_variable="RING",reference_element_count=3,mapping={"mapping":"first"})
    second=save_full_session_state(manifest,project=configured,reference_lattice=lattice,reference_mat_variable="RING",reference_element_count=3,mapping={"mapping":"second"})
    assert first.manifest_path==second.manifest_path==manifest.resolve()
    assert second.mapping=={"mapping":"second"}
    assert len(list((manifest.parent/"lattice").glob("*.mat")))==1
    for role,(path,digest,mtime) in before.items():
        assert second.measurement_paths[role]==path and file_sha256(path)==digest and path.stat().st_mtime_ns==mtime


def test_primary_toolbar_uses_session_actions_and_templates_are_in_file_menu():
    app=QApplication.instance() or QApplication([]); window=MeasureMainWindow()
    try:
        assert tuple(window.session_toolbar_buttons)==("New","Open","Save","Open in FIT")
        assert window.session_toolbar_buttons["Save"].isEnabled()
        assert window.session_export_action.text()=="Export Portable Session…"
        assert window.session_open_folder_action.text()=="Open Portable Session Folder…"
        assert any(action.menu() and action.text()=="Configuration Templates" for action in window.menuBar().actions()[0].menu().actions())
    finally:window.close(); app.processEvents()


def test_save_and_reopen_configured_full_session_before_any_measurement(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([]); root=tmp_path/"empty-session"; root.mkdir(); manifest=root/"measurement-session.pyloco-session.json"
    window=MeasureMainWindow(); window.measurement_label.setText("configured before acquisition"); window.readings.setValue(7)
    monkeypatch.setattr("pyLOCO.measure.main_window.QFileDialog.getSaveFileName",lambda *args:(str(manifest),""))
    try:
        assert window.save_active_session()
        session=load_session(manifest); assert session.files==() and "full_measure_session" in session.metadata
        assert (root/"measure-project.pyloco-measure.json").is_file()
        first_id=session.session_id; assert first_id.startswith("measure-")
        restored=MeasureMainWindow(); messages=[]
        monkeypatch.setattr(QMessageBox,"information",lambda _parent,title,text,*args:messages.append((title,text)) or QMessageBox.Ok)
        restored.open_full_measurement_session(root); app.processEvents()
        assert restored.saved_session_path==manifest.resolve()
        assert restored.measurement_label.text()=="configured before acquisition" and restored.readings.value()==7
        assert load_session(restored.saved_session_path).session_id==first_id
        assert all(value is None for value in restored._measurement_result_paths.values())
        restored.tabs.setCurrentIndex(restored.review_tab_index); app.processEvents()
        assert "no measurements acquired yet" in restored.review_status.text().lower()
        restored.close()
    finally:window.close(); app.processEvents()


def test_empty_full_session_accepts_all_roles_and_repeated_save_does_not_rewrite_hdf5(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([]); root=tmp_path/"append-session"; root.mkdir(); manifest=root/"measurement-session.pyloco-session.json"
    external=tmp_path/"external"; external.mkdir(); files=measurement_files(external)
    window=MeasureMainWindow(); monkeypatch.setattr("pyLOCO.measure.main_window.QFileDialog.getSaveFileName",lambda *args:(str(manifest),"")); assert window.save_active_session()
    selected=iter(str(files[role]) for role in ("bpm_noise","dispersion","orm"))
    monkeypatch.setattr("pyLOCO.measure.main_window.QFileDialog.getOpenFileName",lambda *args:(next(selected),""))
    monkeypatch.setattr(QMessageBox,"question",lambda *args:QMessageBox.Yes)
    try:
        for role in ("bpm_noise","dispersion","orm"):assert window._load_measurement_role(role)
        session=load_session(manifest); assert session.is_complete_for_loco
        paths=session.resolve(manifest); before={role:(path,file_sha256(path),path.stat().st_mtime_ns) for role,path in paths.items()}
        assert window.save_active_session(); window.measurement_label.setText("saved again"); assert window.save_active_session()
        for role,(path,digest,mtime) in before.items():
            assert load_session(manifest).resolve(manifest)[role]==path
            assert file_sha256(path)==digest and path.stat().st_mtime_ns==mtime
    finally:window.close(); app.processEvents()


def test_gui_opens_full_session_disconnected_with_writes_disabled(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([])
    configured,_=project(tmp_path)
    bundle=save_full_session_bundle(tmp_path/"gui.pyloco-measure-session",project=configured,session_manifest=source_session(tmp_path,("bpm_noise",)),reference_lattice=None)
    window=MeasureMainWindow()
    monkeypatch.setattr("pyLOCO.measure.main_window.QFileDialog.getOpenFileName",lambda *args:(str(bundle.manifest_path),""))
    monkeypatch.setattr(QMessageBox,"information",lambda *args:QMessageBox.Ok)
    try:
        window.open_full_measurement_session(); app.processEvents()
        assert window.saved_session_path==bundle.manifest_path
        assert not window.connection_verified
        assert not window.pyaml_magnet_writes.isChecked()
        assert not window.pyaml_rf_writes.isChecked()
        assert isinstance(window.result,BpmNoiseResult)
        assert "disconnected review" in window.statusBar().currentMessage()
        assert window.review_open_button.isEnabled()
    finally:
        window.close(); app.processEvents()


def test_full_session_restores_requested_write_preferences_without_authorization(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([])
    configured,_=project(tmp_path); configured.metadata["pyaml_options"].update(requested_magnet_writes=True,requested_rf_writes=True)
    bundle=save_full_session_bundle(tmp_path/"requested.pyloco-measure-session",project=configured,session_manifest=source_session(tmp_path,("bpm_noise",)),reference_lattice=None)
    window=MeasureMainWindow(); monkeypatch.setattr(QMessageBox,"information",lambda *args:QMessageBox.Ok)
    try:
        window.open_full_measurement_session(bundle.manifest_path); app.processEvents()
        assert window.pyaml_magnet_writes.isChecked() and window.pyaml_rf_writes.isChecked()
        assert not window.connection_verified
        assert "disconnected review" in window.statusBar().currentMessage()
    finally:window.close(); app.processEvents()


def test_saved_bpm_noise_result_is_reconstructed_read_only_for_plots(tmp_path):
    app=QApplication.instance() or QApplication([]); files=measurement_files(tmp_path); window=MeasureMainWindow()
    try:
        result=window._restore_saved_result("bpm_noise",files["bpm_noise"])
        assert tuple(device.name for device in result.devices)==("B1","B2")
        assert result.orbits_x_m.shape==(2,2) and result.orbits_y_m.shape==(2,2)
        before=files["bpm_noise"].read_bytes(); window._measurement_result_paths["bpm_noise"]=files["bpm_noise"]; window._measurement_results["bpm_noise"]=None
        window._measurement_type_changed(); assert window._measurement_results["bpm_noise"] is not None
        assert files["bpm_noise"].read_bytes()==before
    finally:window.close(); app.processEvents()


def test_tab_switch_does_not_assign_one_measurement_file_to_another_role(tmp_path):
    app=QApplication.instance() or QApplication([]); files=measurement_files(tmp_path); window=MeasureMainWindow()
    try:
        window._measurement_result_paths.update(files)
        window.saved_measurement_path=files["bpm_noise"]
        window.result=window._restore_saved_result("bpm_noise",files["bpm_noise"])
        window._store_measurement_view_state("orm")
        assert window._measurement_result_paths["orm"]==files["orm"]
        assert window._measurement_results["orm"] is None
    finally:window.close(); app.processEvents()


def test_october_23_saved_session_restores_all_results_mapping_and_order_read_only(monkeypatch):
    """Local MDT evidence regression; skipped when the operator fixture is absent."""
    root=Path(__file__).resolve().parents[1]/"measurements"
    manifest=root/"measurement-session.pyloco-session.json"
    expected={
        "bpm_noise":root/"pyloco-bpm-noise-20261023-174842.h5",
        "dispersion":root/"pyloco-dispersion-20261023-175303.h5",
        "orm":root/"pyloco-orm-20261023-175520.h5",
    }
    if not manifest.is_file() or not all(path.is_file() for path in expected.values()):
        pytest.skip("October 23 local MDT measurement fixture is not available")
    before={role:hashlib.sha256(path.read_bytes()).hexdigest() for role,path in expected.items()}
    app=QApplication.instance() or QApplication([]); window=MeasureMainWindow()
    monkeypatch.setattr(QMessageBox,"information",lambda *args:QMessageBox.Ok)
    try:
        window.open_full_measurement_session(manifest); app.processEvents()
        assert window.reference_model is not None
        assert window.reference_model.mat_variable=="betamodel" and len(window.reference_model.ring)==4434
        assert isinstance(window._measurement_results["bpm_noise"],BpmNoiseResult)
        assert isinstance(window._measurement_results["dispersion"],DispersionResult)
        orm=window._measurement_results["orm"]; assert isinstance(orm,ORMResult)
        assert orm.response_matrix.shape==(640,10)
        assert len(orm.horizontal_correctors)==5 and len(orm.vertical_correctors)==5
        assert tuple(device.name for device in orm.bpms)==tuple(device.name for device in window._measurement_results["dispersion"].devices)
        assert all(status["validated"] and status["restored"] for status in window._measurement_restore_status.values())
        assert mapping_counts(window.measurement_device_alignment)=={"total":330,"mapped":330,"unresolved":0,"ambiguous":0,"duplicate":0,"invalid":0}
        assert "No mapping saved" in window.device_alignment_status.text()
        assert not window.connection_verified
        for role in ("bpm_noise","dispersion","orm"):
            window.measurement_type.setCurrentIndex(window.measurement_type.findData(role)); app.processEvents()
            assert window.saved_measurement_path==expected[role]
            assert window._measurement_restore_status[role]["restored"]
        assert {role:hashlib.sha256(path.read_bytes()).hexdigest() for role,path in expected.items()}==before
    finally:window.close(); app.processEvents()


def test_ebs_mapping_save_close_open_reconnect_keeps_validated_orm_selection(tmp_path,monkeypatch):
    """Exercise the real saved EBS identities without connecting or writing."""
    yaml=pytest.importorskip("yaml")
    root=Path(__file__).resolve().parents[1]/"measurements"
    manifest=root/"measurement-session.pyloco-session.json"
    if not manifest.is_file():pytest.skip("October 23 local MDT session is not available")
    app=QApplication.instance() or QApplication([]); source=MeasureMainWindow()
    monkeypatch.setattr(QMessageBox,"information",lambda *args:QMessageBox.Ok)
    try:
        source.open_full_measurement_session(manifest); app.processEvents()
        assert mapping_counts(source.measurement_device_alignment)["mapped"]==330
        assert source.machine_mapping_document is not None
        saved_h=tuple(device.name for device in source.selected_hcorrectors)
        saved_v=tuple(device.name for device in source.selected_vcorrectors)
        destination=tmp_path/"saved-ebs-session.pyloco-measure-session"
        bundle=save_full_session_bundle(destination,project=source._collect_project(),session_manifest=manifest,
            reference_lattice=source.reference_model.path,reference_mat_variable=source.reference_model.mat_variable,
            reference_element_count=len(source.reference_model.ring),mapping=source.machine_mapping_document)
        assert bundle.mapping is not None
    finally:source.close(); app.processEvents()

    restored=MeasureMainWindow()
    try:
        restored.open_full_measurement_session(bundle.manifest_path); app.processEvents()
        assert mapping_counts(restored.measurement_device_alignment)["mapped"]==330
        assert restored.alignment_details_button.isEnabled()
        assert restored.save_mapping_button.isEnabled()
        assert "Saved mapping restored and validated offline" in restored.device_alignment_status.text()

        config=Path(restored.project.metadata["pyaml_options"]["config_path"])
        arrays={row["name"]:row["elements"] for row in yaml.safe_load(config.read_text())["arrays"]}
        class Adapter:
            capabilities=frozenset()
            simulator_identity=None
            allow_get_only_restoration=False
            def test_connection(self):return {"bpms":320,"corrector_readback":"available","calibration":"available","rf_readback":352_202_000.0}
            def list_devices(self,kind):
                if kind=="bpm":return tuple({"name":name,"x_channel":"X","y_channel":"Y"} for name in arrays["BPM"])
                names,plane=(arrays["HCorr"],"H") if kind=="hcor" else (arrays["VCorr"],"V")
                return tuple({"name":name,"setpoint_channel":"SP","readback_channel":"RBV","plane":plane} for name in names)
        adapter=Adapter()
        from pyLOCO.measure.main_window import InterfaceRegistry as RealRegistry
        class Registry(RealRegistry):
            def create(self,_key):return SimpleNamespace(adapter=adapter)
        monkeypatch.setattr("pyLOCO.measure.main_window.InterfaceRegistry",Registry)
        restored.pyaml_magnet_writes.setChecked(False); restored.pyaml_rf_writes.setChecked(False)
        restored._test_connection()
        deadline=time.monotonic()+10
        while restored.connection_thread is not None and time.monotonic()<deadline:
            app.processEvents(); time.sleep(.01)
        assert restored.connection_thread is None
        assert tuple(device.name for device in restored.selected_hcorrectors)==saved_h
        assert tuple(device.name for device in restored.selected_vcorrectors)==saved_v
        assert restored.alignment_details_button.isEnabled() and restored.save_mapping_button.isEnabled(),restored.device_alignment_status.text()
        assert "connected inventory matches" in restored.device_alignment_status.text()
        mapping=restored.measurement_device_alignment
        for kind in ("bpm_noise","dispersion","orm"):
            restored.measurement_type.setCurrentIndex(restored.measurement_type.findData(kind)); app.processEvents()
            assert restored.measurement_device_alignment is mapping
        restored.corrector_selection_widgets["hcor"]["search"].setText("c04, c05")
        restored._filter_corrector_table("hcor")
        assert restored.measurement_device_alignment is mapping
    finally:restored.close(); app.processEvents()


def test_ordinary_open_project_redirects_measurement_session_to_full_session_action(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([]); manifest=source_session(tmp_path,("bpm_noise",)); window=MeasureMainWindow(); original=window.project
    messages=[]
    monkeypatch.setattr("pyLOCO.measure.main_window.QFileDialog.getOpenFileName",lambda *args:(str(manifest),""))
    monkeypatch.setattr(QMessageBox,"information",lambda _parent,title,text,*args:messages.append((title,text)) or QMessageBox.Ok)
    try:
        window.open_project(); app.processEvents()
        assert window.project is original
        assert messages and "Use Open to restore the full measurement session" in messages[0][1]
        assert "measurement-session manifest" in messages[0][1]
    finally:
        window.close(); app.processEvents()


def test_full_session_action_opens_older_manifest_and_explains_unrecoverable_settings(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([]); manifest=source_session(tmp_path,("bpm_noise","orm")); window=MeasureMainWindow(); messages=[]
    monkeypatch.setattr("pyLOCO.measure.main_window.QFileDialog.getOpenFileName",lambda *args:(str(manifest),""))
    monkeypatch.setattr(QMessageBox,"information",lambda _parent,title,text,*args:messages.append((title,text)) or QMessageBox.Ok)
    try:
        window.open_full_measurement_session(); app.processEvents()
        assert window.saved_session_path==manifest.resolve()
        assert window.saved_measurement_path.is_file()
        assert "BPM noise" in window.review_details.text() and "ORM" in window.review_details.text()
        assert "File present" in window.review_details.text() and "Missing" in window.review_details.text()
        assert messages and "Older measurement session loaded"==messages[0][0]
        explanation=messages[0][1]
        assert "no Measure project, lattice, selections or mapping" in explanation
        assert "manually" in explanation
    finally:
        window.close(); app.processEvents()


def test_corrupt_modern_session_is_not_reinterpreted_as_legacy_and_preserves_active_state(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([]); configured,_=project(tmp_path); lattice=tmp_path/"ring.mat"; lattice.write_bytes(b"authoritative ring")
    bundle=save_full_session_bundle(tmp_path/"corrupt.pyloco-measure-session",project=configured,session_manifest=source_session(tmp_path,("bpm_noise",)),reference_lattice=lattice)
    Path(bundle.project.reference_model_path).write_bytes(b"corrupted ring")
    window=MeasureMainWindow(); window.measurement_label.setText("active session must survive"); original_project=window.project; warnings=[]
    monkeypatch.setattr(QMessageBox,"warning",lambda _parent,title,text,*args:warnings.append((title,text)) or QMessageBox.Ok)
    try:
        window.open_full_measurement_session(bundle.manifest_path); app.processEvents()
        assert window.project is original_project and window.measurement_label.text()=="active session must survive"
        assert window.saved_session_path is None
        assert warnings and warnings[0][0]=="Cannot open full session"
        assert "integrity validation" in warnings[0][1] and "hash mismatch" in warnings[0][1]
    finally:window.close(); app.processEvents()


def test_opening_session_after_another_clears_stale_role_paths_and_folder_is_supported(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([]); configured,_=project(tmp_path)
    first_source=tmp_path/"first-source"; second_source=tmp_path/"second-source"; first_source.mkdir(); second_source.mkdir()
    first=save_full_session_bundle(tmp_path/"first.pyloco-measure-session",project=configured,session_manifest=source_session(first_source,("bpm_noise",)),reference_lattice=None)
    second=save_full_session_bundle(tmp_path/"second.pyloco-measure-session",project=configured,session_manifest=source_session(second_source,("orm",)),reference_lattice=None)
    window=MeasureMainWindow(); monkeypatch.setattr(QMessageBox,"information",lambda *args:QMessageBox.Ok)
    try:
        window.open_full_measurement_session(first.root); app.processEvents()
        assert window._measurement_result_paths["bpm_noise"].is_file() and window._measurement_result_paths["orm"] is None
        window.open_full_measurement_session(second.manifest_path); app.processEvents()
        assert window._measurement_result_paths["bpm_noise"] is None
        assert window._measurement_result_paths["dispersion"] is None
        assert window._measurement_result_paths["orm"].is_file()
    finally:window.close(); app.processEvents()


@pytest.mark.parametrize("size",[(1280,720),(1440,900)])
@pytest.mark.parametrize("theme",["light","purple"])
def test_review_page_scroll_layout_keeps_manage_actions_readable(size,theme):
    app=QApplication.instance() or QApplication([]); window=MeasureMainWindow(); window.resize(*size); window.apply_theme(theme); window.show(); window.tabs.setCurrentIndex(window.review_tab_index); app.processEvents()
    try:
        assert window.tabs.count()==6 and window.review_scroll.isVisible()
        assert window.review_summary_group.height()>=window.review_summary_group.minimumSizeHint().height()
        assert window.review_roles_group.height()>=window.review_roles_group.minimumSizeHint().height()
        for button in window.role_manage_buttons.values():
            assert button.height()>=30 and button.isVisible()
        assert window.review_scroll.verticalScrollBar().maximum()>=0
        assert window.review_open_button is window.session_toolbar_buttons["Open in FIT"]
    finally:window.close(); app.processEvents()


def test_opened_full_session_marks_configuration_edits_unsaved(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    configured, _ = project(tmp_path)
    bundle = save_full_session_bundle(
        tmp_path / "editable.pyloco-measure-session",
        project=configured,
        session_manifest=source_session(tmp_path),
        reference_lattice=None,
    )
    window = MeasureMainWindow()
    monkeypatch.setattr(
        "pyLOCO.measure.main_window.QFileDialog.getOpenFileName",
        lambda *args: (str(bundle.manifest_path), ""),
    )
    monkeypatch.setattr(QMessageBox, "information", lambda *args: QMessageBox.Ok)
    try:
        window.open_full_measurement_session()
        app.processEvents()
        assert not window._session_dirty
        assert not window.windowTitle().endswith("*")
        assert not window.session_toolbar_buttons["Save"].isEnabled()

        window.measurement_label.setText("edited session label")
        app.processEvents()

        assert window._session_dirty
        assert window.windowTitle().endswith("*")
        assert window.session_toolbar_buttons["Save"].isEnabled()
    finally:
        window._session_dirty = False
        window.close()
        app.processEvents()


def test_independent_measurement_replace_and_remove_preserve_other_roles_and_files(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([]); manifest=source_session(tmp_path); window=MeasureMainWindow()
    monkeypatch.setattr(QMessageBox,"information",lambda *args:QMessageBox.Ok)
    monkeypatch.setattr("pyLOCO.measure.main_window.QFileDialog.getOpenFileName",lambda *args:(str(manifest),"")); window.open_full_measurement_session()
    original=load_session(manifest).resolve(manifest); replacement=tmp_path/"replacement-noise.h5"
    write_bpm_noise(replacement,noise_x_m=[4e-9,5e-9],noise_y_m=[6e-9,7e-9],bpm_names=["B1","B2"],raw_orbits_x_m=np.ones((2,2)),raw_orbits_y_m=np.ones((2,2)))
    monkeypatch.setattr("pyLOCO.measure.main_window.QFileDialog.getOpenFileName",lambda *args:(str(replacement),""))
    monkeypatch.setattr(QMessageBox,"question",lambda *args:QMessageBox.Yes)
    try:
        assert window._load_measurement_role("bpm_noise",replace=True)
        updated=load_session(manifest).resolve(manifest)
        assert updated["bpm_noise"]==replacement.resolve()
        assert updated["dispersion"]==original["dispersion"] and updated["orm"]==original["orm"]
        assert all(path.is_file() for path in original.values())
        assert window._remove_measurement_role("dispersion")
        removed=load_session(manifest).resolve(manifest)
        assert set(removed)=={"bpm_noise","orm"} and original["dispersion"].is_file()
    finally:window.close(); app.processEvents()


def test_resumed_partial_bundle_appends_missing_role_without_replacing_old_file(tmp_path):
    app=QApplication.instance() or QApplication([]); configured,_=project(tmp_path)
    bundle=save_full_session_bundle(tmp_path/"resume.pyloco-measure-session",project=configured,session_manifest=source_session(tmp_path,("bpm_noise",)),reference_lattice=None)
    old_noise=bundle.measurement_paths["bpm_noise"]; window=MeasureMainWindow(); window.project=bundle.project; window.project_path=bundle.project_path; window._load_project_widgets()
    window.saved_session_path=bundle.manifest_path; window.saved_measurement_path=old_noise; window._fresh_measurement_session=False; window.output_directory.setText("measurements"); window.measurement_name.setText("continued-dispersion"); window.nominal_rf.setText("500000000")
    devices=(BpmDevice("B1","x1","y1"),BpmDevice("B2","x2","y2")); raw=np.zeros((2,2))
    states=tuple(DispersionStateResult(label,frequency,frequency,True,raw.copy(),raw.copy(),np.arange(2.),tuple(d.name for d in devices)) for label,frequency in (("reference",500_000_000.),("positive",500_000_100.),("negative",499_999_900.)))
    result=DispersionResult(devices,states,"bipolar",500_000_000.,100.,"restored",1.)
    try:
        window._save_result(result)
        restored=open_full_session_bundle(bundle.manifest_path)
        assert set(restored.measurement_paths)=={"bpm_noise","dispersion"}
        assert old_noise.is_file()
        assert restored.session.metadata["full_measure_session"]["roles"]=={"bpm_noise":True,"dispersion":True,"orm":False}
    finally:
        window.close(); app.processEvents()
