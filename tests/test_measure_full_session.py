from __future__ import annotations

import shutil

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from pyLOCO.data_schema import MeasurementSession, SessionFile, save_session, write_bpm_noise, write_dispersion, write_orm
from pyLOCO.measure.full_session import open_full_session_bundle, save_full_session_bundle
from pyLOCO.measure.project import MeasureProject, load_measure_project, save_measure_project
from pyLOCO.measure.main_window import MeasureMainWindow
from pyLOCO.measure.acquisition import BpmDevice, DispersionResult, DispersionStateResult
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
        assert window.result is None
        assert "requires reconnection" in window.statusBar().currentMessage()
        assert window.review_open_button.isEnabled()
    finally:
        window.close(); app.processEvents()


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
