from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication,QMessageBox

from pyLOCO.data_schema import MeasurementSession,SessionFile,save_session,write_bpm_noise,write_dispersion,write_orm
from pyLOCO.gui.app import build_application
from pyLOCO.gui.backend import LocoRunRequest
from pyLOCO.gui.main_window import MainWindow
from pyLOCO.gui.models.project import ImportedDataset,ProjectMetadata
from pyLOCO.gui import suite
from pyLOCO.gui.suite import inspect_measurement_session
from pyLOCO.measure.main_window import MeasureMainWindow
from pyLOCO.correct.model import load_review
from pyLOCO.correct import app as correct_app


def complete_session(tmp_path):
    bpms=["B1","B2"]; hcors=["H1","H2"]; vcors=["V1"]
    orm=tmp_path/"orm.h5"; write_orm(orm,response_matrix=np.arange(12,dtype=float).reshape(4,3),bpm_names=bpms,horizontal_corrector_names=hcors,vertical_corrector_names=vcors,requested_kick_h_rad=[1e-4,2e-4],requested_kick_v_rad=[3e-4],actual_kick_h_rad=[.9e-4,1.9e-4],actual_kick_v_rad=[2.9e-4],orbit_plus_m=np.ones((3,4)),orbit_minus_m=np.zeros((3,4)),scaled=True,direction="bipolar",metadata={"timestamp_utc":"2026-08-29T10:00:00Z"})
    noise=tmp_path/"noise.h5"; write_bpm_noise(noise,noise_x_m=[1e-7,2e-7],noise_y_m=[3e-7,4e-7],bpm_names=bpms,raw_orbits_x_m=np.zeros((2,2)),raw_orbits_y_m=np.zeros((2,2)),metadata={"timestamp_utc":"2026-08-29T10:01:00Z"})
    dispersion=tmp_path/"dispersion.h5"; write_dispersion(dispersion,measured_eta_x=[1e-6,2e-6],measured_eta_y=[3e-6,4e-6],bpm_names=bpms,rf_frequency_hz=[499e6-1000,499e6+1000],raw_orbits_x_m=np.zeros((2,2)),raw_orbits_y_m=np.zeros((2,2)),rf_step_hz=2000,bidirectional=True,metadata={"timestamp_utc":"2026-08-29T10:02:00Z"})
    manifest=tmp_path/"session.pyloco-session.json"; save_session(manifest,MeasurementSession("suite-mock",(SessionFile("orm",orm.name),SessionFile("bpm_noise",noise.name),SessionFile("dispersion",dispersion.name)),{"machine":"Mock"})); return manifest


def test_exactly_three_primary_launchers():
    text=Path("pyproject.toml").read_text(); scripts=text.split("[project.scripts]",1)[1].split("[",1)[0]; assert set(line.split("=",1)[0].strip() for line in scripts.splitlines() if "=" in line)=={"pyloco-gui","pyloco-measure","pyloco-correct"}


def test_complete_session_preserves_order_kicks_rf_and_units(tmp_path):
    handoff=inspect_measurement_session(complete_session(tmp_path)); assert handoff.available_roles==("orm","bpm_noise","dispersion"); assert handoff.missing_roles==(); orm=handoff.options["orm"]; assert orm["bpm_names"]==["B1","B2"]; assert orm["horizontal_corrector_names"]==["H1","H2"]; assert orm["vertical_corrector_names"]==["V1"]; assert orm["requested_kick_h_rad"]==[1e-4,2e-4]; np.testing.assert_allclose(orm["actual_kick_h_rad"],[.9e-4,1.9e-4]); assert orm["scaled"] is True; assert orm["row_order"]=="horizontal_bpms,vertical_bpms"; assert orm["column_order"]=="horizontal_correctors,vertical_correctors"; assert handoff.options["dispersion"]["rf_step_hz"]==2000.; assert handoff.provenance["session_id"]=="suite-mock"


def test_incomplete_session_is_truthfully_reported(tmp_path):
    noise=tmp_path/"noise.h5"; write_bpm_noise(noise,noise_x_m=[1e-7],noise_y_m=[2e-7],bpm_names=["B1"],raw_orbits_x_m=np.zeros((2,1)),raw_orbits_y_m=np.zeros((2,1))); manifest=tmp_path/"partial.json"; save_session(manifest,MeasurementSession("partial",(SessionFile("bpm_noise",noise.name),)))
    handoff=inspect_measurement_session(manifest); assert handoff.available_roles==("bpm_noise",); assert handoff.missing_roles==("orm","dispersion")


def test_fit_imports_session_and_persists_provenance_without_copying(tmp_path,monkeypatch):
    app=QApplication.instance() or build_application(["suite-test"])
    for name in ("information","warning","critical"):
        monkeypatch.setattr(QMessageBox,name,lambda *args:None)
    monkeypatch.setattr(QMessageBox,"question",lambda *args:QMessageBox.Discard)
    window=MainWindow(); manifest=complete_session(tmp_path); assert window.open_measurement_session(manifest); assert set(window.project.measurements)=={"orm","bpm_noise","dispersion"}; assert window.project.measurements["orm"].path==str((tmp_path/"orm.h5").resolve()); assert window.project.measurement_session["session_id"]=="suite-mock"; assert window.project.loco_config.response_matrix.rfStep==2000.; request=LocoRunRequest.from_project(window.project); assert request.measurement_session["session_id"]=="suite-mock"; np.testing.assert_allclose(request.measurement_options["orm"]["actual_kick_h_rad"],[.9e-4,1.9e-4]); window.close()


def test_main_suite_launches_use_canonical_handoff(monkeypatch):
    app=QApplication.instance() or build_application(["suite-launch"]); calls=[]; monkeypatch.setattr("pyLOCO.gui.main_window.launch_suite_application",lambda application,*args:(calls.append((application,args)) or (True,"process 1"))); window=MainWindow(); window.open_measure_app(); window.open_correct_app()
    assert calls==[("measure",()),("correct",())]
    window.close()


def test_suite_launcher_does_not_create_duplicate_processes(monkeypatch):
    calls=[]
    monkeypatch.setattr("PySide6.QtCore.QProcess.startDetached",lambda executable,arguments:(calls.append((executable,arguments)) or (True,43210)))
    monkeypatch.setattr(suite.os,"kill",lambda pid,signal:None)
    suite._DETACHED_SUITE_PIDS.clear()
    try:
        first=suite.launch_suite_application("correct","--results","/tmp/result")
        second=suite.launch_suite_application("correct","--results","/tmp/result")
        assert first==(True,"process 43210")
        assert second==(True,"already running (process 43210)")
        assert len(calls)==1
    finally:suite._DETACHED_SUITE_PIDS.clear()


def test_suite_launcher_restarts_after_previous_process_exits(monkeypatch):
    calls=[]
    monkeypatch.setattr("PySide6.QtCore.QProcess.startDetached",lambda executable,arguments:(calls.append((executable,arguments)) or (True,54321)))
    monkeypatch.setattr(suite.os,"kill",lambda pid,signal:(_ for _ in ()).throw(ProcessLookupError()))
    suite._DETACHED_SUITE_PIDS.clear(); suite._DETACHED_SUITE_PIDS["measure"]=12345
    try:
        assert suite.launch_suite_application("measure")== (True,"process 54321")
        assert len(calls)==1
    finally:suite._DETACHED_SUITE_PIDS.clear()


def test_fit_to_correct_launches_distinct_app_with_selected_iteration(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr("pyLOCO.gui.main_window.launch_suite_application",lambda application,*args:(calls.append((application,args)) or (True,"process 3")))
    dummy=SimpleNamespace(
        results_workspace=SimpleNamespace(loader=SimpleNamespace(result_dir=tmp_path,iteration=3)),
        statusBar=lambda:SimpleNamespace(showMessage=lambda message:None),
        _launch_suite=lambda application,*args:(calls.append((application,args)) or True),
    )
    assert MainWindow.open_correct_app(dummy)
    assert calls==[("correct",("--results",str(tmp_path),"--iteration","3"))]


def test_measure_to_fit_uses_explicit_portable_session_path(tmp_path,monkeypatch):
    manifest=complete_session(tmp_path); calls=[]
    monkeypatch.setattr("pyLOCO.measure.main_window.launch_suite_application",lambda application,*args:(calls.append((application,args)) or (True,"process 2")))
    dummy=SimpleNamespace(saved_session_path=manifest,statusBar=lambda:SimpleNamespace(showMessage=lambda _message:None))
    MeasureMainWindow.explain_open(dummy)
    assert calls==[("fit",("--measurement-session",str(manifest)))]


def test_correct_handoff_shows_and_activates_before_loading_results(monkeypatch,tmp_path):
    events=[]
    class Status:
        def showMessage(self,message):events.append(("status",message))
    class Window:
        def show(self):events.append("show")
        def raise_(self):events.append("raise")
        def activateWindow(self):events.append("activate")
        def statusBar(self):return Status()
        def _load(self,path,iteration=None):events.append(("load",path,iteration))
    class App:
        def exec(self):events.append("exec"); return 0
    monkeypatch.setattr(correct_app,"CorrectMainWindow",Window); monkeypatch.setattr(correct_app,"build_application",lambda _argv:App())
    monkeypatch.setattr(correct_app.QTimer,"singleShot",lambda _delay,callback:(events.append("scheduled"),callback()))
    assert correct_app.main(["--results",str(tmp_path),"--iteration","2"])==0
    assert events[:3]==["show","raise","activate"]
    assert events.index("scheduled")<events.index(("load",str(tmp_path),2))<events.index("exec")


def test_portable_example_contains_no_machine_specific_paths():
    root=Path("Examples/Suite"); text="\n".join(path.read_text(errors="ignore") for path in root.rglob("*.json")); assert "/Users/" not in text and "/private/tmp/" not in text and "Documents/Codex" not in text


def test_portable_mock_results_open_in_correct_with_session_provenance():
    review=load_review("Examples/Suite/mock_results")
    assert [item.name for item in review.items]==["QF","QD"]
    assert all(item.metadata["measurement_session"]["session_id"]=="pyloco-suite-mock-complete" for item in review.items)
    assert review.to_plan().metadata["source_provenance"]["fit_timestamp"]=="2026-08-29T12:05:00+00:00"


def test_session_provenance_round_trip_stays_portable(tmp_path):
    root=tmp_path/"clone"; root.mkdir(); manifest=complete_session(root)
    handoff=inspect_measurement_session(manifest)
    project=ProjectMetadata(name="Suite handoff")
    project.measurements={role:ImportedDataset(role,str(path),"h5",path.stat().st_size,handoff.options[role]) for role,path in handoff.files.items()}
    project.measurement_session=handoff.provenance
    saved=project.save(root/"fit.pyloco.json"); text=saved.read_text()
    assert str(tmp_path) not in text
    restored=ProjectMetadata.load(saved)
    assert restored.measurement_session["session_id"]=="suite-mock"
    assert Path(restored.measurement_session["manifest"])==manifest.resolve()
    assert all(Path(value).exists() for value in restored.measurement_session["measurement_files"].values())
