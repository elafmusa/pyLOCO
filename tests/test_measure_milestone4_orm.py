from __future__ import annotations

import json, os
from pathlib import Path
from threading import Event

import h5py
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
from PySide6.QtWidgets import QApplication

from pyLOCO.control_system import MockAdapter
from pyLOCO.data_schema import MeasurementSession, SessionFile, load_session, save_session, validate_measurement_file, write_bpm_noise, write_dispersion
from pyLOCO.gui.backend import _load_measurements
from pyLOCO.measure.acquisition import ORMAcquirer, ORMInterrupted, ORMAcquisitionError
from pyLOCO.measure.app import build_application
from pyLOCO.measure.main_window import MeasureMainWindow, build_mock_adapter, default_mock_correctors, default_mock_devices
from pyLOCO.measure.project import load_measure_project, save_measure_project

@pytest.fixture(scope="module")
def app(): return QApplication.instance() or build_application(["orm-test"])

def setup(nbpms=4,nh=3,nv=2,readings=4):
    bpms=default_mock_devices(nbpms); h,v=default_mock_correctors(nh,nv); rows=np.arange(2*nbpms)[:,None]; cols=np.arange(nh+nv)[None,:]; known=2.5e-3*np.sin(.37*(rows+1)+.61*(cols+1)); adapter=build_mock_adapter(bpms,readings=readings,horizontal_correctors=h,vertical_correctors=v,response_matrix=known,enable_orm_writes=True); return bpms,h,v,known,adapter

@pytest.mark.parametrize("direction",["bipolar","positive","negative"])
def test_mock_orm_recovers_known_matrix_and_restores(direction):
    bpms,h,v,known,adapter=setup(); kh=np.array([80,100,130])*1e-6; kv=np.array([90,120])*1e-6
    result=ORMAcquirer(adapter,bpms,h,v).acquire(kh,kv,direction=direction,scaled=True,readings=4,delay_seconds=0,settling_delay_seconds=0)
    np.testing.assert_allclose(result.response_matrix,known,rtol=0,atol=2e-12)
    assert result.response_matrix.shape==(8,5); assert result.restoration_status==("restored",)*5
    assert all(float(adapter.read(c.readback_channel).value)==0 for c in h+v)

def test_unscaled_orm_is_orbit_difference_not_kick_normalized():
    bpms,h,v,known,adapter=setup(); kicks=np.array([80,100,130,90,120])*1e-6
    result=ORMAcquirer(adapter,bpms,h,v).acquire(kicks[:3],kicks[3:],scaled=False,readings=3,delay_seconds=0,settling_delay_seconds=0)
    np.testing.assert_allclose(result.response_matrix,known*kicks[None,:],atol=2e-12)

def test_cancellation_and_failures_attempt_restoration():
    bpms,h,v,_,adapter=setup(2,1,0,3); cancel=Event()
    def stop(event):
        if event.get("event")=="orbit": cancel.set()
    with pytest.raises(ORMInterrupted) as caught: ORMAcquirer(adapter,bpms,h,v).acquire([1e-4],[],readings=3,delay_seconds=0,cancel_event=cancel,progress=stop)
    assert caught.value.restoration_status=="restored" and adapter.read(h[0].readback_channel).value==0

    class FailingMock(MockAdapter):
        def write(self,channel,value):
            if value<0: raise RuntimeError("injected minus-state failure")
            return super().write(channel,value)
    base=build_mock_adapter(bpms,readings=3,horizontal_correctors=h,vertical_correctors=v,response_matrix=np.ones((4,1))*.001)
    failing=FailingMock(base._channels,allow_simulated_writes=True,sequences=base._sequences,device_catalog=base._device_catalog,setpoint_dependent_channels=base._setpoint_dependent_channels)
    with pytest.raises(ORMAcquisitionError) as caught: ORMAcquirer(failing,bpms,h,v).acquire([1e-4],[],readings=3,delay_seconds=0)
    assert caught.value.restoration_status=="restored" and failing.read(h[0].readback_channel).value==0

@pytest.mark.parametrize("phase",["plus","minus","orbit","restore"])
def test_injected_phase_failures_never_complete_and_attempt_restore(phase):
    bpms,h,v,_,base=setup(2,1,0,3)
    class PhaseFailure(MockAdapter):
        def __init__(self):
            super().__init__(base._channels,allow_simulated_writes=True,sequences=base._sequences,device_catalog=base._device_catalog,setpoint_dependent_channels=base._setpoint_dependent_channels); self.perturbed=False
        def write(self,channel,value):
            if phase=="plus" and value>0: raise RuntimeError("plus")
            if phase=="minus" and value<0: raise RuntimeError("minus")
            if phase=="restore" and self.perturbed and value==0: raise RuntimeError("restore")
            if value!=0:self.perturbed=True
            return super().write(channel,value)
        def read(self,channel):
            if phase=="orbit" and self.perturbed and channel.startswith("MOCK/BPM/"): raise RuntimeError("orbit")
            return super().read(channel)
    adapter=PhaseFailure()
    with pytest.raises(ORMAcquisitionError) as caught: ORMAcquirer(adapter,bpms,h,v).acquire([1e-4],[],readings=3,delay_seconds=0)
    expected="restore_failed" if phase=="restore" else "restored"
    assert caught.value.restoration_status==expected

def test_schema_session_and_main_importer(app,tmp_path):
    bpms,h,v,_,adapter=setup(); result=ORMAcquirer(adapter,bpms,h,v).acquire([80e-6,100e-6,130e-6],[90e-6,120e-6],scaled=True,readings=3,delay_seconds=0)
    noise=tmp_path/"noise.h5"; write_bpm_noise(noise,noise_x_m=np.ones(4)*1e-8,noise_y_m=np.ones(4)*2e-8,bpm_names=[b.name for b in bpms],raw_orbits_x_m=np.zeros((2,4)),raw_orbits_y_m=np.zeros((2,4)))
    dispersion=tmp_path/"dispersion.h5"; write_dispersion(dispersion,measured_eta_x=np.zeros(4),measured_eta_y=np.zeros(4),bpm_names=[b.name for b in bpms],rf_frequency_hz=[499_999_000,500_001_000],raw_orbits_x_m=np.zeros((2,4)),raw_orbits_y_m=np.zeros((2,4)),rf_step_hz=2000,bidirectional=True)
    manifest=tmp_path/"measurement-session.pyloco-session.json"; save_session(manifest,MeasurementSession("complete",(SessionFile("bpm_noise",noise.name),SessionFile("dispersion",dispersion.name))),validate_files=False)
    window=MeasureMainWindow(devices=bpms,adapter=adapter,horizontal_correctors=h,vertical_correctors=v); window.measurement_type.setCurrentIndex(2); window.output_directory.setText(str(tmp_path)); window.measurement_name.setText("orm"); window._save_result(result)
    info=validate_measurement_file(window.saved_measurement_path); assert info["kind"]=="orm"
    with h5py.File(window.saved_measurement_path,"r") as f:
        assert f["response_matrix"].shape==(8,5); assert f.attrs["response_matrix_unit"]=="m/rad"; assert f["raw/orbit_plus_m"].shape==(5,3,8); assert len(f["setpoints/restoration_status"])==5
        metadata=json.loads(f["metadata/json"][()].decode() if isinstance(f["metadata/json"][()],bytes) else f["metadata/json"][()])
        assert metadata["machine_identity"]=="Mock / offline"
    window._show_result(result); transaction=window.rf_diagnostics.text()
    assert "Original:" in transaction and "requested +/−:" in transaction
    assert "Readback +/−:" in transaction and "restoration error:" in transaction
    assert "orbit(+) − orbit(−)" in transaction
    assert window.orm_restoration_status.text()=="All correctors restored: ✓ YES"
    assert "Matrix shape: 8 × 5" in window.orm_measurement_summary.text()
    assert len(window.x_plot.figure.axes[0].lines)>=2  # X/Y and H/V boundaries
    window._set_completed_result_view(True)
    assert window.live_plot.isHidden() and window.plan_group.isHidden()
    assert not window.preview_toggle.isHidden() and window.preview_toggle.text()=="Show acquisition preview"
    window.preview_toggle.click(); app.processEvents()
    assert not window.live_plot.isHidden() and window.preview_toggle.text()=="Hide acquisition preview"
    loaded=_load_measurements({"orm":str(window.saved_measurement_path)}); np.testing.assert_allclose(loaded["orm"],result.response_matrix)
    session=load_session(manifest,validate_files=False); assert {entry.role for entry in session.files}=={"orm","bpm_noise","dispersion"}; assert session.missing_roles==(); window.close()

def test_orm_gui_context_preview_kicks_and_project_roundtrip(app,tmp_path):
    window=MeasureMainWindow(); window.measurement_type.setCurrentIndex(2); app.processEvents()
    assert not window.orm_config_group.isHidden() and not window.orm_corrector_group.isHidden(); assert window.start_button.text()=="Start ORM measurement"; assert window.kick_preview.rowCount()==12
    window.corrector_selection_widgets["hcor"]["exclusion"].setText("1"); window._refresh_corrector_preview("hcor"); assert len(window.selected_hcorrectors)==5
    kicks=tmp_path/"kicks.npz"; np.savez(kicks,horizontal=np.arange(1,6)*10e-6,vertical=np.arange(1,7)*20e-6); window.orm_kick_mode.setCurrentIndex(1); window.orm_kick_file.setText(str(kicks)); window._update_kick_preview(); assert window.kick_preview.rowCount()==11
    window.orm_direction.setCurrentIndex(2); window.orm_scaled.setChecked(True); project=window._collect_project(); path=tmp_path/"orm.pyloco-measure.json"; save_measure_project(path,project); loaded=load_measure_project(path); assert loaded.orm_direction=="negative" and loaded.orm_scaled and loaded.excluded_hcor_positions=="1"
    window.close()

def test_measurement_specific_device_selections_and_project_roundtrip(app,tmp_path):
    bpms,h,v,_,adapter=setup(nbpms=30,nh=8,nv=7)
    window=MeasureMainWindow(devices=bpms,adapter=adapter,horizontal_correctors=h,vertical_correctors=v)
    # BPM noise: first 20 BPMs.
    window._select_device_subset("bpm","first",20); assert len(window.selected_devices)==20
    # Dispersion: all BPMs.
    window.measurement_type.setCurrentIndex(window.measurement_type.findData("dispersion")); app.processEvents()
    window._select_all_bpms(); assert len(window.selected_devices)==30
    # ORM: 20 uniformly ordered BPMs and independently chosen H/V correctors.
    window.measurement_type.setCurrentIndex(window.measurement_type.findData("orm")); app.processEvents()
    window._select_device_subset("bpm","uniform",20); window._select_demo_one_each()
    assert len(window.selected_devices)==20 and len(window.selected_hcorrectors)==len(window.selected_vcorrectors)==1
    orm_names=tuple(device.name for device in window.selected_devices)
    # Repeated switching restores, rather than overwrites, each workflow selection.
    window.measurement_type.setCurrentIndex(window.measurement_type.findData("bpm_noise")); app.processEvents(); assert len(window.selected_devices)==20
    window.measurement_type.setCurrentIndex(window.measurement_type.findData("dispersion")); app.processEvents(); assert len(window.selected_devices)==30
    window.measurement_type.setCurrentIndex(window.measurement_type.findData("orm")); app.processEvents(); assert tuple(device.name for device in window.selected_devices)==orm_names
    project=window._collect_project(); path=tmp_path/"independent.pyloco-measure.json"; save_measure_project(path,project); loaded=load_measure_project(path)
    clone=MeasureMainWindow(devices=bpms,adapter=adapter,horizontal_correctors=h,vertical_correctors=v); clone.project=loaded; clone._load_project_widgets(); app.processEvents()
    assert len(clone.selected_devices)==20 and len(clone.selected_hcorrectors)==len(clone.selected_vcorrectors)==1
    clone.measurement_type.setCurrentIndex(clone.measurement_type.findData("dispersion")); app.processEvents(); assert len(clone.selected_devices)==30
    clone.measurement_type.setCurrentIndex(clone.measurement_type.findData("bpm_noise")); app.processEvents(); assert len(clone.selected_devices)==20
    window.close(); clone.close()

def test_generic_selection_helpers_preserve_backend_order(app):
    bpms,h,v,_,adapter=setup(nbpms=21,nh=9,nv=8)
    window=MeasureMainWindow(devices=bpms,adapter=adapter,horizontal_correctors=h,vertical_correctors=v); window.measurement_type.setCurrentIndex(2); app.processEvents()
    window._select_demo_small_orm()
    assert tuple(window.selected_devices)==window._uniform_devices(bpms,20)
    assert tuple(window.selected_hcorrectors)==window._uniform_devices(h,5)
    assert tuple(window.selected_vcorrectors)==window._uniform_devices(v,5)
    window.bpm_search.setText("BPM-00"); window._select_filtered_bpms(); assert all("BPM-00" in item.name for item in window.selected_devices)
    assert window.selection_message.text().endswith("Source: manual selection.")
    window.close()

def test_strict_orm_validation_rejects_sign_normalization_restoration_and_order(app,tmp_path):
    bpms,h,v,_,adapter=setup(nbpms=3,nh=1,nv=1,readings=3); result=ORMAcquirer(adapter,bpms,h,v).acquire([100e-6],[100e-6],scaled=True,readings=3,delay_seconds=0)
    window=MeasureMainWindow(devices=bpms,adapter=adapter,horizontal_correctors=h,vertical_correctors=v); window.measurement_type.setCurrentIndex(2); window.output_directory.setText(str(tmp_path)); window.measurement_name.setText("strict"); window._save_result(result); source=window.saved_measurement_path
    assert validate_measurement_file(source)["kind"]=="orm"
    cases={
        "sign":lambda f:f["response_matrix"].__setitem__((slice(None),0),-f["response_matrix"][:,0]),
        "actual_kick":lambda f:f["kicks/horizontal/actual"].__setitem__(0,2e-4),
        "row_order":lambda f:f.attrs.__setitem__("row_order","vertical_bpms,horizontal_bpms"),
        "restoration_status":lambda f:f["setpoints/restoration_status"].__setitem__(0,"restore_failed"),
        "final_readback":lambda f:f["setpoints/final"].__setitem__(0,1e-6),
    }
    import shutil
    for name,mutate in cases.items():
        target=tmp_path/f"{name}.h5"; shutil.copy2(source,target)
        with h5py.File(target,"r+") as handle:mutate(handle)
        with pytest.raises(ValueError):validate_measurement_file(target)
    window.close()

def test_portable_mock_orm_example():
    path=Path("Examples/Measure/mock_orm.pyloco-measure.json"); project=load_measure_project(path); assert project.measurement_type=="orm"; text=path.read_text(); assert "/Users/" not in text and "/private/tmp/" not in text
