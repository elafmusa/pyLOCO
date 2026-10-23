from __future__ import annotations

import json, os
from pathlib import Path
from threading import Event

import h5py
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
from PySide6.QtCore import Qt
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

def test_bipolar_orm_restores_and_verifies_k0_before_negative_kick():
    bpms,h,_,_,base=setup(2,1,0,2)
    original=17e-6; base.write(h[0].setpoint_channel,original)
    writes=[]; events=[]
    class InstrumentedMock(MockAdapter):
        def write(self,channel,value):
            writes.append((channel,float(value)))
            return super().write(channel,value)
    adapter=InstrumentedMock(base._channels,allow_simulated_writes=True,sequences=base._sequences,device_catalog=base._device_catalog,setpoint_dependent_channels=base._setpoint_dependent_channels)
    result=ORMAcquirer(adapter,bpms,h,()).acquire([100e-6],[],readings=2,delay_seconds=0,settling_delay_seconds=0,progress=events.append)
    np.testing.assert_allclose([value for _,value in writes],[original+50e-6,original,original-50e-6,original],rtol=0,atol=1e-15)
    assert [event["event"] for event in events if event["event"].startswith("intermediate_")]==["intermediate_restore","intermediate_restored"]
    assert result.intermediate_restoration_status==("restored",)
    np.testing.assert_allclose(result.intermediate_restored_setpoints_rad,[original],rtol=0,atol=1e-15)
    np.testing.assert_allclose(result.final_setpoints_rad,[original],rtol=0,atol=1e-15)
    assert np.isfinite(result.raw_reference_before_m).all()
    assert np.isfinite(result.raw_reference_intermediate_m).all()
    assert np.isfinite(result.raw_reference_final_m).all()

def test_negative_kick_never_starts_when_intermediate_restore_fails():
    bpms,h,_,_,base=setup(2,1,0,2); writes=[]
    class RejectRestore(MockAdapter):
        def write(self,channel,value):
            writes.append(float(value))
            if len(writes)>=2 and value==0:raise RuntimeError("injected K0 restore failure")
            return super().write(channel,value)
    adapter=RejectRestore(base._channels,allow_simulated_writes=True,sequences=base._sequences,device_catalog=base._device_catalog,setpoint_dependent_channels=base._setpoint_dependent_channels)
    with pytest.raises(ORMAcquisitionError):ORMAcquirer(adapter,bpms,h,()).acquire([100e-6],[],readings=2,delay_seconds=0)
    assert not any(value<0 for value in writes),"K− must not be applied until K0 restoration succeeds"

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
        np.testing.assert_allclose(f["setpoints/intermediate_restored"][:],f["setpoints/original"][:],rtol=0,atol=1e-12)
        assert all(value.decode()=="restored" for value in f["setpoints/intermediate_restoration_status"][:])
        assert f["raw/reference_before_m"].shape==f["raw/orbit_plus_m"].shape
        assert f["raw/reference_intermediate_m"].shape==f["raw/orbit_plus_m"].shape
        assert f["raw/reference_final_m"].shape==f["raw/orbit_plus_m"].shape
        metadata=json.loads(f["metadata/json"][()].decode() if isinstance(f["metadata/json"][()],bytes) else f["metadata/json"][()])
        assert metadata["machine_identity"]=="Mock / offline"
        assert metadata["readings_per_state"]==3 and metadata["settling_delay_seconds"]==1.0
        assert "diagnostics/state_a_first_sample_minus_later_mean_m" in f
    window._show_result(result); transaction=window.rf_diagnostics.text()
    assert "Original K0:" in transaction and "requested K+/K−:" in transaction
    assert "Readback K+:" in transaction and "Final restore K− → K0:" in transaction
    assert "orbit(+) − orbit(−)" in transaction
    assert window.orm_restoration_status.text()=="✓ ALL CORRECTORS RESTORED"
    assert "Intermediate restore K+ → K0" in transaction
    assert "Final orbit return:" in transaction and "diagnostic-only" in transaction
    assert "Matrix: 8 × 5" in window.orm_measurement_summary.text()
    assert len(window.x_plot.figure.axes[0].lines)>=2  # X/Y and H/V boundaries
    assert window.orm_view_mode.currentData()=="2d" and window.x_plot.figure.axes[0].images[0].get_cmap().name=="viridis"
    assert window.x_plot.figure.axes[0].images[0].origin=="upper"
    state=window._orm_plot_state[window.x_plot]
    event=type("Event",(),{"inaxes":state["axis"],"xdata":4.0,"ydata":7.0})()
    window._on_orm_plot_hover(window.x_plot,event)
    assert h[0].name not in window.orm_element_info.text()
    assert v[1].name in window.orm_element_info.text() and bpms[3].name in window.orm_element_info.text()
    window._on_orm_plot_click(window.x_plot,event); assert v[1].name in window.orm_element_info.text() and "V→Y" in window.orm_element_info.text()
    column_state=window._orm_column_state; column_event=type("Event",(),{"inaxes":column_state["axis"],"xdata":2.0,"ydata":column_state["series"][0][2][2]})()
    window._on_orm_column_hover(column_event); assert bpms[2].name in window.orm_element_info.text() and "plane X" in window.orm_element_info.text()
    window._on_orm_column_click(column_event); assert np.asarray(column_state["selected"].get_offsets()).shape==(1,2)
    window.orm_view_mode.setCurrentIndex(window.orm_view_mode.findData("3d")); app.processEvents(); assert window.x_plot.figure.axes[0].name=="3d"
    window._set_completed_result_view(True)
    assert window.live_plot.isHidden() and window.plan_group.isHidden()
    # The compact completed view no longer exposes a second nested preview
    # toggle; acquisition preview remains in the outer scrollable workspace.
    assert window.preview_toggle.isHidden()
    window.preview_toggle.click(); app.processEvents()
    assert not window.live_plot.isHidden() and window.preview_toggle.text()=="Hide live preview"
    loaded=_load_measurements({"orm":str(window.saved_measurement_path)}); np.testing.assert_allclose(loaded["orm"],result.response_matrix)
    # A newly opened Measure window owns a fresh logical session and must not
    # merge an unrelated manifest merely because the output directory matches.
    session=load_session(manifest,validate_files=False); assert {entry.role for entry in session.files}=={"bpm_noise","dispersion"}
    assert window.saved_session_path!=manifest
    new_session=load_session(window.saved_session_path,validate_files=False); assert {entry.role for entry in new_session.files}=={"orm"}; assert set(new_session.missing_roles)=={"bpm_noise","dispersion"}; window.close()

@pytest.mark.parametrize("direction",["positive","negative"])
def test_one_sided_orm_save_omits_non_applicable_intermediate_reference(app,tmp_path,direction):
    bpms,h,v,_,adapter=setup()
    result=ORMAcquirer(adapter,bpms,h,v).acquire(
        [80e-6,100e-6,130e-6],[90e-6,120e-6],direction=direction,
        scaled=True,readings=3,delay_seconds=0,settling_delay_seconds=0,
    )
    assert not np.isfinite(result.raw_reference_intermediate_m).any()
    window=MeasureMainWindow(devices=bpms,adapter=adapter,horizontal_correctors=h,vertical_correctors=v)
    window.measurement_type.setCurrentIndex(2)
    window.output_directory.setText(str(tmp_path))
    window.measurement_name.setText(f"orm-{direction}")
    window._save_result(result)

    info=validate_measurement_file(window.saved_measurement_path)
    assert info["kind"]=="orm"
    with h5py.File(window.saved_measurement_path,"r") as f:
        assert f.attrs["direction"]==direction
        assert "raw/reference_before_m" in f
        assert "raw/reference_final_m" in f
        assert "raw/reference_intermediate_m" not in f
        assert "setpoints/intermediate_restored" not in f
        metadata=json.loads(f["metadata/json"][()].decode() if isinstance(f["metadata/json"][()],bytes) else f["metadata/json"][()])
        assert metadata["orm_validation_convention"]=="pyloco_one_sided_v1"
        assert "no intermediate reference" in metadata["restored_orbit_role"]
        state="positive" if direction=="positive" else "negative"
        assert metadata["subtraction_convention"]==f"mean({state})-mean(reference)"
    loaded=_load_measurements({"orm":str(window.saved_measurement_path)})
    np.testing.assert_allclose(loaded["orm"],result.response_matrix)
    window.close()

def test_orm_gui_context_preview_kicks_and_project_roundtrip(app,tmp_path):
    window=MeasureMainWindow(); window.measurement_type.setCurrentIndex(2); app.processEvents()
    assert window.readings.value()==10 and window.delay.value()==pytest.approx(.1) and window.orm_settling_delay.value()==pytest.approx(1.0)
    assert window.orm_output_format.text()=="Orbit difference [m] — required for pyLOCO FIT"
    assert "K₀ + ΔK/2" in window.orm_state_explanation.text()
    window.orm_direction.setCurrentIndex(window.orm_direction.findData("positive")); app.processEvents()
    assert window.orm_hkick_label.text()=="Horizontal one-sided kick +ΔK"
    assert "K₀ + ΔK and reference K₀" in window.orm_state_explanation.text()
    assert "/2" not in window.orm_state_explanation.text()
    assert float(window.kick_preview.item(0,3).text()) > 0
    window.orm_direction.setCurrentIndex(window.orm_direction.findData("negative")); app.processEvents()
    assert window.orm_hkick_label.text()=="Horizontal one-sided kick −ΔK"
    assert "K₀ − ΔK and reference K₀" in window.orm_state_explanation.text()
    assert float(window.kick_preview.item(0,3).text()) < 0
    window.orm_direction.setCurrentIndex(window.orm_direction.findData("bipolar")); app.processEvents()
    assert not window.orm_normalized_analysis.isChecked()
    assert not window.orm_config_group.isHidden() and not window.orm_corrector_group.isHidden(); assert window.start_button.text()=="Start ORM measurement"; assert window.kick_preview.rowCount()==12
    assert window.results_tabs.tabText(0)=="Full ORM" and window.results_tabs.currentIndex()==0
    assert [window.results_tabs.tabText(index) for index in range(5)]==["Full ORM","Direct H→X","Direct V→Y","Coupling V→X","Coupling H→Y"]
    assert window.results_tabs.tabBar().elideMode()==Qt.ElideRight
    assert not window.results_tabs.tabBar().usesScrollButtons()
    assert window.results_tabs.tabToolTip(6)=="Measured − Model"
    window.corrector_selection_widgets["hcor"]["exclusion"].setText("1"); window._refresh_corrector_preview("hcor"); assert len(window.selected_hcorrectors)==5
    kicks=tmp_path/"kicks.npz"; np.savez(kicks,horizontal=np.arange(1,6)*10e-6,vertical=np.arange(1,7)*20e-6); window.orm_kick_mode.setCurrentIndex(1); window.orm_kick_file.setText(str(kicks)); window._update_kick_preview(); assert window.kick_preview.rowCount()==11
    window.orm_direction.setCurrentIndex(2); window.orm_normalized_analysis.setChecked(True); project=window._collect_project(); path=tmp_path/"orm.pyloco-measure.json"; save_measure_project(path,project); loaded=load_measure_project(path); assert loaded.orm_direction=="negative" and not loaded.orm_scaled and loaded.metadata["orm_normalized_analysis"] and loaded.excluded_hcor_positions=="1"
    window.close()


@pytest.mark.parametrize("size",[(1200,800),(1600,1000)])
@pytest.mark.parametrize("theme",["light","dark"])
def test_orm_heatmap_remains_readable_in_both_themes(app,theme,size):
    bpms,h,v,_,adapter=setup(); result=ORMAcquirer(adapter,bpms,h,v).acquire([80e-6,100e-6,130e-6],[90e-6,120e-6],readings=2,delay_seconds=0)
    window=MeasureMainWindow(devices=bpms,adapter=adapter,horizontal_correctors=h,vertical_correctors=v); window.resize(*size); window.apply_theme(theme); window.measurement_type.setCurrentIndex(2); window._show_result(result); window.result=result; window._update_diagnostics_visibility(); window.tabs.setCurrentIndex(4); window.show(); app.processEvents()
    axis=window.x_plot.figure.axes[0]; image=axis.images[0]
    assert image.origin=="upper" and image.get_cmap().name=="viridis"
    assert axis.get_xlabel()=="Corrector index" and axis.get_ylabel().startswith("ORM row index")
    assert axis.get_position().width>.55
    assert window.x_plot.figure.axes[1].get_ylabel()=="Response [m]"
    assert window.orm_element_info.isVisible()
    assert window.results_tabs.height()==340
    window.orm_palette.setCurrentIndex(window.orm_palette.findData("red_blue")); app.processEvents(); image=window.x_plot.figure.axes[0].images[0]; assert image.get_cmap().name=="RdBu_r" and image.norm.vmin==pytest.approx(-image.norm.vmax)
    window.close()

def test_normalized_response_is_display_only_and_fit_matrix_stays_in_metres(app,tmp_path):
    bpms,h,v,_,adapter=setup(); result=ORMAcquirer(adapter,bpms,h,v).acquire([80e-6,100e-6,130e-6],[90e-6,120e-6],scaled=False,readings=2,delay_seconds=0)
    original=result.response_matrix.copy(); window=MeasureMainWindow(devices=bpms,adapter=adapter,horizontal_correctors=h,vertical_correctors=v)
    window.measurement_type.setCurrentIndex(window.measurement_type.findData("orm")); window.orm_normalized_analysis.setChecked(True); window._show_result(result)
    np.testing.assert_allclose(window._displayed_orm_matrix,original/result.effective_kicks_rad[np.newaxis,:])
    np.testing.assert_allclose(result.response_matrix,original); assert window._displayed_orm_unit=="m/rad"
    window.output_directory.setText(str(tmp_path)); window.measurement_name.setText("fit-metres"); window._save_result(result)
    with h5py.File(window.saved_measurement_path,"r") as handle:
        assert handle.attrs["response_matrix_unit"]=="m"
        np.testing.assert_allclose(handle["response_matrix"][:],original)
    np.testing.assert_allclose(_load_measurements({"orm":str(window.saved_measurement_path)})["orm"],original)
    window.close()

def test_all_orm_views_keep_titles_and_exact_identity_mapping_after_interaction(app):
    bpms,h,v,_,adapter=setup(); result=ORMAcquirer(adapter,bpms,h,v).acquire([80e-6,100e-6,130e-6],[90e-6,120e-6],readings=2,delay_seconds=0)
    window=MeasureMainWindow(devices=bpms,adapter=adapter,horizontal_correctors=h,vertical_correctors=v); window.measurement_type.setCurrentIndex(window.measurement_type.findData("orm")); window._show_result(result); window.show(); app.processEvents()
    cases=((window.x_plot,7,4,"V→Y",bpms[3].name,v[1].name),(window.y_plot,2,1,"H→X",bpms[2].name,h[1].name),(window.mean_x_plot,1,0,"V→X",bpms[1].name,v[0].name),(window.mean_y_plot,3,2,"H→Y",bpms[3].name,h[2].name),(window.orm_vv_plot,0,1,"V→Y",bpms[0].name,v[1].name))
    callback_counts={canvas:(len(canvas.canvas.callbacks.callbacks.get("motion_notify_event",{})),len(canvas.canvas.callbacks.callbacks.get("button_press_event",{}))) for canvas,*_ in cases}
    for canvas,row,column,block,bpm_name,corrector_name in cases:
        state=window._orm_plot_state[canvas]; title=state["axis"].get_title(); event=type("Event",(),{"inaxes":state["axis"],"xdata":float(column),"ydata":float(row)})()
        window._on_orm_plot_hover(canvas,event); text=window.orm_element_info.text(); assert bpm_name in text and corrector_name in text and block in text and "zero-based" in text
        window._on_orm_plot_click(canvas,event); canvas.canvas.draw(); expected_y="ORM row index" if canvas is window.x_plot else "BPM index"; assert state["axis"].get_title()==title and state["axis"].get_xlabel()=="Corrector index" and state["axis"].get_ylabel().startswith(expected_y)
    window.resize(1600,1000); window.results_tabs.setCurrentIndex(4); window.results_tabs.setCurrentIndex(0); app.processEvents()
    for canvas,*_ in cases:
        assert callback_counts[canvas]==(len(canvas.canvas.callbacks.callbacks.get("motion_notify_event",{})),len(canvas.canvas.callbacks.callbacks.get("button_press_event",{})))
        assert window._orm_plot_state[canvas]["axis"].get_title()
    window.close()

def test_loading_reference_model_after_completed_orm_refreshes_comparison(app,monkeypatch):
    bpms,h,v,known,adapter=setup()
    result=ORMAcquirer(adapter,bpms,h,v).acquire(
        [80e-6,100e-6,130e-6],[90e-6,120e-6],readings=2,delay_seconds=0,
    )
    window=MeasureMainWindow(devices=bpms,adapter=adapter,horizontal_correctors=h,vertical_correctors=v)
    window.measurement_type.setCurrentIndex(window.measurement_type.findData("orm"))
    window.result=result
    window._show_result(result)
    assert not window.orm_model_plot.figure.axes

    monkeypatch.setattr("pyLOCO.measure.main_window.model_orm",lambda *_args,**_kwargs:known*np.asarray(result.effective_kicks_rad)[None,:])
    window.reference_model=object()
    window._refresh_completed_reference_comparison()
    assert window.orm_model_plot.figure.axes[0].get_title()=="MODEL ORM — reference lattice"
    assert window.orm_difference_plot.figure.axes[0].get_title()=="Measured − MODEL ORM"
    assert window.orm_model_plot.figure.axes[0].images
    assert window.orm_difference_plot.figure.axes[0].images
    window.close()

def test_evolving_orm_does_not_invent_identity_for_unacquired_cells(app):
    bpms,h,v,_,adapter=setup(); window=MeasureMainWindow(devices=bpms,adapter=adapter,horizontal_correctors=h,vertical_correctors=v); window.measurement_type.setCurrentIndex(window.measurement_type.findData("orm"))
    matrix=np.full((2*len(bpms),len(h)+len(v)),np.nan); matrix[:,0]=np.arange(2*len(bpms))*1e-6; window._show_orm_progress(matrix,matrix[:,0]); state=window._orm_plot_state[window.x_plot]
    valid=type("Event",(),{"inaxes":state["axis"],"xdata":0.0,"ydata":1.0})(); window._on_orm_plot_hover(window.x_plot,valid); assert bpms[1].name in window.orm_element_info.text() and h[0].name in window.orm_element_info.text()
    missing=type("Event",(),{"inaxes":state["axis"],"xdata":1.0,"ydata":1.0})(); window._on_orm_plot_hover(window.x_plot,missing); assert "not been acquired" in window.orm_element_info.text() and bpms[1].name not in window.orm_element_info.text()
    axis=state["axis"]; image=axis.images[0]
    assert axis.get_title().endswith("1/5 corrector columns acquired")
    assert axis.get_xlim()==pytest.approx((-.5,4.5))
    assert axis.get_ylim()==pytest.approx((7.5,-.5))
    assert np.array_equal(axis.get_xticks(),np.arange(5))
    assert axis.get_ylabel()=="ORM row index (0–3: X; 4–7: Y)"
    assert np.ma.getmaskarray(image.get_array())[:,1:].all()
    assert image.get_cmap().get_bad()[:3]==pytest.approx((89/255,97/255,116/255),abs=.01)
    assert len(axis.lines)==2  # H/V and X/Y half-bin boundaries.
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

def test_correctors_chosen_before_orm_are_retained_for_orm(app):
    bpms,h,v,_,adapter=setup(nbpms=30,nh=8,nv=7)
    window=MeasureMainWindow(devices=bpms,adapter=adapter,horizontal_correctors=h,vertical_correctors=v)
    assert window.measurement_type.currentData()=="bpm_noise"
    window._select_device_subset("hcor","uniform",5); window._select_device_subset("vcor","uniform",5)
    expected_h=tuple(device.name for device in window.selected_hcorrectors)
    expected_v=tuple(device.name for device in window.selected_vcorrectors)
    window.measurement_type.setCurrentIndex(window.measurement_type.findData("orm")); app.processEvents()
    assert tuple(device.name for device in window.selected_hcorrectors)==expected_h
    assert tuple(device.name for device in window.selected_vcorrectors)==expected_v
    window.close()

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
        "intermediate_restoration_status":lambda f:f["setpoints/intermediate_restoration_status"].__setitem__(0,"restore_failed"),
        "intermediate_readback":lambda f:f["setpoints/intermediate_restored"].__setitem__(0,1e-6),
        "missing_restored_orbit":lambda f:f["raw/reference_final_m"].__setitem__((0,0,0),np.nan),
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
