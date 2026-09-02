from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from pyLOCO.control_system import AbstractInterfaceAdapter, AdapterCapability, MockAdapter
from pyLOCO.control_system.backends import BackendSession, InterfaceRegistry
from pyLOCO.data_schema import MeasurementSession, SessionFile, load_session, save_session, validate_measurement_file, write_bpm_noise
from pyLOCO.gui.backend import _load_measurements
from pyLOCO.measure.acquisition import DispersionResult, DispersionStateAcquirer
from pyLOCO.measure.app import build_application
from pyLOCO.measure.main_window import MeasureMainWindow, build_mock_adapter, default_mock_devices
from pyLOCO.measure.dispersion import physical_dispersion, relative_momentum_deviation
from pyLOCO.measure.project import load_measure_project, save_measure_project


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or build_application(["pyloco-measure-dispersion-test"])


def _mock_result(readings=8, step=1500.0):
    devices=default_mock_devices(4); adapter=build_mock_adapter(devices,readings=readings); nominal=500_000_000.0; states=[]
    for label,frequency in (("reference",nominal),("negative",nominal-step),("positive",nominal+step)):
        adapter.set_simulated_rf_state(frequency)
        states.append(DispersionStateAcquirer(adapter,devices).acquire(label,frequency,readings,0,sleeper=lambda _:None))
    result=DispersionResult(devices,tuple(states),"bipolar",nominal,step,"confirmed_by_operator",1.0)
    return adapter,result


def _lattice_metadata(alpha=1.0e-4, inverse_gamma_squared=1.0e-8):
    eta=alpha-inverse_gamma_squared
    return {"momentum_compaction_factor":alpha,
            "relativistic_correction_inverse_gamma_squared":inverse_gamma_squared,
            "slip_factor":-eta,
            "eta_alpha_minus_inverse_gamma_squared":eta}


def test_mock_recovers_known_rf_orbit_response_without_writes():
    adapter,result=_mock_result()
    expected_x=np.arange(1,5)*2.0e-9
    expected_y=-np.arange(1,5)*0.7e-9
    np.testing.assert_allclose(result.response_x_m_per_hz,expected_x,rtol=0,atol=1e-18)
    np.testing.assert_allclose(result.response_y_m_per_hz,expected_y,rtol=0,atol=1e-18)
    np.testing.assert_allclose(result.measured_eta_x,-expected_x*3000.0,rtol=0,atol=1e-18)
    assert result.canonical_rf_step_hz == -3000.0
    assert AdapterCapability.RF_WRITE not in adapter.capabilities
    assert all(operation=="read" for operation,_,_ in adapter.history)


def test_physical_dispersion_uses_actual_full_bipolar_frequency_span():
    f0=500_000_000.0; eta=1.0e-4
    deltas=relative_momentum_deviation([f0+100.0,f0-100.0],f0,eta)
    expected=np.array([0.12,-0.03])
    x_minus=np.array([1e-3,2e-3]); x_plus=x_minus+expected*(deltas[0]-deltas[1])
    measured,span=physical_dispersion(x_minus,x_plus,deltas[1],deltas[0])
    np.testing.assert_allclose(measured,expected,rtol=0,atol=1e-14)
    assert span==pytest.approx(0.004)


def test_dispersion_save_schema_session_and_importer_compatibility(app,tmp_path):
    _,result=_mock_result(); result=replace(result,states=tuple(replace(state,actual_rf_hz=state.requested_rf_hz) for state in result.states)); save_adapter=build_mock_adapter(result.devices,readings=8); save_adapter.backend_metadata=_lattice_metadata(); window=MeasureMainWindow(devices=result.devices,adapter=save_adapter); window._restored_rf_readback=result.nominal_rf_hz
    window.measurement_type.setCurrentIndex(window.measurement_type.findData("dispersion")); window.nominal_rf.setText("500000000"); window.readings.setValue(8); window.output_directory.setText(str(tmp_path)); window.measurement_name.setText("dispersion"); window._save_result(result)
    assert validate_measurement_file(window.saved_measurement_path)["kind"]=="dispersion"
    with h5py.File(window.saved_measurement_path,"r") as handle:
        assert handle["raw/orbits_x_m"].shape==(3,8,4)
        assert handle["raw/mean_orbits_x_m"].shape==(3,4)
        assert handle["raw/std_orbits_y_m"].shape==(3,4)
        assert handle["raw/sample_timestamps_s"].shape==(3,8)
        np.testing.assert_allclose(handle["raw/rf_readback_hz"][:],[state.actual_rf_hz for state in result.states])
        assert handle.attrs["restoration_status"]=="confirmed_by_operator"
        assert handle.attrs["measured_eta_unit"]=="m"
        assert handle.attrs["rf_step_hz"]==-3000.0
        assert handle.attrs["rf_difference_sign_convention"]=="negative_minus_positive"
        assert "mean_orbit_negative - mean_orbit_positive" in handle.attrs["measured_eta_definition"]
        assert handle.attrs["rf_bipolar_separation_hz"]==3000.0
        assert handle.attrs["rf_signed_step_hz"]==-3000.0
        assert handle.attrs["delta_span_definition"]=="delta_negative - delta_positive"
        labels=[x.decode() if isinstance(x,bytes) else x for x in handle["raw/state_labels"][:]]
        means=handle["raw/mean_orbits_x_m"][:]
        np.testing.assert_allclose(handle["measured_eta_x"][:],means[labels.index("negative")]-means[labels.index("positive")])
        assert bool(handle.attrs["bidirectional"])
        assert handle.attrs["momentum_compaction_factor"]==pytest.approx(1e-4)
        assert handle.attrs["relativistic_correction_inverse_gamma_squared"]==pytest.approx(1e-8)
        assert handle.attrs["at_slip_factor"]==pytest.approx(-(1e-4-1e-8))
        assert handle.attrs["slip_factor_eta"]==pytest.approx(1e-4-1e-8)
        assert handle.attrs["rf_restoration_difference_hz"]==0
        assert handle["diagnostics/physical_dispersion_x_m"].shape==(4,)
        assert handle["raw/mean_orbits_x_m"].shape==(3,4)
        assert [x.decode() if isinstance(x,bytes) else x for x in handle["raw/state_labels"][:]]==["reference_before","negative","positive"]
        assert set(handle["raw/states"])=={"reference_before","negative","positive"}
        np.testing.assert_allclose(handle["derived/rf_normalized_response_x_m_per_hz"],result.response_x_m_per_hz)
    orm=tmp_path/"orm.h5"
    with h5py.File(orm,"w") as handle: handle["response_matrix"]=np.zeros((8,2))
    loaded=_load_measurements({"orm":str(orm),"dispersion":str(window.saved_measurement_path)})
    np.testing.assert_allclose(loaded["eta_x"],result.measured_eta_x)
    session=load_session(window.saved_session_path)
    assert session.files[0].role=="dispersion"
    window.close()


def test_dispersion_session_update_preserves_bpm_noise_entry(app,tmp_path):
    _,result=_mock_result(); existing=tmp_path/"noise.h5"; write_bpm_noise(existing,noise_x_m=np.ones(4)*1e-8,noise_y_m=np.ones(4)*2e-8,bpm_names=[d.name for d in result.devices],raw_orbits_x_m=np.zeros((2,4)),raw_orbits_y_m=np.zeros((2,4)))
    manifest=tmp_path/"measurement-session.pyloco-session.json"
    save_session(manifest,MeasurementSession("session",(SessionFile("bpm_noise",existing.name),)),validate_files=False)
    window=MeasureMainWindow(devices=result.devices,adapter=build_mock_adapter(result.devices,readings=8)); window.measurement_type.setCurrentIndex(1); window.nominal_rf.setText("500000000"); window.readings.setValue(8); window.output_directory.setText(str(tmp_path)); window._save_result(result)
    assert {entry.role for entry in load_session(manifest,validate_files=False).files}=={"bpm_noise","dispersion"}
    window.close()


def test_dispersion_gui_is_manual_read_only_and_context_sensitive(app):
    window=MeasureMainWindow(devices=default_mock_devices(3)); assert window.dispersion_config_group.isHidden()
    assert window.stats_group.title()=="Statistics"
    assert window.rf_diagnostics.isHidden() and window.restoration_label.isHidden()
    assert window.restoration_label.text()==""
    window.measurement_type.setCurrentIndex(window.measurement_type.findData("dispersion")); assert not window.dispersion_config_group.isHidden()
    assert window.stats_group.title()=="Statistics and RF diagnostics"
    assert window.stats_group.isHidden() and not window.dispersion_summary.isHidden()
    assert not window.rf_control_mode.model().item(1).isEnabled(); assert "never write RF" in window.rf_safety.text()
    assert not window.start_button.isEnabled(); window.nominal_rf.setText("500000000"); window.refresh_plan(); assert window.start_button.isEnabled()
    assert window.plan_values["measurement"].text()=="Dispersion"; assert window.plan_values["rf_states"].text()=="reference, positive, negative"
    assert "STEP 1 OF 4" in window.step_instruction.text(); assert "Actual RF: not available" in window.step_instruction.text()
    window.measurement_type.setCurrentIndex(window.measurement_type.findData("bpm_noise")); assert window.dispersion_config_group.isHidden(); assert window.start_button.text()=="Start BPM-noise measurement"
    assert all(widget.isHidden() for widget in window.dispersion_plan_widgets)
    assert window.rf_diagnostics.isHidden() and window.restoration_label.isHidden()
    window.close()


def test_connected_pysc_dispersion_presents_automatic_total_200_hz_scan(app,monkeypatch):
    class Interface:
        rf=352_372_169.3993786
        def get_orbit(self):return np.zeros(2),np.zeros(2)
        def get(self,name):return 0.0
        def set(self,name,value):pass
        def get_rf_main_frequency(self):return self.rf
        def set_rf_main_frequency(self,value):self.rf=float(value)
    adapter=AbstractInterfaceAdapter(Interface(),("EBS-B1","EBS-B2"),("EBS-H1",),("EBS-V1",))
    session=BackendSession(InterfaceRegistry().descriptor("pysc"),adapter)
    monkeypatch.setattr(InterfaceRegistry,"create",lambda self,key:session)
    monkeypatch.setattr("pyLOCO.measure.main_window.QMessageBox.information",lambda *args:None)
    window=MeasureMainWindow(); window.adapter_combo.setCurrentIndex(window.adapter_combo.findData("pysc")); window._test_connection()
    window.measurement_type.setCurrentIndex(window.measurement_type.findData("dispersion")); window.refresh_plan()
    nominal=adapter.get_rf_frequency()
    assert window.rf_control_mode.currentData()=="automatic" and not window.rf_control_mode.isEnabled()
    assert "pySC Server RF readback" in window.nominal_rf_source.text()
    assert "READ ONLY" not in window.rf_safety.text() and "restored" in window.rf_safety.text()
    assert window.rf_step.text()=="200 Hz"
    assert window.plan_values["rf_step_±δf"].text()=="100 Hz per side"
    assert window.plan_values["bipolar_separation"].text()=="200 Hz total"
    assert window.plan_values["negative_rf"].text()==f"{nominal-100:.3f} Hz"
    assert window.plan_values["positive_rf"].text()==f"{nominal+100:.3f} Hz"
    assert "Automatic backend RF with restoration protection"==window.plan_values["rf_mode"].text()
    window.close()


def test_dispersion_results_expose_physical_raw_rf_state_and_shift_views(app):
    _,result=_mock_result(step=100.0)
    result=replace(result,states=tuple(replace(state,actual_rf_hz=state.requested_rf_hz) for state in result.states))
    adapter=build_mock_adapter(result.devices,readings=8)
    adapter.backend_metadata=_lattice_metadata()
    window=MeasureMainWindow(devices=result.devices,adapter=adapter)
    window.measurement_type.setCurrentIndex(window.measurement_type.findData("dispersion"))
    window._restored_rf_readback=result.nominal_rf_hz
    window.result=result; window._show_dispersion_result(result)
    assert [window.results_tabs.tabText(i) for i in range(window.results_tabs.count())]==[
        "Physical Dx [mm]","Physical Dy [mm]","RF orbit difference Δx_RF [mm]","RF orbit difference Δy_RF [mm]",
        "RF-state horizontal orbits","RF-state vertical orbits","RF-induced horizontal shifts","RF-induced vertical shifts",
    ]
    window.dispersion_display.setCurrentIndex(window.dispersion_display.findData("physical"))
    assert window.results_tabs.currentWidget() is window.x_plot
    assert "mm" in window.summary_x.text() and "Mean" in window.summary_x.text()
    window.dispersion_display.setCurrentIndex(window.dispersion_display.findData("raw"))
    assert window.results_tabs.currentWidget() is window.raw_x_plot
    assert "mm" in window.summary_x.text() and "RF orbit difference" in window.summary_x.text()
    assert "mean[x(f−)] − mean[x(f+)]" in window.rf_response_formula.text()
    assert "Stored in metres; displayed in mm" in window.rf_response_note.text()
    assert "Mean" in window.rf_response_stats.text() and "RMS" in window.physical_stats.text()
    assert not window.calculation_details_group.isChecked() and window.calculation_details_body.isHidden()
    details=window.calculation_details_body.text()
    assert "Momentum compaction αc:" in details and "Relativistic correction 1/γ²:" in details
    assert "AT ring.get_slip_factor()" in details and "Slip factor η = αc−1/γ²:" in details
    assert "Nominal RF f₀:" in details and "Total RF separation f+ − f−:" in details
    assert "Δδ = δ− − δ+" in details
    assert "η = αc − 1/γ²" in window.physical_formula.text()
    assert window.restoration_status.text()=="RF restoration: ✓ RESTORED"
    assert "diagnostic-only" in window.restoration_diagnostic.text()
    assert window.results_tabs.tabText(window.results_tabs.currentIndex())=="RF orbit difference Δx_RF [mm]"
    window.close()


def test_restored_reference_is_saved_separately_and_excluded_from_dispersion(app,tmp_path):
    _,base=_mock_result(step=100.0)
    states=tuple(replace(state,actual_rf_hz=state.requested_rf_hz) for state in base.states)
    before=next(state for state in states if state.label=="reference")
    after=replace(before,label="reference_after",orbits_x_m=before.orbits_x_m+2e-6,orbits_y_m=before.orbits_y_m-3e-6)
    result=replace(base,states=states+(after,))
    expected_x=result.state("negative").mean_x_m-result.state("positive").mean_x_m
    adapter=build_mock_adapter(result.devices,readings=8); adapter.backend_metadata=_lattice_metadata()
    window=MeasureMainWindow(devices=result.devices,adapter=adapter); window.measurement_type.setCurrentIndex(window.measurement_type.findData("dispersion")); window.nominal_rf.setText(str(result.nominal_rf_hz)); window.output_directory.setText(str(tmp_path)); window.measurement_name.setText("restored-reference"); window._restored_rf_readback=result.nominal_rf_hz; window._save_result(result)
    np.testing.assert_allclose(result.measured_eta_x,expected_x)
    with h5py.File(window.saved_measurement_path,"r") as handle:
        assert set(handle["raw/states"])=={"reference_before","positive","negative","reference_after"}
        assert not bool(handle.attrs["rf_restoration_is_measurement_state"])
        assert bool(handle.attrs["verify_restored_orbit"])
        np.testing.assert_allclose(handle["diagnostics/restored_orbit_difference_x_m"][:],2e-6)
    window.close()


def test_validate_for_pyloco_rejects_positive_minus_negative_dispersion(app,tmp_path):
    _,result=_mock_result(step=100.0)
    result=replace(result,states=tuple(replace(state,actual_rf_hz=state.requested_rf_hz) for state in result.states))
    adapter=build_mock_adapter(result.devices,readings=8); adapter.backend_metadata=_lattice_metadata()
    window=MeasureMainWindow(devices=result.devices,adapter=adapter); window.measurement_type.setCurrentIndex(window.measurement_type.findData("dispersion")); window.nominal_rf.setText(str(result.nominal_rf_hz)); window.output_directory.setText(str(tmp_path)); window.measurement_name.setText("sign-check"); window._restored_rf_readback=result.nominal_rf_hz; window._save_result(result)
    with h5py.File(window.saved_measurement_path,"r+") as handle:
        handle["measured_eta_x"][:] *= -1
    with pytest.raises(ValueError,match="negative-minus-positive"):
        validate_measurement_file(window.saved_measurement_path)
    window.close()


def test_automatic_dispersion_live_preview_names_each_rf_state(app):
    devices=default_mock_devices(2); window=MeasureMainWindow(devices=devices)
    window.measurement_type.setCurrentIndex(window.measurement_type.findData("dispersion"))
    window.rf_control_mode.model().item(window.rf_control_mode.findData("automatic")).setEnabled(True)
    window.rf_control_mode.setCurrentIndex(window.rf_control_mode.findData("automatic")); window.nominal_rf.setText("500000000")
    window.rf_step.setValue(200.0)
    for index,expected in ((0,"reference RF"),(1,"positive RF (+100 Hz)"),(2,"negative RF (−100 Hz)")):
        window._on_automatic_dispersion_progress(index,1,2,0.1,np.zeros(2),np.zeros(2))
        assert expected in window.live_plot.figure.axes[0].get_title()
    assert "3/3 — Negative RF (offset −100 Hz)" in window.reading_status.text()
    window._on_automatic_dispersion_status("restoring",{"offset_hz":0.0,"rf_hz":500_000_000.0})
    assert window.reading_status.text()=="Restoring — RF offset 0 Hz"
    window._on_automatic_dispersion_status("verifying_orbit",{"offset_hz":0.0,"rf_hz":500_000_000.0})
    assert window.reading_status.text()=="Verifying — restored reference orbit"
    assert "RESTORE (not a measurement state)" in window.step_instruction.text()
    window.close()


def test_measurement_splitter_can_shrink_either_pane_and_survives_type_switch(app):
    window=MeasureMainWindow(devices=default_mock_devices(2)); window.resize(1200,800); window.show(); app.processEvents()
    window.measurement_splitter.setSizes([190,900]); app.processEvents(); left_small=window.measurement_splitter.sizes()
    assert left_small[0] <= 220
    window.measurement_splitter.setSizes([900,190]); app.processEvents(); right_small=window.measurement_splitter.sizes()
    assert right_small[1] <= 220
    window.measurement_type.setCurrentIndex(window.measurement_type.findData("dispersion")); app.processEvents()
    assert window.measurement_splitter.sizes()==right_small
    window.close()


def test_guided_manual_sequence_completes_only_after_restore_confirmation(app,tmp_path):
    devices=default_mock_devices(2); window=MeasureMainWindow(devices=devices,adapter=build_mock_adapter(devices,readings=3))
    window.measurement_type.setCurrentIndex(1); window.nominal_rf.setText("500000000"); window.readings.setValue(3); window.delay.setValue(0); window.output_directory.setText(str(tmp_path)); window.measurement_name.setText("guided")
    for expected_step in (1,2,3):
        assert f"STEP {expected_step} OF 4" in window.step_instruction.text(); window.start_acquisition()
        deadline=time.monotonic()+3
        while window.thread is not None and time.monotonic()<deadline:
            app.processEvents()
        assert window.thread is None
    assert window.result is None and "STEP 4 OF 4" in window.step_instruction.text()
    window.start_acquisition()
    assert isinstance(window.result,DispersionResult)
    assert window.result.restoration_status=="confirmed_by_operator"
    assert window.saved_measurement_path.exists()
    window.close()


def test_dispersion_project_round_trip_and_portable_example(tmp_path):
    source=Path("Examples/Measure/mock_dispersion.pyloco-measure.json"); project=load_measure_project(source)
    assert project.measurement_type=="dispersion" and project.nominal_rf_hz==500_000_000.0
    text=source.read_text(); assert "/Users/" not in text and "/private/tmp/" not in text
    destination=tmp_path/"clone"/"mock_dispersion.pyloco-measure.json"; save_measure_project(destination,project)
    assert load_measure_project(destination)==project
    assert str(tmp_path) not in destination.read_text()


def test_bpm_order_mismatch_and_nonfinite_data_are_not_silently_accepted():
    adapter,result=_mock_result(); reordered=(result.devices[1],result.devices[0],*result.devices[2:])
    bad_state=replace(result.states[0],bpm_names=tuple(device.name for device in reordered))
    with pytest.raises(ValueError,match="BPM order changed"):
        replace(result,states=(bad_state,*result.states[1:]))
    channels={channel:0.0 for device in result.devices for channel in (device.x_channel,device.y_channel)}; channels[result.devices[0].x_channel]=float("nan"); adapter=MockAdapter(channels)
    with pytest.raises(ValueError,match="Non-finite BPM data"):
        DispersionStateAcquirer(adapter,result.devices).acquire("reference",500_000_000,2,0,sleeper=lambda _:None)
