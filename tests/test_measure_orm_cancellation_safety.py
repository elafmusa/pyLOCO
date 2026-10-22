from threading import Event

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from pyLOCO.control_system import MockAdapter
from pyLOCO.measure.acquisition import ORMAcquirer, ORMAcquisitionError, ORMInterrupted
from pyLOCO.measure.main_window import ORMAcquisitionWorker, MeasureMainWindow, build_mock_adapter, default_mock_correctors, default_mock_devices


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication(["orm-cancellation-safety"])


def _setup(ncor=2, readings=3):
    bpms=default_mock_devices(2); h,_=default_mock_correctors(ncor,0)
    matrix=np.ones((4,ncor))*1e-3
    adapter=build_mock_adapter(bpms,readings=readings,horizontal_correctors=h,vertical_correctors=(),response_matrix=matrix,enable_orm_writes=True)
    return bpms,h,adapter


@pytest.mark.parametrize("phase",["positive","negative","orbit","settling","between"])
def test_cancel_at_each_orm_phase_restores_and_reports_every_touched_corrector(phase):
    bpms,h,adapter=_setup(); cancel=Event(); sleeps=[]
    def progress(event):
        state=event.get("state","")
        if phase=="positive" and state=="+kick":cancel.set()
        elif phase=="negative" and state=="−kick":cancel.set()
        elif phase=="orbit" and event.get("event")=="orbit" and state=="+kick" and event.get("reading")==1:cancel.set()
        elif phase=="between" and event.get("event")=="orbit" and state=="final restored reference" and event.get("reading")==event.get("readings"):cancel.set()
    def sleeper(seconds):
        sleeps.append(seconds)
        if phase=="settling" and seconds>0 and not cancel.is_set():cancel.set()
    with pytest.raises(ORMInterrupted) as caught:
        ORMAcquirer(adapter,bpms,h,()).acquire([100e-6]*len(h),[],readings=3,delay_seconds=0,settling_delay_seconds=.01,cancel_event=cancel,progress=progress,sleeper=sleeper)
    error=caught.value
    assert error.restoration_status=="restored"
    assert error.restoration_diagnostics["status"]=="restored"
    assert error.restoration_diagnostics["correctors"]
    assert all(row["status"]=="restored" and row["attempts"]>=1 for row in error.restoration_diagnostics["correctors"])
    assert all(float(adapter.read(device.readback_channel).value)==pytest.approx(0,abs=1e-15) for device in h)


def test_all_original_setpoints_and_readbacks_are_snapshotted_before_first_write():
    bpms,h,base=_setup(2); events=[]
    class OrderedMock(MockAdapter):
        def read(self,channel):events.append(("read",channel)); return super().read(channel)
        def write(self,channel,value):events.append(("write",channel)); return super().write(channel,value)
    adapter=OrderedMock(base._channels,allow_simulated_writes=True,sequences=base._sequences,device_catalog=base._device_catalog,setpoint_dependent_channels=base._setpoint_dependent_channels)
    ORMAcquirer(adapter,bpms,h,()).acquire([100e-6,100e-6],[],readings=2,delay_seconds=0)
    first_write=next(index for index,event in enumerate(events) if event[0]=="write")
    snapshot_channels={device.setpoint_channel for device in h}|{device.readback_channel for device in h}
    assert snapshot_channels.issubset({channel for kind,channel in events[:first_write] if kind=="read"})


@pytest.mark.parametrize("direction,expected_transitions",[("bipolar",4),("positive",3),("negative",3)])
def test_settling_delay_runs_after_each_corrector_transition_independent_of_bpm_delay(direction,expected_transitions):
    bpms,h,adapter=_setup(1,readings=2); sleeps=[]
    ORMAcquirer(adapter,bpms,h,()).acquire(
        [100e-6],[],direction=direction,readings=2,delay_seconds=.25,
        settling_delay_seconds=1.5,sleeper=sleeps.append,
    )
    assert sleeps.count(1.5)==expected_transitions
    # Each acquired two-sample orbit contributes one independent BPM-reading delay.
    assert sleeps.count(.25)==(5 if direction=="bipolar" else 4)


@pytest.mark.parametrize("failure",["write","incorrect_readback","unavailable_readback"])
def test_unresolved_restoration_has_complete_diagnostics(failure):
    bpms,h,base=_setup(1); writes=0; readbacks=0
    class BrokenRestore(MockAdapter):
        def write(self,channel,value):
            nonlocal writes; writes+=1
            if failure=="write" and writes>=2 and value==0:raise RuntimeError("restore write unavailable")
            return super().write(channel,value)
        def read(self,channel):
            nonlocal readbacks
            value=super().read(channel)
            if channel==h[0].readback_channel:
                readbacks+=1
                if writes>=2 and failure=="unavailable_readback":raise RuntimeError("restore readback unavailable")
                if writes>=2 and failure=="incorrect_readback":value=type(value)(value.value+1e-6,value.timestamp)
            return value
    adapter=BrokenRestore(base._channels,allow_simulated_writes=True,sequences=base._sequences,device_catalog=base._device_catalog,setpoint_dependent_channels=base._setpoint_dependent_channels)
    with pytest.raises(ORMAcquisitionError) as caught:
        ORMAcquirer(adapter,bpms,h,()).acquire([100e-6],[],readings=2,delay_seconds=0)
    assert caught.value.restoration_status in {"restore_failed","verification_failed"}
    row=caught.value.restoration_diagnostics["correctors"][0]
    assert row["name"]==h[0].name and row["setpoint_channel"]==h[0].setpoint_channel and row["readback_channel"]==h[0].readback_channel
    assert row["original_setpoint_rad"]==0 and row["original_readback_rad"]==0 and row["tolerance_rad"]==1e-12
    assert row["status"] in {"restore_failed","verification_failed"}
    assert row["set_command_issued"] is (failure!="write")


def test_acquisition_exception_restores_and_never_returns_partial_result():
    bpms,h,base=_setup(1)
    class OrbitFailure(MockAdapter):
        def read(self,channel):
            if channel.startswith("MOCK/BPM/") and super().read(h[0].readback_channel).value!=0:raise RuntimeError("BPM connection lost")
            return super().read(channel)
    adapter=OrbitFailure(base._channels,allow_simulated_writes=True,sequences=base._sequences,device_catalog=base._device_catalog,setpoint_dependent_channels=base._setpoint_dependent_channels)
    with pytest.raises(ORMAcquisitionError) as caught:ORMAcquirer(adapter,bpms,h,()).acquire([100e-6],[],readings=2,delay_seconds=0)
    assert caught.value.restoration_status=="restored" and adapter.read(h[0].readback_channel).value==0


def test_gui_blocks_orm_writes_after_unresolved_restore_and_clears_only_after_readback_verification(app,monkeypatch):
    bpms,h,adapter=_setup(1); window=MeasureMainWindow(devices=bpms,adapter=adapter,horizontal_correctors=h,vertical_correctors=())
    window.measurement_type.setCurrentIndex(window.measurement_type.findData("orm")); warnings=[]
    error=ORMInterrupted("cancelled",restoration_status="verification_failed",restoration_diagnostics={"status":"verification_failed","correctors":[{"name":h[0].name,"plane":"H","setpoint_channel":h[0].setpoint_channel,"readback_channel":h[0].readback_channel,"original_setpoint_rad":0.0,"original_readback_rad":0.0,"attempted_restoration_rad":0.0,"final_readback_rad":1e-6,"tolerance_rad":1e-12,"attempts":1,"status":"verification_failed","error":""}]})
    monkeypatch.setattr("pyLOCO.measure.main_window.QMessageBox.critical",lambda *args:warnings.append(args[-1]))
    window._on_cancelled(error)
    assert window._orm_restoration_unresolved and "Corrector restoration could not be verified" in window.reading_status.text()
    window._start_orm(); assert len(warnings)==2 and window.worker is None
    assert window._verify_unresolved_orm_restoration()
    assert not window._orm_restoration_unresolved
    window.close()


def test_gui_cancel_success_message_matches_rf_safety_style(app):
    bpms,h,adapter=_setup(1); window=MeasureMainWindow(devices=bpms,adapter=adapter,horizontal_correctors=h,vertical_correctors=())
    window.measurement_type.setCurrentIndex(window.measurement_type.findData("orm"))
    window._on_cancelled(ORMInterrupted("cancelled",restoration_status="restored",restoration_diagnostics={"status":"restored","verification_level":"independent","correctors":[]}))
    assert window.reading_status.text()=="ORM measurement cancelled. All correctors restored and independently verified."
    window.close()


def test_restoration_diagnostics_use_expanded_terms_and_preserve_values_and_units():
    event=ORMInterrupted("cancelled",restoration_status="verification_failed",restoration_diagnostics={"status":"verification_failed","correctors":[{"name":"H-1","plane":"H","set_command_issued":True,"verification_source":"public readback()","independent_readback_verified":False,"original_setpoint_rad":1e-6,"original_readback_rad":2e-6,"attempted_restoration_rad":1e-6,"final_readback_rad":3e-6,"tolerance_rad":1e-12,"attempts":2,"status":"verification_failed","error":"mismatch"}]})
    text=MeasureMainWindow._restoration_diagnostic_text(event)
    assert "Original setpoint=1e-06 rad" in text
    assert "Original readback=2e-06 rad" in text
    assert "Restoration target=1e-06 rad" in text
    assert "Final readback=3e-06 rad" in text
    assert "Tolerance=1e-12 rad" in text
    assert "Attempts=2" in text and "Status=verification_failed" in text and "Error=mismatch" in text
    assert "Set command issued=True" in text and "Evidence=public readback()" in text and "Independent readback verified=False" in text
    assert "original SP" not in text and "RBV" not in text


def test_adapter_value_message_never_claims_independent_readback(app):
    bpms,h,adapter=_setup(1); window=MeasureMainWindow(devices=bpms,adapter=adapter,horizontal_correctors=h,vertical_correctors=())
    window.measurement_type.setCurrentIndex(window.measurement_type.findData("orm"))
    diagnostics={"status":"restored","verification_level":"adapter_value","correctors":[]}
    window._on_cancelled(ORMInterrupted("cancelled",restoration_status="restored",restoration_diagnostics=diagnostics))
    assert "adapter strength.get() value matched" in window.reading_status.text()
    assert "independent readback unavailable" in window.reading_status.text()
    assert "independently verified" not in window.reading_status.text()
    window.close()


def test_orm_failure_is_never_mislabelled_as_rf_when_gui_selection_changes(app,monkeypatch):
    bpms,h,adapter=_setup(1); window=MeasureMainWindow(devices=bpms,adapter=adapter,horizontal_correctors=h,vertical_correctors=())
    window._active_acquisition_kind="orm"
    window.measurement_type.setCurrentIndex(window.measurement_type.findData("dispersion"))
    dialogs=[]; monkeypatch.setattr(QMessageBox,"critical",lambda *args:dialogs.append((args[1],args[2])))
    diagnostics={"status":"verification_failed","verification_level":"independent","correctors":[{"name":h[0].name,"plane":"H","set_command_issued":True,"verification_source":"independent test readback","independent_readback_verified":False,"original_setpoint_rad":0.0,"original_readback_rad":0.0,"attempted_restoration_rad":0.0,"final_readback_rad":1e-6,"tolerance_rad":1e-12,"attempts":1,"status":"verification_failed","error":"mismatch"}]}
    window._on_failed(ORMAcquisitionError("failure",restoration_status="verification_failed",restoration_diagnostics=diagnostics))
    assert "Corrector restoration could not be verified" in window.reading_status.text()
    assert "RF restoration" not in window.reading_status.text()
    assert window._orm_restoration_unresolved and not window._rf_restoration_unresolved
    assert dialogs and "Corrector restoration diagnostics" in dialogs[-1][1]
    window.close()


def test_rf_failure_remains_rf_when_gui_selection_changes(app,monkeypatch):
    bpms,h,adapter=_setup(1); window=MeasureMainWindow(devices=bpms,adapter=adapter,horizontal_correctors=h,vertical_correctors=())
    window._active_acquisition_kind="dispersion"
    window.measurement_type.setCurrentIndex(window.measurement_type.findData("orm"))
    dialogs=[]; monkeypatch.setattr(QMessageBox,"critical",lambda *args:dialogs.append((args[1],args[2])))
    error=RuntimeError("RF mismatch"); error.restoration_status="verification_failed"; error.restoration_diagnostics={"status":"verification_failed","original_rf_hz":500e6,"attempted_restoration_hz":500e6,"final_readback_hz":500e6+2,"absolute_difference_hz":2,"tolerance_hz":1}
    window._on_failed(error)
    assert "RF restoration could not be verified" in window.reading_status.text()
    assert "Corrector restoration" not in window.reading_status.text()
    assert window._rf_restoration_unresolved and not window._orm_restoration_unresolved
    window.close()


def test_acquisition_status_uses_active_operation_and_returns_to_connected_state(app):
    bpms,h,adapter=_setup(1); window=MeasureMainWindow(devices=bpms,adapter=adapter,horizontal_correctors=h,vertical_correctors=())
    window.status_badge.setText("pyAML • WRITES ENABLED"); window._active_acquisition_kind="orm"
    window._set_acquisition_running(True)
    assert "ACQUIRING ORM" in window.statusBar().currentMessage()
    assert "corrector writes authorized" in window.start_block_reason.text()
    assert not window.measurement_type.isEnabled()
    window._set_acquisition_running(False)
    assert window.statusBar().currentMessage()=="pyAML • WRITES ENABLED"
    assert window.measurement_type.isEnabled()
    window.close()


def test_worker_preserves_structured_restoration_diagnostics_across_qt_signal(app):
    bpms,h,adapter=_setup(1); cancel=Event(); cancel.set(); received=[]
    worker=ORMAcquisitionWorker(ORMAcquirer(adapter,bpms,h,()),[100e-6],[],{"direction":"bipolar","scaled":False,"readings":2,"delay_seconds":0,"settling_delay_seconds":0},cancel)
    worker.cancelled.connect(received.append); worker.run(); app.processEvents()
    assert len(received)==1 and isinstance(received[0],ORMInterrupted)
    assert received[0].restoration_status=="restored" and "correctors" in received[0].restoration_diagnostics
