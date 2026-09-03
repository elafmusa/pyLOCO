from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
from PySide6.QtWidgets import QApplication

from pyLOCO.control_system.petra import CALIBRATION_BASE,MAGNET_BASE,PETRAReadOnlyAdapter,READ_ONLY_ERROR
from pyLOCO.correct.model import CorrectItem,CorrectionReview
from pyLOCO.correct.petra_readonly import PETRACorrectReadOnlyService,MagnetMapping,apply_explicit_mapping,load_mapping,load_snapshot,save_snapshot
from pyLOCO.correct.app import build_application
from pyLOCO.correct.main_window import CorrectMainWindow


class FakePydoocs:
    def __init__(self):self.calls=[]
    def read(self,address):
        self.calls.append(("read",address))
        if address.endswith("STRENGTH.SP"):return {"data":np.asarray(2.0)}
        if address.endswith("CURRENT.RBV"):return {"data":20.0}
        if address.endswith("CURRENT.SP"):return {"data":19.5}
        raise KeyError(address)
    def write(self,*args):raise AssertionError("No write may occur")


class FakeDoocs4py:
    def __init__(self):self.calls=[]
    def get(self,address,*args):
        self.calls.append((address,args))
        if address.endswith("MIN_CURRENT_1"):return SimpleNamespace(value=-25.0)
        if address.endswith("MAX_CURRENT_1"):return SimpleNamespace(value=25.0)
        if address.endswith("STRENGTH2CURRENT"):return SimpleNamespace(value=10.0*float(args[0]))
        raise KeyError(address)


def review():
    items=[CorrectItem(0,"normal_quadrupole","Q1",12,"m^-2",1.8,1.9,.1,.2,"explicit"),CorrectItem(1,"normal_quadrupole","Q2",20,"m^-2",1.0,1.1,.1,.2,"explicit"),CorrectItem(2,"skew_quadrupole","SQ1",30,"m^-2",0.,.01,.01,-.01,"explicit")]
    return CorrectionReview(items,"results",global_scale=.5)


def adapter():
    fake=FakePydoocs(); calibration=FakeDoocs4py(); return PETRAReadOnlyAdapter(pydoocs_module=fake,doocs4py_module=calibration),fake,calibration


def test_explicit_mapping_reports_unmapped_ambiguous_and_duplicate(tmp_path):
    source=tmp_path/"mapping.json"; source.write_text(json.dumps({"mappings":[{"lattice_name":"Q1","lattice_ordinal":12,"control_name":"PQ1"},{"lattice_name":"Q2","control_name":"PQ2"}]})); loaded=load_mapping(source); state=review(); counts=apply_explicit_mapping(state,loaded); assert counts=={"mapped":2,"unmapped":1,"ambiguous":0,"duplicate":0}; assert state.items[0].control_name=="PQ1"; assert state.items[2].metadata["mapping_status"]=="unmapped"
    state=review(); counts=apply_explicit_mapping(state,(MagnetMapping("Q1","PQ1"),MagnetMapping("Q1","PQ1-B"),MagnetMapping("Q2","PQ2"))); assert counts["ambiguous"]==1; assert counts["duplicate"]==0
    state=review(); counts=apply_explicit_mapping(state,(MagnetMapping("Q1","PQ1"),MagnetMapping("Q2","PQ1"))); assert counts["duplicate"]==2; assert all(item.control_name for item in state.items[:2])


def test_real_shaped_readonly_state_calibration_targets_limits_and_addresses():
    state=review(); apply_explicit_mapping(state,(MagnetMapping("Q1","PQ1"),)); item,fake,calibration=adapter(); service=PETRACorrectReadOnlyService(item,sign_difference_names={"PQ1"}); snapshot=service.read_snapshot(state)
    q1=state.items[0]; assert q1.machine_value==2.; assert q1.current_ampere==20.; assert q1.target_value==pytest.approx(2.1); assert q1.target_current_ampere==pytest.approx(21.); assert q1.delta_i_ampere==pytest.approx(1.); assert q1.current_limit_status=="Within limits"; assert q1.current_limit_margin_ampere==pytest.approx(4.); assert q1.calibration_status=="Sign convention warning"
    addresses=[call[1] for call in fake.calls]; assert f"{MAGNET_BASE}/PQ1/STRENGTH.SP" in addresses; assert f"{MAGNET_BASE}/PQ1/CURRENT.RBV" in addresses
    calibration_addresses=[call[0] for call in calibration.calls]; assert f"{CALIBRATION_BASE}/PQ1/STRENGTH2CURRENT" in calibration_addresses; assert f"{CALIBRATION_BASE}/PQ1/MIN_CURRENT_1" in calibration_addresses; assert f"{CALIBRATION_BASE}/PQ1/MAX_CURRENT_1" in calibration_addresses
    assert snapshot.fraction_comparison[0]["max_abs_delta_i_ampere"]==pytest.approx(.2); assert snapshot.fraction_comparison[-1]["max_abs_delta_i_ampere"]==pytest.approx(2.); assert not any(call[0]=="write" for call in fake.calls)


def test_large_calibration_warning_violation_fraction_and_snapshot_roundtrip(tmp_path):
    state=review(); state.set_global_scale(1.0); apply_explicit_mapping(state,(MagnetMapping("Q1","PQ1"),)); item,_,_=adapter(); item.current_limits=lambda _name:(-25.,20.5); service=PETRACorrectReadOnlyService(item,large_difference_names={"Q1"}); snapshot=service.read_snapshot(state); assert state.items[0].calibration_status=="Large calibration discrepancy"; assert "large_calibration_difference" in state.items[0].warnings(state.thresholds); assert state.items[0].current_limit_status=="VIOLATION"; assert snapshot.fraction_comparison[-1]["current_limit_violations"]==1
    destination=save_snapshot(tmp_path/"snapshot.json",snapshot); restored=load_snapshot(destination); assert restored.magnets[0]["target_current_ampere"]==pytest.approx(22.); assert restored.adapter=="PETRAReadOnlyAdapter"
    payload=destination.read_text(); assert "STRENGTH2CURRENT" not in payload; assert json.loads(payload)["file_type"]=="pyloco.petra_readonly_snapshot"


def test_hard_write_guard_and_no_unverified_current_to_strength():
    item,_,_=adapter(); service=PETRACorrectReadOnlyService(item)
    with pytest.raises(PermissionError,match=READ_ONLY_ERROR):service.write("PQ1",1)
    with pytest.raises(PermissionError,match=READ_ONLY_ERROR):item.write("anything",1)
    with pytest.raises(NotImplementedError,match="No verified PETRA CURRENT2STRENGTH"):item.current_to_strength("PQ1",20)


def test_gui_petra_read_is_explicit_and_uses_injected_readonly_adapter(monkeypatch):
    app=QApplication.instance() or build_application(["correct-petra-test"]); item,fake,_=adapter(); monkeypatch.setattr("pyLOCO.correct.main_window.PETRAReadOnlyAdapter",lambda:item); monkeypatch.setattr("pyLOCO.correct.main_window.QMessageBox.critical",lambda *args:None); monkeypatch.setattr("pyLOCO.correct.main_window.QMessageBox.warning",lambda *args:None)
    window=CorrectMainWindow(); window._load(Path("Examples/Correct/mock_corrections.json").resolve()); assert not fake.calls; window.read_petra_state(); app.processEvents(); assert window.badge.text()=="LIVE • PETRA III DOOCS"; assert window.machine_snapshot is not None; assert window.save_snapshot_button.isEnabled(); assert not any(call[0]=="write" for call in fake.calls); window.close()
