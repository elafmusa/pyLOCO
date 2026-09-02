from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
from PySide6.QtWidgets import QApplication

from pyLOCO.control_system import AdapterCapability, PETRAReadOnlyAdapter
from pyLOCO.control_system.petra import BPM_ADDRESS_X, BPM_ADDRESS_Y, READ_ONLY_ERROR
from pyLOCO.measure.app import build_application
from pyLOCO.measure.main_window import MeasureMainWindow


class FakePydoocs:
    def __init__(self,x=(1.,2.,3.),y=(4.,5.,6.)):
        self.x=x; self.y=y; self.calls=[]
    def read(self,address):
        self.calls.append(("read",address))
        if address in {BPM_ADDRESS_X,BPM_ADDRESS_Y}:
            values=self.x if address==BPM_ADDRESS_X else self.y
            return {"data":[(index,value) for index,value in enumerate(values)]+[(0,0),(0,0)]}
        return {"data":11.5 if address.endswith("KICK.SP") else 11.4 if address.endswith("KICK.RBV") else 22.5 if address.endswith("CURRENT.SP") else 22.4}
    def write(self,*args):
        self.calls.append(("write",*args)); raise AssertionError("write must never be called")


class FakeDoocs4py:
    def __init__(self):self.calls=[]
    def get(self,address,*args):
        self.calls.append((address,args))
        if address.endswith("MIN_CURRENT_1"):return SimpleNamespace(value=-100.)
        if address.endswith("MAX_CURRENT_1"):return SimpleNamespace(value=100.)
        return SimpleNamespace(value=42.)


@pytest.fixture(scope="module")
def app():return QApplication.instance() or build_application(["petra-readonly-test"])


def adapter(fake=None,calibration=None):
    return PETRAReadOnlyAdapter(("BPM-A","BPM-B","BPM-C"),("H-A",),("V-A",),pydoocs_module=fake or FakePydoocs(),doocs4py_module=calibration or FakeDoocs4py())


def test_petra_orbit_mapping_units_order_and_no_write_capability():
    fake=FakePydoocs(); item=adapter(fake); x,y=item.read_orbit(); np.testing.assert_allclose(x,[1e-9,2e-9,3e-9]); np.testing.assert_allclose(y,[4e-9,5e-9,6e-9]); assert AdapterCapability.WRITE not in item.capabilities
    devices=item.list_devices("bpm"); samples=item.read_many(channel for device in devices for channel in (device["x_channel"],device["y_channel"])); assert samples["PETRA:BPM:BPM-B:X"].value==2e-9
    with pytest.raises(PermissionError,match="Machine writes are disabled"):item.write("anything",1)
    assert not any(call[0]=="write" for call in fake.calls)


@pytest.mark.parametrize("x,y,match",[((1,2),(1,),"lengths differ"),((1,np.nan,3),(1,2,3),"non-finite"),((1,2),(1,2),"configured BPM list")])
def test_petra_orbit_validation_never_truncates_or_reorders(x,y,match):
    with pytest.raises(ValueError,match=match):adapter(FakePydoocs(x,y)).read_orbit()


def test_corrector_diagnostics_and_calibration_use_verified_addresses_only():
    fake=FakePydoocs(); calibration=FakeDoocs4py(); item=adapter(fake,calibration); values=item.read_corrector_diagnostics("H-A"); assert set(values)=={"KICK.SP","KICK.RBV","CURRENT.SP","CURRENT.RBV"}; assert all(f"PETRA/MAGNET.ML/H-A/{field}" in [call[1] for call in fake.calls] for field in values)
    assert item.strength_to_current("H-A",.25)==42.; assert item.current_limits("H-A")==(-100.,100.)
    assert calibration.calls[0][0]=="PETRA.MAGNETS/MAGNET.ML/H-A/STRENGTH2CURRENT"
    with pytest.raises(NotImplementedError,match="No verified PETRA CURRENT2STRENGTH"):item.current_to_strength("H-A",10)


def test_measure_gui_petra_mode_is_read_only_and_orm_disabled_without_connecting(app):
    window=MeasureMainWindow(); window.adapter_combo.setCurrentIndex(window.adapter_combo.findData("petra")); app.processEvents(); assert window.status_badge.text()=="LIVE • PETRA III DOOCS"; assert isinstance(window.adapter,PETRAReadOnlyAdapter); assert AdapterCapability.WRITE not in window.adapter.capabilities
    assert "PETRA read-only" in window.subtitle.text()
    orm_item=window.measurement_type.model().item(window.measurement_type.findData("orm")); assert not orm_item.isEnabled(); assert not window.orm_unavailable_label.isHidden(); assert window.machine_info["rf_readback"].text().startswith("unavailable")
    assert not window.adapter.history; window.close()


def test_safe_connection_check_and_preview_diagnostics_are_reads_only(app,monkeypatch):
    window=MeasureMainWindow(); window.adapter_combo.setCurrentIndex(window.adapter_combo.findData("petra")); fake=FakePydoocs(tuple(range(1,len(window.devices)+1)),tuple(range(101,101+len(window.devices)))); cal=FakeDoocs4py(); window.adapter._pydoocs_module=fake; window.adapter._doocs4py_module=cal
    monkeypatch.setattr("pyLOCO.measure.main_window.QMessageBox.information",lambda *args:None); monkeypatch.setattr("pyLOCO.measure.main_window.QMessageBox.warning",lambda *args:None)
    window._test_connection(); assert window.machine_info["connection"].text().startswith("Connected"); window._refresh_corrector_preview("hcor"); assert window.corrector_selection_widgets["hcor"]["table"].item(0,4).text()=="11.5"; assert not any(call[0]=="write" for call in fake.calls); window.close()


@pytest.mark.skipif(os.environ.get("PYLOCO_PETRA_HARDWARE_TESTS")!="1",reason="explicit PETRA hardware opt-in required")
def test_opt_in_real_petra_readonly_orbit():
    from pathlib import Path
    names=tuple(line.strip() for line in Path("Examples/PETRAIII/data/BPM_names.txt").read_text().splitlines() if line.strip()); item=PETRAReadOnlyAdapter(names); x,y=item.read_orbit(); assert x.size==y.size==len(names); assert AdapterCapability.WRITE not in item.capabilities
