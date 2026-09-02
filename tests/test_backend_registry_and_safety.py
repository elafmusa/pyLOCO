from pathlib import Path
from threading import Event

import numpy as np
import pytest

from pyLOCO.control_system import AbstractInterfaceAdapter, AdapterCapability, InterfaceRegistry, PETRAReadOnlyAdapter
from pyLOCO.correct.application import CorrectionApplicationService
from pyLOCO.measure.acquisition import BpmDevice, AcquisitionCancelled
from pyLOCO.measure.automatic import AutomaticDispersionAcquirer


class FakeInterface:
    def __init__(self):
        self.values={"H1":0.0,"Q1":1.0}; self.rf=500_000_000.; self.fail_orbit=False
    def get_orbit(self):
        if self.fail_orbit: raise RuntimeError("injected orbit failure")
        offset=self.rf-500_000_000.; return np.array([offset*2e-9]),np.array([-offset*1e-9])
    def get(self,name): return self.values[name]
    def set(self,name,value): self.values[name]=value
    def get_rf_main_frequency(self): return self.rf
    def set_rf_main_frequency(self,value): self.rf=value


def adapter(interface=None):
    return AbstractInterfaceAdapter(interface or FakeInterface(),("B1",),("H1",),())


def test_registry_has_exact_three_persistent_badges(tmp_path):
    descriptors=InterfaceRegistry(repository_root=Path.cwd()).descriptors()
    assert [(d.key,d.badge) for d in descriptors]==[("mock","MOCK • READ ONLY"),("pysc","DEMO • pySC SERVER"),("petra","LIVE • PETRA III DOOCS")]


def test_pysc_registry_uses_its_own_generated_catalog(tmp_path):
    demo=tmp_path/"Examples"/"Demo"; demo.mkdir(parents=True)
    (demo/"pysc_demo_catalog.json").write_text(
        '{"bpms":["SC-B1"],"horizontal_correctors":["SC-H1"],'
        '"vertical_correctors":["SC-V1"],"host":"127.0.0.1","port":13131,"rf_system":"main"}'
    )
    session=InterfaceRegistry(repository_root=tmp_path,interface_loaders={"pysc":FakeInterface}).create("pysc")
    assert [d["name"] for d in session.adapter.list_devices("bpm")]==["SC-B1"]
    assert [d["name"] for d in session.adapter.list_devices("hcor")]==["SC-H1"]
    assert [d["name"] for d in session.adapter.list_devices("vcor")]==["SC-V1"]


def test_pysc_registry_refuses_missing_catalog_instead_of_falling_back(tmp_path):
    with pytest.raises(RuntimeError,match="catalog is missing"):
        InterfaceRegistry(repository_root=tmp_path,interface_loaders={"pysc":FakeInterface}).create("pysc")


def test_petra_registry_uses_only_its_own_catalog_and_is_hard_read_only(tmp_path):
    data=tmp_path/"Examples"/"PETRAIII"/"data"; data.mkdir(parents=True)
    (data/"BPM_names.txt").write_text("P3-B1\nP3-B2\n")
    (data/"HCM_names_control.txt").write_text("P3-H1\n")
    (data/"VCM_names_control.txt").write_text("P3-V1\nP3-V2\n")
    session=InterfaceRegistry(repository_root=tmp_path).create("petra")
    assert isinstance(session.adapter,PETRAReadOnlyAdapter)
    assert AdapterCapability.WRITE not in session.adapter.capabilities
    assert [d["name"] for d in session.adapter.list_devices("bpm")]==["P3-B1","P3-B2"]
    assert [d["name"] for d in session.adapter.list_devices("hcor")]==["P3-H1"]
    assert [d["name"] for d in session.adapter.list_devices("vcor")]==["P3-V1","P3-V2"]


def test_legacy_interface_adapter_exposes_real_names_and_capabilities():
    item=adapter(); assert AdapterCapability.WRITE in item.capabilities
    assert item.list_devices("bpm")[0]["name"]=="B1"
    assert item.read("MAGNET:H1").value==0
    item.write("MAGNET:H1",.001); assert item.interface.values["H1"]==.001


def test_connection_reports_corrector_inventory_without_arbitrary_readback():
    interface=FakeInterface(); item=AbstractInterfaceAdapter(
        interface,("B1",),("H1","H2"),("V1",),
        backend_metadata={"corrector_control_unit":"rad"},
    )
    magnet_gets=[]; original_get=interface.get
    interface.get=lambda name:(magnet_gets.append(name),original_get(name))[1]
    result=item.test_connection()
    assert result["corrector_readback"]=="available — 2 H / 1 V"
    assert result["horizontal_correctors"]==2 and result["vertical_correctors"]==1
    assert result["corrector_unit"]=="rad"
    assert magnet_gets==[]


def test_automatic_dispersion_restores_rf_on_success_failure_and_cancel():
    interface=FakeInterface(); item=adapter(interface); bpm=(BpmDevice("B1","BPM:B1:X","BPM:B1:Y"),)
    result=AutomaticDispersionAcquirer(item,bpm).acquire(1000,2,0,direction="bipolar",sleeper=lambda _:None)
    assert interface.rf==500_000_000.; assert result.restoration_status=="restored"
    assert [state.label for state in result.states]==["reference","positive","negative","reference_after"]
    np.testing.assert_allclose(result.state("reference_after").mean_x_m,result.state("reference").mean_x_m)
    assert result.response_x_m_per_hz[0]==pytest.approx(2e-9)
    interface.fail_orbit=True
    with pytest.raises(RuntimeError,match="injected orbit failure"):
        AutomaticDispersionAcquirer(item,bpm).acquire(1000,2,0,sleeper=lambda _:None)
    assert interface.rf==500_000_000.
    cancel=Event(); cancel.set()
    with pytest.raises(AcquisitionCancelled):
        AutomaticDispersionAcquirer(item,bpm).acquire(1000,2,0,cancel_event=cancel,sleeper=lambda _:None)
    assert interface.rf==500_000_000.


def test_correction_apply_requires_confirmation_and_reads_back():
    class Item:
        included=True; control_name="Q1"; name="Q1"; final_delta=.25
    class Review: items=[Item()]
    service=CorrectionApplicationService(adapter()); changes=service.preview(Review())
    assert changes[0].current==1.; assert changes[0].proposed==1.25
    with pytest.raises(PermissionError):service.apply(changes,confirmed=False)
    applied=service.apply(changes,confirmed=True); assert applied[0].readback==1.25; assert applied[0].status=="success"


def test_p3_vertical_orbit_uses_vertical_address():
    source=Path("p3_interface.py").read_text(encoding="utf-8")
    assert "data_y = doocs4py.get(BPM_ADDRESS_Y)" in source
    assert "data_y = doocs4py.get(BPM_ADDRESS_X)" not in source
