from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from pyLOCO.data_schema import MeasurementSession, SessionFile, save_session, write_bpm_noise, write_dispersion, write_orm
from pyLOCO.gui.main_window import MainWindow
from pyLOCO.gui.measurement_mapping import find_exact_device_mappings, mapping_counts
from pyLOCO.machine_mapping import merge_measurement


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


class BPM:
    def __init__(self,name,device):self.FamName,self.Device=name,device


class Multipole:
    def __init__(self,name,device):
        self.FamName,self.Device=name,device
        self.PolynomA,self.PolynomB=[0.0,0.0],[0.0,0.0]
        self.Length,self.PassMethod=1.0,"StrMPoleSymplectic4Pass"


def _session(tmp_path, *, include_mapping=True, shared_mapping=False):
    bpms=["machine/bpm/2","machine/bpm/1"]
    hcors=["machine/steerer/2-B1L"]
    vcors=["machine/steerer/1-A1L"]
    noise=tmp_path/"noise.h5"; dispersion=tmp_path/"dispersion.h5"; orm=tmp_path/"orm.h5"
    write_bpm_noise(noise,noise_x_m=[1e-9,2e-9],noise_y_m=[2e-9,3e-9],bpm_names=bpms,
                    raw_orbits_x_m=np.zeros((2,2)),raw_orbits_y_m=np.zeros((2,2)))
    write_dispersion(dispersion,measured_eta_x=[1e-5,2e-5],measured_eta_y=[3e-5,4e-5],bpm_names=bpms,
                     rf_frequency_hz=[1,2],rf_setpoint_hz=[1,2],rf_readback_hz=[1,2],
                     raw_orbits_x_m=np.zeros((2,2)),raw_orbits_y_m=np.zeros((2,2)),rf_step_hz=200,bidirectional=True)
    write_orm(orm,response_matrix=np.zeros((4,2)),bpm_names=bpms,horizontal_corrector_names=hcors,
              vertical_corrector_names=vcors,requested_kick_h_rad=[1e-4],requested_kick_v_rad=[1e-4],
              actual_kick_h_rad=[1e-4],actual_kick_v_rad=[1e-4],orbit_plus_m=np.zeros((2,4)),orbit_minus_m=np.zeros((2,4)))
    lattice=tmp_path/"ring.mat"; lattice.write_bytes(b"test lattice")
    ring=[BPM("BPM1","machine/bpm/1"),BPM("BPM2","machine/bpm/2"),
          Multipole("S1","machine/steerer/1"),Multipole("S2","machine/steerer/2")]
    inventory={"bpm":bpms,"horizontal_corrector":hcors,"vertical_corrector":vcors}
    mapping=find_exact_device_mappings(ring,inventory,lattice_path=lattice) if include_mapping else None
    if mapping is not None and shared_mapping:mapping=merge_measurement(None,mapping)
    full={"bundle_version":1,"reference_lattice":{"path":lattice.name,"sha256":mapping["reference_lattice"]["sha256"] if mapping else __import__("hashlib").sha256(lattice.read_bytes()).hexdigest(),"element_count":4,"mat_variable":"RING"},"mapping":mapping}
    manifest=tmp_path/"measurement-session.pyloco-session.json"
    save_session(manifest,MeasurementSession("fit-import",(
        SessionFile("bpm_noise",noise.name),SessionFile("dispersion",dispersion.name),SessionFile("orm",orm.name)),
        {"full_measure_session":full}))
    return manifest,ring,inventory


def test_full_session_import_restores_mapping_and_exact_measured_order(qapp,tmp_path,monkeypatch):
    manifest,ring,inventory=_session(tmp_path)
    monkeypatch.setattr("pyLOCO.gui.main_window.load_project_lattice",lambda *args,**kwargs:ring)
    window=MainWindow(); original_recipe=deepcopy(window.fit_recipe.to_dict())
    monkeypatch.setattr(window,"_choose_measurement_session_import_mode",lambda *args:"full")
    assert window.open_measurement_session(manifest)
    assert mapping_counts(window.project.measurement_device_mapping)["mapped"]==4
    elements=window.project.loco_config.machine_elements
    assert elements.bpm_ords==[1,0]
    assert elements.horizontal_corrector_ords==[3]
    assert elements.vertical_corrector_ords==[2]
    assert window.project.loco_config.element_selection_state["measurement_session_names"]=={
        "bpms":inventory["bpm"],"horizontal_correctors":inventory["horizontal_corrector"],
        "vertical_correctors":inventory["vertical_corrector"]}
    assert [stage["name"] for stage in window.project.fit_recipe["stages"]]==[stage["name"] for stage in original_recipe["stages"]]
    assert "Imported from Measure session / Revalidated" in window.measurement_session_import_summary.text()
    saved=tmp_path/"imported-fit.pyloco.json"; assert window._save_project_to(saved)
    window.project.modified=False; window.close()
    reopened=MainWindow(); reopened.open_project(saved)
    assert mapping_counts(reopened.project.measurement_device_mapping)["mapped"]==4
    assert reopened.project.loco_config.machine_elements.bpm_ords==[1,0]
    assert "Imported from Measure session / Revalidated" in reopened.measurement_session_import_summary.text()
    reopened.project.modified=False; reopened.close()


def test_measurements_only_keeps_fit_lattice_selections_and_recipe(qapp,tmp_path,monkeypatch):
    manifest,_ring,_inventory=_session(tmp_path)
    window=MainWindow(); window.project.lattice.path=str(tmp_path/"existing.mat"); (tmp_path/"existing.mat").write_bytes(b"existing")
    elements=window.project.loco_config.machine_elements
    elements.bpm_ords=[9]; elements.horizontal_corrector_ords=[8]; elements.vertical_corrector_ords=[7]
    before_recipe=deepcopy(window.fit_recipe.to_dict())
    monkeypatch.setattr(window,"_choose_measurement_session_import_mode",lambda *args:"measurements_only")
    assert window.open_measurement_session(manifest)
    assert window.project.lattice.path==str(tmp_path/"existing.mat")
    assert window.project.loco_config.machine_elements.bpm_ords==[9]
    assert window.project.loco_config.machine_elements.horizontal_corrector_ords==[8]
    assert window.project.fit_recipe==before_recipe
    assert set(window.project.measurements)=={"bpm_noise","dispersion","orm"}
    window.project.modified=False; window.close()


def test_full_session_filters_shared_machine_mapping_to_measured_inventory(qapp,tmp_path,monkeypatch):
    manifest,ring,_inventory=_session(tmp_path,shared_mapping=True)
    # A whole-machine shared file may contain unrelated supported roles. FIT
    # must select exact HDF5 identities, never use positional entries.
    data=__import__("json").loads(manifest.read_text())
    mapping=data["metadata"]["full_measure_session"]["mapping"]
    mapping["mappings"].append({"role":"normal_quadrupole","pyaml_device":"Q1","lattice_ordinal":99,
                                "lattice_fam_name":"Q1","lattice_device":"Q1","component":"B2L","unit":"m^-1"})
    manifest.write_text(__import__("json").dumps(data))
    monkeypatch.setattr("pyLOCO.gui.main_window.load_project_lattice",lambda *args,**kwargs:ring)
    window=MainWindow(); monkeypatch.setattr(window,"_choose_measurement_session_import_mode",lambda *args:"full")
    assert window.open_measurement_session(manifest)
    assert mapping_counts(window.project.measurement_device_mapping)["total"]==4
    assert window.project.loco_config.machine_elements.bpm_ords==[1,0]
    window.project.modified=False; window.close()


def test_cancel_is_atomic_and_missing_mapping_is_reported(qapp,tmp_path,monkeypatch):
    manifest,_ring,_inventory=_session(tmp_path,include_mapping=False)
    window=MainWindow(); before=deepcopy(window.project)
    monkeypatch.setattr(window,"_choose_measurement_session_import_mode",lambda *args:None)
    assert window.open_measurement_session(manifest) is False
    assert window.project==before
    monkeypatch.setattr(window,"_choose_measurement_session_import_mode",lambda *args:"full")
    assert window.open_measurement_session(manifest)
    assert window.project.measurement_device_mapping=={}
    assert "Not restored" in window.measurement_session_import_summary.text()
    assert any("No Machine Mapping was saved" in item for item in window.project.compatibility_validation["issues"])
    window.project.modified=False; window.close()


def test_fit_mapping_actions_include_python_helper(qapp):
    window=MainWindow()
    assert window.find_measurement_mappings_button.text()=="Find Machine Mapping"
    assert window.load_measurement_mapping_button.text()=="Load Machine Mapping…"
    assert window.create_measurement_mapping_template_button.text()=="Create Machine Mapping template…"
    assert window.measurement_mapping_python_helper_button.text()=="Python Mapping Helper…"
    assert window.measurement_mapping_details_button.text()=="Mapping Details"
    assert window.save_measurement_mapping_button.text()=="Save validated Machine Mapping…"
    window.project.modified=False; window.close()
