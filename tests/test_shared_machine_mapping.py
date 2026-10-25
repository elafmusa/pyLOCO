from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyLOCO.gui.measurement_mapping import mapping_counts, validate_mapping
from pyLOCO.machine_mapping import (SCHEMA, VERSION, correction_projection, entry,
    complete_template, editable_template, measurement_projection, merge_corrections, merge_measurement,
    mapping_statistics, pyaml_configuration_inventory, reconcile_configured,
    retain_loaded_magnet_rows, template)


def test_shared_template_assigns_roles_and_preserves_same_ordinal_components():
    document=template({"bpm":["B"],"horizontal_corrector":["C-B1L"],"vertical_corrector":["C-A1L"]},
                      {"normal_quadrupole":["Q-B2L"],"skew_quadrupole":["S-A2L"]})
    assert (document["schema"],document["schema_version"])==(SCHEMA,VERSION)
    assert [row["role"] for row in document["mappings"]]==["bpm","horizontal_corrector","vertical_corrector","normal_quadrupole","skew_quadrupole"]
    h,v=document["mappings"][1:3]; h["lattice_ordinal"]=v["lattice_ordinal"]=42
    assert (h["component"],v["component"])==("B1L","A1L") and h["role"]!=v["role"]


def test_editable_template_contains_only_required_mapping_fields():
    document=template({"bpm":["BPM"]})
    document["mappings"][0].update(lattice_ordinal=12,lattice_fam_name="BPM1",
                                    status="proposed_unique_exact_match")
    assert editable_template(document)=={
        "schema":SCHEMA,"schema_version":VERSION,"template":True,
        "mappings":[{"pyaml_device":"BPM","role":"bpm","lattice_ordinal":12}],
    }


def test_measure_projection_ignores_incomplete_quadrupoles_and_preserves_order(tmp_path):
    lattice=tmp_path/"ring.mat"; lattice.write_bytes(b"ring")
    shared=template({"bpm":["B2","B1"],"horizontal_corrector":[],"vertical_corrector":[]},
                    {"normal_quadrupole":["Q-B2L"]})
    shared["mappings"][0]["lattice_ordinal"]=1; shared["mappings"][1]["lattice_ordinal"]=0
    ring=[SimpleNamespace(FamName="B1",Device="B1"),SimpleNamespace(FamName="B2",Device="B2")]
    projected=measurement_projection(shared)
    checked=validate_mapping(projected,ring,{"bpm":["B2","B1"],"horizontal_corrector":[],"vertical_corrector":[]},lattice_path=lattice,require_provenance=False)
    assert mapping_counts(checked)["mapped"]==2
    assert [row["lattice_ordinal"] for row in checked["mappings"]]==[1,0]
    combined=merge_measurement(shared,checked)
    assert any(row["role"]=="normal_quadrupole" and row["lattice_ordinal"] is None for row in combined["mappings"])


def test_measure_and_correct_updates_do_not_modify_each_others_entries(tmp_path):
    shared=template({"bpm":["B"]},{"normal_quadrupole":["Q"]})
    measured={"reference_lattice":{"sha256":"l","element_count":2},"machine_profile":{"configuration_sha256":"c"},
              "inventory":{"sha256":"i","counts":{}},"mapping_provenance":"measure","mappings":[
              {"role":"bpm","measured_name":"B","lattice_ordinal":0,"lattice_name":"B","lattice_device":"DB","component":"","unit":"m","identity_match":"exact_device","status":"mapped","reason":"ok"}]}
    merged=merge_measurement(shared,measured); magnet=next(row for row in merged["mappings"] if row["role"]=="normal_quadrupole")
    bundle={"reference_lattice":{"sha256":"l","element_count":2},"machine_profile":{"configuration_sha256":"c"},
            "inventory":{"sha256":"q","device_count":1},"mapping_provenance":"correct","mappings":[
            {"correction_type":"normal_quadrupole","lattice_name":"Q","lattice_ordinal":1,"lattice_device":"DQ","control_name":"Q-B2L","component":"B2L","unit":"m^-1"}]}
    combined=merge_corrections(merged,bundle)
    assert next(row for row in combined["mappings"] if row["role"]=="bpm")["pyaml_device"]=="B"
    correction=correction_projection(combined)[0]
    assert correction["control_name"]=="Q-B2L" and correction["component"]=="B2L" and correction["unit"]=="m^-1"
    assert combined["inventories"]["measurement"]["sha256"]=="i"
    assert combined["inventories"]["correction"]["sha256"]=="q"


def test_legacy_measure_and_correct_documents_remain_projectable():
    legacy_measure={"schema":"pyloco.measurement_device_mapping","schema_version":1,"mappings":[{"role":"bpm","measured_name":"B"}]}
    assert measurement_projection(legacy_measure) is legacy_measure
    legacy_correct={"mappings":[{"lattice_name":"Q","lattice_ordinal":1,"control_name":"M"}]}
    assert correction_projection(legacy_correct)==legacy_correct["mappings"]


def test_cross_application_merge_rejects_different_machine_identity():
    shared=template({"bpm":["B"]})
    shared["reference_lattice"]={"sha256":"lattice-a","element_count":10}
    shared["machine_profile"]={"configuration_sha256":"config-a"}
    measurement={"reference_lattice":{"sha256":"lattice-b","element_count":10},
                 "machine_profile":{"configuration_sha256":"config-a"},"mappings":[]}
    with pytest.raises(ValueError,match="reference lattice sha256 differs"):
        merge_measurement(shared,measurement)
    bundle={"reference_lattice":{"sha256":"lattice-a","element_count":10},
            "machine_profile":{"configuration_sha256":"config-b"},"mappings":[]}
    with pytest.raises(ValueError,match="configuration fingerprint differs"):
        merge_corrections(shared,bundle)


def test_complete_mapping_uses_configured_pyaml_devices_not_lattice_inventory():
    class BPM:pass
    class Quadrupole:pass
    class Sextupole:pass
    class Drift:pass
    ring=[]
    for cls,family,device in ((BPM,"BPM1","BPM1"),(Quadrupole,"Q1","Q1"),
                              (Sextupole,"S1","S1"),(Drift,"D1","D1"),
                              (Quadrupole,"Q-not-configured","Q-not-configured")):
        item=cls(); item.FamName=family; item.Device=device; ring.append(item)
    configuration={"arrays":[{"name":"CUSTOM_MAGNET_GROUP","elements":["Q1-B2L"]}],"devices":[
        {"class":"site.bpm.BPM","name":"BPM1"},
        {"class":"site.magnet.quadrupole.Quadrupole","name":"Q1-B2L",
         "model":{"unit":"rad","hardware_unit":"str"}},
        {"class":"site.magnet.sextupole.Sextupole","name":"S1-B3L"},
        {"class":"site.service.Diagnostics","name":"NOT-A-PHYSICAL-DEVICE"},
    ]}
    inventory,metadata=pyaml_configuration_inventory(configuration)
    document=complete_template(ring,{role:inventory[role] for role in
        ("bpm","horizontal_corrector","vertical_corrector")},
        {role:inventory[role] for role in ("normal_quadrupole","skew_quadrupole",
         "sextupole","octupole","dipole")},metadata)
    assert [(row["role"],row["pyaml_device"]) for row in document["mappings"]]==[
        ("bpm","BPM1"),("normal_quadrupole","Q1-B2L"),("sextupole","S1-B3L")]
    assert all(row["lattice_ordinal"] is not None for row in document["mappings"])
    assert not any(row["lattice_fam_name"] in {"D1","Q-not-configured"} for row in document["mappings"])
    quad=document["mappings"][1]
    assert quad["component"]=="B2L" and quad["configured_unit"]=="rad"


def test_configured_device_without_unique_lattice_match_remains_unresolved():
    ring=[SimpleNamespace(FamName="Q",Device="Q"),SimpleNamespace(FamName="Q",Device="Q")]
    document=complete_template(ring,{}, {"normal_quadrupole":["Q"]})
    assert len(document["mappings"])==1
    assert document["mappings"][0]["lattice_ordinal"] is None
    assert document["mappings"][0]["status"]=="ambiguous"


def test_complete_mapping_uses_authoritative_backend_lattice_identity():
    ring=[SimpleNamespace(FamName="SH1A",Device="srmag/vst-sh1/c01-a")]
    measured={"bpm":[],"horizontal_corrector":["control/H"],"vertical_corrector":[]}
    metadata={("horizontal_corrector","control/H"):{
        "lattice_names":["srmag/vst-sh1/c01-a"]}}
    row=complete_template(ring,measured,{},metadata)["mappings"][0]
    assert row["lattice_ordinal"]==0
    assert row["status"]=="proposed_unique_exact_match"
    assert row["identity_match"]=="backend_lattice_identity_exact_device"


def test_filled_template_ordinal_is_validated_and_used_before_name_search():
    ring=[SimpleNamespace(FamName="Q",Device="Q"),SimpleNamespace(FamName="Q",Device="Q")]
    filled=template({}, {"normal_quadrupole":["Q"]})
    filled["mappings"][0]["lattice_ordinal"]=1
    proposal=complete_template(ring,{}, {"normal_quadrupole":["Q"]})
    reconciled=reconcile_configured(filled,proposal,ring=ring)
    row=reconciled["mappings"][0]
    assert row["lattice_ordinal"]==1
    assert row["identity_match"]=="explicit_lattice_ordinal"
    assert row["status"]=="proposed_explicit_ordinal"


def test_filled_template_accepts_in_range_ordinal_without_name_matching():
    ring=[SimpleNamespace(FamName="OTHER",Device="OTHER"),
          SimpleNamespace(FamName="Q",Device="Q")]
    filled=template({}, {"normal_quadrupole":["Q"]})
    filled["mappings"][0]["lattice_ordinal"]=0
    proposal=complete_template(ring,{}, {"normal_quadrupole":["Q"]})
    assert proposal["mappings"][0]["lattice_ordinal"]==1
    row=reconcile_configured(filled,proposal,ring=ring)["mappings"][0]
    assert row["lattice_ordinal"]==0
    assert row["identity_match"]=="explicit_lattice_ordinal"
    assert row["status"]=="proposed_explicit_ordinal"


def test_filled_template_rejects_only_out_of_range_ordinal():
    ring=[SimpleNamespace(FamName="Q",Device="Q")]
    filled=template({}, {"normal_quadrupole":["Q"]})
    filled["mappings"][0]["lattice_ordinal"]=4
    row=reconcile_configured(filled,complete_template(ring,{}, {"normal_quadrupole":["Q"]}),
                             ring=ring)["mappings"][0]
    assert row["lattice_ordinal"] is None
    assert row["requested_lattice_ordinal"]==4
    assert row["status"]=="invalid_explicit_ordinal"


def test_reconcile_preserves_validated_mapping_when_details_are_rebuilt():
    ring=[SimpleNamespace(FamName="SH1",Device="AT-COR")]
    validated=template({"horizontal_corrector":["SJ2A-C04-H"]})
    validated["template"]=False
    validated["mappings"][0].update(
        lattice_ordinal=0,lattice_fam_name="SH1",lattice_device="AT-COR",
        identity_match="explicit_lattice_ordinal",status="mapped",
        evidence="Explicit zero-based AT index accepted")
    proposal=complete_template(ring,{"horizontal_corrector":["SJ2A-C04-H"]})
    assert proposal["mappings"][0]["status"]=="unresolved"
    row=reconcile_configured(validated,proposal,ring=ring)["mappings"][0]
    assert row["status"]=="mapped"
    assert row["lattice_ordinal"]==0
    assert row["lattice_fam_name"]=="SH1"
    assert row["lattice_device"]=="AT-COR"


def test_loaded_complete_roles_survive_limited_runtime_proposal():
    ring=[SimpleNamespace(FamName="B",Device="B"),SimpleNamespace(FamName="Q",Device="Q")]
    loaded=template({"bpm":["B"]},{"normal_quadrupole":["Q-B2L"]})
    loaded["mappings"][1]["lattice_ordinal"]=1
    limited=complete_template(ring,{"bpm":["B"]},{})
    retained=retain_loaded_magnet_rows(limited,loaded,ring)
    assert [row["role"] for row in retained["mappings"]]==["bpm","normal_quadrupole"]
    assert retained["mappings"][1]["lattice_ordinal"]==1
    assert mapping_statistics(retained)=={
        "total":2,"mapped":2,"unresolved":0,"ambiguous":0,"duplicate":0,"invalid":0}


def test_fit_project_and_run_request_preserve_complete_machine_mapping():
    from pyLOCO.gui.backend import LocoRunRequest
    from pyLOCO.gui.models.project import ProjectMetadata
    master=template({"bpm":["B"]},{"normal_quadrupole":["Q"],"sextupole":["S"]})
    project=ProjectMetadata(machine_mapping=master)
    restored=ProjectMetadata.from_dict(project.to_dict())
    assert restored.machine_mapping==master
    assert LocoRunRequest.from_project(restored).machine_mapping==master


def test_correct_projects_only_supported_fitted_controls_from_master_mapping():
    master=template({}, {"normal_quadrupole":["Q-B2L"],"sextupole":["S-B3L"]})
    master["mappings"][0].update(lattice_ordinal=7,lattice_fam_name="Q",lattice_device="Q")
    master["mappings"][1].update(lattice_ordinal=8,lattice_fam_name="S",lattice_device="S")
    rows=correction_projection(master)
    assert rows==[{"correction_type":"normal_quadrupole","lattice_name":"Q",
        "lattice_ordinal":7,"lattice_device":"Q","lattice_common_name":"","physical_element_type":"",
        "control_name":"Q-B2L","component":"B2L","unit":"m^-1"}]


def test_correct_merge_updates_fitted_subset_without_dropping_complete_mapping():
    from pyLOCO.machine_mapping import merge_corrections
    master=template({"bpm":["BPM"]},{"normal_quadrupole":["Q1","Q2"],"sextupole":["S1"]})
    for ordinal,row in enumerate(master["mappings"]):row["lattice_ordinal"]=ordinal
    bundle={"mappings":[{"correction_type":"normal_quadrupole","lattice_ordinal":1,
        "lattice_name":"Q1","control_name":"Q1-new","component":"B2L","unit":"m^-1"}]}
    merged=merge_corrections(master,bundle)
    assert len(merged["mappings"])==4
    assert any(row["pyaml_device"]=="Q2" for row in merged["mappings"])
    assert any(row["pyaml_device"]=="S1" for row in merged["mappings"])
    assert any(row["pyaml_device"]=="BPM" for row in merged["mappings"])
    assert any(row["pyaml_device"]=="Q1-new" for row in merged["mappings"])
