from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyLOCO.gui.measurement_mapping import mapping_counts, validate_mapping
from pyLOCO.machine_mapping import (SCHEMA, VERSION, correction_projection, entry,
    measurement_projection, merge_corrections, merge_measurement, template)


def test_shared_template_assigns_roles_and_preserves_same_ordinal_components():
    document=template({"bpm":["B"],"horizontal_corrector":["C-B1L"],"vertical_corrector":["C-A1L"]},
                      {"normal_quadrupole":["Q-B2L"],"skew_quadrupole":["S-A2L"]})
    assert (document["schema"],document["schema_version"])==(SCHEMA,VERSION)
    assert [row["role"] for row in document["mappings"]]==["bpm","horizontal_corrector","vertical_corrector","normal_quadrupole","skew_quadrupole"]
    h,v=document["mappings"][1:3]; h["lattice_ordinal"]=v["lattice_ordinal"]=42
    assert (h["component"],v["component"])==("B1L","A1L") and h["role"]!=v["role"]


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
