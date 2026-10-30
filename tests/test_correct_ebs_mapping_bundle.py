"""Read-only regression checks for the authoritative 4434-element EBS mapping."""
from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from pyLOCO.correct.model import CorrectionReview
from pyLOCO.correct.pyaml_connection import (
    file_hash, inventory_fingerprint, make_mapping_bundle,
    validate_mapping_bundle,
)
from pyLOCO.data_schema import load_correction_plan
from pyLOCO.machine_mapping import merge_corrections, template as machine_mapping_template


_fixture_root = os.environ.get("PYLOCO_EBS_MAPPING_FIXTURES")
EBS = Path(_fixture_root).expanduser() if _fixture_root else Path("__unavailable__")
CONFIG = EBS / "betamodel_conf_pyaml.yaml"
LATTICE = EBS / "betamodel.mat"
MAPPING = EBS / "pyaml_pysc_mapping.json"
FULL_PLAN = EBS / "MOCK_fit_correction.json"
OLD_LATTICE = Path("Examples/EBS/multistage_fit/data/betamodel.mat")
SMOKE_PLAN = Path("mdt_smoke/output/correction/ebs-mdt-offline-dry-run.json")


pytestmark = pytest.mark.skipif(
    not all(path.is_file() for path in (CONFIG, LATTICE, MAPPING, FULL_PLAN)),
    reason="Local validated EBS pySC/pyAML artifacts are unavailable",
)


def connection():
    config = yaml.safe_load(CONFIG.read_text())
    arrays = {row["name"]: row["elements"] for row in config["arrays"]}
    inventory = [
        {"name": name, "family": family, "lattice_name": name}
        for family, key in (("normal_quadrupole", "Quads"),
                            ("skew_quadrupole", "SkewQuads"))
        for name in arrays[key]
    ]
    return SimpleNamespace(
        options={"config_path": str(CONFIG), "mode": "live",
                 "normal_array": "Quads", "skew_array": "SkewQuads"},
        config_hash=file_hash(CONFIG), inventory=inventory,
    )


def review(path):
    return CorrectionReview.from_plan(load_correction_plan(path))


def test_authoritative_ebs_mapping_validates_all_802_entries():
    selected, rows, provenance, lattice = validate_mapping_bundle(
        review(FULL_PLAN), connection(), MAPPING, LATTICE
    )
    assert len(selected) == len(rows) == 802
    assert sum(row["correction_type"] == "normal_quadrupole" for row in rows) == 514
    assert sum(row["correction_type"] == "skew_quadrupole" for row in rows) == 288
    assert all(row["status"] == "mapped" for row in rows)
    assert provenance == "authoritative legacy mapping — revalidated"
    assert lattice["element_count"] == 4434


def test_authoritative_mapping_selects_three_smoke_fit_magnets():
    selected, rows, _provenance, _lattice = validate_mapping_bundle(
        review(SMOKE_PLAN), connection(), MAPPING, LATTICE
    )
    assert [(row["lattice_name"], row["lattice_ordinal"]) for row in rows] == [
        ("QD3J", 27), ("QF4J", 36), ("QD5B", 51),
    ]
    assert len(selected) == 3
    assert all(row["status"] == "mapped" for row in rows)


def test_mapping_rejects_4432_lattice_and_changed_inventory(tmp_path):
    state = review(FULL_PLAN)
    conn = connection()
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_mapping_bundle(state, conn, MAPPING, OLD_LATTICE)
    conn.inventory = conn.inventory[:-1]
    with pytest.raises(ValueError, match="wrong-family mapping target"):
        validate_mapping_bundle(state, conn, MAPPING, LATTICE)


def test_versioned_bundle_revalidates_checksums_and_never_uses_status(tmp_path):
    state = review(FULL_PLAN)
    conn = connection()
    entries, _rows, _provenance, _lattice = validate_mapping_bundle(
        state, conn, MAPPING, LATTICE
    )
    bundle = make_mapping_bundle(
        state, conn, entries, LATTICE, provenance="approved pySC sim_index mapping"
    )
    target = tmp_path / "ebs.mapping.json"
    target.write_text(json.dumps(bundle))
    selected, rows, provenance, _identity = validate_mapping_bundle(
        state, conn, target, LATTICE
    )
    assert len(selected) == 802 and all(row["status"] == "mapped" for row in rows)
    assert provenance == "revalidated saved mapping"
    assert bundle["inventory"]["sha256"] == inventory_fingerprint(conn.inventory)

    stale = deepcopy(bundle)
    stale["machine_profile"]["configuration_sha256"] = "0" * 64
    target.write_text(json.dumps(stale))
    with pytest.raises(ValueError, match="different machine configuration"):
        validate_mapping_bundle(state, conn, target, LATTICE)

    stale = deepcopy(bundle)
    stale["mappings"][0]["mapping_status"] = "mapped"
    stale["mappings"][0]["control_name"] = "not-a-device"
    # Refresh the inventory checksum cannot make an invalid target authoritative.
    target.write_text(json.dumps(stale))
    with pytest.raises(ValueError, match="wrong-family mapping target"):
        validate_mapping_bundle(state, conn, target, LATTICE)


def test_shared_machine_mapping_filters_and_validates_all_802_correction_entries(tmp_path):
    state=review(FULL_PLAN); conn=connection()
    entries,_rows,_provenance,_lattice=validate_mapping_bundle(state,conn,MAPPING,LATTICE)
    bundle=make_mapping_bundle(state,conn,entries,LATTICE,provenance="shared EBS mapping")
    shared=machine_mapping_template({"bpm":["preserved-bpm"]})
    shared=merge_corrections(shared,bundle)
    # This is the form produced by the external Python Mapping Helper: it fills
    # authoritative ordinals, while pyLOCO derives and validates FamName from
    # the active lattice rather than trusting user-authored provenance.
    for row in shared["mappings"]:
        if row["role"] in {"normal_quadrupole", "skew_quadrupole"}:
            row["lattice_fam_name"] = ""
    target=tmp_path/"machine-mapping.json"; target.write_text(json.dumps(shared))
    selected,rows,provenance,identity=validate_mapping_bundle(state,conn,target,LATTICE)
    assert len(selected)==len(rows)==802 and all(row["status"]=="mapped" for row in rows)
    assert any(row["role"]=="bpm" for row in shared["mappings"])
    assert identity["element_count"]==4434 and provenance=="revalidated saved mapping"
    stale=deepcopy(shared); stale["inventories"]["correction"]["sha256"]="0"*64; target.write_text(json.dumps(stale))
    with pytest.raises(ValueError,match="inventory has changed"):validate_mapping_bundle(state,conn,target,LATTICE)
    invalid=deepcopy(shared); next(row for row in invalid["mappings"] if row["role"]=="normal_quadrupole")["unit"]="A"
    target.write_text(json.dumps(invalid))
    with pytest.raises(ValueError,match="Unsupported pyAML unit"):validate_mapping_bundle(state,conn,target,LATTICE)
