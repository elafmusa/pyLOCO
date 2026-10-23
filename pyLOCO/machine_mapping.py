"""Shared, read-only interchange schema for Measure, FIT and Correct mappings."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable
import json


SCHEMA = "pyloco.machine_mapping"
VERSION = 3
MEASUREMENT_ROLES = ("bpm", "horizontal_corrector", "vertical_corrector")
CORRECTION_ROLES = ("normal_quadrupole", "skew_quadrupole")
COMPONENTS = {"bpm":"", "horizontal_corrector":"B1L", "vertical_corrector":"A1L",
              "normal_quadrupole":"B2L", "skew_quadrupole":"A2L"}
UNITS = {"bpm":"m", "horizontal_corrector":"rad", "vertical_corrector":"rad",
         "normal_quadrupole":"m^-1", "skew_quadrupole":"m^-1"}


def entry(*, role: str, pyaml_device: str, lattice_ordinal=None,
          lattice_fam_name="", lattice_device="", component=None, unit=None, **extra):
    if role not in MEASUREMENT_ROLES + CORRECTION_ROLES:
        raise ValueError(f"Unsupported machine-mapping role: {role!r}")
    return {"lattice_ordinal":lattice_ordinal,"lattice_fam_name":str(lattice_fam_name or ""),
            "lattice_device":str(lattice_device or ""),"pyaml_device":str(pyaml_device or ""),
            "role":role,"component":COMPONENTS[role] if component is None else str(component),
            "unit":UNITS[role] if unit is None else str(unit),**extra}


def template(measurement_inventory: dict[str,list[str]], magnet_inventory: dict[str,list[str]]|None=None):
    magnet_inventory=magnet_inventory or {}
    rows=[]
    for role in MEASUREMENT_ROLES:
        rows.extend(entry(role=role,pyaml_device=name) for name in measurement_inventory.get(role,()))
    for role in CORRECTION_ROLES:
        rows.extend(entry(role=role,pyaml_device=name) for name in magnet_inventory.get(role,()))
    return {"schema":SCHEMA,"schema_version":VERSION,
            "description":"Fill lattice_ordinal (AT index, 0-based). Roles come from authoritative pyAML groups; pyLOCO derives and validates remaining lattice/provenance fields.",
            "template":True,"user_fields":["lattice_ordinal","identity_evidence"],
            "derived_on_load":["lattice_fam_name","lattice_device","status","identity_match"],
            "mappings":rows}


def is_document(document: dict[str,Any]) -> bool:
    return document.get("schema")==SCHEMA and document.get("schema_version")==VERSION


def is_template(document: dict[str,Any]) -> bool:
    return is_document(document) and document.get("template") is True and "reference_lattice" not in document


def load(path: str|Path) -> dict[str,Any]:
    source=Path(path).expanduser(); text=source.read_text(encoding="utf-8")
    data=json.loads(text) if source.suffix.lower() not in {".yaml",".yml"} else __import__("yaml").safe_load(text)
    if not isinstance(data,(dict,list)):raise ValueError("Machine Mapping must contain an object or list")
    if isinstance(data,list):return {"mappings":data}
    return data


def measurement_projection(document: dict[str,Any]) -> dict[str,Any]:
    """Project shared entries into the established strict measurement validator."""
    if not is_document(document):return document
    rows=[]
    for row in document.get("mappings",()):
        if row.get("role") not in MEASUREMENT_ROLES:continue
        rows.append({"measured_name":row.get("pyaml_device",""),"role":row.get("role"),
                     "lattice_ordinal":row.get("lattice_ordinal"),
                     "identity_evidence":row.get("identity_evidence",""),
                     "provenance":row.get("provenance","shared Machine Mapping")})
    inventories=document.get("inventories",{})
    return {"schema":"pyloco.measurement_device_mapping","schema_version":1,
            "reference_lattice":deepcopy(document.get("reference_lattice",{})),
            "machine_profile":deepcopy(document.get("machine_profile",{})),
            "inventory":deepcopy(inventories.get("measurement",document.get("inventory",{}))),
            "mapping_provenance":document.get("mapping_provenance","shared Machine Mapping"),
            "user_fields":["lattice_ordinal","identity_evidence"],
            "derived_on_load":["lattice_name","lattice_device","status"],
            "mappings":rows}


def correction_projection(document: dict[str,Any]) -> list[dict[str,Any]]:
    if not is_document(document):
        entries=document.get("mappings",document) if isinstance(document,dict) else document
        return list(entries or ())
    return [dict(correction_type=row["role"],lattice_name=row.get("lattice_fam_name",""),
                 lattice_ordinal=row.get("lattice_ordinal"),lattice_device=row.get("lattice_device",""),control_name=row.get("pyaml_device",""),
                 component=row.get("component"),unit=row.get("unit"))
            for row in document.get("mappings",()) if row.get("role") in CORRECTION_ROLES]


def _require_same_machine(base: dict[str,Any], incoming: dict[str,Any]) -> None:
    """Never preserve entries across incompatible lattice/configuration identities."""
    old_lattice=base.get("reference_lattice",{}); new_lattice=incoming.get("reference_lattice",{})
    for key in ("sha256","element_count"):
        if old_lattice.get(key) is not None and new_lattice.get(key) is not None and old_lattice[key]!=new_lattice[key]:
            raise ValueError(f"Cannot merge Machine Mapping files: reference lattice {key} differs")
    old_profile=base.get("machine_profile",{}); new_profile=incoming.get("machine_profile",{})
    if (old_profile.get("configuration_sha256") and new_profile.get("configuration_sha256")
            and old_profile["configuration_sha256"]!=new_profile["configuration_sha256"]):
        raise ValueError("Cannot merge Machine Mapping files: machine configuration fingerprint differs")


def merge_measurement(document: dict[str,Any]|None, validated: dict[str,Any]) -> dict[str,Any]:
    """Replace only measurement roles; preserve every correction-role entry."""
    base=deepcopy(document) if document and is_document(document) else {"schema":SCHEMA,"schema_version":VERSION,"mappings":[]}
    _require_same_machine(base,validated)
    preserved=[row for row in base.get("mappings",()) if row.get("role") not in MEASUREMENT_ROLES]
    converted=[]
    for row in validated.get("mappings",()):
        converted.append(entry(role=row["role"],pyaml_device=row.get("measured_name",""),
            lattice_ordinal=row.get("lattice_ordinal"),lattice_fam_name=row.get("lattice_name",""),
            lattice_device=row.get("lattice_device",""),component=row.get("component"),unit=row.get("unit"),
            identity_match=row.get("identity_match",""),status=row.get("status",""),
            evidence=row.get("reason",""),identity_evidence=row.get("identity_evidence","")))
    base.update(schema=SCHEMA,schema_version=VERSION,template=False,
                reference_lattice=deepcopy(validated.get("reference_lattice",{})),
                machine_profile=deepcopy(validated.get("machine_profile",{})),
                mapping_provenance=validated.get("mapping_provenance","validated by pyLOCO"))
    inventories=deepcopy(base.get("inventories",{})); inventories["measurement"]=deepcopy(validated.get("inventory",{})); base["inventories"]=inventories
    base["mappings"]=preserved+converted
    return base


def merge_corrections(document: dict[str,Any]|None, bundle: dict[str,Any]) -> dict[str,Any]:
    """Replace only correction roles; preserve measurement entries."""
    base=deepcopy(document) if document and is_document(document) else {"schema":SCHEMA,"schema_version":VERSION,"mappings":[]}
    _require_same_machine(base,bundle)
    preserved=[row for row in base.get("mappings",()) if row.get("role") not in CORRECTION_ROLES]
    converted=[entry(role=row.get("correction_type"),pyaml_device=row.get("control_name",""),
        lattice_ordinal=row.get("lattice_ordinal"),lattice_fam_name=row.get("lattice_name",""),
        lattice_device=row.get("lattice_device",""),component=row.get("component"),unit=row.get("unit"),
        status="mapped",evidence=row.get("provenance","validated by pyLOCO Correct")) for row in bundle.get("mappings",())]
    base.update(schema=SCHEMA,schema_version=VERSION,template=False,
                reference_lattice=deepcopy(bundle.get("reference_lattice",{})),
                machine_profile=deepcopy(bundle.get("machine_profile",{})),
                mapping_provenance=bundle.get("mapping_provenance","validated by pyLOCO Correct"))
    inventories=deepcopy(base.get("inventories",{})); inventories["correction"]=deepcopy(bundle.get("inventory",{})); base["inventories"]=inventories
    base["mappings"]=preserved+converted
    return base
