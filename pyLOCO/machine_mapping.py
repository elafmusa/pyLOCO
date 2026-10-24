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
MACHINE_MAGNET_ROLES = CORRECTION_ROLES + ("sextupole", "octupole", "dipole")
COMPONENTS = {"bpm":"", "horizontal_corrector":"B1L", "vertical_corrector":"A1L",
              "normal_quadrupole":"B2L", "skew_quadrupole":"A2L",
              "sextupole":"B3L", "octupole":"B4L", "dipole":"B1L"}
UNITS = {"bpm":"m", "horizontal_corrector":"rad", "vertical_corrector":"rad",
         "normal_quadrupole":"m^-1", "skew_quadrupole":"m^-1",
         "sextupole":"m^-2", "octupole":"m^-3", "dipole":"rad"}


def entry(*, role: str, pyaml_device: str, lattice_ordinal=None,
          lattice_fam_name="", lattice_device="", physical_element_type="",
          component=None, unit=None, **extra):
    if role not in MEASUREMENT_ROLES + MACHINE_MAGNET_ROLES:
        raise ValueError(f"Unsupported machine-mapping role: {role!r}")
    return {"lattice_ordinal":lattice_ordinal,"lattice_fam_name":str(lattice_fam_name or ""),
            "lattice_device":str(lattice_device or ""),"pyaml_device":str(pyaml_device or ""),
            "role":role,"physical_element_type":str(physical_element_type or ""),
            "component":COMPONENTS[role] if component is None else str(component),
            "unit":UNITS[role] if unit is None else str(unit),**extra}


def template(measurement_inventory: dict[str,list[str]], magnet_inventory: dict[str,list[str]]|None=None):
    magnet_inventory=magnet_inventory or {}
    rows=[]
    for role in MEASUREMENT_ROLES:
        rows.extend(entry(role=role,pyaml_device=name) for name in measurement_inventory.get(role,()))
    for role in MACHINE_MAGNET_ROLES:
        rows.extend(entry(role=role,pyaml_device=name) for name in magnet_inventory.get(role,()))
    return {"schema":SCHEMA,"schema_version":VERSION,
            "description":"One machine mapping row per supported physical device in the selected pyAML configuration. Fill lattice_ordinal (AT index, 0-based) only where automatic exact matching is unresolved; pyLOCO validates lattice and configuration provenance on load.",
            "template":True,"user_fields":["lattice_ordinal","identity_evidence"],
            "derived_on_load":["lattice_fam_name","lattice_device","status","identity_match"],
            "mappings":rows}


def _exact_lattice_match(ring, identity: str, role: str) -> tuple[list[int], str]:
    identities=[identity]
    suffix={"horizontal_corrector":"-B1L","vertical_corrector":"-A1L",
            "normal_quadrupole":"-B2L","skew_quadrupole":"-A2L",
            "sextupole":"-B3L","octupole":"-B4L"}.get(role)
    if suffix and identity.endswith(suffix):identities.append(identity[:-len(suffix)])
    for attribute,method in (("Device","exact_device"),("FamName","exact_fam_name")):
        for position,candidate in enumerate(identities):
            hits=[index for index,element in enumerate(ring)
                  if str(getattr(element,attribute,"") or "")==candidate]
            if hits:
                return hits, method if position==0 else f"legacy_suffix_{method}"
    return [], ""


def pyaml_configuration_inventory(configuration: dict[str,Any]) -> tuple[dict[str,list[str]],dict[tuple[str,str],dict[str,Any]]]:
    """Discover supported physical devices from pyAML device definitions.

    Class identity and, where needed, the terminal integrated-strength
    component determine the role.  Array names are deliberately not used as
    the primary classifier, so custom machine group names remain supported.
    """
    inventories={role:[] for role in MEASUREMENT_ROLES+MACHINE_MAGNET_ROLES}
    metadata={}
    class_roles=(
        ("hcorrector","horizontal_corrector"),("vcorrector","vertical_corrector"),
        ("skewquad","skew_quadrupole"),("quadrupole","normal_quadrupole"),
        ("sextupole","sextupole"),("octupole","octupole"),
        ("dipole","dipole"),("bpm","bpm"),
    )
    suffix_roles={"B1L":"horizontal_corrector","A1L":"vertical_corrector",
                  "B2L":"normal_quadrupole","A2L":"skew_quadrupole",
                  "B3L":"sextupole","B4L":"octupole"}
    for device in configuration.get("devices",()) or ():
        if not isinstance(device,dict):continue
        name=str(device.get("name","") or ""); class_name=str(device.get("class","") or "").lower()
        if not name:continue
        role=next((value for token,value in class_roles if token in class_name),None)
        component=name.rsplit("-",1)[-1].upper() if "-" in name else ""
        if role is None:role=suffix_roles.get(component)
        if role is None:continue
        if name not in inventories[role]:inventories[role].append(name)
        model=device.get("model") if isinstance(device.get("model"),dict) else {}
        metadata[(role,name)]={"pyaml_class":str(device.get("class","") or ""),
            "configured_unit":str(model.get("unit","") or ""),
            "hardware_unit":str(model.get("hardware_unit","") or ""),
            "configured_component":component if component in suffix_roles else ""}
    return inventories,metadata


def complete_template(ring, measurement_inventory: dict[str,list[str]],
                      control_inventory: dict[str,list[str]]|None=None,
                      control_metadata: dict[tuple[str,str],dict[str,Any]]|None=None) -> dict[str,Any]:
    """Create a mapping proposal for every configured pyAML physical device.

    The lattice is a matching target, never an inventory source: AT elements
    absent from the selected pyAML configuration are intentionally omitted.
    """
    controls=control_inventory or {}; metadata=control_metadata or {}
    document=template(measurement_inventory,controls)
    for row in document["mappings"]:
        row.update(metadata.get((row["role"],row["pyaml_device"]),{}))
        hits,method=_exact_lattice_match(ring,row["pyaml_device"],row["role"])
        if len(hits)==1:
            ordinal=hits[0]; element=ring[ordinal]
            row.update(lattice_ordinal=ordinal,lattice_fam_name=str(getattr(element,"FamName","") or ""),
                       lattice_device=str(getattr(element,"Device","") or ""),
                       lattice_common_name=str(getattr(element,"CommonName","") or ""),
                       physical_element_type=type(element).__name__,identity_match=method,
                       status="proposed_unique_exact_match")
        else:row.update(status="ambiguous" if hits else "unresolved",candidate_count=len(hits),
                        candidate_ordinals=list(hits))
    document["scope"]="complete_machine"
    document["configured_inventory"]={"counts":{role:len(names) for role,names in
        {**measurement_inventory,**controls}.items()},"total":len(document["mappings"])}
    return document


def is_document(document: dict[str,Any]) -> bool:
    return document.get("schema")==SCHEMA and document.get("schema_version")==VERSION


def is_template(document: dict[str,Any]) -> bool:
    return is_document(document) and document.get("template") is True


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
                 lattice_ordinal=row.get("lattice_ordinal"),lattice_device=row.get("lattice_device",""),
                 lattice_common_name=row.get("lattice_common_name",""),
                 physical_element_type=row.get("physical_element_type",""),control_name=row.get("pyaml_device",""),
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


def reconcile_configured(document: dict[str,Any]|None, configured: dict[str,Any], *, ring=None) -> dict[str,Any]:
    """Keep verified values for devices still present in the active config.

    Rows absent from the selected pyAML configuration are deliberately not
    carried forward. Newly configured devices retain the proposal produced
    from the active lattice and remain unresolved when no unique match exists.
    """
    result=deepcopy(configured)
    if not document or not is_document(document):return result
    _require_same_machine(document,configured)
    template_source=is_template(document)
    previous={(row.get("role"),row.get("pyaml_device")):row
              for row in document.get("mappings",()) if row.get("pyaml_device")}
    reconciled=[]
    for row in result.get("mappings",()):
        prior=previous.get((row.get("role"),row.get("pyaml_device")),{})
        supplied=prior.get("lattice_ordinal")
        if template_source and type(supplied) is int and ring is not None:
            # Device name and role select the configured row.  For a filled
            # template, an in-range zero-based ordinal is authoritative.
            row=deepcopy(row)
            if 0 <= supplied < len(ring):
                element=ring[supplied]
                row.update(lattice_ordinal=supplied,
                    lattice_fam_name=str(getattr(element,"FamName","") or ""),
                    lattice_device=str(getattr(element,"Device","") or ""),
                    lattice_common_name=str(getattr(element,"CommonName","") or ""),
                    physical_element_type=type(element).__name__,
                    identity_match="explicit_lattice_ordinal",
                    identity_evidence=prior.get("identity_evidence","") or
                        f"Explicit template ordinal {supplied} accepted within the active lattice.",
                    status="proposed_explicit_ordinal")
            else:
                row.update(lattice_ordinal=None,requested_lattice_ordinal=supplied,
                    status="invalid_explicit_ordinal",
                    identity_evidence=(
                        f"Explicit template ordinal {supplied} is outside the active "
                        f"{len(ring)}-element lattice for {row.get('role','device')} "
                        f"{row.get('pyaml_device','')!r}."
                    ))
        elif row.get("status")=="ambiguous" and type(supplied) is int \
                and prior["lattice_ordinal"] in row.get("candidate_ordinals",()):
            row=deepcopy(row); row["lattice_ordinal"]=prior["lattice_ordinal"]
            row["lattice_fam_name"]=prior.get("lattice_fam_name","")
            row["lattice_device"]=prior.get("lattice_device","")
            row["physical_element_type"]=prior.get("physical_element_type","")
            row["identity_match"]=prior.get("identity_match","manual_exact_disambiguation")
            row["identity_evidence"]=prior.get("identity_evidence","")
            row["status"]="proposed_manual_exact_disambiguation"
        reconciled.append(row)
    result["mappings"]=reconciled
    # Active configuration/lattice metadata is authoritative after a merge.
    for key in ("scope","configured_inventory"):
        if key in configured:result[key]=deepcopy(configured[key])
    return result


def merge_measurement(document: dict[str,Any]|None, validated: dict[str,Any]) -> dict[str,Any]:
    """Replace only measurement roles; preserve every correction-role entry."""
    base=deepcopy(document) if document and is_document(document) else {"schema":SCHEMA,"schema_version":VERSION,"mappings":[]}
    _require_same_machine(base,validated)
    preserved=[row for row in base.get("mappings",()) if row.get("role") not in MEASUREMENT_ROLES]
    previous={(row.get("role"),row.get("pyaml_device")):row for row in base.get("mappings",())}
    converted=[]
    for row in validated.get("mappings",()):
        identity=(row["role"],row.get("measured_name","")); prior=previous.get(identity,{})
        metadata={key:prior[key] for key in ("pyaml_class","configured_unit","hardware_unit","configured_component") if key in prior}
        converted.append(entry(role=row["role"],pyaml_device=identity[1],
            lattice_ordinal=row.get("lattice_ordinal"),lattice_fam_name=row.get("lattice_name",""),
            lattice_device=row.get("lattice_device",""),physical_element_type=row.get("element_class",""),component=row.get("component"),unit=row.get("unit"),
            identity_match=row.get("identity_match",""),status=row.get("status",""),
            evidence=row.get("reason",""),identity_evidence=row.get("identity_evidence",""),**metadata))
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
        lattice_device=row.get("lattice_device",""),physical_element_type=row.get("physical_element_type",row.get("element_class","")),component=row.get("component"),unit=row.get("unit"),
        status="mapped",evidence=row.get("provenance","validated by pyLOCO Correct")) for row in bundle.get("mappings",())]
    base.update(schema=SCHEMA,schema_version=VERSION,template=False,
                reference_lattice=deepcopy(bundle.get("reference_lattice",{})),
                machine_profile=deepcopy(bundle.get("machine_profile",{})),
                mapping_provenance=bundle.get("mapping_provenance","validated by pyLOCO Correct"))
    inventories=deepcopy(base.get("inventories",{})); inventories["correction"]=deepcopy(bundle.get("inventory",{})); base["inventories"]=inventories
    base["mappings"]=preserved+converted
    return base
