"""Validated measurement-device to AT-lattice identity mappings.

This module is deliberately independent of machine writes and of Correct's
quadrupole mapping.  It shares the same safety model: source identities,
reference lattice, machine configuration, and inventory are revalidated; a
stored status string is never trusted.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "pyloco.measurement_device_mapping"
VERSION = 1
ROLES = ("bpm", "horizontal_corrector", "vertical_corrector")
ROLE_COMPONENT = {"bpm": "", "horizontal_corrector": "B1L", "vertical_corrector": "A1L"}
STATUS_VALUES = frozenset({"mapped", "unresolved", "ambiguous", "duplicate", "invalid"})


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory_fingerprint(inventory: dict[str, list[str]]) -> str:
    payload = [(role, list(inventory.get(role, ()))) for role in ROLES]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def measurement_inventory(payload: dict[str, Any]) -> dict[str, list[str]]:
    files = payload.get("files", payload)
    orm = files.get("orm", {})
    return {
        "bpm": list(orm.get("bpm_names") or ()),
        "horizontal_corrector": list(orm.get("horizontal_corrector_names") or ()),
        "vertical_corrector": list(orm.get("vertical_corrector_names") or ()),
    }


def _element_identity(element: Any) -> tuple[str, str]:
    name = str(getattr(element, "FamName", None) or getattr(element, "Name", None)
               or getattr(element, "name", ""))
    return name, type(element).__name__


def _device_lookup(ring) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for ordinal, element in enumerate(ring):
        device = getattr(element, "Device", None)
        if device not in (None, ""):
            result.setdefault(str(device), []).append(ordinal)
    return result


def _device_identity(measured_name: str, role: str) -> str | None:
    """Return the exact lattice Device represented by an EBS control name.

    BPM names are Device identities directly.  Steering names are the exact
    Device followed by the role's explicit integrated component.  No fuzzy or
    positional matching is performed.
    """
    if role == "bpm":
        return measured_name
    component = ROLE_COMPONENT[role]
    suffix = f"-{component}"
    return measured_name[:-len(suffix)] if measured_name.endswith(suffix) else None


def _role_compatibility(element: Any, role: str) -> tuple[bool, str]:
    if role == "bpm":
        return True, "exact BPM Device identity"
    from pyLOCO.response_matrix import corrector_steering_compatibility
    plane = 0 if role == "horizontal_corrector" else 1
    compatible, representation = corrector_steering_compatibility(element, plane, calculator="Linear")
    return bool(compatible), str(representation)


def find_exact_device_mappings(ring, inventory: dict[str, list[str]], *,
                               lattice_path: str | Path,
                               configuration_path: str | Path | None = None,
                               provenance: str = "exact AT Device identity") -> dict[str, Any]:
    """Build candidates using exact AT ``Device`` identities only."""
    lookup = _device_lookup(ring)
    rows: list[dict[str, Any]] = []
    used: dict[tuple[str, int], int] = {}
    for role in ROLES:
        component = ROLE_COMPONENT[role]
        for position, measured_name in enumerate(inventory.get(role, ())):
            device = _device_identity(str(measured_name), role)
            hits = [] if device is None else lookup.get(device, [])
            status = "mapped" if len(hits) == 1 else "ambiguous" if len(hits) > 1 else "unresolved"
            ordinal = hits[0] if len(hits) == 1 else None
            reason = "Exact AT Device identity" if status == "mapped" else (
                "More than one AT element has this Device identity" if status == "ambiguous"
                else f"Expected exact Device plus -{component}" if component and device is None
                else "No AT element has this exact Device identity"
            )
            if ordinal is not None:
                compatible, representation = _role_compatibility(ring[ordinal], role)
                if not compatible:
                    status, reason = "invalid", f"AT element cannot represent {component}: {representation}"
                else:
                    reason = f"Exact AT Device identity; {representation}"
                key = (role, ordinal)
                if status == "mapped" and key in used:
                    status, reason = "duplicate", "AT element is assigned more than once in this role"
                if status == "mapped":
                    used[key] = position
                lattice_name, element_class = _element_identity(ring[ordinal])
                lattice_device = str(getattr(ring[ordinal], "Device", ""))
            else:
                lattice_name = element_class = lattice_device = ""
            rows.append({
                "measured_name": str(measured_name), "role": role,
                "lattice_name": lattice_name, "lattice_ordinal": ordinal,
                "lattice_device": lattice_device, "element_class": element_class,
                "component": component, "unit": "rad" if component else "m",
                "status": status, "reason": reason, "provenance": provenance,
            })
    lattice_path = Path(lattice_path).expanduser().resolve()
    configuration = Path(configuration_path).expanduser().resolve() if configuration_path else None
    return {
        "schema": SCHEMA, "schema_version": VERSION,
        "reference_lattice": {"path": str(lattice_path), "sha256": file_sha256(lattice_path),
                              "element_count": len(ring)},
        "machine_profile": {"configuration_path": str(configuration) if configuration else "",
                            "configuration_sha256": file_sha256(configuration) if configuration and configuration.is_file() else ""},
        "inventory": {"sha256": inventory_fingerprint(inventory),
                      "counts": {role: len(inventory.get(role, ())) for role in ROLES}},
        "mapping_provenance": provenance, "mappings": rows,
    }


def mapping_template(inventory: dict[str, list[str]]) -> dict[str, Any]:
    return {
        "description": "Fill in only lattice_ordinal for unresolved devices. pyLOCO validates every identity when loaded.",
        "schema": SCHEMA, "schema_version": VERSION,
        "user_field": "lattice_ordinal",
        "derived_on_load": ["lattice_name", "lattice_device", "element_class", "component", "unit", "status"],
        "mappings": [
            {"measured_name": name, "role": role, "lattice_ordinal": None}
            for role in ROLES for name in inventory.get(role, ())
        ],
    }


def load_document(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser()
    if source.suffix.lower() in {".yaml", ".yml"}:
        import yaml
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
    else:
        data = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(data, list):
        data = {"mappings": data}
    if not isinstance(data, dict) or not isinstance(data.get("mappings"), list):
        raise ValueError("Mapping file must contain a mappings list.")
    return data


def validate_mapping(document: dict[str, Any], ring, inventory: dict[str, list[str]], *,
                     lattice_path: str | Path,
                     configuration_path: str | Path | None = None,
                     require_provenance: bool = True) -> dict[str, Any]:
    """Rebuild trusted rows from source fields; never trust saved status."""
    lattice_path = Path(lattice_path).expanduser().resolve()
    configuration = Path(configuration_path).expanduser().resolve() if configuration_path else None
    if require_provenance:
        if document.get("schema") != SCHEMA or document.get("schema_version") != VERSION:
            raise ValueError("Unsupported measurement-device mapping schema/version.")
        reference = document.get("reference_lattice") or {}
        if reference.get("sha256") != file_sha256(lattice_path) or reference.get("element_count") != len(ring):
            raise ValueError("Mapping belongs to a different reference lattice.")
        profile = document.get("machine_profile") or {}
        if configuration and configuration.is_file() and profile.get("configuration_sha256") != file_sha256(configuration):
            raise ValueError("Mapping belongs to a different machine/control-system configuration.")
        saved_inventory = document.get("inventory") or {}
        if saved_inventory.get("sha256") != inventory_fingerprint(inventory):
            raise ValueError("Mapping device inventory differs from the current measurement inventory.")
    source_rows = document.get("mappings") or []
    keyed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in source_rows:
        role, name = str(row.get("role", "")), str(row.get("measured_name", ""))
        if role not in ROLES or not name:
            raise ValueError("Each mapping needs a supported role and measured_name.")
        if (role, name) in keyed:
            raise ValueError(f"Duplicate mapping source: {role} {name!r}.")
        keyed[(role, name)] = row
    rows = []
    targets = set()
    for role in ROLES:
        component = ROLE_COMPONENT[role]
        for name in inventory.get(role, ()):
            source = keyed.get((role, name))
            ordinal = source.get("lattice_ordinal") if source else None
            status, reason = "unresolved", "No mapping entry"
            lattice_name = lattice_device = element_class = ""
            if type(ordinal) is int and 0 <= ordinal < len(ring):
                lattice_name, element_class = _element_identity(ring[ordinal])
                lattice_device = str(getattr(ring[ordinal], "Device", ""))
                expected_device = _device_identity(name, role)
                if expected_device is None:
                    status, reason = "invalid", "Measured name does not expose the role's authoritative Device identity"
                elif lattice_device != expected_device:
                    status, reason = "invalid", "Selected AT element does not have the authoritative Device identity"
                elif not _role_compatibility(ring[ordinal], role)[0]:
                    compatible, representation = _role_compatibility(ring[ordinal], role)
                    status, reason = "invalid", f"AT element cannot represent {component}: {representation}"
                elif (role, ordinal) in targets:
                    status, reason = "duplicate", "AT element is assigned more than once in this role"
                else:
                    status, reason = "mapped", "Source and AT identity revalidated"
                    targets.add((role, ordinal))
            elif ordinal is not None:
                status, reason = "invalid", "Lattice ordinal is not a valid zero-based AT element index"
            rows.append({"measured_name": name, "role": role, "lattice_name": lattice_name,
                         "lattice_ordinal": ordinal if type(ordinal) is int else None,
                         "lattice_device": lattice_device, "element_class": element_class,
                         "component": component, "unit": "rad" if component else "m",
                         "status": status, "reason": reason,
                         "provenance": str(source.get("provenance", "manual")) if source else ""})
    # Extra source rows must not be silently ignored.
    wanted = {(role, name) for role in ROLES for name in inventory.get(role, ())}
    extras = set(keyed) - wanted
    if extras:
        role, name = sorted(extras)[0]
        raise ValueError(f"Mapping contains a device absent from the measurement inventory: {role} {name!r}.")
    result = find_exact_device_mappings(ring, {role: [] for role in ROLES}, lattice_path=lattice_path,
                                        configuration_path=configuration,
                                        provenance=str(document.get("mapping_provenance", "manual mapping")))
    result["inventory"] = {"sha256": inventory_fingerprint(inventory),
                           "counts": {role: len(inventory.get(role, ())) for role in ROLES}}
    result["mappings"] = rows
    return result


def mapping_counts(document: dict[str, Any]) -> dict[str, int]:
    counts = {status: 0 for status in STATUS_VALUES}
    for row in document.get("mappings", ()):
        status = row.get("status", "unresolved")
        counts[status if status in counts else "invalid"] += 1
    counts["total"] = sum(counts.values())
    return counts


def ordered_ordinals(document: dict[str, Any], role: str) -> list[int] | None:
    rows = [row for row in document.get("mappings", ()) if row.get("role") == role]
    if not rows or any(row.get("status") != "mapped" or type(row.get("lattice_ordinal")) is not int for row in rows):
        return None
    return [row["lattice_ordinal"] for row in rows]
