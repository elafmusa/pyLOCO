"""Validated measurement-device to AT-lattice identity mappings.

This module is deliberately independent of machine writes and of Correct's
quadrupole mapping.  It shares the same safety model: source identities,
reference lattice, machine configuration, and inventory are revalidated; a
stored status string is never trusted.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


SCHEMA = "pyloco.measurement_device_mapping"
VERSION = 1
ROLES = ("bpm", "horizontal_corrector", "vertical_corrector")
ROLE_COMPONENT = {"bpm": "", "horizontal_corrector": "B1L", "vertical_corrector": "A1L"}
STATUS_VALUES = frozenset({"mapped", "unresolved", "ambiguous", "duplicate", "invalid"})
ROLE_LABEL = {"bpm": "BPM", "horizontal_corrector": "H corrector", "vertical_corrector": "V corrector"}


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


def _device_lookup(ring) -> dict[str, dict[str, list[int]]]:
    result: dict[str, dict[str, list[int]]] = {"device": {}, "fam_name": {}, "ebs_alias": {}}
    for ordinal, element in enumerate(ring):
        device = getattr(element, "Device", None)
        if device not in (None, ""):
            result["device"].setdefault(str(device), []).append(ordinal)
        family = getattr(element, "FamName", None)
        if family not in (None, ""):
            result["fam_name"].setdefault(str(family), []).append(ordinal)
        alias = _ebs_fam_cell_alias(element)
        if alias:
            result["ebs_alias"].setdefault(alias, []).append(ordinal)
    return result


def _ebs_fam_cell_alias(element: Any) -> str:
    """Build the measured EBS ``FamName-Cxx`` identity from two AT fields."""
    family = str(getattr(element, "FamName", "") or "")
    device = str(getattr(element, "Device", "") or "")
    match = re.search(r"/c(\d{2})-([a-z])(?:$|/)", device, flags=re.IGNORECASE)
    if not re.fullmatch(r"[A-Z]{2}\d[A-Z]", family, flags=re.IGNORECASE) or not match:
        return ""
    # The final FamName letter and Device subcell must agree; this prevents a
    # coincidental cell-number match from becoming an identity.
    if family[-1].upper() != match.group(2).upper():
        return ""
    return f"{family.upper()}-C{match.group(1)}"


def _device_identities(measured_name: str, role: str) -> tuple[tuple[str, str], ...]:
    """Return supported role-specific suffix interpretations.

    ``-B1L``/``-A1L`` are component conventions and ``-H``/``-V`` are the
    plane conventions used by older EBS measurement files.  Exact identity is
    always attempted before either compatibility interpretation.
    """
    suffixes = {
        "horizontal_corrector": (("legacy_component", "-B1L"), ("plane_suffix", "-H")),
        "vertical_corrector": (("legacy_component", "-A1L"), ("plane_suffix", "-V")),
    }.get(role, ())
    return tuple(
        (method, measured_name[:-len(suffix)])
        for method, suffix in suffixes
        if measured_name.endswith(suffix) and len(measured_name) > len(suffix)
    )


def _identity_candidates(measured_name: str, role: str) -> tuple[tuple[str, str], ...]:
    """Return exact identity comparisons in documented precedence order."""
    candidates = [("exact_device", measured_name), ("exact_fam_name", measured_name)]
    for interpretation, identity in _device_identities(measured_name, role):
        candidates.extend(((f"{interpretation}_device", identity),
                           (f"{interpretation}_fam_name", identity)))
    return tuple(candidates)


def _identity_hits(lookup: dict[str, dict[str, list[int]]], measured_name: str,
                   role: str) -> tuple[list[int], str]:
    for method, identity in _identity_candidates(measured_name, role):
        namespace = "device" if method.endswith("device") else "fam_name"
        hits = lookup[namespace].get(identity, [])
        if hits:return list(hits), method
    for interpretation, identity in _device_identities(measured_name, role):
        if interpretation == "plane_suffix":
            hits = lookup["ebs_alias"].get(identity.upper(), [])
            if hits:return list(hits), "ebs_fam_cell_alias"
    return [], ""


def _identity_match_for_element(measured_name: str, role: str, element: Any) -> tuple[str, str]:
    family = str(getattr(element, "FamName", "") or "")
    device = str(getattr(element, "Device", "") or "")
    for method, identity in _identity_candidates(measured_name, role):
        actual = device if method.endswith("device") else family
        if actual and actual == identity:return method, identity
    alias = _ebs_fam_cell_alias(element)
    if alias:
        for interpretation, identity in _device_identities(measured_name, role):
            if interpretation == "plane_suffix" and alias == identity.upper():
                return "ebs_fam_cell_alias", identity
    return "", ""


def _match_label(method: str) -> str:
    return {
        "exact_device": "exact AT Device",
        "exact_fam_name": "exact AT FamName",
        "legacy_component_device": "legacy component suffix interpreted, then exact AT Device",
        "legacy_component_fam_name": "legacy component suffix interpreted, then exact AT FamName",
        "plane_suffix_device": "H/V plane suffix interpreted, then exact AT Device",
        "plane_suffix_fam_name": "H/V plane suffix interpreted, then exact AT FamName",
        "ebs_fam_cell_alias": "EBS FamName plus exact Device cell identity",
    }.get(method, "no exact identity match")


def _role_compatibility(element: Any, role: str) -> tuple[bool, str]:
    if role == "bpm":
        return True, "exact BPM Device identity"
    from pyLOCO.response_matrix import corrector_steering_compatibility
    plane = 0 if role == "horizontal_corrector" else 1
    compatible, representation = corrector_steering_compatibility(element, plane, calculator="Linear")
    return bool(compatible), str(representation)


def _candidate_description(ring, hits: list[int], role: str) -> str:
    details = []
    for index in hits:
        element = ring[index]
        family, element_class = _element_identity(element)
        device = str(getattr(element, "Device", "") or "Not available")
        compatible, representation = _role_compatibility(element, role)
        details.append(
            f"AT index {index}: FamName={family!r}, Device={device!r}, "
            f"type={element_class}, capability={'supported' if compatible else 'not supported'} ({representation})"
        )
    return "; ".join(details)


def find_exact_device_mappings(ring, inventory: dict[str, list[str]], *,
                               lattice_path: str | Path,
                               configuration_path: str | Path | None = None,
                               provenance: str = "exact AT Device/FamName identity") -> dict[str, Any]:
    """Build candidates using exact AT Device/FamName identities only."""
    lookup = _device_lookup(ring)
    rows: list[dict[str, Any]] = []
    used: dict[tuple[str, int], int] = {}
    for role in ROLES:
        component = ROLE_COMPONENT[role]
        for position, measured_name in enumerate(inventory.get(role, ())):
            measured_name = str(measured_name)
            role_label = ROLE_LABEL[role]
            hits, match_method = _identity_hits(lookup, measured_name, role)
            status = "mapped" if len(hits) == 1 else "ambiguous" if len(hits) > 1 else "unresolved"
            ordinal = hits[0] if len(hits) == 1 else None
            reason = _match_label(match_method) if status == "mapped" else (
                f"Ambiguous identity for measured {role_label} {measured_name!r}: {_match_label(match_method)} "
                f"matches several candidates. {_candidate_description(ring, hits, role)}. Enter the intended "
                "AT index (0-based) and document authoritative identity evidence; the choice will be "
                "revalidated for exact identity and steering capability."
                if status == "ambiguous" else
                f"No exact identity match for measured {role_label} {measured_name!r}: no AT Device or "
                "FamName matched (role-compatible suffixes and the EBS FamName-plus-Device-cell identity were also checked)."
            )
            if ordinal is not None:
                lattice_name, element_class = _element_identity(ring[ordinal])
                lattice_device = str(getattr(ring[ordinal], "Device", ""))
                compatible, representation = _role_compatibility(ring[ordinal], role)
                if not compatible:
                    status, reason = "invalid", (
                        f"Identity passed via {_match_label(match_method)} for measured {role_label} "
                        f"{measured_name!r} at AT index {ordinal} (0-based; FamName={lattice_name!r}, "
                        f"Device={lattice_device!r}), but steering compatibility failed for "
                        f"{component or 'BPM'}: {representation}. Select an element that supports the "
                        "required steering plane."
                    )
                else:
                    reason = f"Identity: {_match_label(match_method)}; steering compatibility: {representation}"
                key = (role, ordinal)
                if status == "mapped" and key in used:
                    status, reason = "duplicate", (
                        f"AT index {ordinal} (0-based; FamName={lattice_name!r}, Device={lattice_device!r}) is "
                        f"assigned more than once to measured {role_label} devices. Choose unique elements."
                    )
                if status == "mapped":
                    used[key] = position
            else:
                lattice_name = element_class = lattice_device = ""
            rows.append({
                "measured_name": measured_name, "role": role,
                "lattice_name": lattice_name, "lattice_ordinal": ordinal,
                "lattice_device": lattice_device, "element_class": element_class,
                "component": component, "unit": "rad" if component else "m",
                "identity_match": match_method,
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
        "description": "Fill in lattice_ordinal (AT index, 0-based). For duplicate identities also record identity_evidence. pyLOCO revalidates every entry when loaded.",
        "schema": SCHEMA, "schema_version": VERSION,
        "user_fields": ["lattice_ordinal", "identity_evidence"],
        "derived_on_load": ["lattice_name", "lattice_device", "element_class", "component", "unit", "status"],
        "mappings": [
            {"measured_name": name, "role": role, "lattice_ordinal": None, "identity_evidence": ""}
            for role in ROLES for name in inventory.get(role, ())
        ],
    }


def fill_missing_mapping_entries(validated: dict[str, Any], discovered: dict[str, Any]) -> dict[str, Any]:
    """Fill absent source rows from a fresh identity-only discovery result.

    Existing loaded rows, including unresolved or manually evidenced rows, are
    preserved. Only rows produced as ``No mapping entry`` by validation are
    eligible for replacement, so a loaded mapping can never silently override
    an explicit operator choice.
    """
    result = {**validated, "mappings": [dict(row) for row in validated.get("mappings", ())]}
    candidates = {
        (str(row.get("role", "")), str(row.get("measured_name", ""))): row
        for row in discovered.get("mappings", ())
    }
    filled = 0
    for index, row in enumerate(result["mappings"]):
        if row.get("status") != "unresolved" or row.get("reason") != "No mapping entry":
            continue
        candidate = candidates.get((str(row.get("role", "")), str(row.get("measured_name", ""))))
        if candidate is not None:
            result["mappings"][index] = dict(candidate)
            filled += 1
    if filled:
        result["mapping_provenance"] = (
            f"{validated.get('mapping_provenance', 'loaded mapping')}; "
            f"{filled} missing row(s) discovered from exact active-lattice identities"
        )
    return result


def is_unvalidated_mapping_template(document: dict[str, Any]) -> bool:
    """Identify the editable template format without trusting it as provenance."""
    return (
        document.get("schema") == SCHEMA
        and document.get("schema_version") == VERSION
        and document.get("user_fields") == ["lattice_ordinal", "identity_evidence"]
        and isinstance(document.get("derived_on_load"), list)
        and "reference_lattice" not in document
        and "inventory" not in document
        and "machine_profile" not in document
    )


def python_mapping_helper_script(template_path: str | Path, lattice_path: str | Path,
                                 mat_variable: str, output_path: str | Path) -> str:
    """Return a readable external helper; pyLOCO never executes this text."""
    template = str(Path(template_path).expanduser()) if str(template_path).strip() else ""
    lattice = str(Path(lattice_path).expanduser()) if str(lattice_path).strip() else ""
    output = str(Path(output_path).expanduser()) if str(output_path).strip() else ""
    return f'''import json
import at

TEMPLATE_PATH = {template!r}
LATTICE_PATH = {lattice!r}
MAT_VARIABLE = {str(mat_variable or '')!r}
OUTPUT_PATH = {output!r}

ring = at.load_mat(LATTICE_PATH, use=MAT_VARIABLE) if MAT_VARIABLE else at.load_lattice(LATTICE_PATH)
with open(TEMPLATE_PATH, "r", encoding="utf-8") as stream:
    d = json.load(stream)
if not isinstance(d, dict) or not isinstance(d.get("mappings"), list):
    raise ValueError("Template must contain a JSON object with a mappings list")

def exact_hits(attribute, identity):
    return [i for i, element in enumerate(ring)
            if str(getattr(element, attribute, "") or "") == identity]

def find_hits(row):
    name = str(row.get("pyaml_device", row.get("measured_name", "")))
    role = str(row["role"])
    identities = [("Device", name), ("FamName", name)]
    suffixes = {{"horizontal_corrector": ("-B1L", "-H"),
                 "vertical_corrector": ("-A1L", "-V"),
                 "normal_quadrupole": ("-B2L",), "skew_quadrupole": ("-A2L",),
                 "sextupole": ("-B3L",), "octupole": ("-B4L",),
                 "dipole": ("-B1L",)}}.get(role, ())
    for suffix in suffixes:
        if name.endswith(suffix) and len(name) > len(suffix):
            base = name[:-len(suffix)]
            identities += [("Device", base), ("FamName", base)]
    for attribute, identity in identities:
        hits = exact_hits(attribute, identity)
        if hits:
            return hits
    return []

matched = 0
for row in d["mappings"]:
    # Current simple template rows contain only pyaml_device, role and
    # lattice_ordinal. Older measured_name templates remain accepted.
    if not row.get("pyaml_device") and not row.get("measured_name"):
        print(f"missing device name: {{row!r}}")
        row["lattice_ordinal"] = None
        continue
    if not row.get("role"):
        print(f"missing role: {{row!r}}")
        row["lattice_ordinal"] = None
        continue
    supplied = row.get("lattice_ordinal")
    if type(supplied) is int:
        if 0 <= supplied < len(ring):
            # An in-range explicit template ordinal is authoritative.
            row["lattice_ordinal"] = supplied
            matched += 1
            continue
        row["lattice_ordinal"] = None
        print(f"invalid ordinal: {{row['role']}} {{row.get('pyaml_device', row.get('measured_name'))}} "
              f"(AT index {{supplied}} is outside a {{len(ring)}}-element lattice)")
        continue
    hits = find_hits(row)
    if len(hits) == 1:
        row["lattice_ordinal"] = hits[0]
        matched += 1
    else:
        row["lattice_ordinal"] = None
        label = "missing" if not hits else "ambiguous"
        print(f"{{label}}: {{row['role']}} {{row.get('pyaml_device', row.get('measured_name'))}} ({{len(hits)}} matches)")

with open(OUTPUT_PATH, "w", encoding="utf-8") as stream:
    json.dump(d, stream, indent=2)
    stream.write("\\n")
print(f"Matched {{matched}} / {{len(d['mappings'])}} devices")
'''


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
                     require_provenance: bool = True,
                     accept_explicit_ordinals: bool = False) -> dict[str, Any]:
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
        role_label = ROLE_LABEL[role]
        for name in inventory.get(role, ()):
            source = keyed.get((role, name))
            ordinal = source.get("lattice_ordinal") if source else None
            status, reason = "unresolved", "No mapping entry"
            lattice_name = lattice_device = element_class = ""
            if type(ordinal) is int and 0 <= ordinal < len(ring):
                lattice_name, element_class = _element_identity(ring[ordinal])
                lattice_device = str(getattr(ring[ordinal], "Device", ""))
                explicit = accept_explicit_ordinals or source.get("identity_match") == "explicit_lattice_ordinal"
                match_method, _identity = (("explicit_lattice_ordinal", name) if explicit else
                    _identity_match_for_element(name, role, ring[ordinal]))
                compatible, representation = ((True, "operator-supplied lattice ordinal") if explicit else
                    _role_compatibility(ring[ordinal], role))
                evidence = str(source.get("identity_evidence", "")).strip() if source else ""
                if explicit:
                    status, reason = "mapped", (
                        f"Explicit zero-based AT index {ordinal} accepted from the loaded mapping for "
                        f"{role_label} {name!r}."
                    )
                else:
                    lookup = _device_lookup(ring)
                    identity_hits, _ = _identity_hits(lookup, name, role)
                    ambiguous_identity = len(identity_hits) > 1 and ordinal in identity_hits
                if not explicit and not match_method:
                    status, reason = "invalid", (
                        f"No exact identity match: measured {role_label} {name!r}; selected AT index {ordinal} (0-based); "
                        f"FamName={lattice_name!r}; Device={lattice_device!r}. Exact Device, exact FamName, "
                        "role-compatible suffixes, and the EBS FamName-plus-Device-cell identity all failed. Select the AT "
                        "element with the matching Device or FamName."
                    )
                elif not explicit and ambiguous_identity and not evidence:
                    status, reason = "ambiguous", (
                        f"The measured {role_label} identity {name!r} matches multiple AT elements. "
                        f"{_candidate_description(ring, identity_hits, role)}. Record authoritative "
                        "identity_evidence for the explicitly selected AT index; a manual index alone is not proof."
                    )
                elif not explicit and not compatible:
                    status, reason = "invalid", (
                        f"Identity passed via {_match_label(match_method)} for measured {role_label} {name!r} at AT "
                        f"index {ordinal} (0-based; FamName={lattice_name!r}, Device={lattice_device!r}), but steering "
                        f"compatibility failed for {component or 'BPM'}: {representation}. Select an element "
                        "that supports the required steering plane."
                    )
                elif not explicit and (role, ordinal) in targets:
                    status, reason = "duplicate", f"AT index {ordinal} (0-based) is assigned more than once in role {role}"
                elif not explicit:
                    status, reason = "mapped", f"Identity: {_match_label(match_method)}; steering compatibility: {representation}"
                    if evidence: reason += f"; operator identity evidence: {evidence}"
                    targets.add((role, ordinal))
            elif ordinal is not None:
                status, reason = "invalid", (
                    f"Invalid manual AT index for measured {role_label} {name!r}: {ordinal!r} is not a valid "
                    f"zero-based AT element index for this {len(ring)}-element lattice."
                )
            rows.append({"measured_name": name, "role": role, "lattice_name": lattice_name,
                         "lattice_ordinal": ordinal if type(ordinal) is int else None,
                         "lattice_device": lattice_device, "element_class": element_class,
                         "component": component, "unit": "rad" if component else "m",
                         "identity_match": match_method if type(ordinal) is int and 0 <= ordinal < len(ring) else "",
                         "status": status, "reason": reason,
                         "identity_evidence": str(source.get("identity_evidence", "")) if source else "",
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
