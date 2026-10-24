"""Individual quadrupole access and conservative lattice-identity matching."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import yaml

from .petra_readonly import load_mapping as load_machine_mapping, apply_explicit_mapping, MagnetMapping
from pyLOCO.machine_mapping import (correction_projection,
    is_document as is_shared_machine_mapping,
    is_template as is_shared_mapping_template)


FAMILIES = ("normal_quadrupole", "skew_quadrupole")
MAPPING_BUNDLE_SCHEMA = "pyloco.machine_mapping"
MAPPING_BUNDLE_VERSION = 2


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory_fingerprint(inventory):
    """Stable fingerprint of the identities exposed by a discovered inventory."""
    rows = sorted((str(row["name"]), str(row["family"]), str(row.get("lattice_name", "")))
                  for row in inventory)
    payload = json.dumps(rows, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_lattice(path):
    import at
    path = Path(path).expanduser().resolve()
    try:
        ring = at.load_lattice(path, use="betamodel")
    except Exception:
        ring = at.load_lattice(path)
    return path, ring


def lattice_identity(path):
    path, ring = _load_lattice(path)
    return dict(path=str(path), sha256=file_hash(path), element_count=len(ring)), ring


def mapping_document(path):
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, (dict, list)):
        raise ValueError("Mapping file must contain a JSON/YAML object or list")
    return data


class PyAMLCorrectionConnection:
    def __init__(self, config_path, *, mode="live", normal_array="Quads",
                 skew_array="SkewQuads", allow_writes=True, accelerator=None):
        if mode not in ("live", "design"):
            raise ValueError("Mode must be live or design")
        self.options = dict(config_path=str(Path(config_path).resolve()), mode=mode,
                            normal_array=normal_array, skew_array=skew_array)
        self.config_hash = file_hash(self.options["config_path"])
        if accelerator is None:
            from pyaml.accelerator import Accelerator
            accelerator = Accelerator.load(self.options["config_path"])
        self.accelerator = accelerator
        self.machine = getattr(accelerator, mode)
        if self.machine is None:
            raise ValueError(f"Configuration has no {mode} machine")
        self.allow_writes = bool(allow_writes)
        self.inventory = self.discover()

    def discover(self):
        inventory = []
        seen = set()
        for family, key in zip(FAMILIES, ("normal_array", "skew_array")):
            array_name = self.options[key].strip()
            if not array_name:
                raise ValueError(f"Specify the {key} name")
            array = self.machine.magnets.get(array_name)
            if array is None:
                raise ValueError(f"Missing pyAML array: {array_name}")
            for name in array.names():
                if name in seen:
                    raise ValueError(f"Duplicate pyAML magnet: {name}")
                seen.add(name)
                magnet = self.machine.magnet.get(name)
                identity = magnet.get_lattice_names() or name
                if isinstance(identity, str):
                    identities = [identity]
                else:
                    try: identities = [str(value) for value in identity if str(value)]
                    except TypeError: identities = [str(identity)]
                identities = list(dict.fromkeys(identities)) or [name]
                inventory.append(dict(name=name, family=family,
                                      lattice_name=identities[0], lattice_names=identities))
        return inventory

    def identity(self):
        if file_hash(self.options["config_path"]) != self.config_hash:
            raise RuntimeError("pyAML configuration changed; reconnect")
        inventory = self.discover()
        if inventory != self.inventory:
            raise RuntimeError("pyAML inventory changed; reconnect")
        return dict(options=self.options, config_hash=self.config_hash, inventory=inventory)

    def _strength(self, name):
        if name not in {row["name"] for row in self.inventory}:
            raise ValueError(f"Magnet outside configured arrays: {name}")
        return self.machine.magnet.get(name).strength

    def read(self, name):
        value = float(self._strength(name).get())
        if not math.isfinite(value):
            raise ValueError(f"Nonfinite strength for {name}")
        return value

    def write(self, name, value):
        if not self.allow_writes:
            raise PermissionError("pyAML magnet writes are disabled")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Magnet target must be finite")
        self._strength(name).set(value)


def item_identity(item):
    return (item.correction_type, item.lattice_ordinal, item.name)


def load_mapping(path):
    data = mapping_document(path)
    if isinstance(data,dict) and is_shared_machine_mapping(data):
        return correction_projection(data)
    entries = data.get("mappings") if isinstance(data, dict) else data
    if isinstance(entries, list) and entries and all(isinstance(row, dict) and "magnet_name" in row for row in entries):
        return entries
    return [dict(lattice_name=row.lattice_name, lattice_ordinal=row.lattice_ordinal,
                 control_name=row.control_name, component=row.component, unit=row.unit)
            for row in load_machine_mapping(path)]


def make_mapping_bundle(review, connection, entries, lattice_path, *, provenance):
    """Create a provenance-bound bundle from already validated mapping entries."""
    lattice, ring = lattice_identity(lattice_path)
    validated = validate_mapping_entries(review, connection, entries, ring)
    if any(row["included"] and row["status"] != "mapped" for row in validated):
        raise ValueError("Cannot save a reusable mapping while included corrections are unresolved")
    rows = []
    by_identity = {(item.correction_type, item.lattice_ordinal, item.name): item
                   for item in review.items}
    for entry in entries:
        family = _entry_family(entry, review)
        key = (family, entry.get("lattice_ordinal"), entry.get("lattice_name"))
        if key not in by_identity:
            continue
        rows.append(dict(
            correction_type=family,
            lattice_name=key[2], lattice_ordinal=key[1],
            lattice_device=str(getattr(ring[key[1]],"Device","") or ""),
            control_name=entry.get("control_name", entry.get("magnet_name")),
            component="B2L" if family == FAMILIES[0] else "A2L", unit="m^-1",
            provenance=str(provenance),
        ))
    return {
        "schema": MAPPING_BUNDLE_SCHEMA,
        "schema_version": MAPPING_BUNDLE_VERSION,
        "reference_lattice": lattice,
        "machine_profile": {
            "backend": "pyaml", "configuration_sha256": connection.config_hash,
            "mode": connection.options["mode"],
            "normal_array": connection.options["normal_array"],
            "skew_array": connection.options["skew_array"],
        },
        "inventory": {"sha256": inventory_fingerprint(connection.inventory),
                      "device_count": len(connection.inventory)},
        "mapping_provenance": str(provenance),
        "mappings": rows,
    }


def _entry_family(entry, review=None):
    family = entry.get("correction_type")
    if family:
        return family
    component = str(entry.get("component") or "").upper()
    if component == "B2L":
        return FAMILIES[0]
    if component == "A2L":
        return FAMILIES[1]
    if review is not None:
        matches = {item.correction_type for item in review.items
                   if item.name == entry.get("lattice_name")
                   and item.lattice_ordinal == entry.get("lattice_ordinal")}
        if len(matches) == 1:
            return matches.pop()
    raise ValueError(
        f"Mapping entry for {entry.get('lattice_name')!r} requires a supported B2L/A2L component"
    )


def validate_mapping_entries(review, connection, entries, ring, *, ordinal_authoritative=False):
    """Validate authoritative source identities before delegating to the strict resolver."""
    inventory = {row["name"]: row for row in connection.inventory}
    sources, targets = set(), set()
    selected = []
    wanted = {(item.correction_type, item.lattice_ordinal) for item in review.items}
    for original in entries:
        entry=dict(original)
        family = _entry_family(entry, review)
        ordinal = entry.get("lattice_ordinal")
        lattice_name = str(entry.get("lattice_name") or "")
        control_name = entry.get("control_name", entry.get("magnet_name"))
        if type(ordinal) is not int or not 0 <= ordinal < len(ring):
            raise ValueError(f"Invalid source lattice ordinal for {lattice_name!r}: {ordinal!r}")
        actual = str(getattr(ring[ordinal], "FamName", ""))
        if ordinal_authoritative:
            # A filled operator template identifies the source by its explicit
            # zero-based AT ordinal. FIT display names may differ from FamName.
            lattice_name=actual; entry["lattice_name"]=actual
        elif not lattice_name:
            lattice_name=actual; entry["lattice_name"]=actual
        if not ordinal_authoritative and actual != lattice_name:
            raise ValueError(
                f"Reference lattice identity mismatch at ordinal {ordinal}: "
                f"mapping has {lattice_name!r}, lattice has {actual!r}"
            )
        declared_device=str(entry.get("lattice_device") or "")
        actual_device=str(getattr(ring[ordinal],"Device","") or "")
        if not ordinal_authoritative and declared_device and declared_device!=actual_device:
            raise ValueError(f"Reference lattice Device mismatch at ordinal {ordinal}: mapping has {declared_device!r}, lattice has {actual_device!r}")
        source = (family, ordinal)
        if source in sources:
            raise ValueError(f"Duplicate mapping source: {source}")
        sources.add(source)
        if control_name in targets:
            raise ValueError(f"Duplicate mapping target: {control_name!r}")
        targets.add(control_name)
        target = inventory.get(control_name)
        if target is None or target["family"] != family:
            raise ValueError(f"Unknown or wrong-family mapping target: {control_name!r}")
        expected = "B2L" if family == FAMILIES[0] else "A2L"
        component = entry.get("component")
        if component and str(component).upper() != expected:
            raise ValueError(f"Unsupported pyAML component for {control_name}: {component}")
        unit = entry.get("unit")
        if unit and str(unit) not in {"m^-1", "m⁻¹"}:
            raise ValueError(f"Unsupported pyAML unit for {control_name}: {unit}")
        if source in wanted:
            selected.append(dict(entry, correction_type=family, component=expected, unit="m^-1"))
    rows = resolve_mapping(review, connection.inventory, selected)
    return rows


def validate_mapping_bundle(review, connection, path, lattice_path, *, operator_selected=False):
    """Load and revalidate a v2 bundle or authoritative legacy mapping."""
    document = mapping_document(path)
    entries = load_mapping(path)
    return validate_mapping_document(
        review, connection, document, lattice_path, entries=entries,
        operator_selected=operator_selected,
    )


def validate_mapping_document(review, connection, document, lattice_path, *, entries=None,
                              operator_selected=False):
    """Revalidate an in-memory FIT/Measure Machine Mapping.

    Loading a saved mapping is not machine authorization.  Its lattice,
    configuration, inventory, source identity, component and unit evidence is
    checked against the exact active connection here.
    """
    if entries is None:
        entries = correction_projection(document) if is_shared_machine_mapping(document) else list(
            document.get("mappings", ()) if isinstance(document, dict) else document or ())
    lattice, ring = lattice_identity(lattice_path)
    provenance = "authoritative legacy mapping — revalidated"
    if isinstance(document, dict) and document.get("schema") == MAPPING_BUNDLE_SCHEMA:
        shared=is_shared_machine_mapping(document)
        template=shared and is_shared_mapping_template(document)
        if not shared and document.get("schema_version") != MAPPING_BUNDLE_VERSION:
            raise ValueError("Unsupported reusable mapping-bundle version")
        if template or operator_selected:
            # Templates deliberately contain only device, role and ordinal.
            # The ordinal is accepted as the operator's source selection;
            # target family/component/unit are still checked below against
            # the connected pyAML inventory before any operation is enabled.
            provenance = (
                "explicitly selected Machine Mapping — AT ordinals and targets validated "
                "against active lattice and connected pyAML inventory"
                if operator_selected else
                "explicit minimal template — validated against connected pyAML inventory"
            )
        else:
            expected_lattice = document.get("reference_lattice", {})
            if expected_lattice.get("sha256") != lattice["sha256"] or expected_lattice.get("element_count") != lattice["element_count"]:
                raise ValueError("Reusable mapping belongs to a different reference lattice")
            profile = document.get("machine_profile", {})
            expected_options = {key: connection.options[key] for key in ("mode", "normal_array", "skew_array")}
            if profile.get("configuration_sha256") != connection.config_hash or any(k in profile and profile.get(k) != v for k, v in expected_options.items()):
                raise ValueError("Reusable mapping belongs to a different machine configuration/profile")
            inventory = document.get("inventories",{}).get("correction",{}) if shared else document.get("inventory", {})
            if inventory and (inventory.get("sha256") != inventory_fingerprint(connection.inventory) or inventory.get("device_count") != len(connection.inventory)):
                raise ValueError("Reusable mapping inventory has changed; revalidation is required")
            provenance = "revalidated saved mapping"
    rows = validate_mapping_entries(
        review, connection, entries, ring,
        ordinal_authoritative=(template or operator_selected)
    )
    unresolved = [row for row in rows if row["included"] and row["status"] != "mapped"]
    if unresolved:
        raise ValueError(f"Authoritative mapping leaves {len(unresolved)} included correction(s) unresolved")
    selected = []
    wanted = {(item.correction_type, item.lattice_ordinal) for item in review.items}
    for entry in entries:
        family = _entry_family(entry, review)
        ordinal=entry.get("lattice_ordinal"); lattice_name=entry.get("lattice_name")
        if not lattice_name and type(ordinal) is int and 0<=ordinal<len(ring):lattice_name=str(getattr(ring[ordinal],"FamName","") or "")
        key = (family, ordinal)
        if key in wanted:
            selected.append(dict(entry, lattice_name=lattice_name, correction_type=family,
                                 component="B2L" if family == FAMILIES[0] else "A2L",
                                 unit="m^-1"))
    return selected, rows, provenance, lattice


def find_reference_lattice(review, connection):
    """Find an existing reference lattice without constructing or modifying one."""
    candidates = []
    for item in review.items[:1]:
        for key in ("reference_lattice_path", "lattice_path", "lattice_file"):
            if item.metadata.get(key):
                candidates.append(Path(item.metadata[key]).expanduser())
        results = item.metadata.get("source_results_directory")
        if results:
            try:
                from pyLOCO.gui.results.results_loader import ResultsLoader
                loader = ResultsLoader(results)
                value = loader.request.get("lattice_path")
                if value:
                    candidates.append(Path(loader._resolve_reference(value)))
            except Exception:
                pass
    config = Path(connection.options["config_path"])
    candidates.append(config.with_name("betamodel.mat"))
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate.resolve()
    raise ValueError(
        "Reference lattice unavailable. Load FIT results that retain their lattice provenance "
        "or use a machine profile with an adjacent betamodel.mat file."
    )


def approved_mapping_candidates(connection, remembered=()):
    """Return explicit profile/remembered mapping candidates; never infer names."""
    config = Path(connection.options["config_path"])
    values = [Path(value).expanduser() for value in remembered if value]
    try:
        document = yaml.safe_load(config.read_text())
        configured = document.get("pyloco_mapping") if isinstance(document, dict) else None
        if configured:
            path = Path(configured).expanduser()
            values.append(path if path.is_absolute() else config.parent / path)
    except Exception:
        pass
    values.extend((config.with_name("pyaml_pysc_mapping.json"),
                   config.with_name(f"{config.stem}_mapping.json"),
                   config.with_name(f"{config.stem}.mapping.json")))
    result = []
    for value in values:
        try:
            resolved = value.resolve()
        except Exception:
            continue
        if resolved.is_file() and resolved not in result:
            result.append(resolved)
    return result


def mapping_template(review):
    """Return the minimal user-editable mapping interchange format.

    The fitted lattice identity is copied from the reviewed FIT result.  Control
    component, units and validation status are deliberately absent: they are
    derived and checked against the connected device inventory by
    :func:`resolve_mapping` when the file is loaded.
    """
    return {"mappings": [dict(
        lattice_ordinal=item.lattice_ordinal, lattice_name=item.name,
        control_name=item.control_name or "")
        for item in review.items if item.correction_type in FAMILIES]}


def resolve_mapping(review, inventory, explicit=()):
    targets = {row["name"]: row for row in inventory}
    identities = defaultdict(list)
    for entry in inventory:
        lattice_names = entry.get("lattice_names") or (entry.get("lattice_name"),)
        if isinstance(lattice_names, str): lattice_names = (lattice_names,)
        # EBS pyAML controls commonly append the integrated-strength
        # component to the exact AT Device identity.  Treat only that
        # terminal, role-specific component as an additional exact identity.
        control_name=str(entry.get("name") or "")
        suffix="-B2L" if entry.get("family")==FAMILIES[0] else \
               "-A2L" if entry.get("family")==FAMILIES[1] else ""
        configured_identities=list(lattice_names)
        if suffix and control_name.endswith(suffix):
            configured_identities.append(control_name[:-len(suffix)])
        for identity in configured_identities:
            if isinstance(identity, str) and identity and not any(char in identity for char in "*?[](),@#"):
                key=(entry["family"],identity)
                if entry["name"] not in identities[key]:identities[key].append(entry["name"])
    overrides = {}
    used = set()
    for row in explicit:
        name = row.get("control_name", row.get("magnet_name"))
        ordinal = row.get("lattice_ordinal")
        family = row.get("correction_type")
        if type(ordinal) is int:
            # Explicit AT ordinal is the authoritative source identity. Do not
            # also require the FIT display name to equal the lattice FamName.
            candidates = [item for item in review.items
                          if item.lattice_ordinal == ordinal
                          and (not family or item.correction_type == family)]
        else:
            candidates = [item for item in review.items
                          if item.name == row["lattice_name"]
                          and (not family or item.correction_type == family)]
        if len(candidates) != 1:
            raise ValueError(f"Mapping source {row['lattice_name']!r} does not identify exactly one fitted lattice element; check lattice_name and lattice_ordinal")
        key = item_identity(candidates[0])
        if key in overrides:
            raise ValueError(f"Duplicate mapping source for {key}; each fitted element may appear only once")
        if name not in targets or targets[name]["family"] != key[0]:
            raise ValueError(f"Unknown or wrong-family mapping target: {name!r}. Select an exact {key[0].replace('_', ' ')} device name from the connected pyAML inventory")
        component = row.get("component")
        expected_component = "B2L" if key[0] == FAMILIES[0] else "A2L"
        if component and str(component).upper() != expected_component:
            raise ValueError(f"Unsupported pyAML component for {name}: {component}. This backend requires integrated {expected_component} controls")
        unit = row.get("unit")
        if unit and str(unit) not in {"m^-1", "m⁻¹"}:
            raise ValueError(f"Unsupported pyAML unit for {name}: {unit}. Integrated {expected_component} controls require m^-1")
        if name in used:
            raise ValueError(f"Duplicate mapping target: {name!r}; one pyAML device cannot represent two fitted elements")
        matched = deepcopy(candidates[0])
        # The source was already resolved above (ordinal first when present),
        # so do not pass it through the legacy name-first mapper again.
        matched.control_name = name
        matched.metadata["mapping_status"] = "mapped"
        matched.metadata["control_component"] = component or expected_component
        matched.metadata["control_unit"] = row.get("unit") or "m^-1"
        overrides[key] = name
        used.add(name)
    rows = []
    for item in review.items:
        key = item_identity(item)
        candidate_identities = list(dict.fromkeys(str(value) for value in (
            item.name,
            item.metadata.get("lattice_fam_name"),
            item.metadata.get("lattice_device"),
            item.metadata.get("at_device"),
            item.metadata.get("lattice_common_name"),
        ) if value))
        matches = list(dict.fromkeys(
            name for identity in candidate_identities
            for name in identities[(item.correction_type, identity)]
        ))
        name = overrides.get(key)
        source = "explicit" if name else "identity"
        if not name and len(matches) == 1:
            name = matches[0]
        status = "mapped" if name else "ambiguous" if len(matches) > 1 else "unmapped"
        if item.correction_type not in FAMILIES:
            status = "unsupported"
        rows.append(dict(correction_type=item.correction_type, lattice_ordinal=item.lattice_ordinal,
                         lattice_name=item.name, magnet_name=name, match_source=source,
                         status=status, included=item.included, length=item.magnetic_length_m,
                         delta=item.final_delta, individual_scale=item.individual_scale))
    counts = Counter(row["magnet_name"] for row in rows)
    for row in rows:
        if row["magnet_name"] and counts[row["magnet_name"]] > 1:
            row["status"] = "duplicate"
    return rows


def review_identity(review, rows):
    source = Path(review.source_result).resolve()
    if not source.exists():
        raise ValueError(f"Correction source is unavailable: {source}")
    paths = [source] if source.is_file() else sorted(
        path for path in source.rglob("*") if path.is_file()
        and path.suffix in {".json", ".yaml", ".yml", ".csv", ".mat", ".npz", ".npy", ".pkl"}
    ) if source.is_dir() else []
    return dict(source=str(source), files={str(path): file_hash(path) for path in paths},
                global_scale=review.global_scale, rows=rows)
