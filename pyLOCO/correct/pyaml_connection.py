"""Individual quadrupole access and conservative lattice-identity matching."""
from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import yaml

from .petra_readonly import load_mapping as load_machine_mapping, apply_explicit_mapping, MagnetMapping


FAMILIES = ("normal_quadrupole", "skew_quadrupole")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
                inventory.append(dict(name=name, family=family, lattice_name=identity))
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
    data = yaml.safe_load(Path(path).read_text())
    entries = data.get("mappings") if isinstance(data, dict) else data
    if isinstance(entries, list) and entries and all(isinstance(row, dict) and "magnet_name" in row for row in entries):
        return entries
    return [dict(lattice_name=row.lattice_name, lattice_ordinal=row.lattice_ordinal,
                 control_name=row.control_name, component=row.component, unit=row.unit)
            for row in load_machine_mapping(path)]


def mapping_template(review):
    return {"mappings": [dict(
        lattice_ordinal=item.lattice_ordinal, lattice_name=item.name, control_name=item.control_name or item.name,
        component="B2L" if item.correction_type == "normal_quadrupole" else "A2L", unit="m^-1")
        for item in review.items if item.correction_type in FAMILIES]}


def resolve_mapping(review, inventory, explicit=()):
    targets = {row["name"]: row for row in inventory}
    identities = defaultdict(list)
    for entry in inventory:
        identity = entry["lattice_name"]
        if isinstance(identity, str) and not any(char in identity for char in "*?[](),@#"):
            identities[(entry["family"], identity)].append(entry["name"])
    overrides = {}
    used = set()
    for row in explicit:
        name = row.get("control_name", row.get("magnet_name"))
        candidates = [item for item in review.items if item.name == row["lattice_name"]
                      and (row.get("lattice_ordinal") is None or item.lattice_ordinal == row["lattice_ordinal"])
                      and (not row.get("correction_type") or item.correction_type == row["correction_type"])]
        if len(candidates) != 1:
            raise ValueError(f"Unknown or duplicate mapping source: {row['lattice_name']}")
        key = item_identity(candidates[0])
        if key in overrides:
            raise ValueError(f"Duplicate mapping source: {key}")
        if name not in targets or targets[name]["family"] != key[0]:
            raise ValueError(f"Unknown or wrong-family mapping target: {name}")
        component = row.get("component")
        if component and component not in (("B2", "B2L") if key[0] == FAMILIES[0] else ("A2", "A2L")):
            raise ValueError(f"Wrong-family component for {name}: {component}")
        if name in used:
            raise ValueError(f"Duplicate mapping target: {name}")
        matched = deepcopy(candidates[0])
        apply_explicit_mapping(SimpleNamespace(items=[matched]),
            [MagnetMapping(row["lattice_name"], name, row.get("lattice_ordinal"), component, row.get("unit"))])
        overrides[key] = matched.control_name
        used.add(name)
    rows = []
    for item in review.items:
        key = item_identity(item)
        matches = identities[(item.correction_type, item.name)]
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
