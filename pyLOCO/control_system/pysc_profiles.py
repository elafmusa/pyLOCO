"""Machine-profile discovery for the generic pySC Server backend."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PySCMachineProfile:
    key: str
    label: str
    scenario: str
    machine: str
    directory: Path
    manifest_path: Path
    configuration: dict[str, Any]

    @property
    def catalog_path(self) -> Path:
        return (self.directory / self.configuration["catalog_file"]).resolve()

    def resolve(self, field: str) -> Path:
        return (self.directory / self.configuration[field]).resolve()


PROFILE_MANIFESTS = {
    "ebs": Path("Examples/pySC_profiles/ebs/validated_demo/profile.yaml"),
    "petra3": Path("Examples/pySC_profiles/petra_iii/official/profile.yaml"),
    "petra3_realistic": Path("Examples/pySC_profiles/petra_iii/realistic_errors/profile.yaml"),
}


def load_pysc_profile(key: str, *, repository_root: Path | None = None) -> PySCMachineProfile:
    root = (repository_root or Path(__file__).resolve().parents[2]).resolve()
    try:
        manifest_path = (root / PROFILE_MANIFESTS[key]).resolve()
    except KeyError as exc:
        raise KeyError(f"Unknown pySC machine profile: {key}") from exc
    if not manifest_path.exists():
        raise RuntimeError(f"pySC profile manifest is missing: {manifest_path}")
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("key") != key:
        raise RuntimeError(f"Invalid pySC profile manifest: {manifest_path}")
    return PySCMachineProfile(
        key=key,
        label=str(data["label"]),
        scenario=str(data["scenario"]),
        machine=str(data["machine"]),
        directory=manifest_path.parent,
        manifest_path=manifest_path,
        configuration=data,
    )


def available_pysc_profiles(*, repository_root: Path | None = None) -> tuple[PySCMachineProfile, ...]:
    return tuple(load_pysc_profile(key, repository_root=repository_root) for key in PROFILE_MANIFESTS)


def load_pysc_catalog(key: str, *, repository_root: Path | None = None) -> dict[str, Any]:
    profile = load_pysc_profile(key, repository_root=repository_root)
    path = profile.catalog_path
    if not path.exists():
        raise RuntimeError(
            f"pySC {profile.label} catalog is missing: {path}. "
            f"Start the '{profile.key}' profile once to generate it from the served SC object."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    for field in ("bpms", "horizontal_correctors", "vertical_correctors"):
        values = data.get(field)
        if not isinstance(values, list) or not values or len(values) != len(set(values)):
            raise RuntimeError(f"Invalid pySC profile catalog field: {field}")
    if data.get("profile") != key and not (key == "ebs" and data.get("machine") == "ESRF-EBS pySC demo"):
        raise RuntimeError(f"pySC catalog does not belong to selected profile '{key}': {path}")
    return data
