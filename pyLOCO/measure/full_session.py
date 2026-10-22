"""Portable, integrity-checked pyLOCO Measure session bundles.

The bundle deliberately reuses the existing Measure project and Measurement
Session schemas.  Its root manifest remains directly consumable by FIT.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime
from hashlib import sha256
from pathlib import Path
import shutil
from typing import Any

from pyLOCO.data_schema import MeasurementSession, SessionFile, load_session, save_session
from .project import MeasureProject, load_measure_project, save_measure_project

BUNDLE_VERSION = 1
PROJECT_NAME = "measure-project.pyloco-measure.json"
MANIFEST_NAME = "measurement-session.pyloco-session.json"


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class FullSessionBundle:
    root: Path
    manifest_path: Path
    project_path: Path
    project: MeasureProject
    session: MeasurementSession
    measurement_paths: dict[str, Path]
    mapping: dict[str, Any] | None
    warnings: tuple[str, ...]


def _copy_verified(source: Path, destination: Path) -> dict[str, Any]:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {"path": destination.as_posix(), "sha256": file_sha256(destination), "size": destination.stat().st_size}


def save_full_session_bundle(destination: str | Path, *, project: MeasureProject,
                             session_manifest: str | Path,
                             reference_lattice: str | Path | None,
                             reference_mat_variable: str = "",
                             reference_element_count: int | None = None,
                             mapping: dict[str, Any] | None = None,
                             include_pyaml_configuration: bool = False) -> FullSessionBundle:
    root = Path(destination).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"Full-session destination already exists: {root}")
    source_manifest = Path(session_manifest).expanduser().resolve()
    source_session = load_session(source_manifest, validate_files=True)
    root.mkdir(parents=True)
    try:
        copied_entries = []
        for entry in source_session.files:
            source = source_session.resolve(source_manifest)[entry.role]
            target = root / "measurements" / source.name
            details = _copy_verified(source, target)
            copied_entries.append(SessionFile(entry.role, target.relative_to(root).as_posix(),
                                               {**entry.metadata, "sha256": details["sha256"], "size": details["size"]}))

        portable = replace(deepcopy(project), output_directory="measurements")
        full: dict[str, Any] = {
            "bundle_version": BUNDLE_VERSION,
            "project": PROJECT_NAME,
            "created": datetime.now().astimezone().isoformat(),
            "roles": {role: role in {entry.role for entry in copied_entries}
                      for role in ("bpm_noise", "dispersion", "orm")},
            "mapping": deepcopy(mapping),
        }

        if reference_lattice:
            lattice_source = Path(reference_lattice).expanduser().resolve()
            target = root / "lattice" / lattice_source.name
            details = _copy_verified(lattice_source, target)
            relative = target.relative_to(root).as_posix()
            portable.reference_model_path = relative
            portable.reference_model_mat_variable = reference_mat_variable
            full["reference_lattice"] = {**details, "path": relative,
                                           "mat_variable": reference_mat_variable,
                                           "element_count": reference_element_count}
        else:
            full["reference_lattice"] = {"status": "not_included"}

        options = dict(portable.metadata.get("pyaml_options", {}))
        config_text = str(options.get("config_path", "")).strip()
        if config_text:
            config = Path(config_text).expanduser().resolve()
            provenance = {"included": False, "external_path": str(config),
                          "sha256": file_sha256(config) if config.is_file() else "",
                          "status": "external" if config.is_file() else "missing"}
            if include_pyaml_configuration:
                target = root / "configuration" / config.name
                details = _copy_verified(config, target)
                relative = target.relative_to(root).as_posix()
                options["config_path"] = relative
                provenance = {**details, "path": relative, "included": True, "status": "included"}
            full["pyaml_configuration"] = provenance
            portable.metadata["pyaml_options"] = options

        project_path = root / PROJECT_NAME
        save_measure_project(project_path, portable)
        metadata = {**source_session.metadata, "full_measure_session": full}
        bundle_session = MeasurementSession(source_session.session_id, tuple(copied_entries), metadata)
        manifest_path = root / MANIFEST_NAME
        save_session(manifest_path, bundle_session)
        return open_full_session_bundle(manifest_path)
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise


def open_full_session_bundle(path: str | Path) -> FullSessionBundle:
    selected = Path(path).expanduser().resolve()
    manifest = selected / MANIFEST_NAME if selected.is_dir() else selected
    session = load_session(manifest, validate_files=False)
    full = session.metadata.get("full_measure_session")
    if not isinstance(full, dict) or full.get("bundle_version") != BUNDLE_VERSION:
        raise ValueError("This is an older measurement-session manifest, not a full Measure session bundle")
    root = manifest.parent
    project_path = root / str(full.get("project", PROJECT_NAME))
    project = load_measure_project(project_path)
    warnings: list[str] = []

    paths = session.resolve(manifest)
    for entry in session.files:
        expected = str(entry.metadata.get("sha256", ""))
        if expected and file_sha256(paths[entry.role]) != expected:
            raise ValueError(f"Measurement file hash mismatch for {entry.role}: {paths[entry.role]}")
    session.validate(base_directory=manifest.parent, validate_files=True)

    lattice = full.get("reference_lattice") or {}
    if lattice.get("path"):
        lattice_path = root / lattice["path"]
        if not lattice_path.is_file():
            raise FileNotFoundError(f"Bundled reference lattice is missing: {lattice_path}")
        if lattice.get("sha256") and file_sha256(lattice_path) != lattice["sha256"]:
            raise ValueError("Bundled reference lattice hash mismatch")
        project.reference_model_path = str(lattice_path)
    else:
        warnings.append("Reference lattice was not included; select the authoritative lattice before model comparison.")

    options = dict(project.metadata.get("pyaml_options", {}))
    config = full.get("pyaml_configuration") or {}
    if config.get("included") and config.get("path"):
        candidate = root / config["path"]
        if not candidate.is_file() or (config.get("sha256") and file_sha256(candidate) != config["sha256"]):
            raise ValueError("Bundled pyAML configuration is missing or has been modified")
        options["config_path"] = str(candidate)
    elif config:
        candidate = Path(str(config.get("external_path", ""))).expanduser()
        if not candidate.is_file():
            warnings.append("External pyAML configuration is missing; select the approved configuration and reconnect.")
            options["config_path"] = str(candidate)
        elif config.get("sha256") and file_sha256(candidate) != config["sha256"]:
            warnings.append("External pyAML configuration checksum changed; device alignment must be revalidated.")
        else:
            options["config_path"] = str(candidate.resolve())
    elif project.adapter == "pyAML":
        warnings.append("No pyAML configuration provenance was recorded; select the approved configuration and reconnect.")
    project.metadata["pyaml_options"] = options
    return FullSessionBundle(root, manifest, project_path, project, session, paths,
                             deepcopy(full.get("mapping")), tuple(warnings))
