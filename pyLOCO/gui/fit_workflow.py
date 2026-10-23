"""Serializable multi-stage FIT recipes and reproducible continuation runs.

Recipes deliberately contain no measurement paths.  A :class:`FitRunSession`
binds one recipe to one lattice and measurement set at execution time.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable


RECIPE_SCHEMA = "pyloco-fit-recipe/v1"
SESSION_SCHEMA = "pyloco-fit-session/v1"


@dataclass(slots=True)
class FitStage:
    name: str
    configuration: dict[str, Any]
    start_from: str = "previous_stage"  # original_model | previous_stage | saved_result
    saved_result: str = ""

    def validate(self, position: int) -> list[str]:
        messages: list[str] = []
        if not self.name.strip():
            messages.append(f"Stage {position + 1} needs a name.")
        if self.start_from not in {"original_model", "previous_stage", "saved_result"}:
            messages.append(f"Stage {position + 1} has an invalid starting state.")
        if position == 0 and self.start_from == "previous_stage":
            messages.append("Stage 1 cannot start from a previous stage.")
        if self.start_from == "saved_result" and not self.saved_result:
            messages.append(f"Stage {position + 1} needs a saved FIT result.")
        mapping = self.configuration.get("backend_mapping", self.configuration)
        required = {"LOCOOptions", "RMConfig", "MachineElements", "FitInitConfig"}
        missing = sorted(required - set(mapping))
        if missing:
            messages.append(f"Stage {position + 1} configuration is incomplete: {', '.join(missing)}")
        return messages


@dataclass(slots=True)
class FitRecipe:
    name: str = "FIT workflow"
    machine_identity: str = ""
    reference_model_checksum: str = ""
    element_identities: dict[str, list[str]] = field(default_factory=dict)
    data_requirements: dict[str, Any] = field(default_factory=dict)
    stages: list[FitStage] = field(default_factory=list)
    schema: str = RECIPE_SCHEMA

    def validate(self) -> list[str]:
        messages = [] if self.schema == RECIPE_SCHEMA else [f"Unsupported recipe schema: {self.schema}"]
        if not self.stages:
            messages.append("The FIT recipe contains no stages.")
        for index, stage in enumerate(self.stages):
            messages.extend(stage.validate(index))
        return messages

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # Scientific guard: measurements belong exclusively to a run session.
        for stage in data["stages"]:
            stage["configuration"].pop("Measurements", None)
        return data

    def save(self, path: str | Path) -> Path:
        target = Path(path).expanduser().resolve()
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "FitRecipe":
        data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        data["stages"] = [FitStage(**item) for item in data.get("stages", [])]
        recipe = cls(**data)
        errors = recipe.validate()
        if errors:
            raise ValueError("\n".join(errors))
        return recipe


@dataclass(slots=True)
class StageCheckpoint:
    stage_index: int
    stage_name: str
    results_dir: str
    ring_file: str
    fit_dict_file: str
    fit_results_file: str
    configuration: dict[str, Any]
    measurement_identity: dict[str, Any]
    parameter_ordering_file: str = "blocks.pkl"
    summary_file: str = "summary.json"

    def validate_files(self) -> list[str]:
        root = Path(self.results_dir)
        required = (self.ring_file, self.fit_dict_file, self.fit_results_file,
                    self.parameter_ordering_file, self.summary_file)
        return [str(root / name) for name in required if not (root / name).is_file()]


@dataclass(slots=True)
class FitRunSession:
    recipe: dict[str, Any]
    lattice_path: str
    lattice_checksum: str
    measurements: dict[str, str]
    measurement_identity: dict[str, Any]
    checkpoints: list[StageCheckpoint] = field(default_factory=list)
    lattice_mat_variable: str = ""
    schema: str = SESSION_SCHEMA

    def save(self, path: str | Path) -> Path:
        target = Path(path).expanduser().resolve()
        target.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "FitRunSession":
        data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        if data.get("schema") != SESSION_SCHEMA:
            raise ValueError(f"Unsupported FIT session schema: {data.get('schema')}")
        data["checkpoints"] = [StageCheckpoint(**item) for item in data.get("checkpoints", [])]
        return cls(**data)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_items(dataset) -> list[str]:
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in dataset[:]]


def inspect_measurement_files(measurements: dict[str, str],
                              measurement_options: dict[str, dict[str, Any]] | None = None,
                              *, trusted_hashes: dict[str, str] | None = None,
                              trusted_identity: dict[str, Any] | None = None) -> dict[str, Any]:
    """Inspect the selected payload, never cached project/session metadata.

    Schema-1.0 files expose identities directly.  Historical files which do
    not contain identity datasets may use recipe metadata only when their
    bytes match a recipe-pinned SHA-256 digest.
    """
    import h5py
    import numpy as np

    options = measurement_options or {}
    trusted_hashes = trusted_hashes or {}
    trusted_identity = trusted_identity or {}
    report: dict[str, Any] = {"files": {}, "errors": []}
    embedded_bpms: dict[str, list[str]] = {}

    def dataset_name(role: str, axis: str | None = None) -> str:
        role_options = options.get(role, {})
        if role == "orm":
            return str(role_options.get("dataset", "response_matrix"))
        names = role_options.get("datasets", {})
        if not isinstance(names, dict):
            names = {}
        key = "horizontal_dataset" if axis == "horizontal" else "vertical_dataset"
        defaults = {("dispersion", "horizontal"): "measured_eta_x",
                    ("dispersion", "vertical"): "measured_eta_y",
                    ("bpm_noise", "horizontal"): "Noise_BPMx",
                    ("bpm_noise", "vertical"): "Noise_BPMy"}
        return str(role_options.get(key, names.get(axis, defaults[(role, axis)])))

    for role in ("orm", "dispersion", "bpm_noise"):
        raw_path = measurements.get(role)
        if not raw_path:
            report["errors"].append(f"Selected measurement is missing required {role} file.")
            continue
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            report["errors"].append(f"Selected {role} file does not exist: {path}")
            continue
        digest = file_sha256(path)
        pinned = trusted_hashes.get(role)
        legacy_trusted = bool(pinned and digest == pinned)
        entry: dict[str, Any] = {"path": str(path), "sha256": digest,
                                 "identity_source": "embedded"}
        try:
            with h5py.File(path, "r") as handle:
                metadata: dict[str, Any] = {}
                if "metadata/json" in handle:
                    raw = handle["metadata/json"][()]
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8")
                    metadata = json.loads(str(raw))
                if role == "orm":
                    name = dataset_name(role)
                    if name not in handle:
                        raise ValueError(f"dataset {name!r} is absent")
                    matrix = handle[name]
                    entry["shape"] = list(matrix.shape)
                    if matrix.ndim != 2:
                        raise ValueError(f"matrix must be two-dimensional, got {matrix.shape}")
                    if not np.isfinite(matrix[:]).all():
                        raise ValueError("matrix contains non-finite values")
                    entry["unit"] = str(handle.attrs.get("response_matrix_unit", metadata.get("response_matrix_unit", "")))
                    entry["row_order"] = str(handle.attrs.get("row_order", metadata.get("row_order", "")))
                    entry["column_order"] = str(handle.attrs.get("column_order", metadata.get("column_order", "")))
                    for key, target in (("names/bpms", "bpm_names"),
                                        ("names/horizontal_correctors", "horizontal_corrector_names"),
                                        ("names/vertical_correctors", "vertical_corrector_names")):
                        if key in handle:
                            entry[target] = _text_items(handle[key])
                    if not all(key in entry for key in ("bpm_names", "horizontal_corrector_names",
                                                        "vertical_corrector_names")):
                        if not legacy_trusted:
                            raise ValueError("device identities are absent and the file does not match a recipe-pinned checksum")
                        entry["identity_source"] = "recipe identity, verified by SHA-256"
                        for key in ("bpm_names", "horizontal_corrector_names", "vertical_corrector_names"):
                            entry[key] = list(trusted_identity.get(key, []))
                        entry["unit"] = entry["unit"] or trusted_identity.get("units", {}).get("orm", "")
                    for dataset, identities, label in (
                        ("kicks/horizontal/actual", entry.get("horizontal_corrector_names", []), "horizontal"),
                        ("kicks/vertical/actual", entry.get("vertical_corrector_names", []), "vertical"),
                    ):
                        if dataset in handle:
                            kick_values = handle[dataset]
                            if kick_values.ndim != 1 or len(kick_values) != len(identities):
                                raise ValueError(
                                    f"{label} actual-kick vector does not align with its ordered corrector identities"
                                )
                else:
                    horizontal = handle[dataset_name(role, "horizontal")]
                    vertical = handle[dataset_name(role, "vertical")]
                    entry["shape"] = [list(horizontal.shape), list(vertical.shape)]
                    if horizontal.ndim != 1 or vertical.ndim != 1 or horizontal.shape != vertical.shape:
                        raise ValueError("horizontal and vertical datasets must be equal-length vectors")
                    if not np.isfinite(horizontal[:]).all() or not np.isfinite(vertical[:]).all():
                        raise ValueError("datasets contain non-finite values")
                    if "names/bpms" in handle:
                        entry["bpm_names"] = _text_items(handle["names/bpms"])
                    elif legacy_trusted:
                        entry["identity_source"] = "recipe identity, verified by SHA-256"
                        entry["bpm_names"] = list(trusted_identity.get("bpm_names", []))
                    else:
                        raise ValueError("BPM identities are absent and the file does not match a recipe-pinned checksum")
                    if role == "dispersion":
                        entry["unit"] = str(handle.attrs.get("measured_eta_unit", metadata.get("canonical_measured_eta_unit", "")))
                        from pyLOCO.dispersion_convention import canonical_dispersion_convention
                        convention = canonical_dispersion_convention(handle.attrs)
                        entry["rf_step_hz"] = convention.rf_step_hz
                        entry["rf_difference_sign_convention"] = "positive_minus_negative"
                        entry["source_rf_difference_sign_convention"] = convention.source_sign
                        if not entry["unit"] and legacy_trusted:
                            entry["unit"] = trusted_identity.get("units", {}).get("dispersion", "")
                            entry["rf_step_hz"] = abs(float(options.get(role, {}).get("rf_step_hz", 0.0)))
                    else:
                        entry["unit"] = str(metadata.get("bpm_noise_unit", ""))
                        if not entry["unit"] and legacy_trusted:
                            entry["unit"] = trusted_identity.get("units", {}).get("bpm_noise", "")
        except Exception as exc:
            report["errors"].append(f"Selected {role} file is incompatible: {exc}")
        report["files"][role] = entry
        if entry.get("bpm_names"):
            embedded_bpms[role] = entry["bpm_names"]

    if embedded_bpms:
        first_role, first_names = next(iter(embedded_bpms.items()))
        for role, names in embedded_bpms.items():
            if names != first_names:
                report["errors"].append(f"BPM identities/order differ between selected {first_role} and {role} files.")
    return report


def _rebase_workflow_optics(results_dir: str | Path, workflow_lattice: str | Path,
                            lattice_mat_variable: str | None = None) -> bool:
    """Make stage optics relative to the lattice that started the workflow.

    The single-stage backend correctly uses the resumed checkpoint as its
    local reference.  For a multi-stage FIT, however, the final scientific
    result must be compared with the original workflow input, as in the
    historical LOCO notebooks.
    """
    import at
    import numpy as np
    from .lattice_loader import load_project_lattice

    root = Path(results_dir)
    optics_path = root / "optics_results.npz"
    fitted_path = root / "final_lattice.mat"
    if not fitted_path.is_file():
        fitted_path = root / "ring_pyloco.mat"
    if not optics_path.is_file() or not fitted_path.is_file():
        return False

    with np.load(optics_path, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key]) for key in archive.files}
    reference = load_project_lattice(str(workflow_lattice), lattice_mat_variable)
    fitted = at.load_lattice(str(fitted_path))
    reference.disable_6d(); fitted.disable_6d()
    if len(reference) != len(fitted):
        raise ValueError("Workflow input and fitted lattices have different element counts.")
    refpts = np.arange(len(reference), dtype=np.uint32)
    beta_ref = np.asarray(reference.get_optics(refpts=refpts)[2].beta, dtype=float)
    beta_fit = np.asarray(fitted.get_optics(refpts=refpts)[2].beta, dtype=float)
    arrays.update({
        "reference_kind": np.asarray("workflow_input_lattice"),
        "s": np.asarray(reference.get_s_pos(refpts), dtype=float),
        "beta_x_reference": beta_ref[:, 0], "beta_y_reference": beta_ref[:, 1],
        "beta_x_fitted": beta_fit[:, 0], "beta_y_fitted": beta_fit[:, 1],
        "beta_beating_x": np.divide(beta_fit[:, 0] - beta_ref[:, 0], beta_ref[:, 0],
                                     out=np.full(len(reference), np.nan), where=beta_ref[:, 0] != 0),
        "beta_beating_y": np.divide(beta_fit[:, 1] - beta_ref[:, 1], beta_ref[:, 1],
                                     out=np.full(len(reference), np.nan), where=beta_ref[:, 1] != 0),
    })
    bpm_ords = arrays.get("dispersion_bpm_ords")
    if bpm_ords is not None:
        bpm_ords = np.asarray(bpm_ords, dtype=np.uint32)
        dispersion = np.asarray(reference.get_optics(refpts=bpm_ords)[2].dispersion, dtype=float)
        conversion = (
            -float(arrays["dispersion_momentum_compaction"])
            * float(arrays["dispersion_rf_frequency_hz"])
            / float(arrays["dispersion_rf_step_hz"])
        )
        arrays["dispersion_s"] = np.asarray(reference.get_s_pos(bpm_ords), dtype=float)
        arrays["dispersion_x_initial"] = dispersion[:, 0] / conversion
        arrays["dispersion_y_initial"] = dispersion[:, 2] / conversion
    np.savez_compressed(optics_path, **arrays)
    return True


def preflight_recipe(recipe: FitRecipe, *, lattice_path: str, measurement_identity: dict[str, Any],
                     lattice_element_names: list[str] | None = None,
                     measurements: dict[str, str] | None = None,
                     measurement_options: dict[str, dict[str, Any]] | None = None,
                     measurement_device_mapping: dict[str, Any] | None = None,
                     lattice_mat_variable: str | None = None,
                     comparison_only: bool = False) -> dict[str, Any]:
    """Validate identity/order and explicitly report any stable-name remapping."""
    # A reference-model comparison consumes the measured-device inventory and
    # response configuration, but no FIT stage, fitted magnet group, Jacobian,
    # or optimizer option. Keep execution validation strict while allowing the
    # read-only diagnostic before a fitting recipe has been configured.
    errors = ([] if recipe.schema == RECIPE_SCHEMA else
              [f"Unsupported recipe schema: {recipe.schema}"]) if comparison_only else recipe.validate()
    checksum = file_sha256(lattice_path)
    if recipe.reference_model_checksum and checksum != recipe.reference_model_checksum:
        errors.append("Reference lattice checksum does not match the FIT recipe.")
    machine = str(measurement_identity.get("machine_profile") or measurement_identity.get("machine") or "")
    if recipe.machine_identity and machine and machine != recipe.machine_identity:
        errors.append(f"Measurement machine/profile {machine!r} does not match recipe {recipe.machine_identity!r}.")
    # Cached session identity is useful for old callers, but must never decide
    # replacement-file compatibility when actual paths were supplied.
    if measurements is None:
        for key in ("bpm_names", "horizontal_corrector_names", "vertical_corrector_names",
                    "orm_convention", "dispersion_convention", "rf_convention", "units"):
            expected = recipe.data_requirements.get(key)
            actual = measurement_identity.get(key)
            if expected not in (None, [], "") and actual != expected:
                errors.append(f"Measurement {key} does not match the recipe.")
    payload = None
    if measurements is not None:
        payload = inspect_measurement_files(
            measurements, measurement_options,
            trusted_hashes=recipe.data_requirements.get("measurement_sha256", {}),
            trusted_identity=recipe.data_requirements,
        )
        errors.extend(payload["errors"])
        files = payload["files"]
        orm = files.get("orm", {})
        actual_bpms = list(orm.get("bpm_names") or [])
        actual_h = list(orm.get("horizontal_corrector_names") or [])
        actual_v = list(orm.get("vertical_corrector_names") or [])
        expected_bpms = recipe.data_requirements.get("bpm_names", [])
        expected_h = recipe.data_requirements.get("horizontal_corrector_names", [])
        expected_v = recipe.data_requirements.get("vertical_corrector_names", [])
        mapped_ordinals = None
        if measurement_device_mapping:
            try:
                import at
                from .lattice_loader import load_project_lattice
                from .measurement_mapping import (measurement_inventory, ordered_ordinals,
                                                  validate_mapping as validate_device_mapping)
                ring = (
                    load_project_lattice(str(lattice_path), lattice_mat_variable)
                    if lattice_mat_variable else at.load_lattice(str(lattice_path))
                )
                inventory = measurement_inventory(payload)
                configured = (measurement_device_mapping.get("machine_profile") or {}).get("configuration_path")
                mapping = validate_device_mapping(
                    measurement_device_mapping, ring, inventory, lattice_path=lattice_path,
                    configuration_path=configured or None, require_provenance=True,
                )
                mapped_ordinals = {
                    role: ordered_ordinals(mapping, role)
                    for role in ("bpm", "horizontal_corrector", "vertical_corrector")
                }
                incomplete = [role for role, values in mapped_ordinals.items() if values is None]
                if incomplete:
                    errors.append("Measurement-device mapping is incomplete for: " + ", ".join(incomplete) + ".")
            except Exception as exc:
                errors.append(f"Measurement-device mapping is incompatible: {exc}")
                mapped_ordinals = None
        count_bpms = actual_bpms if mapped_ordinals is not None else (expected_bpms or actual_bpms)
        count_h = actual_h if mapped_ordinals is not None else (expected_h or actual_h)
        count_v = actual_v if mapped_ordinals is not None else (expected_v or actual_v)
        expected_shape = [2 * len(count_bpms), len(count_h) + len(count_v)]
        if orm.get("shape") and orm["shape"] != expected_shape:
            errors.append(f"Selected ORM shape {tuple(orm['shape'])} does not satisfy the recipe/stages; expected {tuple(expected_shape)}.")
        if mapped_ordinals is None:
            for key, expected in (("bpm_names", expected_bpms),
                                  ("horizontal_corrector_names", expected_h),
                                  ("vertical_corrector_names", expected_v)):
                if expected and orm.get(key) != expected:
                    errors.append(f"Selected ORM {key} identities/order do not match the recipe.")
        for role in ("dispersion", "bpm_noise"):
            entry = files.get(role, {})
            expected_role_bpms = actual_bpms if mapped_ordinals is not None else count_bpms
            if entry.get("bpm_names") != expected_role_bpms:
                errors.append(f"Selected {role} BPM identities/order do not match the ORM measurement/recipe.")
            if entry.get("shape") and entry["shape"] != [[len(count_bpms)], [len(count_bpms)]]:
                errors.append(f"Selected {role} vectors do not match the required BPM count {len(count_bpms)}.")
        expected_units = recipe.data_requirements.get("units", {})
        for role in ("orm", "dispersion", "bpm_noise"):
            expected_unit = expected_units.get(role)
            actual_unit = files.get(role, {}).get("unit")
            if expected_unit and actual_unit != expected_unit:
                errors.append(f"Selected {role} unit {actual_unit!r} does not match required {expected_unit!r}.")
        expected_rf = recipe.data_requirements.get("rf_step_hz")
        if expected_rf is not None and files.get("dispersion", {}).get("rf_step_hz") != float(expected_rf):
            errors.append("Selected dispersion RF bipolar separation does not match the recipe.")
        stage_roles = (("bpm_ords", len(count_bpms), "BPMs", "bpm"),
                       ("horizontal_corrector_ords", len(count_h), "horizontal correctors", "horizontal_corrector"),
                       ("vertical_corrector_ords", len(count_v), "vertical correctors", "vertical_corrector"))
        for position, stage in enumerate(() if comparison_only else recipe.stages, 1):
            mapping = stage.configuration.get("backend_mapping", stage.configuration)
            elements = mapping.get("MachineElements", {})
            for key, count, label, role in stage_roles:
                selected = elements.get(key)
                if selected is not None and len(selected) != count:
                    errors.append(
                        f"Stage {position} ({stage.name}) selects {len(selected)} {label}, "
                        f"but the selected measurements/recipe require {count}."
                    )
                if mapped_ordinals is not None and selected is not None and mapped_ordinals.get(role) is not None:
                    if list(selected) != mapped_ordinals[role]:
                        errors.append(
                            f"Stage {position} ({stage.name}) {label} do not match the validated "
                            "measurement-device mapping in identity and order."
                        )
            exclusions = (
                ("BadBPMPositions", len(count_bpms), "BPM"),
                ("horizontal", len(count_h), "horizontal-corrector"),
                ("vertical", len(count_v), "vertical-corrector"),
            )
            excluded_correctors = mapping.get("ExcludedCorrectorPositions", {})
            for key, size, label in exclusions:
                values = mapping.get(key, []) if key == "BadBPMPositions" else excluded_correctors.get(key, [])
                if len(values) != len(set(values)) or any(type(value) is not int or value < 0 or value >= size for value in values):
                    errors.append(f"Stage {position} ({stage.name}) has invalid {label} exclusion positions.")
            fit_list = mapping.get("FitInitConfig", {}).get("fit_list", [])
            for parameter, key in (("quads", "normal_quadrupole_ords"),
                                   ("skew_quads", "skew_quadrupole_ords")):
                if parameter in fit_list and key in elements and not elements.get(key):
                    errors.append(f"Stage {position} ({stage.name}) fits {parameter} but selects no lattice elements.")
    remapping: dict[str, list[int]] = {}
    if lattice_element_names is not None:
        lookup: dict[str, list[int]] = {}
        for ordinal, name in enumerate(lattice_element_names):
            lookup.setdefault(str(name), []).append(ordinal)
        for role, names in recipe.element_identities.items():
            ambiguous = [name for name in names if len(lookup.get(name, [])) != 1]
            if ambiguous:
                errors.append(f"{role} identity is missing or ambiguous: {ambiguous[0]}")
            else:
                remapping[role] = [lookup[name][0] for name in names]
    return {"compatible": not errors, "errors": errors, "remapping": remapping,
            "lattice_checksum": checksum, "measurement_payload": payload}


def execute_workflow(base_request, recipe: FitRecipe, *, session_path: str | Path,
                     runner: Callable[..., Any], log_callback=None, progress_callback=None,
                     resume_after_stage: int = -1, resume_session: FitRunSession | None = None) -> FitRunSession:
    """Execute stages through the existing single-fit runner and disk checkpoints."""
    identity = copy.deepcopy(getattr(base_request, "measurement_session", {}) or {})
    session = resume_session or FitRunSession(
        recipe=recipe.to_dict(), lattice_path=base_request.lattice_path,
        lattice_checksum=file_sha256(base_request.lattice_path),
        measurements=copy.deepcopy(base_request.measurements), measurement_identity=identity,
        lattice_mat_variable=getattr(base_request, "lattice_mat_variable", ""),
    )
    session_target = Path(session_path).expanduser().resolve()
    previous: StageCheckpoint | None = session.checkpoints[-1] if session.checkpoints else None
    resume_after_stage = max(resume_after_stage, len(session.checkpoints) - 1)
    stage_count = len(recipe.stages)
    for index, stage in enumerate(recipe.stages):
        if index <= resume_after_stage:
            continue
        request = copy.deepcopy(base_request)
        base_output = copy.deepcopy(request.backend_mapping.get("Output", {}))
        request.backend_mapping = copy.deepcopy(
            stage.configuration.get("backend_mapping", stage.configuration)
        )
        stage_output = request.backend_mapping.setdefault("Output", {})
        if not stage_output.get("directory"):
            stage_output["directory"] = base_output.get("directory")
        request.project_name = f"{base_request.project_name} — {stage.name}"
        resume_dir = stage.saved_result if stage.start_from == "saved_result" else (
            previous.results_dir if stage.start_from == "previous_stage" and previous else ""
        )
        request.backend_mapping["Resume"] = {
            "enabled": bool(resume_dir), "directory": resume_dir,
            "ring_file": "ring_pyloco.mat", "fit_dict_file": "fit_dict.pkl",
            "fit_results_file": "fit_results.npy",
        }
        request.backend_mapping["Workflow"] = {
            "stage": index + 1, "stages": stage_count,
            "reference_lattice": base_request.lattice_path,
        }
        def stage_progress(event, *, _index=index, _stage=stage):
            if progress_callback is None:
                return
            detailed = copy.deepcopy(event or {})
            stage_fraction = min(1.0, max(0.0, float(detailed.get("workflow_fraction", 0.0))))
            detailed.update({
                "workflow_stage": _index + 1,
                "workflow_stages": stage_count,
                "stage_name": _stage.name,
                "stage_fraction": stage_fraction,
                "workflow_fraction": (_index + stage_fraction) / stage_count,
            })
            progress_callback(detailed)

        stage_progress({"phase": "stage_start", "message": f"Starting {stage.name}.",
                        "workflow_fraction": 0.0})
        result = runner(request, log_callback=log_callback, progress_callback=stage_progress)
        try:
            selected_variable = getattr(base_request, "lattice_mat_variable", "")
            if selected_variable:
                _rebase_workflow_optics(
                    result.results_dir, base_request.lattice_path, selected_variable,
                )
            else:
                _rebase_workflow_optics(result.results_dir, base_request.lattice_path)
        except Exception as exc:
            if log_callback is not None:
                log_callback(f"Warning: could not rebase workflow optics to the original lattice: {exc}")
        checkpoint = StageCheckpoint(index, stage.name, result.results_dir, "ring_pyloco.mat",
                                     "fit_dict.pkl", "fit_results.npy",
                                     copy.deepcopy(stage.configuration), identity)
        missing = checkpoint.validate_files()
        if missing:
            raise RuntimeError("Incomplete continuation checkpoint: " + ", ".join(missing))
        session.checkpoints.append(checkpoint)
        previous = checkpoint
        session.save(session_target)
    return session
