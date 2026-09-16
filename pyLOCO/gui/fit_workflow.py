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


def _rebase_workflow_optics(results_dir: str | Path, workflow_lattice: str | Path) -> bool:
    """Make stage optics relative to the lattice that started the workflow.

    The single-stage backend correctly uses the resumed checkpoint as its
    local reference.  For a multi-stage FIT, however, the final scientific
    result must be compared with the original workflow input, as in the
    historical LOCO notebooks.
    """
    import at
    import numpy as np

    root = Path(results_dir)
    optics_path = root / "optics_results.npz"
    fitted_path = root / "final_lattice.mat"
    if not fitted_path.is_file():
        fitted_path = root / "ring_pyloco.mat"
    if not optics_path.is_file() or not fitted_path.is_file():
        return False

    with np.load(optics_path, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key]) for key in archive.files}
    reference = at.load_lattice(str(workflow_lattice))
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
                     lattice_element_names: list[str] | None = None) -> dict[str, Any]:
    """Validate identity/order and explicitly report any stable-name remapping."""
    errors = recipe.validate()
    checksum = file_sha256(lattice_path)
    if recipe.reference_model_checksum and checksum != recipe.reference_model_checksum:
        errors.append("Reference lattice checksum does not match the FIT recipe.")
    machine = str(measurement_identity.get("machine_profile") or measurement_identity.get("machine") or "")
    if recipe.machine_identity and machine and machine != recipe.machine_identity:
        errors.append(f"Measurement machine/profile {machine!r} does not match recipe {recipe.machine_identity!r}.")
    for key in ("bpm_names", "horizontal_corrector_names", "vertical_corrector_names",
                "orm_convention", "dispersion_convention", "rf_convention", "units"):
        expected = recipe.data_requirements.get(key)
        actual = measurement_identity.get(key)
        if expected not in (None, [], "") and actual != expected:
            errors.append(f"Measurement {key} does not match the recipe.")
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
            "lattice_checksum": checksum}


def execute_workflow(base_request, recipe: FitRecipe, *, session_path: str | Path,
                     runner: Callable[..., Any], log_callback=None, progress_callback=None,
                     resume_after_stage: int = -1, resume_session: FitRunSession | None = None) -> FitRunSession:
    """Execute stages through the existing single-fit runner and disk checkpoints."""
    identity = copy.deepcopy(getattr(base_request, "measurement_session", {}) or {})
    session = resume_session or FitRunSession(
        recipe=recipe.to_dict(), lattice_path=base_request.lattice_path,
        lattice_checksum=file_sha256(base_request.lattice_path),
        measurements=copy.deepcopy(base_request.measurements), measurement_identity=identity,
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
