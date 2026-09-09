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
    for index, stage in enumerate(recipe.stages):
        if index <= resume_after_stage:
            continue
        request = copy.deepcopy(base_request)
        request.backend_mapping = copy.deepcopy(
            stage.configuration.get("backend_mapping", stage.configuration)
        )
        request.project_name = f"{base_request.project_name} — {stage.name}"
        resume_dir = stage.saved_result if stage.start_from == "saved_result" else (
            previous.results_dir if stage.start_from == "previous_stage" and previous else ""
        )
        request.backend_mapping["Resume"] = {
            "enabled": bool(resume_dir), "directory": resume_dir,
            "ring_file": "ring_pyloco.mat", "fit_dict_file": "fit_dict.pkl",
            "fit_results_file": "fit_results.npy",
        }
        if progress_callback:
            progress_callback({"workflow_stage": index + 1, "workflow_stages": len(recipe.stages),
                               "stage_name": stage.name})
        result = runner(request, log_callback=log_callback, progress_callback=progress_callback)
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
