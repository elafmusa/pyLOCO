"""UI-only project state for the pyLOCO GUI."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pyLOCO.config import DEFAULT_INIT_POLICY

PROJECT_FILE_SUFFIX = ".pyloco.json"
REQUIRED_MEASUREMENTS = ("orm",)


@dataclass(slots=True)
class ImportedDataset:
    """Metadata for a measurement file imported into the GUI project."""

    role: str
    path: str
    file_type: str
    size_bytes: int = 0

    @property
    def name(self) -> str:
        return Path(self.path).name


@dataclass(slots=True)
class LatticeSelection:
    """Metadata for a selected lattice/model file."""

    path: str = ""
    file_type: str = ""
    element_count: int | None = None

    @property
    def name(self) -> str:
        return Path(self.path).name if self.path else "No lattice selected"


@dataclass(slots=True)
class ElementSelection:
    """Selected lattice ordinals for one machine-element role."""

    ords: list[int] = field(default_factory=list)


@dataclass(slots=True)
class MachineElementsConfig:
    """Machine element selections stored as lattice ordinals."""

    bpm_ords: list[int] = field(default_factory=list)
    horizontal_corrector_ords: list[int] = field(default_factory=list)
    vertical_corrector_ords: list[int] = field(default_factory=list)
    normal_quadrupole_ords: list[int] = field(default_factory=list)
    skew_quadrupole_ords: list[int] = field(default_factory=list)
    cavity_ords: list[int] = field(default_factory=list)


@dataclass(slots=True)
class ResponseMatrixConfig:
    """GUI state matching the backend RMConfig constructor fields."""

    bpm_ords: list[int] = field(default_factory=list)
    cm_ords: tuple[list[int], list[int]] = field(default_factory=lambda: ([], []))
    cav_ords: list[int] = field(default_factory=list)
    dkick_h: float = 1e-5
    dkick_v: float = 1e-5
    bidirectional: bool = True
    includeDispersion: bool = False
    rfStep: float = -3000.0
    delta_coupling: float = 1e-6
    coupling_orm: bool = False
    calculator: str = "Linear"
    NewVectorizedMethod: bool = True
    fixedpathlength: bool = False
    log_info: bool = False
    HCMCoupling: str = ""
    VCMCoupling: str = ""
    Frequency: str = ""
    HarmNumber: str = ""
    RFAttr: str = "Frequency"

    def to_rm_config_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments compatible with pyloco_config.RMConfig."""

        calculator = "Linear" if self.calculator == "Linear" else "Tracking"
        data = {
            "bpm_ords": list(self.bpm_ords),
            "cm_ords": (list(self.cm_ords[0]), list(self.cm_ords[1])),
            "cav_ords": list(self.cav_ords),
            "dkick": self.dkick_value(),
            "bidirectional": self.bidirectional,
            "includeDispersion": self.includeDispersion,
            "rfStep": self.rfStep,
            "delta_coupling": self.delta_coupling,
            "coupling_orm": self.coupling_orm,
            "calculator": calculator,
            "NewVectorizedMethod": self.NewVectorizedMethod,
            "fixedpathlength": self.fixedpathlength,
            "log_info": self.log_info,
            "HCMCoupling": _literal_or_none(self.HCMCoupling),
            "VCMCoupling": _literal_or_none(self.VCMCoupling),
            "Frequency": _literal_or_none(self.Frequency),
            "HarmNumber": _literal_or_none(self.HarmNumber),
            "RFAttr": self.RFAttr,
        }
        return data

    def dkick_value(self) -> Any:
        return (float(self.dkick_h), float(self.dkick_v))


@dataclass(slots=True)
class SolverConfig:
    """GUI state matching LOCO solver option names."""

    algorithm: str = "lm"
    nIter: int = 1
    nLMIter: int = 10
    Starting_Lambda: float = 1e-3
    max_lm_lambda: float = 15.0
    scaled: bool = True


@dataclass(slots=True)
class SVDConfig:
    """GUI state for SVD selection options consumed by pyloco."""

    svd_selection_method: str = "threshold"
    svd_threshold: float = 1e-7
    cut_: int = 397
    show_svd_plot: bool = True


@dataclass(slots=True)
class RejectionConfig:
    """Iteration, normalization, and outlier controls."""

    outlier_rejection: bool = True
    sigma_outlier: float = 10.0
    apply_normalization: bool = False
    normalization_mode: str = "component"
    includeDispersion: bool = False
    hor_dispersion_weight: float = 1.0
    ver_dispersion_weight: float = 1.0
    auto_correct_delta: bool = True
    fixedpathlength: bool = False
    individuals: bool = True
    remove_coupling_: bool = True
    plot_fit_parameters: bool = False


@dataclass(slots=True)
class ConstraintConfigState:
    """Duck-typed UI state for the backend constraint_cfg argument."""

    enable: bool = False
    quad_sigma: float = 0.0
    skew_sigma: float = 0.0
    quad_weights: str = ""
    skew_weights: str = ""
    quad_mask: str = ""
    skew_mask: str = ""

    def to_constraint_config_kwargs(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("quad_weights", "skew_weights", "quad_mask", "skew_mask"):
            data[key] = _literal_or_none(data[key])
        return data



@dataclass(slots=True)
class CMStepConfig:
    """Corrector-step source and values for LOCO initialization."""

    mode: str = "uniform"
    horizontal: float = 1e-5
    vertical: float = 1e-5
    file: str = ""

    def value(self, n_hcor: int | None = None, n_vcor: int | None = None) -> Any:
        if self.mode == "file":
            return load_cmstep_npz(self.file, n_hcor, n_vcor)
        return (float(self.horizontal), float(self.vertical))


@dataclass(slots=True)
class ParameterSelectionConfig:
    """Selected LOCO fit blocks compatible with FitInitConfig.fit_list."""

    quads: bool = True
    skew_quads: bool = False
    quads_tilt: bool = False
    hbpm_gain: bool = True
    vbpm_gain: bool = True
    hbpm_coupling: bool = False
    vbpm_coupling: bool = False
    hcor_cal: bool = True
    vcor_cal: bool = True
    hcor_coupling: bool = False
    vcor_coupling: bool = False
    HCMEnergyShift: bool = True
    VCMEnergyShift: bool = False
    delta_rf: bool = False
    individuals: bool = True
    init_policy: str = ""
    init_policy_overrides: dict[str, str] = field(default_factory=dict)
    cmstep: CMStepConfig = field(default_factory=CMStepConfig)
    rfStep: float = -3000.0
    init: str = ""
    quads_attr: str = "PolynomB"
    quads_attr_index: int = 1
    skew_attr: str = "PolynomA"
    skew_attr_index: int = 1
    quads_tilt_attr_R1: str = "R1"
    quads_tilt_attr_R2: str = "R2"
    quads_tilt_method: str = "set"

    def fit_list(self) -> list[str]:
        order = (
            "quads",
            "skew_quads",
            "quads_tilt",
            "hbpm_gain",
            "vbpm_gain",
            "hbpm_coupling",
            "vbpm_coupling",
            "hcor_cal",
            "vcor_cal",
            "hcor_coupling",
            "vcor_coupling",
            "HCMEnergyShift",
            "VCMEnergyShift",
            "delta_rf",
        )
        return [name for name in order if bool(getattr(self, name))]

    def to_fit_init_config_kwargs(self) -> dict[str, Any]:
        return {
            "fit_list": self.fit_list(),
            "init_policy": self.init_policy_value(),
            "CMstep": self.cmstep_value(),
            "rfStep": self.rfStep,
            "individuals": self.individuals,
            "init": _literal_or_none(self.init),
            "quads_attr": self.quads_attr,
            "quads_attr_index": self.quads_attr_index,
            "skew_attr": self.skew_attr,
            "skew_attr_index": self.skew_attr_index,
            "quads_tilt_attr_R1": self.quads_tilt_attr_R1,
            "quads_tilt_attr_R2": self.quads_tilt_attr_R2,
            "quads_tilt_method": self.quads_tilt_method,
        }

    def init_policy_value(self) -> dict[str, str] | None:
        if self.init_policy.strip():
            value = _literal_or_none(self.init_policy)
            if not isinstance(value, dict):
                raise ValueError("init_policy must be a dictionary when edited as a literal")
            return value
        policy = dict(DEFAULT_INIT_POLICY)
        for key, value in self.init_policy_overrides.items():
            if str(value).strip():
                policy[key] = str(value).strip()
        return policy

    def cmstep_value(self, n_hcor: int | None = None, n_vcor: int | None = None) -> Any:
        return self.cmstep.value(n_hcor, n_vcor)

    @property
    def CMstep_mode(self) -> str:
        return self.cmstep.mode

    @CMstep_mode.setter
    def CMstep_mode(self, value: str) -> None:
        self.cmstep.mode = value

    @property
    def CMstep_h(self) -> float:
        return self.cmstep.horizontal

    @CMstep_h.setter
    def CMstep_h(self, value: Any) -> None:
        self.cmstep.horizontal = _parse_float(value, "horizontal corrector step")

    @property
    def CMstep_v(self) -> float:
        return self.cmstep.vertical

    @CMstep_v.setter
    def CMstep_v(self, value: Any) -> None:
        self.cmstep.vertical = _parse_float(value, "vertical corrector step")

    @property
    def CMstep_file(self) -> str:
        return self.cmstep.file

    @CMstep_file.setter
    def CMstep_file(self, value: str) -> None:
        self.cmstep.file = value


def _parse_float(text: Any, label: str) -> float:
    try:
        value = float(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    import math
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def load_cmstep_npz(path: str | Path, n_hcor: int | None = None, n_vcor: int | None = None) -> list[Any]:
    import numpy as np
    source = Path(path).expanduser()
    with np.load(source) as data:
        missing = [name for name in ("hor", "ver") if name not in data]
        if missing:
            raise ValueError("CM-step file must contain datasets: hor and ver")
        values = []
        for name, expected in (("hor", n_hcor), ("ver", n_vcor)):
            arr = np.asarray(data[name])
            if arr.ndim != 1:
                raise ValueError(f"CM-step dataset {name!r} must be one-dimensional")
            if not np.issubdtype(arr.dtype, np.number):
                raise ValueError(f"CM-step dataset {name!r} must be numeric")
            arr = arr.astype(float, copy=False)
            if expected is not None and arr.size != expected:
                raise ValueError(f"CM-step dataset {name!r} length {arr.size} does not match selected corrector count {expected}")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"CM-step dataset {name!r} must contain only finite values")
            values.append(arr)
    return values


def _literal_or_none(text: Any) -> Any:
    """Parse an optional GUI literal while preserving blank values as None."""

    if text is None or not isinstance(text, str):
        return text
    stripped = text.strip()
    if not stripped:
        return None
    import ast

    try:
        return ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        return stripped


@dataclass(slots=True)
class FixedParameterConfig:
    """GUI state matching FixedParameters."""

    Frequency: str = "499664399.4230182"
    HarmNumber: int = 3840
    rfstep: float = -3000.0
    dk: str = ""
    delta_skew: float = 1e-3
    delta_q_tilt: float = 1e-6

    def to_fixed_parameters_kwargs(self) -> dict[str, Any]:
        data = asdict(self)
        data["Frequency"] = _parse_float(self.Frequency, "RF frequency")
        if int(self.HarmNumber) <= 0:
            raise ValueError("Harmonic number must be a positive integer")
        data["HarmNumber"] = int(self.HarmNumber)
        data["dk"] = _literal_or_none(self.dk)
        return data

@dataclass(slots=True)
class LocoConfiguration:
    """Complete GUI LOCO configuration without importing numerical backend code."""

    machine_elements: MachineElementsConfig = field(default_factory=MachineElementsConfig)
    response_matrix: ResponseMatrixConfig = field(default_factory=ResponseMatrixConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    svd: SVDConfig = field(default_factory=SVDConfig)
    rejection: RejectionConfig = field(default_factory=RejectionConfig)
    constraints: ConstraintConfigState = field(default_factory=ConstraintConfigState)
    parameters: ParameterSelectionConfig = field(default_factory=ParameterSelectionConfig)
    fixed_parameters: FixedParameterConfig = field(default_factory=FixedParameterConfig)
    mcf_source: str = "automatic"
    mcf_user_value: str = ""
    output_directory: str = ""
    bad_bpm_positions: list[int] = field(default_factory=list)

    def to_backend_mapping(self) -> dict[str, Any]:
        """Return a serializable mapping of backend-compatible constructor data."""

        options = asdict(self.solver) | asdict(self.svd) | asdict(self.rejection)
        options["fit_list"] = self.parameters.fit_list()
        options["includeDispersion"] = self.rejection.includeDispersion
        self._sync_response_matrix_elements()
        cmstep = self.parameters.cmstep_value(len(self.machine_elements.horizontal_corrector_ords) or None, len(self.machine_elements.vertical_corrector_ords) or None)
        return {
            "LOCOOptions": options,
            "RMConfig": self.response_matrix.to_rm_config_kwargs() | {"dkick": cmstep, "Frequency": self.fixed_parameters.to_fixed_parameters_kwargs()["Frequency"], "HarmNumber": self.fixed_parameters.to_fixed_parameters_kwargs()["HarmNumber"]},
            "MachineElements": asdict(self.machine_elements),
            "FitInitConfig": self.parameters.to_fit_init_config_kwargs(),
            "ConstraintConfig": self.constraints.to_constraint_config_kwargs(),
            "FixedParameters": self.fixed_parameters.to_fixed_parameters_kwargs(),
            "MomentumCompaction": self.to_mcf_kwargs(),
            "Output": {"directory": self.output_directory},
            "BadBPMPositions": list(self.bad_bpm_positions),
        }

    def to_mcf_kwargs(self) -> dict[str, Any]:
        if self.mcf_source == "user":
            return {"source": "user", "value": _parse_float(self.mcf_user_value, "momentum compaction factor")}
        return {"source": "automatic", "value": None}

    def _sync_response_matrix_elements(self) -> None:
        self.response_matrix.bpm_ords = list(self.machine_elements.bpm_ords)
        self.response_matrix.cm_ords = (
            list(self.machine_elements.horizontal_corrector_ords),
            list(self.machine_elements.vertical_corrector_ords),
        )
        self.response_matrix.cav_ords = list(self.machine_elements.cavity_ords)

    def summary_lines(self) -> list[str]:
        self._sync_response_matrix_elements()
        fit_list = ", ".join(self.parameters.fit_list()) or "none"
        return [
            f"Response matrix: {self.response_matrix.calculator}, dispersion={self.response_matrix.includeDispersion}, coupling={self.response_matrix.coupling_orm}, bidirectional={self.response_matrix.bidirectional}",
            f"Solver: {self.solver.algorithm.upper()}, iterations={self.solver.nIter}, LM inner={self.solver.nLMIter}, scaled={self.solver.scaled}",
            f"SVD: method={self.svd.svd_selection_method}, threshold={self.svd.svd_threshold:g}, rank={self.svd.cut_}",
            f"Outliers: enabled={self.rejection.outlier_rejection}, sigma={self.rejection.sigma_outlier:g}, normalization={self.rejection.normalization_mode if self.rejection.apply_normalization else 'off'}",
            f"Constraints: enabled={self.constraints.enable}, quad_sigma={self.constraints.quad_sigma:g}, skew_sigma={self.constraints.skew_sigma:g}",
            f"Fit parameters: {fit_list}",
        ]

    def save(self, path: str | Path) -> Path:
        target = Path(path).expanduser()
        data = asdict(self)
        if target.suffix.lower() in {".yaml", ".yml"}:
            if importlib.util.find_spec("yaml") is None:
                raise RuntimeError("PyYAML is required to export YAML configuration files.")
            import yaml

            target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        else:
            target.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "LocoConfiguration":
        source = Path(path).expanduser()
        if source.suffix.lower() in {".yaml", ".yml"}:
            if importlib.util.find_spec("yaml") is None:
                raise RuntimeError("PyYAML is required to import YAML configuration files.")
            import yaml

            data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        else:
            data = json.loads(source.read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LocoConfiguration":
        if any(key in data for key in ("loco", "measurement", "rf", "elements", "data")):
            data = _example_config_to_gui(data)
        response_matrix = dict(data.get("response_matrix", {}))
        for key, label in (("dkick_h", "horizontal response-matrix kick step"), ("dkick_v", "vertical response-matrix kick step")):
            if key in response_matrix:
                response_matrix[key] = _parse_float(response_matrix[key], label)

        parameters = dict(data.get("parameters", {}))
        cmstep = dict(parameters.get("cmstep", {}))
        legacy_cmstep = {
            "CMstep_mode": "mode",
            "CMstep_h": "horizontal",
            "CMstep_v": "vertical",
            "CMstep_file": "file",
        }
        for legacy_key, cmstep_key in legacy_cmstep.items():
            if legacy_key in parameters:
                cmstep[cmstep_key] = parameters.pop(legacy_key)
        for key, label in (("horizontal", "horizontal corrector step"), ("vertical", "vertical corrector step")):
            if key in cmstep:
                cmstep[key] = _parse_float(cmstep[key], label)
        parameters["cmstep"] = CMStepConfig(**cmstep)

        return cls(
            machine_elements=MachineElementsConfig(**data.get("machine_elements", {})),
            response_matrix=ResponseMatrixConfig(**response_matrix),
            solver=SolverConfig(**data.get("solver", {})),
            svd=SVDConfig(**data.get("svd", {})),
            rejection=RejectionConfig(**data.get("rejection", {})),
            constraints=ConstraintConfigState(**data.get("constraints", {})),
            parameters=ParameterSelectionConfig(**parameters),
            fixed_parameters=FixedParameterConfig(**data.get("fixed_parameters", {})),
            mcf_source=data.get("mcf_source", "automatic"),
            mcf_user_value=data.get("mcf_user_value", ""),
            output_directory=data.get("output_directory", ""),
            bad_bpm_positions=[int(value) for value in data.get("bad_bpm_positions", [])],
        )


@dataclass(slots=True)
class ProjectMetadata:
    """Serializable GUI project state that does not touch numerical pyLOCO code."""

    name: str = "Untitled LOCO Project"
    mode: str = "Basic"
    path: str = ""
    modified: bool = False
    lattice: LatticeSelection = field(default_factory=LatticeSelection)
    measurements: dict[str, ImportedDataset] = field(default_factory=dict)
    loco_config: LocoConfiguration = field(default_factory=LocoConfiguration)
    recent_projects: list[str] = field(default_factory=list)

    @property
    def is_saved(self) -> bool:
        return bool(self.path) and not self.modified

    def validation_messages(self) -> list[str]:
        """Return missing required inputs for enabling LOCO execution."""

        messages: list[str] = []
        if not self.name.strip():
            messages.append("Project name is required.")
        if not self.lattice.path:
            messages.append("A lattice/model file is required.")
        elif not Path(self.lattice.path).expanduser().exists():
            messages.append(f"Lattice/model file does not exist: {self.lattice.path}")
        for role in REQUIRED_MEASUREMENTS:
            if role not in self.measurements:
                messages.append(f"{role.replace('_', ' ').title()} data is required.")
        for role, dataset in self.measurements.items():
            if not Path(dataset.path).expanduser().exists():
                messages.append(f"{role.replace('_', ' ').title()} file does not exist: {dataset.path}")
        include_dispersion = (
            self.loco_config.response_matrix.includeDispersion
            or self.loco_config.rejection.includeDispersion
        )
        if include_dispersion and "dispersion" not in self.measurements:
            messages.append("Dispersion data is required when dispersion fitting is enabled.")
        if not self.loco_config.parameters.fit_list():
            messages.append("At least one fitted parameter must be selected.")
        return messages

    @property
    def is_complete(self) -> bool:
        return not self.validation_messages()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["modified"] = False
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectMetadata":
        lattice_data = data.get("lattice") or {}
        measurement_data = data.get("measurements") or {}
        project = cls(
            name=data.get("name", "Untitled LOCO Project"),
            mode=data.get("mode", "Basic"),
            path=data.get("path", ""),
            modified=False,
            lattice=LatticeSelection(**lattice_data),
            measurements={
                key: ImportedDataset(**value) for key, value in measurement_data.items()
            },
            loco_config=LocoConfiguration.from_dict(data.get("loco_config", {})),
            recent_projects=list(data.get("recent_projects", [])),
        )
        return project

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path or self.path).expanduser()
        if not str(target).endswith(PROJECT_FILE_SUFFIX):
            target = target.with_suffix(PROJECT_FILE_SUFFIX)
        self.path = str(target)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        self.modified = False
        self.add_recent_project(target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "ProjectMetadata":
        source = Path(path).expanduser()
        project = cls.from_dict(json.loads(source.read_text(encoding="utf-8")))
        project.path = str(source)
        project.modified = False
        project.add_recent_project(source)
        return project

    def add_recent_project(self, path: str | Path) -> None:
        normalized = str(Path(path).expanduser())
        self.recent_projects = [p for p in self.recent_projects if p != normalized]
        self.recent_projects.insert(0, normalized)
        del self.recent_projects[5:]


def _example_config_to_gui(data: dict[str, Any]) -> dict[str, Any]:
    """Translate the maintained example YAML vocabulary to GUI state.

    The example files remain the source format; this adapter only maps their
    public settings onto the same backend-facing GUI configuration objects.
    """

    loco = data.get("loco") or {}
    measurement = data.get("measurement") or {}
    rf = data.get("rf") or {}
    if data.get("fit_parameters") is not None:
        from pyLOCO.user_config import selected_fit_parameters
        fit_names = set(selected_fit_parameters(data))
    else:
        fit_names = set(loco.get("fit_list") or loco.get("standard_fit_list") or ["quads"])
    kick = measurement.get("corrector_kick_rad", 1e-5)
    output = data.get("output") or {}
    output_directory = output.get("directory") or output.get("standard") or ""
    parameter_names = {
        name: name in fit_names
        for name in (
            "quads", "skew_quads", "quads_tilt", "hbpm_gain", "vbpm_gain",
            "hbpm_coupling", "vbpm_coupling", "hcor_cal", "vcor_cal",
            "hcor_coupling", "vcor_coupling", "HCMEnergyShift",
            "VCMEnergyShift", "delta_rf",
        )
    }
    parameters = parameter_names | {
        "cmstep": {
            "mode": "file" if (data.get("data") or {}).get("corrector_steps") else "uniform",
            "horizontal": kick,
            "vertical": kick,
            "file": (data.get("data") or {}).get("corrector_steps", ""),
        },
        "rfStep": rf.get("step_hz", -3000.0),
    }
    constraint_source = data.get("constraints") or {}
    quad_constraint = constraint_source.get("quadrupoles") or {}
    skew_constraint = constraint_source.get("skew_quadrupoles") or {}
    constraints = {
        "enable": bool(constraint_source.get("enable", False)),
        "quad_sigma": float(quad_constraint.get("sigma", quad_constraint.get("relative_sigma", 0.0))),
        "skew_sigma": float(skew_constraint.get("sigma", 0.0)),
        "quad_weights": (str(quad_constraint["weights"]) if "weights" in quad_constraint else ""),
        "skew_weights": (str(skew_constraint["weights"]) if "weights" in skew_constraint else ""),
        "quad_mask": (str(quad_constraint["mask"]) if "mask" in quad_constraint else ""),
        "skew_mask": (str(skew_constraint["mask"]) if "mask" in skew_constraint else ""),
    }
    return {
        "response_matrix": {
            "calculator": measurement.get("response_matrix_calculator", "Linear"),
            "dkick_h": kick,
            "dkick_v": kick,
            "rfStep": rf.get("step_hz", -3000.0),
            "includeDispersion": bool(loco.get("include_dispersion", False)),
        },
        "solver": {key: loco[key] for key in ("nIter", "nLMIter", "Starting_Lambda", "max_lm_lambda", "scaled") if key in loco},
        "svd": {
            **{key: loco[key] for key in ("svd_selection_method", "svd_threshold", "show_svd_plot") if key in loco},
            **({"cut_": loco["cut"]} if loco.get("cut") is not None else {}),
        },
        "rejection": {
            "outlier_rejection": loco.get("outlier_rejection", False),
            "sigma_outlier": loco.get("sigma_outlier", 10.0),
            "apply_normalization": loco.get("apply_normalization", False),
            "normalization_mode": loco.get("normalization_mode", "component"),
            "includeDispersion": bool(loco.get("include_dispersion", False)),
            "hor_dispersion_weight": loco.get("horizontal_dispersion_weight", 1.0),
            "ver_dispersion_weight": loco.get("vertical_dispersion_weight", 1.0),
            "remove_coupling_": loco.get("remove_coupling", True),
        },
        "parameters": parameters,
        "constraints": constraints,
        "fixed_parameters": {
            "Frequency": str(rf.get("frequency_hz", 499664399.4230182)),
            "rfstep": rf.get("step_hz", -3000.0),
        },
        "output_directory": output_directory,
        "bad_bpm_positions": data.get("bad_bpm_positions", []),
    }


def load_example_project_data(path: str | Path) -> tuple[LocoConfiguration, dict[str, str], str]:
    """Load configuration plus lattice/measurement paths from either YAML schema."""

    source = Path(path).expanduser().resolve()
    if source.suffix.lower() not in {".yaml", ".yml"}:
        return LocoConfiguration.load(source), {}, ""
    if importlib.util.find_spec("yaml") is None:
        raise RuntimeError("PyYAML is required to import YAML configuration files.")
    import yaml

    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    cfg = LocoConfiguration.from_dict(raw)
    base = source.parent
    data = raw.get("data") or {}
    measurements = {
        role: str((base / value).resolve())
        for role, value in data.items()
        if role in {"orm", "dispersion", "bpm_noise", "bad_bpms"} and value
    }
    lattice_value = (raw.get("lattice") or {}).get("file", "")
    lattice = str((base / lattice_value).resolve()) if lattice_value else ""
    cmstep = cfg.parameters.cmstep
    if cmstep.mode == "file" and cmstep.file:
        cmstep.file = str((base / cmstep.file).resolve())
    if cfg.output_directory:
        cfg.output_directory = str((base / cfg.output_directory).resolve())
    return cfg, measurements, lattice


def resolve_example_machine_elements(path: str | Path, lattice) -> MachineElementsConfig:
    """Resolve optional example name/index files against an already-loaded lattice."""

    source = Path(path).expanduser().resolve()
    if source.suffix.lower() not in {".yaml", ".yml"} or importlib.util.find_spec("yaml") is None:
        return MachineElementsConfig()
    import numpy as np
    import yaml

    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    data = raw.get("data") or {}
    base = source.parent

    def common_name_indices(key: str) -> list[int]:
        value = data.get(key)
        if not value:
            return []
        names_path = (base / value).resolve()
        if not names_path.exists():
            raise ValueError(f"Configured name file does not exist: {names_path}")
        return resolve_element_name_file(lattice, names_path)

    def index_file(key: str) -> list[int]:
        value = data.get(key)
        if not value:
            return []
        index_path = (base / value).resolve()
        if not index_path.exists():
            raise ValueError(f"Configured index file does not exist: {index_path}")
        values = np.asarray(np.load(index_path, allow_pickle=False))
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"Index file must contain a one-dimensional integer array: {index_path}")
        if np.any(values < 0) or np.any(values >= len(lattice)):
            raise ValueError(f"Index file contains ordinals outside lattice range 0..{len(lattice)-1}: {index_path}")
        return values.astype(int).tolist()

    return MachineElementsConfig(
        bpm_ords=common_name_indices("bpm_names"),
        horizontal_corrector_ords=common_name_indices("horizontal_corrector_names"),
        vertical_corrector_ords=common_name_indices("vertical_corrector_names"),
        normal_quadrupole_ords=index_file("quadrupole_indices"),
        skew_quadrupole_ords=index_file("skew_indices"),
    )


def resolve_element_name_file(lattice, path: str | Path, attribute: str = "auto") -> list[int]:
    """Resolve a text file of element names to lattice ordinals.

    Selection follows lattice order, matching the ORM ordering convention used
    by the maintained measured-data examples. ``auto`` checks common AT naming
    attributes without imposing one machine-specific convention.
    """

    source = Path(path).expanduser()
    if not source.exists():
        raise ValueError(f"Element-name file does not exist: {source}")
    ordered_names = [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not ordered_names:
        raise ValueError(f"Element-name file is empty: {source}")
    if len(set(ordered_names)) != len(ordered_names):
        raise ValueError(f"Element-name file contains duplicate names: {source}")
    names = set(ordered_names)
    allowed = ("CommonName", "FamName", "Name", "name")
    if attribute != "auto" and attribute not in allowed:
        raise ValueError(f"Unsupported element-name attribute: {attribute}")
    attributes = allowed if attribute == "auto" else (attribute,)

    def matched_name(element) -> str:
        return next(
            (str(getattr(element, key)) for key in attributes if str(getattr(element, key, "")) in names),
            "",
        )

    indices = [i for i, element in enumerate(lattice) if matched_name(element)]
    found = {matched_name(lattice[i]) for i in indices}
    missing = names - found
    if missing:
        sample = ", ".join(sorted(missing)[:5])
        raise ValueError(f"{len(missing)} element name(s) were not found in the lattice: {sample}")
    return indices
