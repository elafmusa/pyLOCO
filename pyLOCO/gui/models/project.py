"""UI-only project state for the pyLOCO GUI."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROJECT_FILE_SUFFIX = ".pyloco.json"
REQUIRED_MEASUREMENTS = ("orm", "dispersion", "bpm_noise")


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
class ResponseMatrixConfig:
    """GUI state matching the backend RMConfig constructor fields."""

    bpm_ords: str = ""
    cm_ords: str = ""
    cav_ords: str = ""
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
            "bpm_ords": _literal_or_none(self.bpm_ords),
            "cm_ords": _literal_or_none(self.cm_ords),
            "cav_ords": _literal_or_none(self.cav_ords),
            "dkick": (self.dkick_h, self.dkick_v),
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
        return asdict(self)


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
    CMstep_h: float = 1e-5
    CMstep_v: float = 1e-5
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
            "init_policy": _literal_or_none(self.init_policy),
            "CMstep": (self.CMstep_h, self.CMstep_v),
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

    Frequency: float = 499664399.4230182
    HarmNumber: int = 3840
    rfstep: float = -3000.0
    dk: str = ""
    delta_skew: float = 1e-3
    delta_q_tilt: float = 1e-6

    def to_fixed_parameters_kwargs(self) -> dict[str, Any]:
        data = asdict(self)
        data["dk"] = _literal_or_none(self.dk)
        return data

@dataclass(slots=True)
class LocoConfiguration:
    """Complete GUI LOCO configuration without importing numerical backend code."""

    response_matrix: ResponseMatrixConfig = field(default_factory=ResponseMatrixConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    svd: SVDConfig = field(default_factory=SVDConfig)
    rejection: RejectionConfig = field(default_factory=RejectionConfig)
    constraints: ConstraintConfigState = field(default_factory=ConstraintConfigState)
    parameters: ParameterSelectionConfig = field(default_factory=ParameterSelectionConfig)
    fixed_parameters: FixedParameterConfig = field(default_factory=FixedParameterConfig)

    def to_backend_mapping(self) -> dict[str, Any]:
        """Return a serializable mapping of backend-compatible constructor data."""

        options = asdict(self.solver) | asdict(self.svd) | asdict(self.rejection)
        options["fit_list"] = self.parameters.fit_list()
        options["includeDispersion"] = self.rejection.includeDispersion
        return {
            "LOCOOptions": options,
            "RMConfig": self.response_matrix.to_rm_config_kwargs(),
            "FitInitConfig": self.parameters.to_fit_init_config_kwargs(),
            "ConstraintConfig": self.constraints.to_constraint_config_kwargs(),
            "FixedParameters": self.fixed_parameters.to_fixed_parameters_kwargs(),
        }

    def summary_lines(self) -> list[str]:
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
        return cls(
            response_matrix=ResponseMatrixConfig(**data.get("response_matrix", {})),
            solver=SolverConfig(**data.get("solver", {})),
            svd=SVDConfig(**data.get("svd", {})),
            rejection=RejectionConfig(**data.get("rejection", {})),
            constraints=ConstraintConfigState(**data.get("constraints", {})),
            parameters=ParameterSelectionConfig(**data.get("parameters", {})),
            fixed_parameters=FixedParameterConfig(**data.get("fixed_parameters", {})),
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
        for role in REQUIRED_MEASUREMENTS:
            if role not in self.measurements:
                messages.append(f"{role.replace('_', ' ').title()} data is required.")
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
