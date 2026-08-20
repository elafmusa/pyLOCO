"""Generic, YAML-driven preparation and execution for measured machines."""
from __future__ import annotations

import copy
import fnmatch
import pickle
import re
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import at
import h5py
import numpy as np
import yaml

from pyLOCO.config import ConstraintConfig, FitInitConfig, FitResumeConfig, RMConfig, fixed_parameters
from pyLOCO.user_config import build_constraints, selected_fit_parameters
from pyLOCO.pyloco import pyloco, remove_bad_bpms
from pyLOCO.response_matrix import response_matrix
from .diagnostics import make_diagnostic_plots, save_run_results


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream) or {}
    for section in ("lattice", "data", "rf", "loco", "output"):
        if section not in cfg:
            raise ValueError(f"Missing YAML section: {section}")
    return cfg


# Historical name retained for PETRA example imports.
load_yaml = load_config


def _require_datasets(stream: h5py.File, path: Path, names: tuple[str, ...]) -> None:
    missing = [name for name in names if name not in stream]
    if missing:
        raise ValueError(
            f"{path} is missing required dataset(s): {', '.join(missing)}"
        )


def _load_family_groups(path: Path) -> list[list[int]]:
    groups = np.load(path, allow_pickle=True).tolist()
    if not isinstance(groups, list) or not groups:
        raise ValueError(f"Quadrupole family file {path} must contain a non-empty list")
    result = []
    for number, group in enumerate(groups):
        members = np.atleast_1d(group).astype(int).tolist()
        if not members:
            raise ValueError(f"Quadrupole family {number} in {path} is empty")
        result.append(members)
    return result


def _names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _validate_indices(indices: Any, ring: at.Lattice, label: str) -> np.ndarray:
    result = np.asarray(indices, dtype=int).ravel()
    if result.size != len(set(result.tolist())):
        raise ValueError(f"{label} selection contains duplicate lattice indices")
    invalid = result[(result < 0) | (result >= len(ring))]
    if invalid.size:
        raise ValueError(f"{label} lattice index {int(invalid[0])} is outside 0..{len(ring)-1}")
    return result


def select_elements(ring: at.Lattice, spec: dict[str, Any], base: Path,
                    label: str) -> np.ndarray:
    """Resolve a common YAML element selector without machine-specific code."""
    if not isinstance(spec, dict):
        raise ValueError(f"elements.{label} must be a mapping")
    methods = [name for name in (
        "indices", "indices_file", "names_file", "element_type", "family_name",
        "pattern", "regex") if spec.get(name) is not None]
    if len(methods) != 1:
        raise ValueError(f"elements.{label} must define exactly one selection method")
    method = methods[0]
    attribute = str(spec.get("name_attribute", "CommonName"))
    if method == "indices":
        indices = spec[method]
    elif method == "indices_file":
        indices = np.load(base / spec[method], allow_pickle=False)
    elif method == "names_file":
        requested = _names(base / spec[method])
        by_name: dict[str, list[int]] = {}
        for index, element in enumerate(ring):
            value = getattr(element, attribute, None)
            if value is not None:
                by_name.setdefault(str(value), []).append(index)
        missing = [name for name in requested if name not in by_name]
        ambiguous = [name for name in requested if len(by_name.get(name, ())) > 1]
        if missing:
            raise ValueError(f"elements.{label} names not found via {attribute}: {missing[:5]}")
        if ambiguous and not bool(spec.get("allow_repeated_names", False)):
            raise ValueError(f"elements.{label} names are ambiguous via {attribute}: {ambiguous[:5]}")
        indices = [index for name in requested for index in by_name[name]]
        if str(spec.get("order", "names_file")) == "lattice":
            indices = sorted(indices)
    elif method == "element_type":
        element_class = getattr(at.elements, str(spec[method]), None)
        if element_class is None:
            raise ValueError(f"Unknown Accelerator Toolbox element type: {spec[method]}")
        indices = at.get_refpts(ring, element_class)
    else:
        matcher = (lambda value: value == str(spec[method])) if method == "family_name" else (
            (lambda value: fnmatch.fnmatch(value, str(spec[method]))) if method == "pattern"
            else (lambda value: re.search(str(spec[method]), value) is not None)
        )
        indices = [i for i, element in enumerate(ring)
                   if matcher(str(getattr(element, attribute, "")))]
    result = _validate_indices(indices, ring, f"elements.{label}")
    if not result.size and not bool(spec.get("optional", False)):
        raise ValueError(f"elements.{label} matched no lattice elements")
    return result


def _file_spec(value: Any, *, dataset: str | None = None) -> dict[str, Any]:
    return ({"file": value, **({"dataset": dataset} if dataset else {})}
            if isinstance(value, (str, Path)) else dict(value or {}))


def _read_hdf5(base: Path, spec: dict[str, Any], datasets: tuple[str, ...]) -> list[np.ndarray]:
    path = base / spec["file"]
    with h5py.File(path, "r") as stream:
        _require_datasets(stream, path, datasets)
        return [np.asarray(stream[name]) for name in datasets]


def _legacy_elements(cfg: dict[str, Any]) -> dict[str, Any]:
    data = cfg["data"]
    return {
        "bpms": {"names_file": data["bpm_names"], "name_attribute": "CommonName",
                  "allow_repeated_names": True, "order": "lattice"},
        "horizontal_correctors": {"names_file": data["horizontal_corrector_names"],
                                  "name_attribute": "CommonName", "allow_repeated_names": True,
                                  "order": "lattice"},
        "vertical_correctors": {"names_file": data["vertical_corrector_names"],
                                "name_attribute": "CommonName", "allow_repeated_names": True,
                                "order": "lattice"},
        "quadrupoles": {"indices_file": data["quadrupole_indices"],
                        "mode": data.get("quadrupole_mode", "individual"),
                        "family_groups_file": data.get("quadrupole_family_groups")},
        "skew_quadrupoles": {"indices_file": data["skew_indices"], "optional": True},
        "cavities": {"element_type": "RFCavity", "optional": True},
    }


def _remove_configured_correctors(
    ring: at.Lattice, measured_orm: np.ndarray, hcor: np.ndarray,
    vcor: np.ndarray, cm_step: list[np.ndarray], removals: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Remove measured ORM columns and matching correctors requested in YAML."""
    h_positions: list[int] = []
    v_positions: list[int] = []
    for number, item in enumerate(removals or []):
        if not isinstance(item, dict):
            raise ValueError(f"data.orm.remove_correctors item {number} must be a mapping")
        plane = str(item.get("plane", "")).lower()
        if plane not in {"horizontal", "vertical"}:
            raise ValueError("Removed corrector plane must be horizontal or vertical")
        selected = hcor if plane == "horizontal" else vcor
        if "position" in item:
            position = int(item["position"])
            if position < 0 or position >= len(selected):
                raise ValueError(f"Removed {plane} corrector position {position} is out of range")
        else:
            if "name" not in item:
                raise ValueError("Removed corrector requires either position or name")
            attribute = str(item.get("name_attribute", "CommonName"))
            matches = [i for i, lattice_index in enumerate(selected)
                       if str(getattr(ring[int(lattice_index)], attribute, "")) == str(item["name"])]
            if len(matches) != 1:
                raise ValueError(
                    f"Removed {plane} corrector {item['name']!r} matched {len(matches)} "
                    f"selected elements via {attribute}; expected exactly one")
            position = matches[0]
        (h_positions if plane == "horizontal" else v_positions).append(position)
    if len(h_positions) != len(set(h_positions)) or len(v_positions) != len(set(v_positions)):
        raise ValueError("data.orm.remove_correctors contains duplicate selections")
    columns = h_positions + [len(hcor) + position for position in v_positions]
    if columns:
        measured_orm = np.delete(measured_orm, columns, axis=1)
        hcor = np.delete(hcor, h_positions)
        vcor = np.delete(vcor, v_positions)
        cm_step = [np.delete(cm_step[0], h_positions), np.delete(cm_step[1], v_positions)]
    return measured_orm, hcor, vcor, cm_step


def prepare_measurement(config_path: Path) -> dict[str, Any]:
    """Load and validate a measured-machine problem described entirely by YAML."""
    config_path = Path(config_path).resolve()
    cfg = load_config(config_path)
    base = config_path.parent
    data_cfg = cfg["data"]
    ring = at.load_lattice(base / cfg["lattice"]["file"])
    if bool(cfg["lattice"].get("disable_6d", True)):
        ring.disable_6d()

    elements = cfg.get("elements") or _legacy_elements(cfg)
    bpms = select_elements(ring, elements["bpms"], base, "bpms")
    hcor = select_elements(ring, elements["horizontal_correctors"], base,
                           "horizontal_correctors")
    vcor = select_elements(ring, elements["vertical_correctors"], base,
                           "vertical_correctors")
    quad_spec = elements["quadrupoles"]
    individual_quad_indices = select_elements(ring, quad_spec, base, "quadrupoles")
    skew_indices = select_elements(
        ring, elements.get("skew_quadrupoles", {"indices": [], "optional": True}),
        base, "skew_quadrupoles")
    cavities = select_elements(
        ring, elements.get("cavities", {"element_type": "RFCavity", "optional": True}),
        base, "cavities")

    quadrupole_mode = str(quad_spec.get("mode", "individual")).lower()
    if quadrupole_mode == "individual":
        quad_indices = individual_quad_indices
        quad_individuals = True
    elif quadrupole_mode == "family":
        family_file = quad_spec.get("family_groups_file")
        if not family_file:
            raise ValueError(
                "elements.quadrupoles.family_groups_file is required in family mode"
            )
        quad_indices = _load_family_groups(base / family_file)
        quad_individuals = False
    else:
        raise ValueError("data.quadrupole_mode must be 'individual' or 'family'")
    groups_to_check = (
        [[int(index)] for index in quad_indices] if quad_individuals else quad_indices
    )
    physical_set = set(individual_quad_indices.tolist())
    mapped: list[int] = []
    for number, group in enumerate(groups_to_check):
        if any(index < 0 or index >= len(ring) for index in group):
            raise ValueError(f"Quadrupole family/parameter {number} contains an invalid lattice index")
        if not set(group).issubset(physical_set):
            raise ValueError(f"Quadrupole family {number} contains magnets outside the physical selection")
        mapped.extend(group)
        if not quad_individuals:
            strengths = np.asarray([ring[index].PolynomB[1] for index in group], dtype=float)
            if not np.allclose(strengths, strengths[0], rtol=1.0e-10, atol=1.0e-14):
                raise ValueError(
                    f"Quadrupole family {number} contains unequal nominal PolynomB[1] strengths"
                )
    if len(mapped) != len(set(mapped)):
        raise ValueError("Quadrupole family mapping contains duplicate physical magnets")

    orm_spec = _file_spec(data_cfg["orm"], dataset="response_matrix")
    orm_dataset = str(orm_spec.get("dataset", "response_matrix"))
    measured_orm = _read_hdf5(base, orm_spec, (orm_dataset,))[0]
    if bool(orm_spec.get("transpose", False)):
        measured_orm = measured_orm.T
    measured_orm = np.asarray(measured_orm, dtype=float) * float(orm_spec.get("scale", 1.0))

    step_cfg = orm_spec.get("corrector_steps", data_cfg.get("corrector_steps", 1e-5))
    if isinstance(step_cfg, dict):
        cm_step = [
            np.full(len(hcor), float(step_cfg.get("horizontal", step_cfg.get("horizontal_rad")))),
            np.full(len(vcor), float(step_cfg.get("vertical", step_cfg.get("vertical_rad")))),
        ]
    elif isinstance(step_cfg, (str, Path)):
        steps = np.load(base / step_cfg)
        cm_step = [np.asarray(steps[str(orm_spec.get("horizontal_step_dataset", "hor"))]),
                   np.asarray(steps[str(orm_spec.get("vertical_step_dataset", "ver"))])]
    else:
        cm_step = [np.full(len(hcor), float(step_cfg)), np.full(len(vcor), float(step_cfg))]
    if len(cm_step[0]) != len(hcor) or len(cm_step[1]) != len(vcor):
        raise ValueError("Corrector-step arrays do not match the selected H/V correctors")

    dispersion_spec = _file_spec(data_cfg.get("dispersion"))
    legacy_dispersion_names = data_cfg.get("dispersion_datasets", {})
    dispersion_enabled = bool(dispersion_spec.get("enable", bool(dispersion_spec.get("file"))))
    if dispersion_enabled:
        x_name = str(dispersion_spec.get("horizontal_dataset",
                     legacy_dispersion_names.get("horizontal", "measured_eta_x")))
        y_name = str(dispersion_spec.get("vertical_dataset",
                     legacy_dispersion_names.get("vertical", "measured_eta_y")))
        eta_x, eta_y = _read_hdf5(base, dispersion_spec, (x_name, y_name))
        eta_x = np.asarray(eta_x, dtype=float).ravel() * float(dispersion_spec.get("horizontal_scale", 1.0))
        eta_y = np.asarray(eta_y, dtype=float).ravel() * float(dispersion_spec.get("vertical_scale", 1.0))
    else:
        eta_x = np.zeros(len(bpms)); eta_y = np.zeros(len(bpms))

    noise_spec = _file_spec(data_cfg.get("bpm_noise"))
    legacy_noise_names = data_cfg.get("bpm_noise_datasets", {})
    noise_enabled = bool(noise_spec.get("enable", bool(noise_spec.get("file"))))
    if noise_enabled:
        x_name = str(noise_spec.get("horizontal_dataset",
                     legacy_noise_names.get("horizontal", "Noise_BPMx")))
        y_name = str(noise_spec.get("vertical_dataset",
                     legacy_noise_names.get("vertical", "Noise_BPMy")))
        noise_x, noise_y = _read_hdf5(base, noise_spec, (x_name, y_name))
        noise_x = np.asarray(noise_x, dtype=float).ravel() * float(noise_spec.get("horizontal_scale", 1.0))
        noise_y = np.asarray(noise_y, dtype=float).ravel() * float(noise_spec.get("vertical_scale", 1.0))
    else:
        sigma = float(noise_spec.get("default_sigma", 1.0))
        noise_x = np.full(len(bpms), sigma); noise_y = np.full(len(bpms), sigma)

    if bool(cfg["loco"].get("include_dispersion", False)) and not dispersion_enabled:
        raise ValueError("loco.include_dispersion requires data.dispersion.enable: true")
    if bool(cfg["loco"].get("include_dispersion", False)) and not cavities.size:
        raise ValueError("Dispersion fitting requires at least one selected RF cavity")

    expected_raw = (2 * len(bpms), len(hcor) + len(vcor))
    if measured_orm.shape != expected_raw:
        raise ValueError(f"ORM shape {measured_orm.shape} does not match selected elements {expected_raw}")
    row_order = list(orm_spec.get("row_order", ["horizontal", "vertical"]))
    column_order = list(orm_spec.get("column_order", ["horizontal", "vertical"]))
    if row_order == ["vertical", "horizontal"]:
        measured_orm = np.vstack((measured_orm[len(bpms):], measured_orm[:len(bpms)]))
    elif row_order != ["horizontal", "vertical"]:
        raise ValueError("data.orm.row_order must be [horizontal, vertical] or [vertical, horizontal]")
    if column_order == ["vertical", "horizontal"]:
        measured_orm = np.hstack((measured_orm[:, len(vcor):], measured_orm[:, :len(vcor)]))
    elif column_order != ["horizontal", "vertical"]:
        raise ValueError("data.orm.column_order must be [horizontal, vertical] or [vertical, horizontal]")

    measured_orm, hcor, vcor, cm_step = _remove_configured_correctors(
        ring, measured_orm, hcor, vcor, cm_step, orm_spec.get("remove_correctors"))

    bad = np.asarray(cfg.get("bad_bpm_positions", []), dtype=int)
    if np.any((bad < 0) | (bad >= len(bpms))):
        raise ValueError("bad_bpm_positions contains an out-of-range position")
    cleaned_orm, _ = remove_bad_bpms(
        measured_orm, bad, total_bpms=len(bpms), axis=0, input_type="positions"
    )
    good_bpms = np.delete(bpms, bad)
    result = {
        "cfg": cfg,
        "config_path": config_path,
        "ring": ring,
        "quad_indices": quad_indices,
        "individual_quad_indices": individual_quad_indices,
        "quad_individuals": quad_individuals,
        "skew_indices": skew_indices,
        "correctors": [hcor, vcor],
        "bpms": good_bpms,
        "cavities": cavities,
        "cm_step": cm_step,
        "orm": cleaned_orm,
        "eta_x": np.delete(eta_x, bad),
        "eta_y": np.delete(eta_y, bad),
        "weights": np.concatenate((np.delete(noise_x, bad), np.delete(noise_y, bad))),
    }
    expected = (2 * len(good_bpms), len(hcor) + len(vcor))
    if cleaned_orm.shape != expected:
        raise ValueError(f"Cleaned ORM shape {cleaned_orm.shape} does not match {expected}")
    if len(result["weights"]) != cleaned_orm.shape[0]:
        raise ValueError("BPM uncertainty vector does not match the cleaned ORM")
    if len(result["eta_x"]) != len(good_bpms) or len(result["eta_y"]) != len(good_bpms):
        raise ValueError("Dispersion vectors do not match the cleaned BPM selection")
    expected_counts = cfg.get("expected_counts", {})
    actual_counts = {
        "lattice_elements": len(ring),
        "bpms_before_removal": len(bpms), "bpms_after_removal": len(good_bpms),
        "horizontal_correctors": len(hcor), "vertical_correctors": len(vcor),
        "quadrupole_families": len(quad_indices),
        "physical_quadrupoles": len(individual_quad_indices),
        "skew_quadrupoles": len(skew_indices),
    }
    for name, expected_count in expected_counts.items():
        if name not in actual_counts:
            raise ValueError(f"Unknown expected_counts field: {name}")
        if actual_counts[name] != int(expected_count):
            raise ValueError(
                f"Expected {expected_count} {name.replace('_', ' ')}, found {actual_counts[name]}"
            )
    result["resume"] = load_resume_state(cfg.get("resume"), base)
    if result["resume"] is not None:
        result["ring"] = result["resume"]["ring"]
    return result


def load_resume_state(value: Any, base: Path) -> dict[str, Any] | None:
    """Load the fitted lattice and parameters from a previous example run."""
    raw = value or {}
    resume = FitResumeConfig(
        enabled=bool(raw.get("enabled", False)), directory=raw.get("directory"),
        ring_file=str(raw.get("ring_file", "ring_pyloco.mat")),
        fit_dict_file=str(raw.get("fit_dict_file", "fit_dict.pkl")),
        fit_results_file=raw.get("fit_results_file", "fit_results.npy"),
    )
    if not resume.enabled:
        return None
    directory = (base / Path(resume.directory)).resolve()
    results = directory if directory.name == "results" else directory / "results"
    ring_path = results / resume.ring_file
    fit_dict_path = results / resume.fit_dict_file
    missing = [str(path) for path in (ring_path, fit_dict_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Previous fit is missing: " + ", ".join(missing))
    ring = at.load_lattice(ring_path)
    ring.disable_6d()
    with fit_dict_path.open("rb") as stream:
        fit_dict = pickle.load(stream)
    if not isinstance(fit_dict, dict) or not fit_dict:
        raise ValueError(f"Previous fit dictionary {fit_dict_path} is empty or invalid")
    fit_results = None
    if resume.fit_results_file:
        fit_results_path = results / resume.fit_results_file
        if fit_results_path.is_file():
            fit_results = np.load(fit_results_path, allow_pickle=True).tolist()
    return {"ring": ring, "fit_dict": fit_dict, "fit_results": fit_results,
            "results_directory": results}


def build_constraint_config(data: dict[str, Any]) -> ConstraintConfig | None:
    """Translate YAML constraints without affecting fitted-parameter selection."""
    nominal = [
        float(getattr(data["ring"][group[0] if not np.isscalar(group) else group], "K"))
        for group in data["quad_indices"]
    ]
    bpms = data.get("bpms", ())
    correctors = data.get("correctors", ((), ()))
    return build_constraints(
    data["cfg"],
    quad_nominal=nominal,
    n_skew=len(data["skew_indices"]),
        n_hbpm=len(bpms),
        n_vbpm=len(bpms),
        n_hcor=len(correctors[0]),
        n_vcor=len(correctors[1]),
)


def _optional_mask(value: Any, size: int, label: str) -> np.ndarray | None:
    if value is None:
        return None
    mask = np.asarray(value, dtype=bool)
    if mask.size != size:
        raise ValueError(f"The {label} constraint mask has {mask.size} entries; expected {size}")
    return mask


def _required_vector(value: Any, size: int, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float).ravel()
    if vector.size != size:
        raise ValueError(f"The {label} vector has {vector.size} entries; expected {size}")
    return vector


def _scalar_or_vector(value: Any, size: int, label: str) -> float | np.ndarray:
    vector = np.asarray(value, dtype=float).ravel()
    if vector.size == 1:
        return float(vector[0])
    return _required_vector(vector, size, label)


def model_orm(data: dict[str, Any]) -> np.ndarray:
    cfg = data["cfg"]
    rm_cfg = RMConfig(
        bpm_ords=data["bpms"], cm_ords=data["correctors"], cav_ords=data["cavities"],
        dkick=data["cm_step"], bidirectional=True, includeDispersion=False,
        rfStep=float(cfg["rf"]["step_hz"]), Frequency=float(cfg["rf"]["frequency_hz"]),
        HarmNumber=int(cfg["rf"].get("harmonic_number", fixed_parameters.HarmNumber)),
        fixedpathlength=False,
    )
    return response_matrix(data["ring"], config=rm_cfg)


def model_dispersion(data: dict[str, Any], ring: at.Lattice) -> dict[str, np.ndarray]:
    """Calculate the RF minus/plus orbit difference using pyLOCO's ORM convention."""
    if not len(data["cavities"]):
        zeros = np.zeros(len(data["bpms"]))
        return {"x": zeros.copy(), "y": zeros.copy()}
    cfg = data["cfg"]
    rm_cfg = RMConfig(
        bpm_ords=data["bpms"], cm_ords=data["correctors"], cav_ords=data["cavities"],
        dkick=data["cm_step"], bidirectional=True, includeDispersion=True,
        rfStep=float(cfg["rf"]["step_hz"]), Frequency=float(cfg["rf"]["frequency_hz"]),
        HarmNumber=int(cfg["rf"].get("harmonic_number", fixed_parameters.HarmNumber)),
        fixedpathlength=False,
    )
    matrix = response_matrix(ring, config=rm_cfg)
    vector = matrix[:, -1]
    n_bpm = len(data["bpms"])
    return {"x": vector[:n_bpm], "y": vector[n_bpm:]}


def run_fit(data: dict[str, Any], *, coupling: bool | None = None,
            constrained: bool | None = None) -> dict[str, Any]:
    """Run the fit described by YAML; legacy flags no longer change its policy."""
    cfg = data["cfg"]
    loco = cfg["loco"]
    fit_list = selected_fit_parameters(cfg)
    coupling = any(name in fit_list for name in (
        "skew_quads", "quads_tilt", "hbpm_coupling", "vbpm_coupling",
        "hcor_coupling", "vcor_coupling", "VCMEnergyShift"))
    fit_cfg = FitInitConfig(
        fit_list=fit_list, CMstep=data["cm_step"], rfStep=float(cfg["rf"]["step_hz"]),
        individuals=data["quad_individuals"], quads_attr="PolynomB", quads_attr_index=1,
        skew_attr=str(loco.get("skew_attribute", "PolynomB")), skew_attr_index=1,
    )
    constraint_cfg = build_constraint_config(data)
    constrained = constraint_cfg is not None
    temporary = TemporaryDirectory(prefix="pyloco_measured_machine_")
    original_frequency = fixed_parameters.Frequency
    original_harmonic = fixed_parameters.HarmNumber
    fixed_parameters.Frequency = float(cfg["rf"]["frequency_hz"])
    fixed_parameters.HarmNumber = int(cfg["rf"].get("harmonic_number", original_harmonic))
    initial_chi2: list[float] = []
    try:
        measured_for_fit = data["orm"]
        if bool(loco["include_dispersion"]):
            eta = np.concatenate((data["eta_x"], data["eta_y"]))
            measured_for_fit = np.hstack((measured_for_fit, eta[:, None]))
        result = pyloco(
            copy.deepcopy(data["ring"]), algorithm=str(loco.get("algorithm", "lm")),
            nIter=int(loco["nIter"]),
            used_bpms_ords=data["bpms"], used_cor_ords=data["correctors"],
            quads_ords=data["quad_indices"],
            skew_ords=data["skew_indices"] if coupling else np.array([], dtype=int),
            CAVords=data["cavities"], nHBPM=len(data["bpms"]), nVBPM=len(data["bpms"]),
            nHorCOR=len(data["correctors"][0]), nVerCOR=len(data["correctors"][1]),
            quads_tilt_ind=data["individual_quad_indices"], orm_measured=measured_for_fit,
            weights=data["weights"], includeDispersion=bool(loco["include_dispersion"]),
            measured_eta_x=data["eta_x"], measured_eta_y=data["eta_y"],
            hor_dispersion_weight=float(loco["horizontal_dispersion_weight"]),
            ver_dispersion_weight=float(loco["vertical_dispersion_weight"]),
            CMstep=data["cm_step"], rfStep=float(cfg["rf"]["step_hz"]),
            Frequency=float(cfg["rf"]["frequency_hz"]), fit_list=fit_list,
            quad_individuals=data["quad_individuals"],
            skew_individuals=bool(loco.get("skew_individuals", True)),
            tilt_individuals=bool(loco.get("tilt_individuals", True)),
            remove_coupling_=bool(loco.get("remove_coupling", not coupling)),
            outlier_rejection=bool(loco["outlier_rejection"]),
            sigma_outlier=float(loco["sigma_outlier"]),
            apply_normalization=bool(loco["apply_normalization"]),
            normalization_mode=str(loco["normalization_mode"]),
            svd_selection_method=str(loco["svd_selection_method"]),
            svd_threshold=float(loco["svd_threshold"]), cut_=loco.get("cut"),
            show_svd_plot=bool(loco["show_svd_plot"]), nLMIter=int(loco["nLMIter"]),
            Starting_Lambda=float(loco["Starting_Lambda"]),
            max_lm_lambda=float(loco["max_lm_lambda"]), scaled=bool(loco["scaled"]),
            plot_fit_parameters=bool(loco.get("plot_fit_parameters", False)),
            auto_correct_delta=bool(loco.get("auto_correct_delta", True)),
            fixedpathlength=bool(loco.get("fixedpathlength", False)),
            fixedmomentum=False, fit_cfg=fit_cfg, constraint_cfg=constraint_cfg,
            calculate_delta_chi2=bool(loco.get("calculate_delta_chi2", False)),
            initial_chi2_callback=initial_chi2.append,
            continue_from_previous=data.get("resume") is not None,
            previous_ring=(copy.deepcopy(data["resume"]["ring"])
                           if data.get("resume") is not None else None),
            previous_fit_dict=(data["resume"]["fit_dict"]
                               if data.get("resume") is not None else None),
            previous_fit_results=(data["resume"]["fit_results"]
                                  if data.get("resume") is not None else None),
            output_dir=temporary.name,
        )
        # ------------------------------------------------------
        # Preserve Jacobians produced inside pyLOCO's temporary
        # working directory before TemporaryDirectory is deleted.
        # ------------------------------------------------------
        temporary_output = Path(temporary.name)

        permanent_output = output_directory(
            data,
            coupling=coupling,
            constrained=constrained,
        )

        source_jacobians = temporary_output / "jacobians"
        destination_jacobians = permanent_output / "jacobians"

        if source_jacobians.exists():
            shutil.copytree(
                source_jacobians,
                destination_jacobians,
                dirs_exist_ok=True,
            )

            print(
                f"[Jacobian] Preserved Jacobians in "
                f"{destination_jacobians}"
            )
    finally:
        fixed_parameters.Frequency = original_frequency
        fixed_parameters.HarmNumber = original_harmonic
        temporary.cleanup()
    fit_results, fit_dict, fitted_ring, fitted_matrix, c_bpms, chi2, delta_chi2, blocks = result
    include_dispersion = bool(loco["include_dispersion"])
    fitted_orm = fitted_matrix[:, :-1] if include_dispersion else fitted_matrix
    if include_dispersion:
        n_bpm = len(data["bpms"])
        fitted_dispersion = {"x": fitted_matrix[:n_bpm, -1], "y": fitted_matrix[n_bpm:, -1]}
    else:
        fitted_dispersion = model_dispersion(data, fitted_ring)
    return {"fit_results": fit_results, "fit_dict": fit_dict, "ring": fitted_ring,
            "orm": fitted_orm, "chi2": chi2, "fit_list": fit_list,
            "constraint_cfg": constraint_cfg, "constrained": constrained,
            "c_bpms": c_bpms, "delta_chi2": delta_chi2, "blocks": blocks,
            "initial_chi2": initial_chi2[0] if initial_chi2 else None,
            "resumed_from": (str(data["resume"]["results_directory"])
                             if data.get("resume") is not None else None),
            "initial_dispersion": model_dispersion(data, data["ring"]),
            "fitted_dispersion": fitted_dispersion}


def fit_modes(fit: dict[str, Any]) -> tuple[bool, bool]:
    """Return coupling/constraint presentation modes from an actual fit."""
    coupling = any(name in fit["fit_list"] for name in (
        "skew_quads", "quads_tilt", "hbpm_coupling", "vbpm_coupling",
        "hcor_coupling", "vcor_coupling"))
    return coupling, fit["constraint_cfg"] is not None


def output_directory(data: dict[str, Any], *, coupling: bool, constrained: bool) -> Path:
    output_cfg = data["cfg"]["output"]
    configured = output_cfg.get("directory")
    if configured is None and output_cfg.get("root") is not None:
        run_name = str(output_cfg.get("run_name", "")).strip()
        if not run_name:
            raise ValueError("output.run_name is required when output.root is used")
        configured = Path(output_cfg["root"]) / run_name
    if configured is None:
        configured = output_cfg["constrained" if constrained else ("coupling" if coupling else "standard")]
    return (data["config_path"].parent / configured).resolve()


def make_plots(data: dict[str, Any], initial_orm: np.ndarray, fit: dict[str, Any], *, coupling: bool, constrained: bool = False) -> Path:
    output = output_directory(data, coupling=coupling, constrained=constrained)
    save_run_results(data, initial_orm, fit, output)
    make_diagnostic_plots(data, initial_orm, fit, output, coupling=coupling)
    return output


def print_summary(data: dict[str, Any], initial_orm: np.ndarray, fit: dict[str, Any], *, coupling: bool, constrained: bool = False) -> None:
    before = float(np.sqrt(np.mean((initial_orm - data["orm"]) ** 2)))
    after = float(np.sqrt(np.mean((fit["orm"] - data["orm"]) ** 2)))
    machine = data["cfg"].get("machine", {}).get("name", "Measured machine")
    title = f"{machine} constrained fit" if constrained else (
        f"{machine} coupling fit" if coupling else f"{machine} measured-ORM fit"
    )
    print(f"\n{title}")
    print("-" * 46)
    print(f"Measured ORM shape : {data['orm'].shape}")
    print(f"Retained BPMs      : {len(data['bpms'])} per plane")
    print(f"Correctors         : {len(data['correctors'][0])} H, {len(data['correctors'][1])} V")
    print(f"Fitted parameters  : {', '.join(fit['fit_list'])}")
    if fit.get("resumed_from"):
        print(f"Resumed from       : {fit['resumed_from']}")
    print(f"ORM RMS before     : {1e6*before:.6f} µm")
    print(f"ORM RMS after      : {1e6*after:.6f} µm")
    print(f"Improvement        : {before/after:.3f}x")
    metrics = fit.get("metrics", {})
    if metrics:
        print(f"Dispersion x RMS   : {1e3*metrics['dispersion_x_rms_initial_m']:.6f} → "
              f"{1e3*metrics['dispersion_x_rms_fitted_m']:.6f} mm")
        print(f"Dispersion y RMS   : {1e3*metrics['dispersion_y_rms_initial_m']:.6f} → "
              f"{1e3*metrics['dispersion_y_rms_fitted_m']:.6f} mm")
        if metrics.get("optics_available", True):
            print(f"Beta beating RMS   : {metrics['beta_beating_x_rms_percent']:.6f}% x, "
                  f"{metrics['beta_beating_y_rms_percent']:.6f}% y")
            print(f"Tune initial/fitted: {metrics['initial_tune']} → {metrics['fitted_tune']}")
            print(f"Chromaticity       : {metrics['initial_chromaticity']} → "
                  f"{metrics['fitted_chromaticity']}")
        else:
            print(f"Optics diagnostics : {metrics.get('optics_warning', 'unavailable')}")
        if "family_correction_rms" in metrics:
            print("Quadrupole correction convention: ΔK_apply = K_model,initial − K_model,fitted")
            print("Machine application           : K_machine,new = K_machine,current + ΔK_apply")
            print(f"Family ΔK_apply [m⁻²]: min {metrics['family_delta_k_min']:.6g}, "
                  f"max {metrics['family_delta_k_max']:.6g}, "
                  f"RMS {metrics['family_delta_k_rms']:.6g}")
            print(f"Expanded ΔK_apply [m⁻²]: min {metrics['expanded_delta_k_min']:.6g}, "
                  f"max {metrics['expanded_delta_k_max']:.6g}, "
                  f"RMS {metrics['expanded_delta_k_rms']:.6g}")
            print(f"Family ΔK/K [%]    : min {metrics['family_correction_min']:.6g}, "
                  f"max {metrics['family_correction_max']:.6g}, "
                  f"RMS {metrics['family_correction_rms']:.6g}")
            print(f"Expanded ΔK/K [%]  : min {metrics['expanded_correction_min']:.6g}, "
                  f"max {metrics['expanded_correction_max']:.6g}, "
                  f"RMS {metrics['expanded_correction_rms']:.6g}")
        if "skew_correction_rms" in metrics:
            print(f"Skew correction    : min {metrics['skew_correction_min']:.6g}, "
                  f"max {metrics['skew_correction_max']:.6g}, "
                  f"RMS {metrics['skew_correction_rms']:.6g}")
    if fit.get("runtime_seconds") is not None:
        seconds = float(fit["runtime_seconds"])
        minutes, remainder = divmod(seconds, 60.0)
        print(f"Analysis runtime   : {int(minutes)}m {remainder:.2f}s ({seconds:.2f} s)")
