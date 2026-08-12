"""Shared data preparation and plotting for the PETRA III examples."""
from __future__ import annotations

import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import at
import h5py
import numpy as np
import yaml

from pyLOCO.config import ConstraintConfig, FitInitConfig, RMConfig, fixed_parameters
from pyLOCO.pyloco import pyloco, remove_bad_bpms
from pyLOCO.response_matrix import response_matrix
from petra_diagnostics import make_diagnostic_plots, save_run_results


HERE = Path(__file__).resolve().parent


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream) or {}
    for section in ("lattice", "data", "bad_bpm_positions", "rf", "loco", "output"):
        if section not in cfg:
            raise ValueError(f"Missing YAML section: {section}")
    return cfg


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


def _indices_by_common_name(ring: at.Lattice, names: list[str]) -> np.ndarray:
    selected = {name for name in names}
    return np.asarray(
        [i for i, element in enumerate(ring) if getattr(element, "CommonName", None) in selected],
        dtype=int,
    )


def prepare_measurement(config_path: Path) -> dict[str, Any]:
    """Load the PETRA III model, measured ORM, uncertainty, and element selections."""
    cfg = load_yaml(config_path)
    base = config_path.parent
    data_cfg = cfg["data"]
    ring = at.load_lattice(base / cfg["lattice"]["file"])
    ring.disable_6d()
    individual_quad_indices = np.load(base / data_cfg["quadrupole_indices"]).astype(int)
    quadrupole_mode = str(data_cfg.get("quadrupole_mode", "individual")).lower()
    if quadrupole_mode == "individual":
        quad_indices = individual_quad_indices
        quad_individuals = True
    elif quadrupole_mode == "family":
        family_file = data_cfg.get("quadrupole_family_groups")
        if not family_file:
            raise ValueError(
                "data.quadrupole_family_groups is required when quadrupole_mode is 'family'"
            )
        quad_indices = _load_family_groups(base / family_file)
        quad_individuals = False
    else:
        raise ValueError("data.quadrupole_mode must be 'individual' or 'family'")
    groups_to_check = (
        [[int(index)] for index in quad_indices] if quad_individuals else quad_indices
    )
    for number, group in enumerate(groups_to_check):
        if any(index < 0 or index >= len(ring) for index in group):
            raise ValueError(f"Quadrupole family/parameter {number} contains an invalid lattice index")
        if not quad_individuals:
            strengths = np.asarray([ring[index].PolynomB[1] for index in group], dtype=float)
            if not np.allclose(strengths, strengths[0], rtol=1.0e-10, atol=1.0e-14):
                raise ValueError(
                    f"Quadrupole family {number} contains unequal nominal PolynomB[1] strengths"
                )
    skew_indices = np.load(base / data_cfg["skew_indices"]).astype(int)
    hcor = _indices_by_common_name(ring, _names(base / data_cfg["horizontal_corrector_names"]))
    vcor = _indices_by_common_name(ring, _names(base / data_cfg["vertical_corrector_names"]))
    bpms = _indices_by_common_name(ring, _names(base / data_cfg["bpm_names"]))
    step_cfg = data_cfg["corrector_steps"]
    if isinstance(step_cfg, dict):
        cm_step = [
            np.full(len(hcor), float(step_cfg["horizontal_rad"])),
            np.full(len(vcor), float(step_cfg["vertical_rad"])),
        ]
    else:
        steps = np.load(base / step_cfg)
        cm_step = [steps["hor"], steps["ver"]]

    orm_path = base / data_cfg["orm"]
    with h5py.File(orm_path, "r") as stream:
        _require_datasets(stream, orm_path, ("response_matrix",))
        measured_orm = np.asarray(stream["response_matrix"])
    dispersion_path = base / data_cfg["dispersion"]
    dispersion_datasets = data_cfg.get("dispersion_datasets", {})
    eta_x_name = str(dispersion_datasets.get("horizontal", "measured_eta_x"))
    eta_y_name = str(dispersion_datasets.get("vertical", "measured_eta_y"))
    with h5py.File(dispersion_path, "r") as stream:
        _require_datasets(stream, dispersion_path, (eta_x_name, eta_y_name))
        eta_x = np.asarray(stream[eta_x_name])
        eta_y = np.asarray(stream[eta_y_name])
    noise_path = base / data_cfg["bpm_noise"]
    noise_datasets = data_cfg.get("bpm_noise_datasets", {})
    noise_x_name = str(noise_datasets.get("horizontal", "Noise_BPMx"))
    noise_y_name = str(noise_datasets.get("vertical", "Noise_BPMy"))
    with h5py.File(noise_path, "r") as stream:
        _require_datasets(stream, noise_path, (noise_x_name, noise_y_name))
        noise_x = np.asarray(stream[noise_x_name])
        noise_y = np.asarray(stream[noise_y_name])

    bad = np.asarray(cfg["bad_bpm_positions"], dtype=int)
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
        "cavities": np.asarray(at.get_refpts(ring, at.elements.RFCavity), dtype=int),
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
    return result


def build_constraint_config(data: dict[str, Any]) -> ConstraintConfig | None:
    """Translate the YAML constraint section to pyLOCO's supported API."""
    constraints = data["cfg"].get("constraints", {}) or {}
    if not bool(constraints.get("enable", False)):
        return None

    n_quad = len(data["quad_indices"])
    n_skew = len(data["skew_indices"])
    quad_cfg = constraints.get("quadrupoles", {}) or {}
    skew_cfg = constraints.get("skew_quadrupoles", {}) or {}

    relative_sigma = quad_cfg.get("relative_sigma")
    absolute_sigma = quad_cfg.get("sigma")
    if relative_sigma is not None and absolute_sigma is not None:
        raise ValueError("Set only one of constraints.quadrupoles.sigma or relative_sigma")
    if relative_sigma is not None:
        nominal = np.asarray([
            float(getattr(data["ring"][group[0] if not np.isscalar(group) else group], "K"))
            for group in data["quad_indices"]
        ])
        quad_sigma: float | np.ndarray = np.maximum(
            np.abs(nominal) * float(relative_sigma),
            float(quad_cfg.get("minimum_sigma", 1.0e-12)),
        )
    else:
        quad_sigma = _scalar_or_vector(
            absolute_sigma if absolute_sigma is not None else 0.01, n_quad, "quadrupole sigma"
        )

    explicit_quad_weights = quad_cfg.get("weights")
    quad_weights = (
        np.full(n_quad, float(quad_cfg.get("default_weight", 1.0)))
        if explicit_quad_weights is None
        else _required_vector(explicit_quad_weights, n_quad, "quadrupole weights")
    )
    selected_families = quad_cfg.get("selected_families")
    selected_weight = quad_cfg.get("selected_weight")
    if (selected_families is None) != (selected_weight is None):
        raise ValueError(
            "constraints.quadrupoles.selected_families and selected_weight must be set together"
        )
    if selected_families is not None:
        selected = np.asarray(selected_families, dtype=int).ravel()
        if selected.size != len(set(selected.tolist())):
            raise ValueError("constraints.quadrupoles.selected_families contains duplicates")
        invalid = selected[(selected < 0) | (selected >= n_quad)]
        if invalid.size:
            raise ValueError(
                f"Selected quadrupole family {int(invalid[0])} is outside 0..{n_quad - 1}"
            )
        quad_weights[selected] = float(selected_weight)
    for raw_index, raw_weight in (quad_cfg.get("weighted_families", {}) or {}).items():
        index = int(raw_index)
        if index < 0 or index >= n_quad:
            raise ValueError(f"Weighted quadrupole family {index} is outside 0..{n_quad - 1}")
        quad_weights[index] = float(raw_weight)

    return ConstraintConfig(
        enable=True,
        quad_sigma=quad_sigma,
        quad_weights=quad_weights,
        quad_mask=_optional_mask(quad_cfg.get("mask"), n_quad, "quadrupole"),
        skew_sigma=_scalar_or_vector(skew_cfg.get("sigma", 0.001), n_skew, "skew sigma"),
        skew_weights=(
            np.full(n_skew, float(skew_cfg.get("default_weight", 1.0)))
            if skew_cfg.get("weights") is None
            else _required_vector(skew_cfg["weights"], n_skew, "skew weights")
        ),
        skew_mask=_optional_mask(skew_cfg.get("mask"), n_skew, "skew quadrupole"),
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


def run_fit(data: dict[str, Any], *, coupling: bool, constrained: bool = False) -> dict[str, Any]:
    """Run the configured standard or coupling-aware measured-data fit."""
    cfg = data["cfg"]
    loco = cfg["loco"]
    list_name = "constrained_fit_list" if constrained else (
        "coupling_fit_list" if coupling else "standard_fit_list"
    )
    fit_list = list(loco[list_name])
    fit_cfg = FitInitConfig(
        fit_list=fit_list, CMstep=data["cm_step"], rfStep=float(cfg["rf"]["step_hz"]),
        individuals=data["quad_individuals"], quads_attr="PolynomB", quads_attr_index=1,
        skew_attr=str(loco.get("skew_attribute", "PolynomB")), skew_attr_index=1,
    )
    constraint_cfg = build_constraint_config(data) if constrained else None
    if constrained and constraint_cfg is None:
        raise ValueError("A constrained fit requires constraints.enable: true")
    temporary = TemporaryDirectory(prefix="pyloco_petra_")
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
            output_dir=temporary.name,
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
            "initial_dispersion": model_dispersion(data, data["ring"]),
            "fitted_dispersion": fitted_dispersion}


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
    title = "PETRA III constrained fit" if constrained else (
        "PETRA III coupling fit" if coupling else "PETRA III measured-ORM fit"
    )
    print(f"\n{title}")
    print("-" * 46)
    print(f"Measured ORM shape : {data['orm'].shape}")
    print(f"Retained BPMs      : {len(data['bpms'])} per plane")
    print(f"Correctors         : {len(data['correctors'][0])} H, {len(data['correctors'][1])} V")
    print(f"Fitted parameters  : {', '.join(fit['fit_list'])}")
    print(f"ORM RMS before     : {1e6*before:.6f} µm")
    print(f"ORM RMS after      : {1e6*after:.6f} µm")
    print(f"Improvement        : {before/after:.3f}x")
    metrics = fit.get("metrics", {})
    if metrics:
        print(f"Dispersion x RMS   : {1e3*metrics['dispersion_x_rms_initial_m']:.6f} → "
              f"{1e3*metrics['dispersion_x_rms_fitted_m']:.6f} mm")
        print(f"Dispersion y RMS   : {1e3*metrics['dispersion_y_rms_initial_m']:.6f} → "
              f"{1e3*metrics['dispersion_y_rms_fitted_m']:.6f} mm")
        print(f"Beta beating RMS   : {metrics['beta_beating_x_rms_percent']:.6f}% x, "
              f"{metrics['beta_beating_y_rms_percent']:.6f}% y")
        print(f"Tune initial/fitted: {metrics['initial_tune']} → {metrics['fitted_tune']}")
        print(f"Chromaticity       : {metrics['initial_chromaticity']} → "
              f"{metrics['fitted_chromaticity']}")
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
