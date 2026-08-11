"""Shared data preparation and plotting for the PETRA III examples."""
from __future__ import annotations

import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import at
import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml

from pyLOCO.analysis import plot_matrices
from pyLOCO.config import FitInitConfig, RMConfig
from pyLOCO.pyloco import pyloco, remove_bad_bpms
from pyLOCO.response_matrix import response_matrix


HERE = Path(__file__).resolve().parent


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream) or {}
    for section in ("lattice", "data", "bad_bpm_positions", "rf", "loco", "output"):
        if section not in cfg:
            raise ValueError(f"Missing YAML section: {section}")
    return cfg


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
    quad_indices = np.load(base / data_cfg["quadrupole_indices"]).astype(int)
    skew_indices = np.load(base / data_cfg["skew_indices"]).astype(int)
    steps = np.load(base / data_cfg["corrector_steps"])
    cm_step = [steps["hor"], steps["ver"]]
    hcor = _indices_by_common_name(ring, _names(base / data_cfg["horizontal_corrector_names"]))
    vcor = _indices_by_common_name(ring, _names(base / data_cfg["vertical_corrector_names"]))
    bpms = _indices_by_common_name(ring, _names(base / data_cfg["bpm_names"]))

    with h5py.File(base / data_cfg["orm"], "r") as stream:
        measured_orm = np.asarray(stream["response_matrix"])
    with h5py.File(base / data_cfg["dispersion"], "r") as stream:
        eta_x = np.asarray(stream["measured_eta_x"])
        eta_y = np.asarray(stream["measured_eta_y"])
    with h5py.File(base / data_cfg["bpm_noise"], "r") as stream:
        noise_x = np.asarray(stream["Noise_BPMx"])
        noise_y = np.asarray(stream["Noise_BPMy"])

    bad = np.asarray(cfg["bad_bpm_positions"], dtype=int)
    if np.any((bad < 0) | (bad >= len(bpms))):
        raise ValueError("bad_bpm_positions contains an out-of-range position")
    cleaned_orm, _ = remove_bad_bpms(
        measured_orm, bad, total_bpms=len(bpms), axis=0, input_type="positions"
    )
    good_bpms = np.delete(bpms, bad)
    result = {
        "cfg": cfg,
        "ring": ring,
        "quad_indices": quad_indices,
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
    return result


def model_orm(data: dict[str, Any]) -> np.ndarray:
    cfg = data["cfg"]
    rm_cfg = RMConfig(
        bpm_ords=data["bpms"], cm_ords=data["correctors"], cav_ords=data["cavities"],
        dkick=data["cm_step"], bidirectional=True, includeDispersion=False,
        rfStep=float(cfg["rf"]["step_hz"]), Frequency=float(cfg["rf"]["frequency_hz"]),
        fixedpathlength=False,
    )
    return response_matrix(data["ring"], config=rm_cfg)


def run_fit(data: dict[str, Any], *, coupling: bool) -> dict[str, Any]:
    """Run the configured standard or coupling-aware measured-data fit."""
    cfg = data["cfg"]
    loco = cfg["loco"]
    fit_list = list(loco["coupling_fit_list"] if coupling else loco["standard_fit_list"])
    fit_cfg = FitInitConfig(
        fit_list=fit_list, CMstep=data["cm_step"], rfStep=float(cfg["rf"]["step_hz"]),
        individuals=True, quads_attr="PolynomB", quads_attr_index=1,
        skew_attr="PolynomB", skew_attr_index=1,
    )
    temporary = TemporaryDirectory(prefix="pyloco_petra_coupling_" if coupling else "pyloco_petra_orm_")
    try:
        result = pyloco(
            copy.deepcopy(data["ring"]), algorithm="lm", nIter=int(loco["nIter"]),
            used_bpms_ords=data["bpms"], used_cor_ords=data["correctors"],
            quads_ords=data["quad_indices"],
            skew_ords=data["skew_indices"] if coupling else np.array([], dtype=int),
            CAVords=data["cavities"], nHBPM=len(data["bpms"]), nVBPM=len(data["bpms"]),
            nHorCOR=len(data["correctors"][0]), nVerCOR=len(data["correctors"][1]),
            quads_tilt_ind=data["quad_indices"], orm_measured=data["orm"],
            weights=data["weights"], includeDispersion=bool(loco["include_dispersion"]),
            measured_eta_x=data["eta_x"], measured_eta_y=data["eta_y"],
            hor_dispersion_weight=float(loco["horizontal_dispersion_weight"]),
            ver_dispersion_weight=float(loco["vertical_dispersion_weight"]),
            CMstep=data["cm_step"], rfStep=float(cfg["rf"]["step_hz"]),
            Frequency=float(cfg["rf"]["frequency_hz"]), fit_list=fit_list,
            quad_individuals=True, skew_individuals=True, tilt_individuals=True,
            remove_coupling_=not coupling, outlier_rejection=bool(loco["outlier_rejection"]),
            sigma_outlier=float(loco["sigma_outlier"]),
            apply_normalization=bool(loco["apply_normalization"]),
            normalization_mode=str(loco["normalization_mode"]),
            svd_selection_method=str(loco["svd_selection_method"]),
            svd_threshold=float(loco["svd_threshold"]), cut_=loco.get("cut"),
            show_svd_plot=bool(loco["show_svd_plot"]), nLMIter=int(loco["nLMIter"]),
            Starting_Lambda=float(loco["Starting_Lambda"]),
            max_lm_lambda=float(loco["max_lm_lambda"]), scaled=bool(loco["scaled"]),
            plot_fit_parameters=False, auto_correct_delta=True, fixedpathlength=False,
            fixedmomentum=False, fit_cfg=fit_cfg, output_dir=temporary.name,
        )
    finally:
        temporary.cleanup()
    fit_results, fit_dict, fitted_ring, fitted_orm, _, chi2, _, _ = result
    return {"fit_results": fit_results, "fit_dict": fit_dict, "ring": fitted_ring,
            "orm": fitted_orm, "chi2": chi2, "fit_list": fit_list}


def make_plots(data: dict[str, Any], initial_orm: np.ndarray, fit: dict[str, Any], *, coupling: bool) -> Path:
    cfg = data["cfg"]
    output = (HERE / cfg["output"]["coupling" if coupling else "standard"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    before = initial_orm - data["orm"]
    after = fit["orm"] - data["orm"]
    plot_matrices(before, after, titles=[
        f"Initial model − measurement ({1e6*np.sqrt(np.mean(before**2)):.3f} µm RMS)",
        f"Fitted model − measurement ({1e6*np.sqrt(np.mean(after**2)):.3f} µm RMS)",
    ], cmap="viridis", plot_type="3d", same_scale=True,
       save_path=output / "orm_residual_before_after.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(np.arange(1, len(fit["chi2"]) + 1), fit["chi2"], "o-")
    ax.set(xlabel="LOCO iteration", ylabel="Normalized chi-square", title="Fit convergence")
    ax.grid(alpha=0.25); fig.tight_layout(); fig.savefig(output / "fit_convergence.png", dpi=180)
    plt.close(fig)

    refpts = np.arange(len(data["ring"]) + 1)
    _, _, twiss0 = at.get_optics(data["ring"], refpts)
    _, _, twissf = at.get_optics(fit["ring"], refpts)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for plane, ax in enumerate(axes):
        beating = 100 * (twissf.beta[:, plane] - twiss0.beta[:, plane]) / twiss0.beta[:, plane]
        ax.plot(twiss0.s_pos, beating, color="C0", linestyle="--", label="Fitted lattice")
        ax.set_ylabel(rf"$Δ\beta_{{{'xy'[plane]}}}/\beta_{{{'xy'[plane]}}}$ [%]")
        ax.grid(alpha=.2); ax.legend()
    axes[-1].set_xlabel("Longitudinal position s [m]")
    fig.suptitle("Fitted optics relative to the initial PETRA III model")
    fig.tight_layout(); fig.savefig(output / "beta_beating.png", dpi=180); plt.close(fig)

    if coupling:
        n_bpm = len(data["bpms"]); n_hcor = len(data["correctors"][0])
        plot_matrices(
                      before[:n_bpm, n_hcor:], before[n_bpm:, :n_hcor],
                      after[:n_bpm, n_hcor:], after[n_bpm:, :n_hcor],
                      titles=["Initial XY residual", "Initial YX residual",
                              "Fitted XY residual", "Fitted YX residual"],
                      cmap="viridis", plot_type="3d", same_scale=True,
                      save_path=output / "coupling_blocks_before_after.png")
    return output


def print_summary(data: dict[str, Any], initial_orm: np.ndarray, fit: dict[str, Any], *, coupling: bool) -> None:
    before = float(np.sqrt(np.mean((initial_orm - data["orm"]) ** 2)))
    after = float(np.sqrt(np.mean((fit["orm"] - data["orm"]) ** 2)))
    print("\nPETRA III coupling fit" if coupling else "\nPETRA III measured-ORM fit")
    print("-" * 46)
    print(f"Measured ORM shape : {data['orm'].shape}")
    print(f"Retained BPMs      : {len(data['bpms'])} per plane")
    print(f"Correctors         : {len(data['correctors'][0])} H, {len(data['correctors'][1])} V")
    print(f"Fitted parameters  : {', '.join(fit['fit_list'])}")
    print(f"ORM RMS before     : {1e6*before:.6f} µm")
    print(f"ORM RMS after      : {1e6*after:.6f} µm")
    print(f"Improvement        : {before/after:.3f}x")
