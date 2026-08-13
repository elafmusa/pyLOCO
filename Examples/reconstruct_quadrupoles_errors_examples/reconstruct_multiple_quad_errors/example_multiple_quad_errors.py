#!/usr/bin/env python3
"""Reconstruct several deliberately assigned EBS quadrupole errors with pyLOCO.

The simulated measurement is intentionally simple:

    ideal AT lattice -> copied machine lattice -> assign quadrupole errors
    -> AT orbit-response matrix -> reproducible BPM noise -> pyLOCO fit
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import at
import matplotlib.pyplot as plt
import numpy as np
import yaml

from pyLOCO.config import FitInitConfig, RMConfig
from pyLOCO.analysis import plot_matrices
from pyLOCO.pyloco import pyloco
from pyLOCO.response_matrix import response_matrix
from pyLOCO.user_config import build_constraints, selected_fit_parameters


HERE = Path(__file__).resolve().parent


def load_config(path: Path) -> dict[str, Any]:
    """Load the small user-facing YAML file and validate its required keys."""
    with path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream) or {}

    required = {
        "lattice": ("file", "use"),
        "elements": ("bpm_type", "corrector_pattern", "quad_groups"),
        "assigned_errors": (),
        "measurement": (
            "corrector_kick_rad", "bpm_noise_rms_m", "random_seed",
        ),
        "rf": ("frequency_hz", "step_hz"),
        "loco": (
            "nIter", "nLMIter", "Starting_Lambda", "max_lm_lambda",
            "svd_selection_method", "svd_threshold",
        ),
        "output": ("directory",),
    }
    missing_sections = [name for name in required if name not in cfg]
    if missing_sections:
        raise ValueError(f"Missing required YAML section(s): {', '.join(missing_sections)}")
    for section, keys in required.items():
        missing = [key for key in keys if key not in cfg[section]]
        if missing:
            raise ValueError(
                f"Missing required key(s) in YAML section '{section}': "
                + ", ".join(missing)
            )
    errors = cfg["assigned_errors"]
    if not isinstance(errors, list) or not errors:
        raise ValueError("assigned_errors must be a non-empty YAML list")
    for number, error in enumerate(errors, start=1):
        missing = [key for key in ("quad_index", "relative_error") if key not in error]
        if missing:
            raise ValueError(f"assigned_errors item {number} is missing: {', '.join(missing)}")
    return cfg


def element_indices(ring: at.Lattice, cfg: dict[str, Any]):
    """Resolve the element selectors kept in YAML."""
    elements = cfg["elements"]
    try:
        bpm_class = getattr(at.elements, elements["bpm_type"])
    except AttributeError as exc:
        raise ValueError(f"Unknown AT BPM element type: {elements['bpm_type']}") from exc

    bpm_indices = np.asarray(at.get_refpts(ring, bpm_class), dtype=int)
    corrector_indices = np.asarray(
        at.get_refpts(ring, elements["corrector_pattern"]), dtype=int
    )
    groups = [np.asarray(at.get_refpts(ring, pattern), dtype=int)
              for pattern in elements["quad_groups"]]
    empty = [pattern for pattern, indices in zip(elements["quad_groups"], groups)
             if indices.size == 0]
    if empty:
        raise ValueError(f"Quadrupole selector(s) matched no elements: {empty}")
    quad_indices = np.unique(np.concatenate(groups))
    if not (bpm_indices.size and corrector_indices.size and quad_indices.size):
        raise ValueError("BPM, corrector, and quadrupole selectors must all be non-empty")
    return bpm_indices, corrector_indices, quad_indices


def assign_quadrupole_error(
    machine: at.Lattice, quad_index: int, relative_error: float
) -> tuple[float, float]:
    """Scale K and explicitly keep AT's PolynomB[1] representation consistent."""
    if not 0 <= quad_index < len(machine):
        raise IndexError(f"quad_index={quad_index} is outside the lattice")
    element = machine[quad_index]
    if not isinstance(element, at.elements.Quadrupole):
        raise TypeError(
            f"Lattice element {quad_index} ({element.FamName}) is not a quadrupole"
        )

    nominal_k = float(element.K)
    if not np.isclose(nominal_k, float(element.PolynomB[1])):
        raise ValueError(
            f"Before assigning the error, {element.FamName}.K and PolynomB[1] disagree"
        )
    assigned_k = nominal_k * (1.0 + relative_error)
    polynom_b = np.array(element.PolynomB, copy=True)
    polynom_b[1] = assigned_k
    element.PolynomB = polynom_b
    element.K = assigned_k
    if not np.isclose(float(element.K), float(element.PolynomB[1])):
        raise RuntimeError("AT quadrupole K and PolynomB[1] became inconsistent")
    return nominal_k, assigned_k - nominal_k


def apply_quadrupole_changes(
    ring: at.Lattice, quad_indices: np.ndarray, delta_k: np.ndarray
) -> None:
    """Add strength changes while keeping K and PolynomB[1] consistent."""
    for index, change in zip(quad_indices, delta_k):
        element = ring[int(index)]
        new_k = float(element.K) + float(change)
        polynom_b = np.array(element.PolynomB, copy=True)
        polynom_b[1] = new_k
        element.PolynomB = polynom_b
        element.K = new_k


def orm_config(bpm_indices, corrector_indices, kick_rad: float) -> RMConfig:
    kicks = [[kick_rad] * len(corrector_indices)] * 2
    return RMConfig(
        bpm_ords=bpm_indices,
        cm_ords=[corrector_indices, corrector_indices],
        dkick=kicks,
        bidirectional=True,
        includeDispersion=False,
        fixedpathlength=False,
        HCMCoupling=np.zeros(len(corrector_indices)),
        VCMCoupling=np.zeros(len(corrector_indices)),
    )


def optics_beta(ring: at.Lattice, refpts: np.ndarray):
    _, _, twiss = at.get_optics(ring, refpts)
    return twiss.s_pos, twiss.beta


def print_optics_summary(
    ideal: at.Lattice,
    lattices: list[tuple[str, at.Lattice]],
    orm_ideal: np.ndarray,
    orms: list[np.ndarray],
) -> None:
    """Print a compact AT-only replacement for the old pySC analyze_ring output."""
    refpts = np.arange(len(ideal) + 1)
    _, beta_ideal = optics_beta(ideal, refpts)
    print("\nOptics and ORM comparison with the ideal lattice")
    print("------------------------------------------------")
    print(f"{'Lattice':<20} {'βx RMS [%]':>12} {'βy RMS [%]':>12} {'ORM RMS [µm]':>14}")
    for (label, ring), orm in zip(lattices, orms):
        _, beta = optics_beta(ring, refpts)
        beating = 100.0 * (beta - beta_ideal) / beta_ideal
        bx_rms, by_rms = np.sqrt(np.mean(beating**2, axis=0))
        orm_rms = 1e6 * np.sqrt(np.mean((orm - orm_ideal) ** 2))
        print(f"{label:<20} {bx_rms:12.6f} {by_rms:12.6f} {orm_rms:14.6f}")


def make_figures(
    output_dir: Path,
    ideal: at.Lattice,
    machine: at.Lattice,
    fitted: at.Lattice,
    corrected: at.Lattice,
    quad_indices: np.ndarray,
    assigned_delta: np.ndarray,
    fitted_delta: np.ndarray,
    measured_orm: np.ndarray,
    initial_orm: np.ndarray,
    fitted_orm: np.ndarray,
    chi2_history: list[float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    quad_s = at.get_s_pos(ideal, quad_indices)
    assigned_slots = np.flatnonzero(assigned_delta)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    markerline, _, _ = ax.stem(quad_s, 100.0 * fitted_delta /
                               np.array([ideal[i].K for i in quad_indices]),
                               linefmt="C0-", markerfmt="C0o", basefmt=" ",
                               label="pyLOCO reconstructed error")
    markerline.set_markersize(3)
    ax.scatter(quad_s[assigned_slots],
               100.0 * assigned_delta[assigned_slots] /
               np.array([ideal[quad_indices[i]].K for i in assigned_slots]),
               marker="D", s=45, color="C3", zorder=4, label="Assigned error")
    for slot in assigned_slots:
        ax.axvline(quad_s[slot], color="C3", alpha=0.15, lw=6)
    ax.set(xlabel="Longitudinal position s [m]", ylabel=r"$\Delta K/K$ [%]",
           title="Assigned and reconstructed quadrupole errors")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "quadrupole_error_reconstruction.png", dpi=180)
    plt.close(fig)

    before = initial_orm - measured_orm
    after = fitted_orm - measured_orm
    plot_matrices(
        before,
        after,
        titles=[
            f"Ideal model − measurement (RMS {1e6 * np.sqrt(np.mean(before**2)):.3f} µm)",
            f"Fitted model − measurement (RMS {1e6 * np.sqrt(np.mean(after**2)):.3f} µm)",
        ],
        cmap="viridis",
        plot_type="3d",
        same_scale=True,
        save_path=output_dir / "orm_residual_before_after.png",
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(np.arange(1, len(chi2_history) + 1), chi2_history, "o-")
    ax.set(xlabel="LOCO iteration", ylabel="Normalized chi-square",
           title="LOCO fit convergence")
    ax.grid(alpha=0.25)
    ax.set_xticks(np.arange(1, len(chi2_history) + 1))
    fig.tight_layout()
    fig.savefig(output_dir / "fit_convergence.png", dpi=180)
    plt.close(fig)

    refpts = np.arange(len(ideal) + 1)
    s, beta_ideal = optics_beta(ideal, refpts)
    _, beta_machine = optics_beta(machine, refpts)
    _, beta_fitted = optics_beta(fitted, refpts)
    _, beta_corrected = optics_beta(corrected, refpts)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for plane, ax in enumerate(axes):
        ax.plot(s, 100.0 * (beta_machine[:, plane] - beta_ideal[:, plane]) /
                beta_ideal[:, plane], label="Modified machine", color="C3")
        ax.plot(s, 100.0 * (beta_fitted[:, plane] - beta_ideal[:, plane]) /
                beta_ideal[:, plane], label="Fitted lattice", color="C0", linestyle="--")
        ax.plot(s, 100.0 * (beta_corrected[:, plane] - beta_ideal[:, plane]) /
                beta_ideal[:, plane], label="Corrected machine", color="C2", linestyle="-.")
        ax.set_ylabel(rf"$\Delta\beta_{{{'xy'[plane]}}}/\beta_{{{'xy'[plane]}}}$ [%]")
        ax.grid(alpha=0.2)
        ax.legend()
    axes[-1].set_xlabel("Longitudinal position s [m]")
    fig.suptitle("Optics validation against the ideal lattice")
    fig.tight_layout()
    fig.savefig(output_dir / "beta_beating_before_after.png", dpi=180)
    plt.close(fig)


def main(config_path: Path) -> None:
    cfg = load_config(config_path)
    lattice_path = (config_path.parent / cfg["lattice"]["file"]).resolve()
    ideal = at.load_lattice(lattice_path, use=cfg["lattice"]["use"])
    ideal.disable_6d()
    machine = copy.deepcopy(ideal)

    bpm_indices, corrector_indices, quad_indices = element_indices(ideal, cfg)
    assigned = []
    for error in cfg["assigned_errors"]:
        quad_index = int(error["quad_index"])
        relative_error = float(error["relative_error"])
        if quad_index not in quad_indices:
            raise ValueError(
                f"Assigned quad_index={quad_index} is not selected by elements.quad_groups"
            )
        nominal_k, assigned_dk = assign_quadrupole_error(
            machine, quad_index, relative_error
        )
        assigned.append((quad_index, nominal_k, assigned_dk))

    measurement = cfg["measurement"]
    kick_rad = float(measurement["corrector_kick_rad"])
    noise_rms_m = float(measurement["bpm_noise_rms_m"])
    if kick_rad <= 0.0 or noise_rms_m <= 0.0:
        raise ValueError("corrector_kick_rad and bpm_noise_rms_m must be positive")
    rm_cfg = orm_config(bpm_indices, corrector_indices, kick_rad)
    orm_ideal = response_matrix(ideal, config=rm_cfg)
    orm_machine = response_matrix(machine, config=rm_cfg)
    rng = np.random.default_rng(int(measurement["random_seed"]))
    measured_orm = orm_machine + rng.normal(0.0, noise_rms_m, orm_machine.shape)

    # Each ORM entry is a BPM position measured after the configured corrector
    # kick, so its independent one-sigma uncertainty is the BPM noise in metres.
    sigma_w = np.full(2 * len(bpm_indices), noise_rms_m)
    cm_step = [[kick_rad] * len(corrector_indices)] * 2
    cavity_indices = np.asarray(at.get_refpts(ideal, at.elements.RFCavity), dtype=int)
    if cavity_indices.size == 0:
        raise ValueError("The lattice contains no RFCavity element")
    frequency_hz = float(cfg["rf"]["frequency_hz"])
    rf_step_hz = float(cfg["rf"]["step_hz"])
    loco_cfg = cfg["loco"]
    fit_list = selected_fit_parameters(cfg)
    fit_cfg = FitInitConfig(
        fit_list=fit_list, CMstep=cm_step, individuals=True,
        quads_attr="PolynomB", quads_attr_index=1,
    )
    output_dir = (config_path.parent / cfg["output"]["directory"]).resolve()
    temporary_fit_output = TemporaryDirectory(prefix="pyloco_multiple_quads_")
    result = pyloco(
        copy.deepcopy(ideal),
        algorithm=str(loco_cfg.get("algorithm", "lm")),
        nIter=int(loco_cfg["nIter"]),
        used_bpms_ords=bpm_indices,
        used_cor_ords=[corrector_indices, corrector_indices],
        quads_ords=quad_indices,
        skew_ords=np.array([], dtype=int),
        CAVords=cavity_indices,
        nHBPM=len(bpm_indices), nVBPM=len(bpm_indices),
        nHorCOR=len(corrector_indices), nVerCOR=len(corrector_indices),
        quads_tilt_ind=quad_indices,
        orm_measured=measured_orm,
        weights=sigma_w,
        includeDispersion=bool(loco_cfg.get("include_dispersion", False)),
        measured_eta_x=np.zeros(len(bpm_indices)),
        measured_eta_y=np.zeros(len(bpm_indices)),
        CMstep=cm_step,
        rfStep=rf_step_hz,
        Frequency=frequency_hz,
        fit_list=fit_list,
        quad_individuals=True,
        remove_coupling_=bool(loco_cfg.get("remove_coupling", True)),
        outlier_rejection=bool(loco_cfg.get("outlier_rejection", False)),
        apply_normalization=bool(loco_cfg.get("apply_normalization", False)),
        normalization_mode=str(loco_cfg.get("normalization_mode", "component")),
        svd_selection_method=str(loco_cfg["svd_selection_method"]),
        svd_threshold=float(loco_cfg["svd_threshold"]),
        cut_=loco_cfg.get("cut"),
        show_svd_plot=bool(loco_cfg.get("show_svd_plot", False)),
        nLMIter=int(loco_cfg["nLMIter"]),
        Starting_Lambda=float(loco_cfg["Starting_Lambda"]),
        max_lm_lambda=float(loco_cfg["max_lm_lambda"]),
        scaled=bool(loco_cfg.get("scaled", True)),
        plot_fit_parameters=bool(loco_cfg.get("plot_fit_parameters", False)),
        auto_correct_delta=bool(loco_cfg.get("auto_correct_delta", True)),
        fixedpathlength=bool(loco_cfg.get("fixedpathlength", False)),
        fixedmomentum=False,
        fit_cfg=fit_cfg,
        constraint_cfg=build_constraints(cfg, quad_nominal=[ideal[i].K for i in quad_indices], n_skew=0),
        output_dir=temporary_fit_output.name,
    )
    _, fit_dict, fitted, fitted_orm, _, chi2_history, _, _ = result
    temporary_fit_output.cleanup()

    fitted_k = np.array([fitted[i].PolynomB[1] for i in quad_indices])
    ideal_k = np.array([ideal[i].PolynomB[1] for i in quad_indices])
    fitted_delta = fitted_k - ideal_k
    assigned_delta = np.zeros_like(fitted_delta)
    for quad_index, _, assigned_dk in assigned:
        slot = int(np.flatnonzero(quad_indices == quad_index)[0])
        assigned_delta[slot] = assigned_dk
    correction_delta = -fitted_delta
    corrected_machine = copy.deepcopy(machine)
    apply_quadrupole_changes(corrected_machine, quad_indices, correction_delta)
    corrected_orm = response_matrix(corrected_machine, config=rm_cfg)
    rms_before = float(np.sqrt(np.mean((orm_ideal - measured_orm) ** 2)))
    rms_after = float(np.sqrt(np.mean((fitted_orm - measured_orm) ** 2)))

    make_figures(
        output_dir, ideal, machine, fitted, corrected_machine, quad_indices,
        assigned_delta, fitted_delta, measured_orm, orm_ideal, fitted_orm,
        chi2_history,
    )
    print("\nMultiple-quadrupole reconstruction summary")
    print("-------------------------------------------")
    print(f"{'Name':<7} {'Index':>6} {'s [m]':>10} {'Assigned [%]':>14} {'Fitted [%]':>12} {'Recovery [%]':>14}")
    for quad_index, nominal_k, assigned_dk in assigned:
        slot = int(np.flatnonzero(quad_indices == quad_index)[0])
        fitted_dk = fitted_delta[slot]
        s_position = float(np.asarray(at.get_s_pos(ideal, quad_index)).item())
        print(
            f"{ideal[quad_index].FamName:<7} {quad_index:6d} {s_position:10.3f} "
            f"{100.0 * assigned_dk / nominal_k:14.6f} "
            f"{100.0 * fitted_dk / nominal_k:12.6f} "
            f"{100.0 * fitted_dk / assigned_dk:14.3f}"
        )
    print(f"ORM RMS before LOCO               : {1e6 * rms_before:.6f} µm")
    print(f"ORM RMS after LOCO                : {1e6 * rms_after:.6f} µm")
    print(f"ORM improvement factor            : {rms_before / rms_after:.3f}x")
    print(f"Corrected machine ORM RMS vs ideal: {1e6 * np.sqrt(np.mean((corrected_orm - orm_ideal)**2)):.6f} µm")
    print_optics_summary(
        ideal,
        [("Modified machine", machine), ("Fitted lattice", fitted),
         ("Corrected machine", corrected_machine)],
        orm_ideal,
        [orm_machine, fitted_orm, corrected_orm],
    )
    print(f"Figures written to                : {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=HERE / "pyloco_config.yaml",
        help="YAML configuration file (default: pyloco_config.yaml beside this script)",
    )
    args = parser.parse_args()
    main(args.config.resolve())
