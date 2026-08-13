#!/usr/bin/env python3
"""Measure, fit, and correct an EBS machine using pySC and pyLOCO.

pySC supplies the machine with errors and performs the simulated measurements.
pyLOCO reconstructs those errors from the measured orbit-response matrix. The
fitted quadrupole changes are then applied back through pySC and the machine is
remeasured to demonstrate the correction.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from tempfile import TemporaryDirectory

import at
import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml

from pyLOCO.analysis import plot_matrices
from pyLOCO.config import FitInitConfig, RMConfig
from pyLOCO.pyloco import pyloco
from pyLOCO.response_matrix import response_matrix
from pyLOCO.user_config import selected_fit_parameters


HERE = Path(__file__).resolve().parent


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream) or {}
    for section in ("lattice", "pysc", "data", "elements", "rf", "loco", "output"):
        if section not in cfg:
            raise ValueError(f"Missing YAML section: {section}")
    return cfg


def load_pysc_machine(config_path: Path, cfg: dict):
    try:
        from pySC import SimulatedCommissioning, disable_pySC_rich
    except ImportError as exc:
        raise RuntimeError(
            "This example requires pySC. Install pySC in addition to the base pyLOCO dependencies."
        ) from exc
    disable_pySC_rich()
    base = config_path.parent
    return SimulatedCommissioning.from_json(
        str(base / cfg["pysc"]["state_file"]),
        lattice_file=str(base / cfg["lattice"]["file"]),
    )


def load_measurement(base: Path, cfg: dict):
    """Load the preserved pySC measurement when cached mode is requested."""
    with h5py.File(base / cfg["data"]["orm"], "r") as stream:
        orm = np.asarray(stream["response_matrix"])
    with h5py.File(base / cfg["data"]["dispersion"], "r") as stream:
        eta_x = np.asarray(stream["measured_eta_x"])
        eta_y = np.asarray(stream["measured_eta_y"])
    with h5py.File(base / cfg["data"]["bpm_noise"], "r") as stream:
        noise_x = np.asarray(stream["Noise_BPMx"])
        noise_y = np.asarray(stream["Noise_BPMy"])
    return orm, eta_x, eta_y, np.concatenate((noise_x, noise_y))


def measure_with_pysc(commissioning, cfg: dict):
    """Perform the ORM, RF-response, and BPM-repeatability measurements in pySC."""
    from pySC.tuning.averaging import get_average_orbit
    from pySC.tuning.response_measurements import (
        measure_OrbitResponseMatrix,
        measure_RFFrequencyOrbitResponse,
    )

    measurement = cfg["measurement"]
    kick = float(measurement["corrector_kick_rad"])
    rf_step = float(cfg["rf"]["step_hz"])
    repetitions = int(measurement["bpm_noise_repetitions"])
    if repetitions < 2:
        raise ValueError("bpm_noise_repetitions must be at least 2")

    # pySC changes each corrector in both directions and reads every BPM. The
    # result is already in the [horizontal+vertical BPM, H+V corrector] layout
    # expected by pyLOCO.
    orm = measure_OrbitResponseMatrix(
        commissioning,
        commissioning.tuning.HCORR,
        commissioning.tuning.VCORR,
        dkick=kick,
        normalize=False,
        bipolar=True,
    )

    # A bipolar RF-frequency change provides the horizontal and vertical
    # dispersion-like orbit response used when dispersion fitting is enabled.
    eta = measure_RFFrequencyOrbitResponse(
        commissioning, delta_frf=rf_step, normalize=False, bipolar=True
    )
    eta_x, eta_y = np.split(np.asarray(eta), 2)

    # Repeated unperturbed orbit readings estimate the uncertainty assigned to
    # each horizontal and vertical BPM row of the ORM.
    _, _, noise_x, noise_y = get_average_orbit(commissioning, repetitions)
    weights = np.concatenate((noise_x, noise_y))
    return np.asarray(orm), eta_x, eta_y, weights


def measure_orm_with_pysc(commissioning, cfg: dict) -> np.ndarray:
    """Remeasure only the ORM, used after applying the pyLOCO correction."""
    from pySC.tuning.response_measurements import measure_OrbitResponseMatrix

    return np.asarray(measure_OrbitResponseMatrix(
        commissioning,
        commissioning.tuning.HCORR,
        commissioning.tuning.VCORR,
        dkick=float(cfg["measurement"]["corrector_kick_rad"]),
        normalize=False,
        bipolar=True,
    ))


def acquire_measurement(commissioning, base: Path, cfg: dict):
    """Choose live pySC measurement or the preserved reproducible dataset."""
    source = str(cfg["measurement"]["source"]).lower()
    if source == "simulate":
        print("Measuring ORM, RF response, and BPM repeatability with pySC ...")
        return measure_with_pysc(commissioning, cfg)
    if source == "cached":
        print("Loading the cached measurement previously produced by pySC ...")
        return load_measurement(base, cfg)
    raise ValueError("measurement.source must be 'simulate' or 'cached'")


def selected_indices(ring: at.Lattice, cfg: dict):
    bpm = np.asarray(at.get_refpts(ring, at.elements.Monitor), dtype=int)
    correctors = np.asarray(at.get_refpts(ring, cfg["elements"]["corrector_pattern"]), dtype=int)
    groups = [np.asarray(at.get_refpts(ring, pattern), dtype=int)
              for pattern in cfg["elements"]["quadrupole_groups"]]
    quads = np.unique(np.concatenate(groups))
    cavities = np.asarray(at.get_refpts(ring, at.elements.RFCavity), dtype=int)
    return bpm, correctors, quads, cavities


def run_fit(ideal, bpm, correctors, quads, cavities, orm, eta_x, eta_y, weights, cfg):
    loco = cfg["loco"]
    kick = float(cfg["measurement"]["corrector_kick_rad"])
    cm_step = [[kick] * len(correctors)] * 2
    fit_list = selected_fit_parameters(cfg)
    fit_cfg = FitInitConfig(fit_list=fit_list, CMstep=cm_step, individuals=True,
                            quads_attr="PolynomB", quads_attr_index=1)
    temporary = TemporaryDirectory(prefix="pyloco_ebs_pysc_")
    try:
        return pyloco(
            copy.deepcopy(ideal), algorithm=str(loco.get("algorithm", "lm")), nIter=int(loco["nIter"]),
            used_bpms_ords=bpm, used_cor_ords=[correctors, correctors], quads_ords=quads,
            skew_ords=np.array([], dtype=int), CAVords=cavities,
            nHBPM=len(bpm), nVBPM=len(bpm), nHorCOR=len(correctors), nVerCOR=len(correctors),
            quads_tilt_ind=quads, orm_measured=orm, weights=weights,
            includeDispersion=bool(loco["include_dispersion"]), measured_eta_x=eta_x,
            measured_eta_y=eta_y, CMstep=cm_step, rfStep=float(cfg["rf"]["step_hz"]),
            Frequency=float(cfg["rf"]["frequency_hz"]), fit_list=fit_list,
            quad_individuals=True, remove_coupling_=True,
            outlier_rejection=bool(loco["outlier_rejection"]),
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


def apply_pyloco_correction(commissioning, ideal, fitted, quads) -> np.ndarray:
    """Apply the opposite reconstructed quadrupole error through pySC settings."""
    reconstructed_error = np.asarray([
        fitted[int(index)].PolynomB[1] - ideal[int(index)].PolynomB[1]
        for index in quads
    ])
    correction = -reconstructed_error
    for index, delta_k in zip(quads, correction):
        setting = commissioning.magnet_settings.index_mapping[int(index)] + "/B2"
        present_value = commissioning.magnet_settings.get(setting)
        commissioning.magnet_settings.set(setting, present_value + float(delta_k))
    return correction


def optics_error_metrics(ideal, lattice) -> tuple[float, float, float, float]:
    """Return RMS beta beating [%] and dispersion error [mm] in both planes."""
    refpts = np.arange(len(ideal) + 1)
    ideal_twiss = at.get_optics(ideal, refpts)[2]
    lattice_twiss = at.get_optics(lattice, refpts)[2]
    beta_error = 100.0 * (
        lattice_twiss.beta - ideal_twiss.beta
    ) / ideal_twiss.beta
    eta_x_error_mm = 1e3 * (
        lattice_twiss.dispersion[:, 0] - ideal_twiss.dispersion[:, 0]
    )
    eta_y_error_mm = 1e3 * (
        lattice_twiss.dispersion[:, 2] - ideal_twiss.dispersion[:, 2]
    )
    return (
        float(np.sqrt(np.mean(beta_error[:, 0] ** 2))),
        float(np.sqrt(np.mean(beta_error[:, 1] ** 2))),
        float(np.sqrt(np.mean(eta_x_error_mm ** 2))),
        float(np.sqrt(np.mean(eta_y_error_mm ** 2))),
    )


def print_optics_summary(ideal, machine, fitted, corrected) -> None:
    """Print the numerical counterpart of the beta-beating/dispersion plots."""
    rows = (
        ("pySC machine with errors", machine),
        ("pyLOCO fitted lattice", fitted),
        ("Corrected pySC machine", corrected),
    )
    print("\nOptics errors relative to the ideal EBS lattice")
    print("-" * 84)
    print(f"{'Lattice':29s} {'βx RMS [%]':>12s} {'βy RMS [%]':>12s} "
          f"{'ηx RMS [mm]':>13s} {'ηy RMS [mm]':>13s}")
    for label, lattice in rows:
        beta_x, beta_y, eta_x, eta_y = optics_error_metrics(ideal, lattice)
        print(f"{label:29s} {beta_x:12.6f} {beta_y:12.6f} "
              f"{eta_x:13.6f} {eta_y:13.6f}")


def make_plots(output: Path, ideal, machine, fitted, corrected, quads, orm,
               corrected_orm, initial_orm, fitted_orm, chi2):
    output.mkdir(parents=True, exist_ok=True)
    before, after = initial_orm - orm, fitted_orm - orm
    plot_matrices(before, after, titles=[
        f"Ideal model − pySC measurement ({1e6*np.sqrt(np.mean(before**2)):.3f} µm RMS)",
        f"Fitted model − pySC measurement ({1e6*np.sqrt(np.mean(after**2)):.3f} µm RMS)"],
        cmap="viridis", plot_type="3d", same_scale=True,
        save_path=output / "orm_residual_before_after.png")
    correction_before = orm - initial_orm
    correction_after = corrected_orm - initial_orm
    plot_matrices(
        correction_before, correction_after,
        titles=[
            f"pySC machine − ideal ({1e6*np.sqrt(np.mean(correction_before**2)):.3f} µm RMS)",
            f"Corrected pySC machine − ideal ({1e6*np.sqrt(np.mean(correction_after**2)):.3f} µm RMS)",
        ],
        cmap="viridis", plot_type="3d", same_scale=True,
        save_path=output / "orm_correction_on_machine.png",
    )
    fig, ax = plt.subplots(figsize=(7, 4)); ax.semilogy(range(1, len(chi2)+1), chi2, "o-")
    ax.set(xlabel="LOCO iteration", ylabel="Normalized chi-square", title="Fit convergence")
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(output / "fit_convergence.png", dpi=180); plt.close(fig)
    refpts = np.arange(len(ideal)+1)
    twiss = [at.get_optics(r, refpts)[2] for r in (ideal, machine, fitted, corrected)]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for plane, ax in enumerate(axes):
        for label, optics, style, color in (("pySC machine", twiss[1], "-", "C3"),
                                             ("Fitted lattice", twiss[2], "--", "C0"),
                                             ("Corrected pySC machine", twiss[3], "-.", "C2")):
            beat = 100*(optics.beta[:, plane]-twiss[0].beta[:, plane])/twiss[0].beta[:, plane]
            ax.plot(twiss[0].s_pos, beat, style, color=color, label=label)
        ax.set_ylabel(rf"$Δ\beta_{{{'xy'[plane]}}}/\beta_{{{'xy'[plane]}}}$ [%]")
        ax.grid(alpha=.2); ax.legend()
    axes[-1].set_xlabel("Longitudinal position s [m]")
    fig.suptitle("Beta beating before and after applying the pyLOCO correction")
    fig.tight_layout()
    fig.savefig(output / "beta_beating_before_after_correction.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    dispersion_components = (0, 2)
    for plane, (ax, component) in enumerate(zip(axes, dispersion_components)):
        for label, optics, style, color in (("pySC machine", twiss[1], "-", "C3"),
                                             ("Fitted lattice", twiss[2], "--", "C0"),
                                             ("Corrected pySC machine", twiss[3], "-.", "C2")):
            dispersion_error_mm = 1e3 * (
                optics.dispersion[:, component] - twiss[0].dispersion[:, component]
            )
            ax.plot(twiss[0].s_pos, dispersion_error_mm, style, color=color, label=label)
        ax.set_ylabel(rf"$Δ\eta_{{{'xy'[plane]}}}$ [mm]")
        ax.grid(alpha=.2)
        ax.legend()
    axes[-1].set_xlabel("Longitudinal position s [m]")
    fig.suptitle("Dispersion error before and after applying the pyLOCO correction")
    fig.tight_layout()
    fig.savefig(output / "dispersion_before_after_correction.png", dpi=180)
    plt.close(fig)
    truth = np.array([machine[i].PolynomB[1]-ideal[i].PolynomB[1] for i in quads])
    reconstructed = np.array([fitted[i].PolynomB[1]-ideal[i].PolynomB[1] for i in quads])
    s = at.get_s_pos(ideal, quads)
    fig, ax = plt.subplots(figsize=(11, 4)); ax.plot(s, truth, "C3.", label="pySC assigned errors")
    ax.plot(s, reconstructed, "C0--", label="pyLOCO reconstructed errors")
    ax.set(xlabel="Longitudinal position s [m]", ylabel=r"$Δ K$ [m$^{-2}$]",
           title="Quadrupole error reconstruction"); ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    fig.savefig(output / "quadrupole_errors.png", dpi=180); plt.close(fig)


def main(config_path: Path) -> None:
    cfg = load_config(config_path); base = config_path.parent
    commissioning = load_pysc_machine(config_path, cfg)
    ideal = commissioning.lattice.design
    # Keep the original erroneous machine for the before/after plots because
    # applying settings below updates commissioning.lattice.ring in place.
    machine = copy.deepcopy(commissioning.lattice.ring)
    ideal.disable_6d(); machine.disable_6d()
    bpm, correctors, quads, cavities = selected_indices(ideal, cfg)
    orm, eta_x, eta_y, weights = acquire_measurement(commissioning, base, cfg)
    if orm.shape != (2*len(bpm), 2*len(correctors)):
        raise ValueError("Saved pySC ORM dimensions do not match the selected BPMs/correctors")
    kick = float(cfg["measurement"]["corrector_kick_rad"])
    rm_cfg = RMConfig(bpm_ords=bpm, cm_ords=[correctors, correctors],
                      dkick=[[kick]*len(correctors)]*2, bidirectional=True)
    initial_orm = response_matrix(ideal, config=rm_cfg)
    result = run_fit(ideal, bpm, correctors, quads, cavities, orm, eta_x, eta_y, weights, cfg)
    _, _, fitted, fitted_orm, _, chi2, _, _ = result

    print("Applying the opposite reconstructed quadrupole errors through pySC ...")
    correction = apply_pyloco_correction(commissioning, ideal, fitted, quads)
    corrected = copy.deepcopy(commissioning.lattice.ring)
    corrected.disable_6d()
    corrected_orm = measure_orm_with_pysc(commissioning, cfg)

    output = (base / cfg["output"]["directory"]).resolve()
    make_plots(output, ideal, machine, fitted, corrected, quads, orm,
               corrected_orm, initial_orm, fitted_orm, chi2)
    before = np.sqrt(np.mean((initial_orm-orm)**2)); after = np.sqrt(np.mean((fitted_orm-orm)**2))
    corrected_residual = np.sqrt(np.mean((corrected_orm-initial_orm)**2))
    print("\nEBS pySC + pyLOCO summary\n--------------------------")
    print(f"ORM shape           : {orm.shape}")
    print(f"Configured errors   : {cfg['pysc']['error_configuration']}")
    print(f"Fitted parameters   : {', '.join(selected_fit_parameters(cfg))}")
    print(f"ORM RMS before      : {1e6*before:.6f} µm")
    print(f"ORM RMS after       : {1e6*after:.6f} µm")
    print(f"Improvement         : {before/after:.3f}x")
    print(f"Corrections applied : {len(correction)} quadrupoles")
    print(f"Corrected ORM vs ideal: {1e6*corrected_residual:.6f} µm")
    print_optics_summary(ideal, machine, fitted, corrected)
    print(f"Figures             : {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "pyloco_config.yaml")
    main(parser.parse_args().config.resolve())
