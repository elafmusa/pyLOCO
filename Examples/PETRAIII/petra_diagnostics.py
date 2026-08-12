"""Result extraction, persistence, and diagnostics for PETRA III examples."""
from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

from pyLOCO.analysis import plot_matrices
from pyLOCO.pyloco import get_fit_param_block, last_by_sorted_key


def rms(values: Any) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(array**2)))


def safe_relative_percent(delta: Any, nominal: Any, threshold: float = 1.0e-14) -> np.ndarray:
    delta = np.asarray(delta, dtype=float)
    nominal = np.asarray(nominal, dtype=float)
    return np.divide(100.0 * delta, nominal, out=np.full(delta.shape, np.nan), where=np.abs(nominal) > threshold)


def final_fit_state(fit_dict: dict[Any, Any]) -> dict[str, Any]:
    state = last_by_sorted_key(fit_dict)
    if not isinstance(state, dict):
        raise TypeError("The final pyLOCO fit state is not a parameter dictionary")
    return state


def extract_corrections(data: dict[str, Any], fit: dict[str, Any]) -> dict[str, np.ndarray]:
    """Extract machine corrections; correction = initial strength - fitted strength."""
    state = final_fit_state(fit["fit_dict"])
    groups = [([int(group)] if np.isscalar(group) else [int(i) for i in group]) for group in data["quad_indices"]]
    fitted_family_k = np.asarray(state.get("quads", []), dtype=float).ravel()
    if "quads" not in fit["fit_list"]:
        groups = []
    if "quads" in fit["fit_list"] and fitted_family_k.size != len(groups):
        raise ValueError(f"Fitted quadrupole block has {fitted_family_k.size} values; expected {len(groups)}")
    nominal_family_k = np.asarray([_normal_k(data["ring"][group[0]]) for group in groups])
    delta_q_families = nominal_family_k - fitted_family_k

    expanded_indices = np.asarray([index for group in groups for index in group], dtype=int)
    expanded_families = np.asarray([family for family, group in enumerate(groups) for _ in group], dtype=int)
    expanded_nominal = np.asarray([_normal_k(data["ring"][index]) for index in expanded_indices])
    expanded_fitted = np.asarray([_normal_k(fit["ring"][index]) for index in expanded_indices])
    delta_q_expanded = expanded_nominal - expanded_fitted

    fitted_skew = np.asarray(state.get("skew_quads", []), dtype=float).ravel()
    skew_indices = np.asarray(data["skew_indices"], dtype=int)
    if "skew_quads" in fit["fit_list"] and fitted_skew.size != len(skew_indices):
        raise ValueError(f"Fitted skew block has {fitted_skew.size} values; expected {len(skew_indices)}")
    skew_attribute = str(data["cfg"]["loco"].get(
        "skew_correction_reference_attribute",
        data["cfg"]["loco"].get("skew_attribute", "PolynomB"),
    ))
    if fitted_skew.size == 0:
        skew_indices = np.asarray([], dtype=int)
    nominal_skew = np.asarray([_skew_k(data["ring"][index], skew_attribute) for index in skew_indices])
    delta_skew = nominal_skew - fitted_skew if fitted_skew.size else np.asarray([])

    return {
        "family_indices": np.arange(len(groups)), "representative_indices": np.asarray([g[0] for g in groups]),
        "nominal_family_k": nominal_family_k, "fitted_family_k": fitted_family_k,
        "delta_q_families": delta_q_families,
        "expanded_family_indices": expanded_families, "expanded_indices": expanded_indices,
        "expanded_nominal_k": expanded_nominal, "expanded_fitted_k": expanded_fitted,
        "delta_q_expanded": delta_q_expanded, "skew_indices": skew_indices,
        "nominal_skew_k": nominal_skew, "fitted_skew_k": fitted_skew, "delta_skew": delta_skew,
    }


def save_run_results(data: dict[str, Any], initial_orm: np.ndarray, fit: dict[str, Any], output: Path) -> dict[str, Any]:
    plots, correction, results = _directories(output)
    np.save(results / "fit_results.npy", np.asarray(fit["fit_results"], dtype=object), allow_pickle=True)
    np.save(results / "orm_model_after.npy", fit["orm"])
    np.save(results / "chi2_history.npy", np.asarray(fit["chi2"]))
    if len(fit["delta_chi2"]):
        np.save(results / "delta_chi2_history.npy", np.asarray(fit["delta_chi2"], dtype=object), allow_pickle=True)
    if fit["c_bpms"] is not None:
        np.save(results / "C_bpms_after.npy", fit["c_bpms"])
    with (results / "fit_dict.pkl").open("wb") as stream:
        pickle.dump(fit["fit_dict"], stream)
    if fit["blocks"] is not None:
        with (results / "blocks.pkl").open("wb") as stream:
            pickle.dump(fit["blocks"], stream)
    try:
        import at
        at.save_lattice(fit["ring"], str(results / "ring_pyloco.mat"))
    except Exception as exc:
        fit["lattice_save_warning"] = str(exc)

    corrections = extract_corrections(data, fit)
    _save_corrections(data, fit, corrections, correction)
    metrics = calculate_metrics(data, initial_orm, fit, corrections)
    fit["output"] = output
    summary = _run_summary(data, fit, metrics)
    (results / "run_summary.yaml").write_text(yaml.safe_dump(_plain(summary), sort_keys=False), encoding="utf-8")
    fit["metrics"] = metrics
    fit["corrections"] = corrections
    return summary


def make_diagnostic_plots(data: dict[str, Any], initial_orm: np.ndarray, fit: dict[str, Any], output: Path, *, coupling: bool) -> None:
    plots, _, _ = _directories(output)
    measured, fitted = data["orm"], fit["orm"]
    before, after = initial_orm - measured, fitted - measured
    plot_matrices(measured, initial_orm, fitted, titles=[f"Measured ORM\n{_stats_text(measured, 1e6, 'µm')}", f"Initial model ORM\n{_stats_text(initial_orm, 1e6, 'µm')}", f"Fitted pyLOCO ORM\n{_stats_text(fitted, 1e6, 'µm')}"], cmap="viridis", plot_type="3d", same_scale=True, save_path=plots / "orm_comparison.png")
    plot_matrices(before, after, titles=[f"Initial model − measurement\n{_stats_text(before, 1e6, 'µm')}", f"Fitted model − measurement\n{_stats_text(after, 1e6, 'µm')}"], cmap="viridis", plot_type="3d", same_scale=True, save_path=plots / "orm_residual_before_after.png")

    initial_chi2 = fit.get("initial_chi2")
    chi2_values = np.asarray(fit["chi2"], dtype=float)
    if initial_chi2 is not None:
        chi2_values = np.concatenate(([float(initial_chi2)], chi2_values))
        iterations = np.arange(len(chi2_values))
    else:
        iterations = np.arange(1, len(chi2_values) + 1)
    fig, ax = plt.subplots(figsize=(7, 4)); ax.semilogy(iterations, chi2_values, "o-")
    ax.set(xlabel="LOCO iteration", ylabel="Normalized chi-square", title="Fit convergence"); ax.grid(alpha=.25)
    if chi2_values.size:
        for index, label in ((0, "Initial"), (-1, "Final")):
            ax.annotate(f"{label}: {chi2_values[index]:.4g}", (iterations[index], chi2_values[index]), xytext=(8, 8 if index == 0 else -18), textcoords="offset points", bbox=dict(boxstyle="round", fc="white", alpha=.8), arrowprops=dict(arrowstyle="->", alpha=.5))
    _save(fig, plots / "fit_convergence.png")

    n_bpm, n_hcor = len(data["bpms"]), len(data["correctors"][0])
    if coupling:
        coupling_arrays = (before[:n_bpm, n_hcor:], before[n_bpm:, :n_hcor], after[:n_bpm, n_hcor:], after[n_bpm:, :n_hcor])
        coupling_names = ("Initial XY residual", "Initial YX residual", "Fitted XY residual", "Fitted YX residual")
        plot_matrices(*coupling_arrays, titles=[f"{name}\n{_stats_text(values, 1e6, 'µm')}" for name, values in zip(coupling_names, coupling_arrays)], cmap="viridis", plot_type="3d", same_scale=True, save_path=plots / "coupling_blocks_before_after.png")
    _plot_orm_blocks(measured, initial_orm, fitted, n_bpm, n_hcor, plots)

    _plot_dispersion(data, fit, plots)
    _plot_optics(data, fit, plots)
    _plot_corrections(data, fit, plots)
    _plot_parameter_blocks(fit, plots)


def calculate_metrics(data: dict[str, Any], initial_orm: np.ndarray, fit: dict[str, Any], corrections: dict[str, np.ndarray]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "orm_rms_initial_m": rms(initial_orm - data["orm"]),
        "orm_rms_fitted_m": rms(fit["orm"] - data["orm"]),
    }
    for plane in ("x", "y"):
        measured = data[f"eta_{plane}"]
        initial = fit["initial_dispersion"][plane]
        fitted = fit["fitted_dispersion"][plane]
        metrics[f"dispersion_{plane}_rms_initial_m"] = rms(initial - measured)
        metrics[f"dispersion_{plane}_rms_fitted_m"] = rms(fitted - measured)
    optics = _optics_data(data["ring"], fit["ring"])
    metrics.update(optics["metrics"])
    for prefix, values in (
        ("family_delta_k", corrections["delta_q_families"]),
        ("expanded_delta_k", corrections["delta_q_expanded"]),
    ):
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            metrics.update({f"{prefix}_min": float(finite.min()), f"{prefix}_max": float(finite.max()), f"{prefix}_rms": rms(finite)})
    for prefix, values, nominal in (
        ("family_correction", corrections["delta_q_families"], corrections["nominal_family_k"]),
        ("expanded_correction", corrections["delta_q_expanded"], corrections["expanded_nominal_k"]),
        ("skew_correction", corrections["delta_skew"], np.ones_like(corrections["delta_skew"])),
    ):
        relative = safe_relative_percent(values, nominal) if "correction" in prefix and prefix != "skew_correction" else values
        finite = relative[np.isfinite(relative)]
        if finite.size:
            metrics.update({f"{prefix}_min": float(finite.min()), f"{prefix}_max": float(finite.max()), f"{prefix}_rms": rms(finite)})
    return metrics


def _save_corrections(data: dict[str, Any], fit: dict[str, Any], c: dict[str, np.ndarray], directory: Path) -> None:
    weights = np.ones(len(c["family_indices"]))
    constraint = fit.get("constraint_cfg")
    if constraint is not None and constraint.quad_weights is not None:
        weights = np.asarray(constraint.quad_weights)
    sigma = np.full(len(weights), np.nan) if constraint is None else np.broadcast_to(constraint.quad_sigma, weights.shape)
    s_pos = _s_positions(data["ring"])
    family_rows = []
    for i, index in enumerate(c["representative_indices"]):
        family_rows.append({"family_index": i, "representative_lattice_index": int(index), "representative_name": _element_name(data["ring"][index]), "representative_s_position_m": s_pos[index], "nominal_K": c["nominal_family_k"][i], "fitted_K": c["fitted_family_k"][i], "delta_K": c["delta_q_families"][i], "delta_K_over_K_percent": safe_relative_percent(c["delta_q_families"][i:i+1], c["nominal_family_k"][i:i+1])[0], "constraint_sigma": sigma[i], "constraint_weight": weights[i]})
    _write_csv(directory / "quadrupole_family_corrections.csv", family_rows)
    expanded_rows = []
    for i, index in enumerate(c["expanded_indices"]):
        family = c["expanded_family_indices"][i]
        expanded_rows.append({"family_index": int(family), "lattice_index": int(index), "element_name": _element_name(data["ring"][index]), "s_position_m": s_pos[index], "nominal_K": c["expanded_nominal_k"][i], "fitted_K": c["expanded_fitted_k"][i], "delta_K": c["delta_q_expanded"][i], "delta_K_over_K_percent": safe_relative_percent(c["delta_q_expanded"][i:i+1], c["expanded_nominal_k"][i:i+1])[0], "constraint_weight": weights[family]})
    _write_csv(directory / "quadrupole_corrections_expanded.csv", expanded_rows)
    skew_rows = [{"lattice_index": int(index), "element_name": _element_name(data["ring"][index]), "s_position_m": s_pos[index], "nominal_skew": c["nominal_skew_k"][i], "fitted_skew": c["fitted_skew_k"][i], "delta_skew": c["delta_skew"][i]} for i, index in enumerate(c["skew_indices"])]
    if skew_rows:
        _write_csv(directory / "skew_corrections.csv", skew_rows)
    if c["delta_q_families"].size:
        np.save(directory / "delta_q_families.npy", c["delta_q_families"]); np.save(directory / "delta_q_expanded.npy", c["delta_q_expanded"])
        np.save(directory / "quad_indices_expanded.npy", c["expanded_indices"]); np.save(directory / "s_positions_expanded.npy", s_pos[c["expanded_indices"]])
    if c["delta_skew"].size: np.save(directory / "delta_skew.npy", c["delta_skew"])
    normal_delta = [float(delta) for delta in c["delta_q_expanded"]]
    normal_length = [float(getattr(data["ring"][index], "Length", 0.0)) for index in c["expanded_indices"]]
    skew_delta = [float(delta) for delta in c["delta_skew"]]
    skew_length = [float(getattr(data["ring"][index], "Length", 0.0)) for index in c["skew_indices"]]
    family_length = [float(getattr(data["ring"][index], "Length", 0.0)) for index in c["representative_indices"]]
    payload = {
        "correction_definition": {
            "symbol": "delta_K_apply",
            "equation": "delta_K_apply = K_model_initial - K_model_fitted",
            "machine_application": "K_machine_new = K_machine_current + delta_K_apply",
            "units": "m^-2",
        },
        # Keep the historical machine-application schema intact.
        "normal_quads": {"delta": normal_delta, "length": normal_length},
        "skew_quads": {"delta": skew_delta, "length": skew_length},
        # Exactly one value per fitted family (193 for case C).
        "normal_quads_family": {
            "family_index": c["family_indices"].tolist(),
            "delta": c["delta_q_families"].tolist(),
            "representative_lattice_index": c["representative_indices"].tolist(),
            "length": family_length,
            "statistics": _statistics(c["delta_q_families"]),
        },
        "normal_quads_expanded": {
            "delta": c["delta_q_expanded"].tolist(),
            "lattice_index": c["expanded_indices"].tolist(),
            "statistics": _statistics(c["delta_q_expanded"]),
        },
        "skew_correction_statistics": _statistics(c["delta_skew"]),
    }
    (directory / "quad_skew_deltas_lengths.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _plot_dispersion(data: dict[str, Any], fit: dict[str, Any], plots: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for plane, ax in zip(("x", "y"), axes):
        measured = 1e3 * data[f"eta_{plane}"]
        initial = 1e3 * fit["initial_dispersion"][plane]
        fitted = 1e3 * fit["fitted_dispersion"][plane]
        initial_error, fitted_error = initial - measured, fitted - measured
        ax.plot(measured, color="black", linewidth=1.4, label="Measurement")
        ax.plot(initial, color="C1", linestyle="--", label="Initial model")
        ax.plot(fitted, color="C0", linestyle="-", label="pyLOCO fit")
        ax.plot(initial_error, color="C3", linestyle=":", alpha=.8, label="Initial − measurement error")
        ax.plot(fitted_error, color="C2", linestyle="-.", alpha=.9, label="Fit − measurement error")
        ax.set_ylabel(rf"$\eta_{plane}$ RF orbit difference [mm]"); ax.grid(alpha=.2); ax.legend(ncol=2, fontsize=8)
        _stats_box(ax, fitted_error, unit="mm", label="Fit error")
    axes[-1].set_xlabel("Retained BPM index"); _save(fig, plots / "dispersion_fit.png")


def _plot_orm_blocks(measured: np.ndarray, initial: np.ndarray, fitted: np.ndarray, n_bpm: int, n_hcor: int, plots: Path) -> None:
    blocks = (
        (slice(0, n_bpm), slice(0, n_hcor), "XX"),
        (slice(0, n_bpm), slice(n_hcor, None), "XY"),
        (slice(n_bpm, None), slice(0, n_hcor), "YX"),
        (slice(n_bpm, None), slice(n_hcor, None), "YY"),
    )
    matrices = (("Measurement", measured), ("Initial model", initial), ("Fitted model", fitted))
    fig, axes = plt.subplots(4, 3, figsize=(15, 15), constrained_layout=True)
    scale = max(float(np.nanmax(np.abs(matrix))) for _, matrix in matrices)
    for row, (rows, columns, block_name) in enumerate(blocks):
        for column, (model_name, matrix) in enumerate(matrices):
            image = axes[row, column].imshow(matrix[rows, columns], aspect="auto", cmap="RdBu_r", vmin=-scale, vmax=scale)
            axes[row, column].set_title(f"{model_name} {block_name}")
            axes[row, column].set(xlabel="Corrector index", ylabel="BPM index")
    fig.colorbar(image, ax=axes, label="Orbit difference for configured kick [m]", shrink=.7)
    fig.savefig(plots / "orm_blocks_comparison.png", dpi=180)
    plt.close(fig)


def _plot_optics(data: dict[str, Any], fit: dict[str, Any], plots: Path) -> None:
    optics = _optics_data(data["ring"], fit["ring"]); s = optics["s"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for plane, ax in enumerate(axes):
        values = 100 * (optics["beta_fitted"][:, plane] - optics["beta_initial"][:, plane]) / optics["beta_initial"][:, plane]
        ax.plot(s, values); ax.set_ylabel(rf"$\Delta\beta_{{{'xy'[plane]}}}/\beta_{{{'xy'[plane]}}}$ [%]"); ax.grid(alpha=.2); _stats_box(ax, values, unit="%")
    axes[-1].set_xlabel("s [m]"); _save(fig, plots / "beta_beating.png")
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for plane, ax in enumerate(axes):
        values = 1e3 * (optics["dispersion_fitted"][:, plane] - optics["dispersion_initial"][:, plane])
        ax.plot(s, values, linestyle="--" if plane == 0 else "-"); ax.set_ylabel(rf"$\Delta\eta_{{{'xy'[plane]}}}$ [mm]"); ax.grid(alpha=.2); _stats_box(ax, values, unit="mm")
    axes[-1].set_xlabel("s [m]"); _save(fig, plots / "dispersion_beating.png")


def _plot_corrections(data: dict[str, Any], fit: dict[str, Any], plots: Path) -> None:
    c = fit["corrections"]; s = _s_positions(data["ring"])
    if c["delta_q_families"].size:
        family_indices = np.arange(len(c["delta_q_families"]))
        _single_plot(family_indices, c["delta_q_families"], "Fitted family index", r"Applied machine correction $\Delta K_{apply}$ [m$^{-2}$]", plots / "quadrupole_family_delta_k.png", unit=r"m$^{-2}$", title=r"Absolute correction: $K_{machine,new}=K_{machine,current}+\Delta K_{apply}$")
        _single_plot(family_indices, safe_relative_percent(c["delta_q_families"], c["nominal_family_k"]), "Fitted family index", r"Relative applied correction $100\,\Delta K_{apply}/K_{initial}$ [%]", plots / "quadrupole_family_correction.png", unit="%", title="Normalized quadrupole correction")
        pyloco_relative = safe_relative_percent(c["delta_q_expanded"], c["expanded_nominal_k"])
        reference = _expanded_reference_correction(data, c)
        _correction_along_lattice_plot(
            s[c["expanded_indices"]], pyloco_relative,
            plots / "quadrupole_correction_along_lattice.png",
            reference=reference,
            ylabel=r"Relative applied correction $100\,\Delta K_{apply}/K_{initial}$ [%]",
            unit="%",
            title="Normalized quadrupole correction along the lattice",
        )
        _correction_along_lattice_plot(
            s[c["expanded_indices"]], c["delta_q_expanded"],
            plots / "quadrupole_delta_k_along_lattice.png",
            ylabel=r"Applied machine correction $\Delta K_{apply}$ [m$^{-2}$]",
            unit=r"m$^{-2}$",
            title=r"Absolute correction: $K_{machine,new}=K_{machine,current}+\Delta K_{apply}$",
        )
    if c["delta_skew"].size: _single_plot(s[c["skew_indices"]], c["delta_skew"], "s [m]", "Machine skew correction ΔPolynomA[1] [m⁻²]", plots / "skew_correction_along_lattice.png")


def _expanded_reference_correction(data: dict[str, Any], corrections: dict[str, np.ndarray]) -> np.ndarray | None:
    relative_path = data["cfg"]["data"].get("quadrupole_reference_relative")
    if relative_path is None:
        return None
    path = data["config_path"].parent / relative_path
    family_values = np.asarray(np.load(path), dtype=float).ravel()
    if family_values.size != len(corrections["family_indices"]):
        raise ValueError(
            f"Reference quadrupole correction has {family_values.size} families; "
            f"expected {len(corrections['family_indices'])}"
        )
    return 100.0 * family_values[corrections["expanded_family_indices"]]


def _correction_along_lattice_plot(
    x: Any,
    pyloco: Any,
    path: Path,
    *,
    reference: Any = None,
    ylabel: str = r"Relative applied correction $100\,\Delta K_{apply}/K_{initial}$ [%]",
    unit: str = "%",
    title: str | None = None,
) -> None:
    x_values = np.asarray(x, dtype=float)
    pyloco_values = np.asarray(pyloco, dtype=float)
    order = np.argsort(x_values, kind="stable")
    x_values, pyloco_values = x_values[order], pyloco_values[order]
    fig, ax = plt.subplots(figsize=(12, 5))
    if reference is not None:
        reference_values = np.asarray(reference, dtype=float)[order]
        ax.plot(x_values, reference_values, color="darkorange", linewidth=1.0, marker=".", markersize=2.5, label="Reference")
    ax.plot(x_values, pyloco_values, color="navy", linestyle="--", linewidth=1.0, marker=".", markersize=2.5, label="pyLOCO")
    ax.axhline(0.0, color="0.3", linewidth=.7)
    ax.set(xlabel="Longitudinal position s [m]", ylabel=ylabel, title=title)
    ax.grid(alpha=.2); ax.legend(loc="lower right")
    _stats_box(ax, pyloco_values, unit=unit, label="pyLOCO")
    _save(fig, path)


def _plot_parameter_blocks(fit: dict[str, Any], plots: Path) -> None:
    pairs = [("bpm_gains.png", "BPM gain", ("hbpm_gain", "vbpm_gain")), ("bpm_coupling.png", "BPM coupling", ("hbpm_coupling", "vbpm_coupling")), ("corrector_calibration.png", "Corrector kick calibration [rad]", ("hcor_cal", "vcor_cal")), ("corrector_coupling.png", "Corrector coupling", ("hcor_coupling", "vcor_coupling"))]
    for filename, ylabel, names in pairs:
        if not any(name in fit["fit_list"] for name in names): continue
        fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=False)
        for ax, name in zip(axes, names):
            values = get_fit_param_block(fit["fit_dict"], name); ax.plot(values); ax.set(ylabel=ylabel, title=name); ax.grid(alpha=.2)
            _stats_box(ax, values)
        axes[-1].set_xlabel("Device index"); _save(fig, plots / filename)
    if "HCMEnergyShift" in fit["fit_list"]:
        _single_plot(np.arange(len(get_fit_param_block(fit["fit_dict"], "HCMEnergyShift"))), get_fit_param_block(fit["fit_dict"], "HCMEnergyShift"), "Horizontal corrector index", "Relative momentum/energy shift Δp/p", plots / "hcm_energy_shift.png")


def _optics_data(initial_ring: Any, fitted_ring: Any) -> dict[str, Any]:
    import at
    refpts = np.arange(len(initial_ring) + 1)
    _, _, initial = at.get_optics(initial_ring, refpts); _, _, fitted = at.get_optics(fitted_ring, refpts)
    beta_beat = 100 * (fitted.beta - initial.beta) / initial.beta
    result = {"s": initial.s_pos, "beta_initial": initial.beta, "beta_fitted": fitted.beta, "dispersion_initial": initial.dispersion[:, (0, 2)], "dispersion_fitted": fitted.dispersion[:, (0, 2)]}
    result["metrics"] = {"beta_beating_x_rms_percent": rms(beta_beat[:, 0]), "beta_beating_y_rms_percent": rms(beta_beat[:, 1]), "initial_tune": np.asarray(at.get_tune(initial_ring)).tolist(), "fitted_tune": np.asarray(at.get_tune(fitted_ring)).tolist(), "initial_chromaticity": _chromaticity(initial_ring), "fitted_chromaticity": _chromaticity(fitted_ring)}
    return result


def _chromaticity(ring: Any) -> list[float] | None:
    import at
    try:
        ring_data, _, _ = at.get_optics(ring, get_chrom=True)
        return np.asarray(ring_data.chromaticity, dtype=float).tolist()
    except Exception:
        return None


def _run_summary(data: dict[str, Any], fit: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    cfg = data["cfg"]
    runtime = fit.get("runtime_seconds")
    return {"run_name": cfg["output"].get("run_name", fit["output"].name if "output" in fit else None), "configuration_file": str(data["config_path"]), "lattice_file": cfg["lattice"]["file"], "measurement_files": {key: cfg["data"].get(key) for key in ("orm", "dispersion", "bpm_noise", "corrector_steps")}, "retained_bpms_per_plane": len(data["bpms"]), "horizontal_correctors": len(data["correctors"][0]), "vertical_correctors": len(data["correctors"][1]), "quadrupole_mode": "individual" if data["quad_individuals"] else "family", "quadrupole_fit_parameters": len(data["quad_indices"]), "skew_quadrupoles": len(data["skew_indices"]), "fit_list": fit["fit_list"], "constraints_enabled": fit["constraint_cfg"] is not None, "constraint_settings": cfg.get("constraints", {}), "dispersion_enabled": cfg["loco"]["include_dispersion"], "horizontal_dispersion_weight": cfg["loco"]["horizontal_dispersion_weight"], "vertical_dispersion_weight": cfg["loco"]["vertical_dispersion_weight"], "runtime_seconds": runtime, "runtime_readable": _format_duration(runtime) if runtime is not None else None, "metrics": metrics, "initial_chi2": fit.get("initial_chi2"), "chi2_history": fit["chi2"], "final_chi2": fit["chi2"][-1] if fit["chi2"] else None, "correction_sign_convention": {"delta_K_apply": "K_model_initial - K_model_fitted", "machine_application": "K_machine_new = K_machine_current + delta_K_apply", "relative_percent": "100 * delta_K_apply / K_model_initial"}, "lattice_save_warning": fit.get("lattice_save_warning")}


def _format_duration(seconds: float) -> str:
    minutes, secs = divmod(float(seconds), 60.0)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:d}h {minutes:02d}m {secs:05.2f}s" if hours else f"{minutes:d}m {secs:05.2f}s"


def _directories(output: Path) -> tuple[Path, Path, Path]:
    paths = tuple(output / name for name in ("plots", "correction", "results"))
    for path in paths: path.mkdir(parents=True, exist_ok=True)
    return paths


def _normal_k(element: Any) -> float:
    poly = getattr(element, "PolynomB", None)
    value = float(np.asarray(poly)[1]) if poly is not None and len(poly) > 1 else float(element.K)
    if hasattr(element, "K") and not np.isclose(value, float(element.K), rtol=1e-10, atol=1e-14):
        raise ValueError("Quadrupole K and PolynomB[1] are inconsistent")
    return value


def _skew_k(element: Any, attribute: str) -> float:
    poly = np.asarray(getattr(element, attribute)); return float(poly[1])


def _s_positions(ring: Any) -> np.ndarray:
    import at
    return np.asarray(at.get_s_pos(ring, np.arange(len(ring) + 1)))[:-1]


def _element_name(element: Any) -> str:
    return str(getattr(element, "CommonName", getattr(element, "FamName", "")))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows: return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def _single_plot(x: Any, y: Any, xlabel: str, ylabel: str, path: Path, *, unit: str = "", title: str | None = None) -> None:
    x_values = np.asarray(x)
    y_values = np.asarray(y)
    if x_values.size != y_values.size:
        raise ValueError(f"Plot x/y size mismatch: {x_values.size} != {y_values.size}")
    order = np.argsort(x_values, kind="stable")
    x_values, y_values = x_values[order], y_values[order]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_values, y_values, "--", linewidth=.8, alpha=.7)
    ax.scatter(x_values, y_values, s=13, zorder=3)
    ax.axhline(0.0, color="0.25", linewidth=.8, linestyle="--")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title); ax.grid(alpha=.2); _stats_box(ax, y_values, unit=unit); _save(fig, path)


def _stats_text(values: Any, scale: float = 1.0, unit: str = "") -> str:
    finite = np.asarray(values, dtype=float).ravel()
    finite = finite[np.isfinite(finite)] * scale
    if not finite.size:
        return "No finite values"
    suffix = f" {unit}" if unit else ""
    return f"min {finite.min():.4g}{suffix} | max {finite.max():.4g}{suffix} | RMS {rms(finite):.4g}{suffix}"


def _statistics(values: Any) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=float).ravel()
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {"minimum": None, "maximum": None, "rms": None}
    return {"minimum": float(finite.min()), "maximum": float(finite.max()), "rms": rms(finite)}


def _stats_box(ax: Any, values: Any, *, unit: str = "", label: str = "") -> None:
    prefix = f"{label}\n" if label else ""
    ax.text(.985, .97, prefix + _stats_text(values, unit=unit), transform=ax.transAxes, ha="right", va="top", fontsize=8, bbox=dict(boxstyle="round", fc="white", alpha=.82))


def _save(fig: Any, path: Path) -> None:
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _plain(value: Any) -> Any:
    if isinstance(value, dict): return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_plain(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    return value
