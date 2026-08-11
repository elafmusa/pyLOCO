#!/usr/bin/env python3
"""Compare a PETRA III pyLOCO fit with the saved MATLAB LOCO result."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from petra_workflow import HERE, model_orm, prepare_measurement, print_summary, run_fit


def parameter_block_slices(data: dict) -> dict[str, slice]:
    """Map the standard pyLOCO/MATLAB vector layout to readable fit blocks."""
    sizes = {
        "Horizontal BPM gain": len(data["bpms"]),
        "Vertical BPM gain": len(data["bpms"]),
        "Horizontal corrector calibration": len(data["correctors"][0]),
        "Vertical corrector calibration": len(data["correctors"][1]),
        "Horizontal corrector energy shift": len(data["correctors"][0]),
        "Quadrupole strength": len(data["quad_indices"]),
    }
    blocks: dict[str, slice] = {}
    start = 0
    for name, size in sizes.items():
        blocks[name] = slice(start, start + size)
        start += size
    return blocks


def compare_with_matlab(data: dict, fit: dict) -> dict[str, np.ndarray | float]:
    """Validate vector compatibility and calculate transparent difference metrics."""
    reference_path = HERE / data["cfg"]["matlab_reference"]["standard_one_iteration"]
    matlab = np.asarray(np.load(reference_path), dtype=float).ravel()
    pyloco = np.asarray(fit["fit_results"][-1], dtype=float).ravel()
    if pyloco.shape != matlab.shape:
        raise ValueError(
            f"pyLOCO vector {pyloco.shape} is incompatible with MATLAB reference {matlab.shape}"
        )
    difference = pyloco - matlab
    scale = np.maximum(np.abs(matlab), np.finfo(float).eps)
    blocks = parameter_block_slices(data)
    if blocks["Quadrupole strength"].stop != len(pyloco):
        raise ValueError("Configured parameter blocks do not span the fitted vector")
    block_rms = {
        name: float(np.sqrt(np.mean(difference[indices] ** 2)))
        for name, indices in blocks.items()
    }
    block_max = {
        name: float(np.max(np.abs(difference[indices])))
        for name, indices in blocks.items()
    }
    return {
        "pyloco": pyloco,
        "matlab": matlab,
        "difference": difference,
        "rms_difference": float(np.sqrt(np.mean(difference**2))),
        "max_abs_difference": float(np.max(np.abs(difference))),
        "relative_rms": float(np.sqrt(np.mean((difference / scale) ** 2))),
        "blocks": blocks,
        "block_rms": block_rms,
        "block_max": block_max,
    }


def make_comparison_plots(data: dict, comparison: dict) -> Path:
    output = (HERE / data["cfg"]["output"]["matlab_comparison"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    pyloco = comparison["pyloco"]
    matlab = comparison["matlab"]
    difference = comparison["difference"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    indices = np.arange(len(pyloco))
    axes[0].plot(indices, matlab, color="C3", label="MATLAB LOCO")
    axes[0].plot(indices, pyloco, color="C0", linestyle="--", label="pyLOCO")
    axes[0].set_ylabel("Fitted parameter value")
    axes[0].set_title("PETRA III fitted parameter vectors")
    axes[0].grid(alpha=0.2); axes[0].legend()
    axes[1].plot(indices, difference, color="C2")
    axes[1].axhline(0.0, color="0.25", linewidth=0.8)
    axes[1].set(xlabel="Parameter-vector index", ylabel="pyLOCO − MATLAB")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "pyloco_vs_matlab_parameters.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(matlab, pyloco, s=10, alpha=0.55)
    lower = min(float(matlab.min()), float(pyloco.min()))
    upper = max(float(matlab.max()), float(pyloco.max()))
    ax.plot([lower, upper], [lower, upper], "k--", label="Exact agreement")
    ax.set(xlabel="MATLAB fitted value", ylabel="pyLOCO fitted value",
           title="Parameter-by-parameter agreement")
    ax.grid(alpha=0.2); ax.legend(); fig.tight_layout()
    fig.savefig(output / "pyloco_vs_matlab_scatter.png", dpi=180)
    plt.close(fig)

    blocks = comparison["blocks"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
    for ax, (name, indices) in zip(axes.ravel(), blocks.items()):
        local_index = np.arange(indices.stop - indices.start)
        ax.plot(local_index, matlab[indices], color="C3", label="MATLAB")
        ax.plot(local_index, pyloco[indices], color="C0", linestyle="--", label="pyLOCO")
        ax.set_title(name)
        ax.set_xlabel("Element within block")
        ax.set_ylabel("Fitted value")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("pyLOCO and MATLAB comparison by fitted parameter family")
    fig.savefig(output / "pyloco_vs_matlab_by_parameter_block.png", dpi=180)
    plt.close(fig)

    names = list(blocks)
    rms = np.asarray([comparison["block_rms"][name] for name in names])
    maximum = np.asarray([comparison["block_max"][name] for name in names])
    positions = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.38
    ax.bar(positions - width / 2, rms, width, label="RMS |pyLOCO − MATLAB|", color="C0")
    ax.bar(positions + width / 2, maximum, width, label="Maximum |pyLOCO − MATLAB|", color="C3")
    ax.set_yscale("log")
    ax.set_xticks(positions, names, rotation=25, ha="right")
    ax.set_ylabel("Absolute difference")
    ax.set_title("Agreement by fitted parameter family")
    ax.grid(axis="y", alpha=0.2); ax.legend(); fig.tight_layout()
    fig.savefig(output / "pyloco_vs_matlab_block_errors.png", dpi=180)
    plt.close(fig)
    return output


def print_comparison_summary(comparison: dict) -> None:
    print("\npyLOCO and MATLAB parameter comparison")
    print("---------------------------------------")
    print(f"Parameters compared       : {len(comparison['pyloco'])}")
    print(f"RMS absolute difference   : {comparison['rms_difference']:.9e}")
    print(f"Maximum absolute difference: {comparison['max_abs_difference']:.9e}")
    print(f"RMS relative difference   : {comparison['relative_rms']:.9e}")
    print("\nDifference by parameter family")
    print(f"{'Parameter family':38s} {'RMS':>14s} {'Maximum':>14s}")
    for name in comparison["blocks"]:
        print(f"{name:38s} {comparison['block_rms'][name]:14.6e} "
              f"{comparison['block_max'][name]:14.6e}")


def main(config_path: Path) -> None:
    data = prepare_measurement(config_path)
    initial_orm = model_orm(data)
    fit = run_fit(data, coupling=False)
    comparison = compare_with_matlab(data, fit)
    output = make_comparison_plots(data, comparison)
    print_summary(data, initial_orm, fit, coupling=False)
    print_comparison_summary(comparison)
    print(f"Figures                   : {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "pyloco_config.yaml")
    main(parser.parse_args().config.resolve())
