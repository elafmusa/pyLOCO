#!/usr/bin/env python3
"""Compare PETRA-IV normal-quadrupole dispersion derivatives by ORM backend.

This benchmark evaluates only dD/dK through pyLOCO's existing central finite-
difference and adaptive-step implementation.  It does not run LOCO or change
the analytical normal-quadrupole formulas.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import at
import h5py  # noqa: F401 - fail early if the production environment is incomplete
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from pyLOCO.config import FitInitConfig, RMConfig, fixed_parameters
from pyLOCO.pyloco import calculate_quads_dispersion_jacobian
from pyLOCO.response_matrix import response_matrix


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LATTICE = SCRIPT_DIR / "p4_H6BA_v4_5_for_pySC.mat"
DEFAULT_CORRECTORS = SCRIPT_DIR / "selected_correctors_v_4_5_covarage_all.npz"
BACKENDS = ("Linear", "Analytical", "Tracking")
EXPECTED = {"bpms": 786, "hcors": 10, "vcors": 10, "quads": 1350, "cavities": 1}
RF_STEP_HZ = 40.0
RF_FREQUENCY_HZ = 499654096.6666667
CM_STEP_RAD = 100.0e-6


def _resolved_file(value, label):
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    return path


def _selection(ring, corrector_path, count):
    correctors = np.load(corrector_path)
    all_quads = np.asarray(at.get_refpts(ring, at.elements.Quadrupole), dtype=int)
    selected = {
        "bpms": np.asarray(at.get_refpts(ring, at.Monitor), dtype=int),
        "hcors": np.asarray(correctors["hcor_inds"], dtype=int),
        "vcors": np.asarray(correctors["vcor_inds"], dtype=int),
        "quads": all_quads,
        "cavities": np.asarray(at.get_refpts(ring, at.elements.RFCavity), dtype=int),
    }
    actual = {name: len(values) for name, values in selected.items()}
    if actual != EXPECTED:
        raise RuntimeError(f"Production selection mismatch: expected {EXPECTED}, got {actual}")
    if selected["cavities"].tolist() != [21085]:
        raise RuntimeError(f"Expected cavity ordinal 21085, got {selected['cavities'].tolist()}")
    selected["quads"] = all_quads[:count]
    return selected


def _plane_metrics(reference, candidate):
    difference = candidate - reference
    reference_norm = float(np.linalg.norm(reference))
    return {
        "reference_rms": float(np.sqrt(np.mean(reference ** 2))),
        "candidate_rms": float(np.sqrt(np.mean(candidate ** 2))),
        "rms_difference": float(np.sqrt(np.mean(difference ** 2))),
        "maximum_absolute_difference": float(np.max(np.abs(difference))),
        "relative_norm_difference": (
            float(np.linalg.norm(difference) / reference_norm)
            if reference_norm else None
        ),
        "reference_finite_count": int(np.count_nonzero(np.isfinite(reference))),
        "candidate_finite_count": int(np.count_nonzero(np.isfinite(candidate))),
        "element_count": int(reference.size),
    }


def _comparison(reference_name, candidate_name, arrays, n_bpms):
    reference = arrays[reference_name]
    candidate = arrays[candidate_name]
    return {
        "reference": reference_name,
        "candidate": candidate_name,
        "horizontal": _plane_metrics(reference[:, :n_bpms], candidate[:, :n_bpms]),
        "vertical": _plane_metrics(reference[:, n_bpms:], candidate[:, n_bpms:]),
    }


def _plots(output_dir, runtimes, comparisons):
    colors = ("#4472C4", "#ED7D31", "#70AD47")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(BACKENDS, [runtimes[name] for name in BACKENDS], color=colors)
    ax.set_ylabel("Dispersion derivative runtime [s]")
    ax.set_title("PETRA-IV dD/dK runtime")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "runtime_backends.png", dpi=180)
    plt.close(fig)

    labels = ("Analytical", "Tracking")
    x = np.arange(len(labels))
    width = 0.36
    def plotted(value):
        return np.nan if value is None else value

    horizontal = [plotted(comparisons[f"Linear_vs_{name}"]["horizontal"]["relative_norm_difference"]) for name in labels]
    vertical = [plotted(comparisons[f"Linear_vs_{name}"]["vertical"]["relative_norm_difference"]) for name in labels]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(x - width / 2, horizontal, width, label="Horizontal BPM rows")
    ax.bar(x + width / 2, vertical, width, label="Vertical BPM rows")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Relative norm difference vs Linear")
    ax.set_title("PETRA-IV dD/dK backend agreement")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    if any(np.isnan(vertical)):
        fig.text(
            0.5, 0.01,
            "Vertical relative difference undefined when the reference norm is zero",
            ha="center", va="bottom", fontsize=8,
        )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output_dir / "relative_difference_vs_linear.png", dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lattice", type=Path, default=DEFAULT_LATTICE)
    parser.add_argument("--corrector-selection", type=Path, default=DEFAULT_CORRECTORS)
    parser.add_argument("--quadrupoles", type=int, choices=(10, 40), default=10)
    parser.add_argument(
        "--output-root", type=Path,
        default=SCRIPT_DIR / "dispersion_derivative_backend_benchmark",
    )
    args = parser.parse_args(argv)

    lattice_path = _resolved_file(args.lattice, "Lattice")
    corrector_path = _resolved_file(args.corrector_selection, "Corrector selection")
    output_dir = args.output_root.expanduser().resolve() / f"first_{args.quadrupoles}_quads"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = (
        output_dir / "dispersion_derivatives.npz",
        output_dir / "summary.json",
        output_dir / "runtime_backends.png",
        output_dir / "relative_difference_vs_linear.png",
    )
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"Benchmark output already exists; refusing to overwrite: {existing}")

    ring = at.load_lattice(lattice_path)
    ring.disable_6d()
    selection = _selection(ring, corrector_path, args.quadrupoles)
    bpms, hcors, vcors, quads, cavities = (
        selection[name] for name in ("bpms", "hcors", "vcors", "quads", "cavities")
    )
    cm_step = (
        np.full(len(hcors), CM_STEP_RAD, dtype=float),
        np.full(len(vcors), CM_STEP_RAD, dtype=float),
    )
    calibration = np.eye(2 * len(bpms))
    fit_config = FitInitConfig(fit_list=("quads",), CMstep=cm_step, rfStep=RF_STEP_HZ, individuals=True)

    original_frequency = fixed_parameters.Frequency
    arrays, steps, runtimes, model_runtimes = {}, {}, {}, {}
    try:
        fixed_parameters.Frequency = RF_FREQUENCY_HZ
        for backend in BACKENDS:
            config = RMConfig(
                bpm_ords=bpms, cm_ords=(hcors, vcors), cav_ords=cavities,
                dkick=cm_step, includeDispersion=True, rfStep=RF_STEP_HZ,
                Frequency=RF_FREQUENCY_HZ, calculator=backend,
            )
            model_started = time.perf_counter()
            model = response_matrix(ring, config=config)
            model_runtimes[backend] = time.perf_counter() - model_started

            started = time.perf_counter()
            arrays[backend], steps[backend] = calculate_quads_dispersion_jacobian(
                ring=ring, C_model=model, dkick=cm_step,
                used_cor_ind=(hcors, vcors), bpm_indexes=bpms,
                quads_ind=quads, dk=fixed_parameters.dk, C=calibration,
                individuals=True, HCMCoupling=np.zeros(len(hcors)),
                VCMCoupling=np.zeros(len(vcors)), rf_step=RF_STEP_HZ,
                CAVords=cavities, auto_correct_delta=True,
                fit_cfg=fit_config, orm_calculator=backend,
                use_mp=False,
            )
            runtimes[backend] = time.perf_counter() - started
    finally:
        fixed_parameters.Frequency = original_frequency

    comparisons = {
        "Linear_vs_Analytical": _comparison("Linear", "Analytical", arrays, len(bpms)),
        "Linear_vs_Tracking": _comparison("Linear", "Tracking", arrays, len(bpms)),
        "Analytical_vs_Tracking": _comparison("Analytical", "Tracking", arrays, len(bpms)),
    }
    step_comparisons = {
        "Linear_vs_Analytical": bool(np.array_equal(steps["Linear"], steps["Analytical"])),
        "Linear_vs_Tracking": bool(np.array_equal(steps["Linear"], steps["Tracking"])),
        "Analytical_vs_Tracking": bool(np.array_equal(steps["Analytical"], steps["Tracking"])),
    }
    summary = {
        "configuration": {
            "lattice": str(lattice_path),
            "corrector_selection": str(corrector_path),
            "bpms": len(bpms), "horizontal_correctors": len(hcors),
            "vertical_correctors": len(vcors), "normal_quadrupoles": len(quads),
            "quadrupole_ordinals": quads.tolist(), "cavity_ordinals": cavities.tolist(),
            "rf_step_hz": RF_STEP_HZ, "rf_frequency_hz": RF_FREQUENCY_HZ,
            "cm_step_rad": CM_STEP_RAD, "thick_quadrupoles": True,
            "thick_steerers": False, "dispersion_enabled": True,
            "adaptive_step_selection": True, "multiprocessing": False,
        },
        "runtime_seconds": runtimes,
        "nominal_model_orm_runtime_seconds": model_runtimes,
        "speed_ratios": {
            "Analytical_over_Linear": runtimes["Analytical"] / runtimes["Linear"],
            "Tracking_over_Linear": runtimes["Tracking"] / runtimes["Linear"],
        },
        "finite_difference_steps": {name: values.tolist() for name, values in steps.items()},
        "identical_quadrupole_ordering": True,
        "identical_finite_difference_steps": step_comparisons,
        "comparisons": comparisons,
    }
    np.savez_compressed(
        outputs[0],
        quadrupole_ordinals=quads,
        bpm_ordinals=bpms,
        linear=arrays["Linear"], analytical=arrays["Analytical"], tracking=arrays["Tracking"],
        steps_linear=steps["Linear"], steps_analytical=steps["Analytical"],
        steps_tracking=steps["Tracking"],
    )
    outputs[1].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plots(output_dir, runtimes, comparisons)
    print(json.dumps(summary, indent=2))
    print(f"Saved benchmark outputs to {output_dir}")
    return summary


if __name__ == "__main__":
    main()
