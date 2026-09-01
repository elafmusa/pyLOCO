#!/usr/bin/env python3
"""Run one production-selection PETRA-IV normal analytical Jacobian.

This is deliberately not a LOCO driver.  It loads the PETRA-IV design
lattice, computes the nominal Linear ORM required by the adaptive dispersion
finite difference, and evaluates exactly one normal-quadrupole Jacobian.

Example Maxwell/Slurm invocation::

    srun --cpus-per-task=16 python benchmark_analytical_jacobian.py

The production default is the vectorized analytical implementation with
multiprocessing.  The preserved reference implementation can be selected
explicitly with ``--implementation legacy``.
"""

from __future__ import annotations

import argparse
import ast
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import at
import h5py
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from pyLOCO.config import FitInitConfig, RMConfig, fixed_parameters
from pyLOCO.pyloco import compute_jacobian
from pyLOCO.response_matrix import response_matrix


EXPECTED_COUNTS = {
    "bpms": 786,
    "horizontal_correctors": 10,
    "vertical_correctors": 10,
    "normal_quadrupoles": 1350,
    "rf_cavities": 1,
}


def _load_production_settings():
    """Read literal scientific settings without importing the pySC driver."""
    script_dir = Path(__file__).resolve().parent
    source = script_dir / "improve_pyLOCO_latest.py"
    wanted = {
        "CORRECTOR_KICK_RAD", "INCLUDE_DISPERSION", "RF_STEP_HZ",
        "PETRA_IV_RF_FREQUENCY_HZ", "AUTO_CORRECT_DELTA",
        "STAGE1_RESPONSE_MATRIX_CALCULATOR",
    }
    values = {}
    for node in ast.parse(source.read_text(encoding="utf-8")).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in wanted:
            values[target.id] = ast.literal_eval(node.value)
    missing = wanted - values.keys()
    if missing:
        raise RuntimeError(
            f"Could not read production PETRA-IV settings {sorted(missing)} from {source}"
        )
    return SimpleNamespace(
        **values,
        LATTICE_FILE=script_dir / "p4_H6BA_v4_5_for_pySC.mat",
        CORRECTOR_FILE=script_dir / "selected_correctors_v_4_5_covarage_all.npz",
        SOURCE_FILE=source,
    )


production = _load_production_settings()


def _json_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _resolve_input_path(value, label):
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    return path


def _selection(ring, corrector_selection):
    correctors = np.load(corrector_selection)
    selection = {
        "bpms": np.asarray(at.get_refpts(ring, at.Monitor), dtype=int),
        "horizontal_correctors": np.asarray(correctors["hcor_inds"], dtype=int),
        "vertical_correctors": np.asarray(correctors["vcor_inds"], dtype=int),
        "normal_quadrupoles": np.asarray(
            at.get_refpts(ring, at.elements.Quadrupole), dtype=int
        ),
        "rf_cavities": np.asarray(
            at.get_refpts(ring, at.elements.RFCavity), dtype=int
        ),
    }
    actual = {name: len(ordinals) for name, ordinals in selection.items()}
    if actual != EXPECTED_COUNTS:
        raise RuntimeError(
            "PETRA-IV production selection changed; refusing to benchmark a "
            f"different system. Expected {EXPECTED_COUNTS}, got {actual}."
        )
    return selection


def _timing_summary(events):
    def latest(key, default=0.0):
        values = [event[key] for event in events if key in event]
        return float(values[-1]) if values else float(default)

    formula = latest("derivative_seconds")
    formula_assembly = latest("assembly_and_calibration_seconds")
    hybrid_assembly = latest("hybrid_assembly_seconds")
    worker_values = [event["workers"] for event in events if "workers" in event]
    return {
        "shared_optics_seconds": latest("optics_preparation_seconds"),
        "analytical_formula_seconds": formula,
        "numerical_dispersion_seconds": latest("dispersion_derivative_seconds"),
        "assembly_seconds": formula_assembly + hybrid_assembly,
        "formula_assembly_seconds": formula_assembly,
        "dispersion_assembly_seconds": hybrid_assembly,
        "hdf5_write_seconds": latest("hdf5_save_seconds"),
        "multiprocessing_worker_count": (
            int(max(worker_values)) if worker_values else 1
        ),
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lattice", type=Path, default=production.LATTICE_FILE,
        help="PETRA-IV design lattice (default: repository production lattice).",
    )
    parser.add_argument(
        "--corrector-selection", type=Path, default=production.CORRECTOR_FILE,
        help="Production H/V corrector-selection NPZ file.",
    )
    parser.add_argument(
        "--implementation", choices=("vectorized", "legacy"),
        default="vectorized",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path(__file__).resolve().parent / "analytical_jacobian_benchmark",
    )
    parser.add_argument(
        "--no-multiprocessing", action="store_true",
        help="Diagnostic override; Maxwell production validation should omit this.",
    )
    parser.add_argument(
        "--formula-workers", type=int, default=None,
        help="Formula workers; 0 selects serial. Unspecified preserves legacy coupling.",
    )
    parser.add_argument(
        "--dispersion-workers", type=int, default=None,
        help="Dispersion workers; 0 selects serial. Unspecified preserves legacy coupling.",
    )
    parser.add_argument(
        "--dispersion-worker", choices=("legacy_full_orm", "rf_only"),
        default="legacy_full_orm",
    )
    parser.add_argument(
        "--dispersion-difference", choices=("central", "forward"),
        default="central",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    implementation = args.implementation
    use_mp = not args.no_multiprocessing
    formula_use_mp = use_mp if args.formula_workers is None else args.formula_workers > 0
    dispersion_use_mp = use_mp if args.dispersion_workers is None else args.dispersion_workers > 0
    formula_workers = args.formula_workers if formula_use_mp else None
    dispersion_workers = args.dispersion_workers if dispersion_use_mp else None
    lattice_path = _resolve_input_path(args.lattice, "Lattice")
    corrector_selection_path = _resolve_input_path(
        args.corrector_selection, "Corrector selection"
    )
    output_dir = args.output_root.resolve() / implementation
    output_dir.mkdir(parents=True, exist_ok=True)
    jacobian_path = output_dir / "petra_iv_normal_analytical_jacobian.h5"
    summary_path = output_dir / "timing_summary.json"
    if jacobian_path.exists() or summary_path.exists():
        raise FileExistsError(
            f"Benchmark output already exists in {output_dir}; choose a new "
            "--output-root or remove it explicitly. Campaign output is never overwritten."
        )

    wall_started = time.perf_counter()
    ring = at.load_lattice(lattice_path)
    ring.disable_6d()
    selection = _selection(ring, corrector_selection_path)
    bpms = selection["bpms"]
    hcors = selection["horizontal_correctors"]
    vcors = selection["vertical_correctors"]
    quads = selection["normal_quadrupoles"]
    cavities = selection["rf_cavities"]
    cm_step = [
        np.full(len(hcors), production.CORRECTOR_KICK_RAD, dtype=float),
        np.full(len(vcors), production.CORRECTOR_KICK_RAD, dtype=float),
    ]

    # The production wrapper passes this RF frequency to pyloco().  The
    # dispersion helper resolves it through fixed_parameters, so scope the
    # override to this benchmark and restore it on every exit path.
    original_frequency = fixed_parameters.Frequency
    original_working_directory = Path.cwd()
    timing_events = []
    try:
        fixed_parameters.Frequency = production.PETRA_IV_RF_FREQUENCY_HZ
        # The existing multiprocessing dispersion helper writes diagnostic
        # logs below a relative ``output/`` directory. Keep those artifacts
        # inside this benchmark directory rather than beside campaign data.
        os.chdir(output_dir)
        rm_config = RMConfig(
            bpm_ords=bpms,
            cm_ords=[hcors, vcors],
            cav_ords=cavities,
            dkick=cm_step,
            includeDispersion=production.INCLUDE_DISPERSION,
            rfStep=production.RF_STEP_HZ,
            calculator=production.STAGE1_RESPONSE_MATRIX_CALCULATOR,
            Frequency=production.PETRA_IV_RF_FREQUENCY_HZ,
        )
        orm_started = time.perf_counter()
        model_orm = response_matrix(ring, config=rm_config)
        initial_orm_seconds = time.perf_counter() - orm_started

        n_bpm_rows = 2 * len(bpms)
        fit_config = FitInitConfig(
            fit_list=("quads",),
            CMstep=cm_step,
            rfStep=production.RF_STEP_HZ,
            individuals=True,
        )
        jacobian_started = time.perf_counter()
        jacobian, delta, _, _ = compute_jacobian(
            ring,
            C_model=model_orm,
            dkick=cm_step,
            dk=fixed_parameters.dk,
            bpm_indexes=bpms,
            CMords=[hcors, vcors],
            quads_ind=quads,
            nHorCOR=len(hcors),
            nVerCOR=len(vcors),
            nHBPM=len(bpms),
            nVBPM=len(bpms),
            C=np.eye(n_bpm_rows),
            CAVords=cavities,
            includeDispersion=True,
            include_quads=True,
            include_skew=False,
            rf_step=production.RF_STEP_HZ,
            quad_individuals=True,
            auto_correct_delta=production.AUTO_CORRECT_DELTA,
            Frequency=production.PETRA_IV_RF_FREQUENCY_HZ,
            fit_cfg=fit_config,
            quad_jacobian_file=jacobian_path,
            force_recompute=True,
            output_dir=output_dir,
            save_jacobians=True,
            quad_jacobian_calculator="Analytical",
            analytical_implementation=implementation,
            analytical_thick_quadrupole=True,
            analytical_thick_steerers=False,
            analytical_verbose=False,
            analytical_use_mp=use_mp,
            analytical_formula_use_mp=formula_use_mp,
            analytical_formula_workers=formula_workers,
            analytical_dispersion_use_mp=dispersion_use_mp,
            analytical_dispersion_workers=dispersion_workers,
            analytical_dispersion_worker=args.dispersion_worker,
            analytical_dispersion_difference=args.dispersion_difference,
            response_matrix_calculator=production.STAGE1_RESPONSE_MATRIX_CALCULATOR,
            analytical_timing_callback=timing_events.append,
        )
        jacobian_call_seconds = time.perf_counter() - jacobian_started
    finally:
        os.chdir(original_working_directory)
        fixed_parameters.Frequency = original_frequency

    timings = _timing_summary(timing_events)
    with h5py.File(jacobian_path, "r") as handle:
        compute_seconds = float(handle.attrs["computation_seconds"])
        saved_implementation = str(handle.attrs["analytical_implementation"])
        saved_dispersion_calculator = str(
            handle.attrs["analytical_dispersion_calculator"]
        )

    array = np.asarray(jacobian)
    finite_count = int(np.count_nonzero(np.isfinite(array)))
    summary = {
        "benchmark": "PETRA-IV normal analytical Jacobian",
        "production_settings_source": str(production.SOURCE_FILE.resolve()),
        "lattice": str(lattice_path),
        "corrector_selection": str(corrector_selection_path),
        "implementation": saved_implementation,
        "analytical_dispersion_calculator": saved_dispersion_calculator,
        "analytical_use_mp": use_mp,
        "analytical_formula_use_mp": formula_use_mp,
        "analytical_formula_workers_requested": formula_workers,
        "analytical_dispersion_use_mp": dispersion_use_mp,
        "analytical_dispersion_workers_requested": dispersion_workers,
        "analytical_dispersion_worker": args.dispersion_worker,
        "analytical_dispersion_difference": args.dispersion_difference,
        "formula_worker_count_resolved": int(next((
            event["formula_worker_count"] for event in reversed(timing_events)
            if "formula_worker_count" in event
        ), 1)),
        "dispersion_worker_count_resolved": int(next((
            event["dispersion_worker_count"] for event in reversed(timing_events)
            if "dispersion_worker_count" in event
        ), 1)),
        "dispersion_worker_payload_bytes": int(next((
            event["dispersion_worker_payload_bytes"] for event in reversed(timing_events)
            if "dispersion_worker_payload_bytes" in event
        ), 0)),
        "adaptive_step_evaluation_counts": next((
            event["adaptive_step_evaluation_counts"] for event in reversed(timing_events)
            if "adaptive_step_evaluation_counts" in event
        ), []),
        "multiprocessing_worker_count": timings["multiprocessing_worker_count"],
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "host_cpu_count": multiprocessing.cpu_count(),
        "include_dispersion": True,
        "thick_quadrupoles": True,
        "thick_steerers": False,
        "response_matrix_calculator": production.STAGE1_RESPONSE_MATRIX_CALCULATOR,
        "cm_step_rad": production.CORRECTOR_KICK_RAD,
        "rf_step_hz": production.RF_STEP_HZ,
        "rf_frequency_hz": production.PETRA_IV_RF_FREQUENCY_HZ,
        "selection_counts": EXPECTED_COUNTS,
        "jacobian_shape": list(array.shape),
        "finite_count": finite_count,
        "element_count": int(array.size),
        "all_finite": bool(finite_count == array.size),
        "rms": float(np.sqrt(np.mean(np.square(array)))),
        "max_abs": float(np.max(np.abs(array))),
        "shared_optics_seconds": timings["shared_optics_seconds"],
        "analytical_formula_seconds": timings["analytical_formula_seconds"],
        "numerical_dispersion_seconds": timings["numerical_dispersion_seconds"],
        "assembly_seconds": timings["assembly_seconds"],
        "formula_assembly_seconds": timings["formula_assembly_seconds"],
        "dispersion_assembly_seconds": timings["dispersion_assembly_seconds"],
        "hdf5_write_seconds": timings["hdf5_write_seconds"],
        "initial_model_orm_seconds": initial_orm_seconds,
        "total_jacobian_compute_seconds": compute_seconds,
        "jacobian_call_including_hdf5_seconds": jacobian_call_seconds,
        "total_wall_seconds": time.perf_counter() - wall_started,
        "finite_difference_steps_shape": list(np.asarray(delta).shape),
        "finite_difference_steps": np.asarray(delta).tolist(),
        "jacobian_file": str(jacobian_path),
        "summary_file": str(summary_path),
        "timing_events": timing_events,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_value) + "\n",
        encoding="utf-8",
    )

    labels = (
        ("shared optics time", "shared_optics_seconds"),
        ("analytical formula time", "analytical_formula_seconds"),
        ("numerical dispersion time", "numerical_dispersion_seconds"),
        ("assembly time", "assembly_seconds"),
        ("HDF5 write time", "hdf5_write_seconds"),
        ("total Jacobian compute time", "total_jacobian_compute_seconds"),
        ("total wall time", "total_wall_seconds"),
        ("Jacobian shape", "jacobian_shape"),
        ("finite count", "finite_count"),
        ("RMS", "rms"),
        ("max abs", "max_abs"),
        ("implementation", "implementation"),
        ("multiprocessing worker count", "multiprocessing_worker_count"),
    )
    print("\nPETRA-IV Analytical Jacobian benchmark")
    for label, key in labels:
        print(f"{label:34s}: {summary[key]}")
    print(f"Jacobian file                     : {jacobian_path}")
    print(f"Timing summary                    : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
