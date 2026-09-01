#!/usr/bin/env python3
"""Benchmark exactly one production-size PETRA-IV analytical skew Jacobian.

This wrapper does not run LOCO or correction cycles. It builds one nominal
Stage-2 model ORM and evaluates the 388-parameter analytical skew Jacobian,
including its configured numerical dispersion derivative.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import resource
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
from pyLOCO.parallel import available_worker_count
from pyLOCO.pyloco import compute_jacobian
from pyLOCO.response_matrix import response_matrix


EXPECTED_COUNTS = {
    "bpms": 786,
    "horizontal_correctors": 10,
    "vertical_correctors": 10,
    "skew_quadrupoles": 388,
    "rf_cavities": 1,
}
EXPECTED_SKEW_FAMILY_COUNTS = {"CYSF": 144, "CXYSF": 172, "CXSF": 72}


def _load_production_settings():
    script_dir = Path(__file__).resolve().parent
    source = script_dir / "improve_pyLOCO_latest.py"
    wanted = {
        "CORRECTOR_KICK_RAD", "INCLUDE_DISPERSION", "RF_STEP_HZ",
        "PETRA_IV_RF_FREQUENCY_HZ", "AUTO_CORRECT_DELTA",
        "SKEW_ANALYTICAL_DISPERSION_CALCULATOR",
        "SKEW_ANALYTICAL_DISPERSION_WORKER",
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
        raise RuntimeError(f"Missing production settings {sorted(missing)} in {source}")
    return SimpleNamespace(
        **values,
        SOURCE_FILE=source,
        LATTICE_FILE=script_dir / "p4_H6BA_v4_5_for_pySC.mat",
        CORRECTOR_FILE=script_dir / "selected_correctors_v_4_5_covarage_all.npz",
    )


production = _load_production_settings()


def _resolve_input_path(value, label):
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    return path


def _skew_family_group(name):
    upper = str(name).upper()
    for prefix in ("CXYSF", "CYSF", "CXSF"):
        if upper.startswith(prefix):
            return prefix
    return None


def _selection(ring, corrector_path):
    correctors = np.load(corrector_path)
    skew_ords = np.asarray(sorted(
        index for index, element in enumerate(ring)
        if _skew_family_group(element.FamName) is not None
    ), dtype=int)
    family_counts = {name: 0 for name in EXPECTED_SKEW_FAMILY_COUNTS}
    for ordinal in skew_ords:
        family_counts[_skew_family_group(ring[int(ordinal)].FamName)] += 1
    selection = {
        "bpms": np.asarray(at.get_refpts(ring, at.Monitor), dtype=int),
        "horizontal_correctors": np.asarray(correctors["hcor_inds"], dtype=int),
        "vertical_correctors": np.asarray(correctors["vcor_inds"], dtype=int),
        "skew_quadrupoles": skew_ords,
        "rf_cavities": np.asarray(at.get_refpts(ring, at.elements.RFCavity), dtype=int),
    }
    actual = {name: len(values) for name, values in selection.items()}
    if actual != EXPECTED_COUNTS:
        raise RuntimeError(f"Production selection mismatch: expected {EXPECTED_COUNTS}, got {actual}")
    if family_counts != EXPECTED_SKEW_FAMILY_COUNTS:
        raise RuntimeError(
            f"Production skew-family selection mismatch: expected "
            f"{EXPECTED_SKEW_FAMILY_COUNTS}, got {family_counts}"
        )
    if selection["rf_cavities"].tolist() != [21085]:
        raise RuntimeError(f"Expected cavity ordinal 21085, got {selection['rf_cavities'].tolist()}")
    return selection, family_counts


def _latest(events, key, default=0.0):
    values = [event[key] for event in events if key in event]
    return float(values[-1]) if values else float(default)


def _peak_rss_bytes():
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lattice", type=Path, default=production.LATTICE_FILE)
    parser.add_argument("--corrector-selection", type=Path, default=production.CORRECTOR_FILE)
    parser.add_argument("--model-orm", choices=("Linear", "Tracking"), default="Linear")
    parser.add_argument(
        "--dispersion-calculator", choices=("Linear", "Tracking"),
        default=production.SKEW_ANALYTICAL_DISPERSION_CALCULATOR,
    )
    parser.add_argument(
        "--dispersion-worker", choices=("legacy_full_orm", "rf_only"),
        default=production.SKEW_ANALYTICAL_DISPERSION_WORKER,
    )
    parser.add_argument(
        "--dispersion-difference", choices=("central", "forward"),
        default="central",
    )
    parser.add_argument("--implementation", choices=("legacy", "vectorized"), default="vectorized")
    parser.add_argument(
        "--output-root", type=Path,
        default=Path(__file__).resolve().parent / "analytical_skew_jacobian_benchmark",
    )
    parser.add_argument("--no-multiprocessing", action="store_true")
    parser.add_argument("--formula-workers", type=int, default=None,
                        help="Formula workers; 0 selects serial.")
    parser.add_argument("--dispersion-workers", type=int, default=None,
                        help="Dispersion workers; 0 selects serial.")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    lattice_path = _resolve_input_path(args.lattice, "Lattice")
    corrector_path = _resolve_input_path(args.corrector_selection, "Corrector selection")
    use_mp = not args.no_multiprocessing
    formula_use_mp = use_mp if args.formula_workers is None else args.formula_workers > 0
    dispersion_use_mp = use_mp if args.dispersion_workers is None else args.dispersion_workers > 0
    tag = (
        f"model_{args.model_orm.lower()}__disp_{args.dispersion_calculator.lower()}__"
        f"worker_{args.dispersion_worker}__{args.implementation}"
    )
    output_dir = args.output_root.expanduser().resolve() / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    jacobian_path = output_dir / "petra_iv_skew_analytical_jacobian.h5"
    summary_path = output_dir / "timing_summary.json"
    if jacobian_path.exists() or summary_path.exists():
        raise FileExistsError(
            f"Benchmark output already exists in {output_dir}; choose a new --output-root."
        )

    wall_started = time.perf_counter()
    peak_before = _peak_rss_bytes()
    ring = at.load_lattice(lattice_path)
    ring.disable_6d()
    selection, family_counts = _selection(ring, corrector_path)
    bpms = selection["bpms"]
    hcors = selection["horizontal_correctors"]
    vcors = selection["vertical_correctors"]
    skews = selection["skew_quadrupoles"]
    cavities = selection["rf_cavities"]
    cm_step = [
        np.full(len(hcors), production.CORRECTOR_KICK_RAD, dtype=float),
        np.full(len(vcors), production.CORRECTOR_KICK_RAD, dtype=float),
    ]
    calibration = np.eye(2 * len(bpms))
    fit_config = FitInitConfig(
        fit_list=("skew_quads",), CMstep=cm_step,
        rfStep=production.RF_STEP_HZ, individuals=True,
    )
    timing_events = []
    original_frequency = fixed_parameters.Frequency
    original_directory = Path.cwd()
    try:
        fixed_parameters.Frequency = production.PETRA_IV_RF_FREQUENCY_HZ
        os.chdir(output_dir)
        model_config = RMConfig(
            bpm_ords=bpms, cm_ords=[hcors, vcors], cav_ords=cavities,
            dkick=cm_step, includeDispersion=True,
            rfStep=production.RF_STEP_HZ, calculator=args.model_orm,
            Frequency=production.PETRA_IV_RF_FREQUENCY_HZ,
        )
        model_started = time.perf_counter()
        model_orm = response_matrix(ring, config=model_config)
        model_seconds = time.perf_counter() - model_started
        jacobian_started = time.perf_counter()
        jacobian, _, steps, _ = compute_jacobian(
            ring, C_model=model_orm, dkick=cm_step, dk=None,
            bpm_indexes=bpms, CMords=[hcors, vcors], quads_ind=[],
            skew_ind=skews, nHorCOR=len(hcors), nVerCOR=len(vcors),
            nHBPM=len(bpms), nVBPM=len(bpms), C=calibration,
            CAVords=cavities, includeDispersion=True,
            include_quads=False, include_skew=True,
            skew_jacobian_calculator="Analytical",
            response_matrix_calculator=args.model_orm,
            analytical_thick_skew=True,
            analytical_skew_thick_steerers=False,
            analytical_skew_verbose=False,
            analytical_skew_use_mp=use_mp,
            skew_analytical_formula_use_mp=formula_use_mp,
            skew_analytical_formula_workers=(args.formula_workers if formula_use_mp else None),
            skew_analytical_dispersion_use_mp=dispersion_use_mp,
            skew_analytical_dispersion_workers=(args.dispersion_workers if dispersion_use_mp else None),
            skew_analytical_implementation=args.implementation,
            skew_analytical_dispersion_calculator=args.dispersion_calculator,
            skew_analytical_dispersion_worker=args.dispersion_worker,
            skew_analytical_dispersion_difference=args.dispersion_difference,
            skew_individuals=True, delta_skew_=fixed_parameters.delta_skew,
            auto_correct_delta=production.AUTO_CORRECT_DELTA,
            rf_step=production.RF_STEP_HZ,
            Frequency=production.PETRA_IV_RF_FREQUENCY_HZ,
            fit_cfg=fit_config, skew_jacobian_file=jacobian_path,
            force_recompute=True, output_dir=output_dir, save_jacobians=True,
            skew_analytical_timing_callback=timing_events.append,
        )
        jacobian_call_seconds = time.perf_counter() - jacobian_started
    finally:
        os.chdir(original_directory)
        fixed_parameters.Frequency = original_frequency

    append_started = time.perf_counter()
    with h5py.File(jacobian_path, "a") as handle:
        total_skew_jacobian_seconds = float(handle.attrs["computation_seconds"])
        if "finite_difference_steps" not in handle:
            handle.create_dataset("finite_difference_steps", data=np.asarray(steps))
        handle.create_dataset("skew_ordinals", data=skews)
        handle.attrs["model_orm_calculator"] = args.model_orm
        handle.attrs["production_settings_source"] = str(production.SOURCE_FILE.resolve())
    append_seconds = time.perf_counter() - append_started
    hdf5_seconds = _latest(timing_events, "hdf5_save_seconds") + append_seconds

    array = np.asarray(jacobian)
    steps_array = np.asarray(steps)
    worker_count = available_worker_count(task_count=len(skews)) if use_mp else 1
    selection_digest = hashlib.sha256(skews.astype("<i8").tobytes()).hexdigest()
    summary = {
        "benchmark": "PETRA-IV Stage-2 analytical skew Jacobian",
        "production_settings_source": str(production.SOURCE_FILE.resolve()),
        "lattice": str(lattice_path),
        "corrector_selection": str(corrector_path),
        "selection_counts": EXPECTED_COUNTS,
        "skew_family_counts": family_counts,
        "skew_ordinal_sha256": selection_digest,
        "model_orm": args.model_orm,
        "implementation": args.implementation,
        "dispersion_calculator": args.dispersion_calculator,
        "dispersion_worker": args.dispersion_worker,
        "dispersion_difference": args.dispersion_difference,
        "multiprocessing": use_mp,
        "formula_multiprocessing": formula_use_mp,
        "formula_workers_requested": args.formula_workers,
        "dispersion_multiprocessing": dispersion_use_mp,
        "dispersion_workers_requested": args.dispersion_workers,
        "worker_count": worker_count,
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
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "host_cpu_count": multiprocessing.cpu_count(),
        "include_dispersion": True,
        "thick_skew": True,
        "thick_steerers": False,
        "cm_step_rad": production.CORRECTOR_KICK_RAD,
        "rf_step_hz": production.RF_STEP_HZ,
        "rf_frequency_hz": production.PETRA_IV_RF_FREQUENCY_HZ,
        "calibration": "identity BPM gain/coupling matrix",
        "nominal_model_orm_seconds": model_seconds,
        "matching_dispersion_reference_orm_seconds": _latest(
            timing_events, "dispersion_reference_seconds"
        ),
        "matching_dispersion_reference_reused": bool(next(
            (event["dispersion_reference_reused"] for event in reversed(timing_events)
             if "dispersion_reference_reused" in event), False
        )),
        "optics_preparation_seconds": _latest(timing_events, "optics_preparation_seconds"),
        "analytical_skew_formula_seconds": _latest(timing_events, "derivative_seconds"),
        "dispersion_derivative_seconds": _latest(timing_events, "numerical_dispersion_seconds"),
        "assembly_calibration_seconds": (
            _latest(timing_events, "assembly_and_calibration_seconds")
            + _latest(timing_events, "dispersion_assembly_seconds")
        ),
        "hdf5_write_seconds": hdf5_seconds,
        "total_skew_jacobian_seconds": total_skew_jacobian_seconds,
        "jacobian_call_including_hdf5_seconds": jacobian_call_seconds,
        "total_wall_seconds": time.perf_counter() - wall_started,
        "finite_difference_steps": steps_array.tolist(),
        "finite_difference_step_count": int(steps_array.size),
        "jacobian_shape": list(array.shape),
        "finite_count": int(np.count_nonzero(np.isfinite(array))),
        "element_count": int(array.size),
        "rms": float(np.sqrt(np.mean(np.square(array)))),
        "max_abs": float(np.max(np.abs(array))),
        "result_array_bytes": int(array.nbytes),
        "hdf5_file_bytes": int(jacobian_path.stat().st_size),
        "peak_rss_before_bytes": peak_before,
        "peak_rss_after_bytes": _peak_rss_bytes(),
        "jacobian_file": str(jacobian_path),
        "summary_file": str(summary_path),
        "timing_events": timing_events,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for label, key in (
        ("nominal model ORM time", "nominal_model_orm_seconds"),
        ("matching reference ORM time", "matching_dispersion_reference_orm_seconds"),
        ("optics preparation", "optics_preparation_seconds"),
        ("analytical skew formula", "analytical_skew_formula_seconds"),
        ("dispersion derivative", "dispersion_derivative_seconds"),
        ("assembly/calibration", "assembly_calibration_seconds"),
        ("HDF5 write", "hdf5_write_seconds"),
        ("total skew Jacobian", "total_skew_jacobian_seconds"),
        ("worker count", "worker_count"),
        ("result array bytes", "result_array_bytes"),
        ("peak RSS after", "peak_rss_after_bytes"),
    ):
        print(f"{label:34s}: {summary[key]}")
    print(f"finite-difference steps           : {steps_array.tolist()}")
    print(f"HDF5                              : {jacobian_path}")
    print(f"JSON                              : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
