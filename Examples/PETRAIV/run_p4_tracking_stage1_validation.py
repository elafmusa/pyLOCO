#!/usr/bin/env python3
"""One-iteration, Stage-1-only PETRA-IV Tracking/Numerical validation."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np

import improve_pyLOCO_latest as drv
from pyLOCO.config import FitInitConfig
from run_p4_calculator_case import prepare_sc, selection


class Tee:
    def __init__(self, *streams):
        self.streams = streams
        self.lines: list[str] = []

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        self.lines.extend(value.splitlines())
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_trials(lines):
    pattern = re.compile(
        r"LM inner (?P<inner>\d+): chi² (?P<trial>[0-9.eE+-]+) "
        r"\(previous (?P<before>[0-9.eE+-]+)\), λ=(?P<lambda>[0-9.eE+-]+)"
    )
    trials = []
    for line in lines:
        match = pattern.search(line)
        if match:
            values = match.groupdict()
            before = float(values["before"])
            trial = float(values["trial"])
            trials.append({
                "inner_iteration": int(values["inner"]),
                "lambda": float(values["lambda"]),
                "chi2_before": before,
                "chi2_trial": trial,
                "finite_orm": True,
                "accepted": trial < before,
            })
        elif "rejecting non-finite" in line and "LM inner" in line:
            inner = re.search(r"LM inner (\d+)", line)
            damping = re.search(r"λ=([0-9.eE+-]+)", line)
            invalid = re.search(r"\((\d+) invalid values\)", line)
            trials.append({
                "inner_iteration": int(inner.group(1)) if inner else None,
                "lambda": float(damping.group(1)) if damping else None,
                "chi2_before": None,
                "chi2_trial": None,
                "finite_orm": False,
                "nonfinite_entries": int(invalid.group(1)) if invalid else None,
                "accepted": False,
            })
    return trials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-file", required=True, type=Path)
    parser.add_argument("--measured", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)

    sc = prepare_sc(args.seed_file)
    bpms, correctors, quads, skews, cavities, _, kicks = selection(sc)
    ring = sc.lattice.design.deepcopy()
    ring.disable_6d()
    with np.load(args.measured) as saved:
        measured_orm = saved["measured_orm"].copy()
        measured_x = saved["measured_eta_x"].copy()
        measured_y = saved["measured_eta_y"].copy()
        sigma = saved["sigma_w"].copy()
        persisted = {
            "rf_step_hz": float(saved["rf_step_hz"]),
            "rf_frequency_hz": float(saved["rf_frequency_hz"]),
            "bpm_order_matches": bool(np.array_equal(saved["used_bpms_ords"], bpms)),
            "hcor_order_matches": bool(np.array_equal(saved["hcor_inds"], correctors[0])),
            "vcor_order_matches": bool(np.array_equal(saved["vcor_inds"], correctors[1])),
            "hcor_kicks_match": bool(np.array_equal(saved["CMstep_h"], kicks[0])),
            "vcor_kicks_match": bool(np.array_equal(saved["CMstep_v"], kicks[1])),
        }
    if not all(value for key, value in persisted.items() if key.endswith("matches")):
        raise RuntimeError(f"Persisted selection/configuration mismatch: {persisted}")

    fit_list = ["quads", "hbpm_gain", "vbpm_gain", "hcor_cal", "vcor_cal"]
    fit_cfg = FitInitConfig(
        fit_list=fit_list, CMstep=kicks, rfStep=drv.RF_STEP_HZ,
        individuals=True,
    )
    iteration_metrics = []

    def capture_metrics(item):
        model = np.asarray(item["orm_model"])
        iteration_metrics.append({
            "iteration": int(item["iteration"]),
            "chi2_before": float(item["chi2_before"]),
            "chi2_after": float(item["chi2_after"]),
            "timings": dict(item["timings"]),
            "final_orm_finite": int(np.isfinite(model).sum()),
            "final_orm_total": int(model.size),
        })
    log_path = args.output / "stage1_validation.log"
    original_stdout = sys.stdout
    with log_path.open("w", buffering=1) as log, warnings.catch_warnings(record=True) as caught:
        tee = Tee(original_stdout, log)
        sys.stdout = tee
        warnings.simplefilter("always")
        try:
            result = drv.pyloco(
                ring,
                algorithm=drv.ALGORITHM,
                nIter=1,
                fit_list=fit_list,
                remove_coupling_=True,
                svd_selection_method=drv.SVD_SELECTION_METHOD,
                cut_=drv.STAGE1_SVD_CUT,
                fit_cfg=fit_cfg,
                response_matrix_calculator="Tracking",
                iteration_metrics_callback=capture_metrics,
                output_dir=str(args.output / "pyloco_stage1"),
                used_bpms_ords=bpms,
                used_cor_ords=correctors,
                quads_ords=quads,
                skew_ords=skews,
                CAVords=cavities,
                nHBPM=len(bpms), nVBPM=len(bpms),
                nHorCOR=len(correctors[0]), nVerCOR=len(correctors[1]),
                quads_tilt_ind=quads,
                inetial_fit_parameters=None,
                orm_measured=measured_orm,
                weights=sigma,
                includeDispersion=drv.INCLUDE_DISPERSION,
                measured_eta_x=measured_x,
                measured_eta_y=measured_y,
                hor_dispersion_weight=drv.H_DISPERSION_WEIGHT,
                ver_dispersion_weight=drv.V_DISPERSION_WEIGHT,
                CMstep=[kicks[0].copy(), kicks[1].copy()],
                rfStep=drv.RF_STEP_HZ,
                Frequency=drv.PETRA_IV_RF_FREQUENCY_HZ,
                quad_individuals=True, skew_individuals=True,
                tilt_individuals=True,
                outlier_rejection=drv.OUTLIER_REJECTION,
                sigma_outlier=drv.SIGMA_OUTLIER,
                apply_normalization=drv.APPLY_NORMALIZATION,
                normalization_mode=drv.NORMALIZATION_MODE,
                svd_threshold=drv.SVD_THRESHOLD,
                show_svd_plot=False,
                nLMIter=drv.N_LM_ITER,
                Starting_Lambda=drv.STARTING_LAMBDA,
                max_lm_lambda=drv.MAX_LM_LAMBDA,
                scaled=drv.SCALED_LM,
                plot_fit_parameters=False,
                auto_correct_delta=drv.AUTO_CORRECT_DELTA,
                fixedpathlength=drv.FIXED_PATH_LENGTH,
                fixedmomentum=drv.FIXED_MOMENTUM,
                quad_jacobian_file=None,
                skew_jacobian_file=None,
                quads_tilt_jacobian_file=None,
                quad_jacobian_calculator="Numerical",
                skew_jacobian_calculator="Numerical",
                force_recompute=True,
                save_jacobians=True,
                constraint_cfg=drv.make_constraint_config(ring, quads, skews),
                calculate_delta_chi2=False,
            )
        finally:
            sys.stdout = original_stdout

    trials = parse_trials(tee.lines)
    warning_records = [{
        "category": item.category.__name__,
        "message": str(item.message),
    } for item in caught]
    orbit_warnings = [item for item in warning_records if
                      "convergence" in item["message"].lower() or
                      "maximum number of iterations" in item["message"].lower()]
    accepted = next((trial for trial in trials if trial["accepted"]), None)
    summary = {
        "seed_file": str(args.seed_file.resolve()),
        "seed_sha256": sha256(args.seed_file),
        "measured_file": str(args.measured.resolve()),
        "measured_sha256": sha256(args.measured),
        "configuration": persisted | {
            "orm_calculator": "Tracking",
            "jacobian_calculator": "Numerical",
            "stage": 1,
            "outer_iterations": 1,
            "lm_inner_iterations": drv.N_LM_ITER,
            "starting_lambda": drv.STARTING_LAMBDA,
            "svd_cut": drv.STAGE1_SVD_CUT,
            "fit_list": fit_list,
        },
        "iteration_metrics": iteration_metrics,
        "trials": trials,
        "lambda_sequence": [trial["lambda"] for trial in trials],
        "nonfinite_trial_orm_count": sum(not trial["finite_orm"] for trial in trials),
        "warnings": warning_records,
        "closed_orbit_warnings": orbit_warnings,
        "step_accepted": accepted is not None,
        "accepted_trial": accepted,
        "returned_lattice_polynomials_finite": all(
            np.all(np.isfinite(getattr(element, attr)))
            for element in result[2]
            for attr in ("PolynomA", "PolynomB")
            if hasattr(element, attr)
        ),
        "full_jacobian": str(
            args.output / "pyloco_stage1/jacobians/full/J_fit_filtered.npy"
        ),
    }
    summary_path = args.output / "validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
