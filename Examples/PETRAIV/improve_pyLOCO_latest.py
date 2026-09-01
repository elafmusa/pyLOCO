#!/usr/bin/env python3
"""
PETRA IV pySC + latest production pyLOCO workflow.

This script is a modernized replacement for the historical
    improve_pyLOCO_weights.py + test_weight.py + pyloco_config.py
combination.

Scientific workflow intentionally preserved:
  * 2 outer machine-correction cycles
  * bipolar, unscaled ORM measurement
  * legacy sign correction on the first 10 ORM columns
  * bipolar ±40 Hz RF-orbit response
  * 100-orbit BPM repeatability measurement
  * LOCO Stage 1:
      4 iterations
      quads + H/V BPM gains + H/V corrector calibration
      dispersion weights = 5 / 5
      SVD user cut = 2500
      coupling removed
  * LOCO Stage 2:
      4 iterations
      skew quads + H/V BPM coupling + H/V corrector coupling
      dispersion weights = 5 / 5
      SVD user cut = 1750
      continues from Stage 1
  * apply reconstructed normal/skew corrections to pySC
  * orbit, tune and chromaticity correction

Important migration changes:
  * uses pyLOCO.pyloco production API (NOT pyLOCO.pyloco_test)
  * no dependency on test_weight.py or local pyloco_config.py
  * passes the pure ORM separately from measured_eta_x/y
    (does NOT manually append the dispersion column)
  * uses explicit PETRA IV RF frequency instead of the current pyLOCO default
  * uses separate output directories for each outer-cycle / LOCO stage
  * supports seed index OR exact seed filename/stem safely
  * leaves old scripts/files untouched

Run, for example:
    python improve_pyLOCO_latest.py 0
    python improve_pyLOCO_latest.py 111
    python improve_pyLOCO_latest.py my_seed.json

For a quick local integration check, set LOCAL_TEST=True below.
"""

from __future__ import annotations

from pathlib import Path
import json
import pickle
import sys
import time
import traceback
import warnings

import numpy as np
import at
from at import get_refpts

from pySC import disable_pySC_rich
from pySC import SimulatedCommissioning
from pySC.tuning.response_measurements import (
    measure_OrbitResponseMatrix,
    measure_RFFrequencyOrbitResponse,
)
from pySC.tuning.averaging import get_average_orbit

from pyLOCO.pyloco import pyloco, get_fit_param_block
from pyLOCO.config import FitInitConfig, ConstraintConfig

# Existing PETRA-IV helper modules. These stay local to Examples/P4_LOCO.
from set_correction import set_correction
from analyze_ring import analyze_ring
from for_script import find_seed_files, save_corrected_lattice


# =============================================================================
# PATHS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_DIR = SCRIPT_DIR / "pre-loco"
LATTICE_FILE = SCRIPT_DIR / "p4_H6BA_v4_5_for_pySC.mat"
CORRECTOR_FILE = SCRIPT_DIR / "selected_correctors_v_4_5_covarage_all.npz"

# Keep this completely separate from the historical output directory.
OUTPUT_DIR = SCRIPT_DIR / "loco_latest_pyloco_test"

FILE_PATTERNS = ["*.json"]


# =============================================================================
# RUN CONTROL
# =============================================================================

# False = preserve the full historical settings below.
# True  = lightweight plumbing/integration test. It still performs real pySC
#         measurements, but reduces LOCO iterations and outer correction cycles.
LOCAL_TEST = False

CORRECTION_CYCLES = 1
STAGE1_NITER = 4
STAGE2_NITER = 4

if LOCAL_TEST:
    CORRECTION_CYCLES = 1
    STAGE1_NITER = 1
    STAGE2_NITER = 1


# =============================================================================
# VALIDATED PETRA IV / HISTORICAL LOCO SETTINGS
# =============================================================================

ALGORITHM = "lm"
N_LM_ITER = 10
STARTING_LAMBDA = 1.0e-3
MAX_LM_LAMBDA = 15.0
SCALED_LM = True

OUTLIER_REJECTION = True
SIGMA_OUTLIER = 10.0
APPLY_NORMALIZATION = True
NORMALIZATION_MODE = "component"

INCLUDE_DISPERSION = True
H_DISPERSION_WEIGHT = 5.0
V_DISPERSION_WEIGHT = 5.0

# Explicit historical PETRA IV values: do not use current pyLOCO defaults.
RF_STEP_HZ = 40.0
PETRA_IV_RF_FREQUENCY_HZ = 499654096.6666667

CORRECTOR_KICK_RAD = 100.0e-6
BPM_AVERAGES = 100

STAGE1_SVD_CUT = 2500
STAGE2_SVD_CUT = 1750
SVD_SELECTION_METHOD = "user_input"
SVD_THRESHOLD = 1.0e-7

FIXED_PATH_LENGTH = True
FIXED_MOMENTUM = False
AUTO_CORRECT_DELTA = True

# Preserve the established numerical/linear workflow for the first migration.
# These can be benchmarked against Analytical only AFTER old-vs-new equivalence
# is established.
RESPONSE_MATRIX_CALCULATOR = "Linear"
STAGE1_RESPONSE_MATRIX_CALCULATOR = "Linear"
STAGE2_RESPONSE_MATRIX_CALCULATOR = "Linear"
QUAD_JACOBIAN_CALCULATOR = "Numerical"
SKEW_JACOBIAN_CALCULATOR = "Numerical"
ANALYTICAL_IMPLEMENTATION = "vectorized"
ANALYTICAL_USE_MP = True
ANALYTICAL_DISPERSION_CALCULATOR = "Linear"
SKEW_ANALYTICAL_DISPERSION_CALCULATOR = "Linear"
SKEW_ANALYTICAL_DISPERSION_WORKER = "rf_only"
FORCE_RECOMPUTE_JACOBIANS = True

# Avoid creating multiple very large Jacobian artifacts during the first
# migration test. Change to True later if you explicitly want them persisted.
SAVE_JACOBIANS = False

# Historical workaround in the original P4 driver.
FLIP_FIRST_N_ORM_COLUMNS = 10


# =============================================================================
# UTILITIES
# =============================================================================

def _json_default(obj):
    """JSON helper for numpy/path values in analysis output."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, default=_json_default)


def benchmark_stage_response_matrix_calculators(
    stage1_calculator: str,
) -> tuple[str, str]:
    """Return the validated Stage-1/Stage-2 ORM pair for a benchmark case."""
    stage1 = str(stage1_calculator).strip()
    stage2 = "Linear" if stage1.lower() == "analytical" else stage1
    return stage1, stage2


def resolve_seed(argument: str) -> tuple[Path, int | None]:
    """
    Resolve either:
      * sorted seed-list index: "111"
      * exact filename: "seed_111.json"
      * exact stem: "seed_111"
    """
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")

    seed_files = find_seed_files(INPUT_DIR, FILE_PATTERNS)
    if not seed_files:
        raise RuntimeError(f"No seed files found in {INPUT_DIR}")

    if argument.isdigit():
        idx = int(argument)
        if idx < 0 or idx >= len(seed_files):
            raise IndexError(
                f"Seed index {idx} out of range: available indices are "
                f"0..{len(seed_files)-1}"
            )
        return Path(seed_files[idx]), idx

    candidate = INPUT_DIR / argument
    if candidate.exists():
        return candidate, None

    matches = [
        Path(p) for p in seed_files
        if Path(p).name == argument or Path(p).stem == argument
    ]
    if len(matches) == 1:
        return matches[0], None
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous seed name {argument!r}: {matches}")

    raise FileNotFoundError(
        f"Could not resolve seed {argument!r} in {INPUT_DIR}"
    )


def current_skew_strength(element) -> float:
    """
    Read the actual skew-quadrupole A1 coefficient used by the modern
    FitInitConfig (PolynomA[1]). Falls back to 0 when unavailable.

    This avoids subtracting element.K for the skew block: the skew fit is
    represented by PolynomA, not the normal-quadrupole K/PolynomB term.
    """
    pa = getattr(element, "PolynomA", None)
    if pa is None:
        return 0.0
    arr = np.asarray(pa, dtype=float).ravel()
    return float(arr[1]) if arr.size > 1 else 0.0


def make_constraint_config(ring, quad_indices, skew_quad_indices):
    """
    Preserve the historical constraint configuration from test_weight.py.
    """
    # Sigma is an uncertainty magnitude. Quadrupole K legitimately carries
    # the focusing/defocusing sign, but that sign must never be transferred to
    # a standard deviation.
    sigma_quads = np.asarray(
        [abs(float(ring[i].K)) * 1.0e-5 for i in quad_indices],
        dtype=float,
    )
    if np.any(sigma_quads <= 0):
        invalid = np.asarray(quad_indices, dtype=int)[sigma_quads <= 0]
        raise ValueError(
            "PETRA IV quadrupole constraints require positive |K| values; "
            f"zero-strength selected quadrupole ordinals: {invalid.tolist()}"
        )
    quad_weights = np.ones(len(quad_indices), dtype=float)

    skew_sigma = np.full(len(skew_quad_indices), 1.0e-40, dtype=float)
    skew_weights = np.zeros(len(skew_quad_indices), dtype=float)

    return ConstraintConfig(
        enable=True,
        quad_sigma=sigma_quads,
        quad_weights=quad_weights,
        skew_sigma=skew_sigma,
        skew_weights=skew_weights,
    )


def save_stage_returns(
    stage_dir: Path,
    *,
    fit_results,
    fit_dict,
    ring_fit,
    fitted_orm,
    c_bpms_after,
    chi2_history,
    delta_chi2_history,
    blocks,
) -> None:
    """
    Persist lightweight returned objects in addition to pyLOCO's own output.
    """
    stage_dir.mkdir(parents=True, exist_ok=True)

    np.save(stage_dir / "fit_results_return.npy",
            np.asarray(fit_results, dtype=object),
            allow_pickle=True)
    np.save(stage_dir / "fitted_orm_return.npy",
            np.asarray(fitted_orm))
    np.save(stage_dir / "C_bpms_after_return.npy",
            np.asarray(c_bpms_after))

    if chi2_history is not None:
        np.save(stage_dir / "chi2_history_return.npy",
                np.asarray(chi2_history))
    if delta_chi2_history is not None:
        np.save(stage_dir / "delta_chi2_history_return.npy",
                np.asarray(delta_chi2_history))

    with (stage_dir / "fit_dict_return.pkl").open("wb") as f:
        pickle.dump(fit_dict, f)

    with (stage_dir / "blocks_return.pkl").open("wb") as f:
        pickle.dump(blocks, f)

    try:
        ring_fit.save(str(stage_dir / "ring_pyloco_return.mat"), mat_key="ring")
    except Exception as exc:
        print(f"[warning] Could not save returned fitted ring: {exc}")


# =============================================================================
# LATEST pyLOCO TWO-STAGE FIT
# =============================================================================

def run_latest_pyloco_two_stage(
    *,
    model_ring,
    CMstep,
    CAVords,
    quad_indices,
    skew_quad_indices,
    used_cor_ords,
    used_bpms_ords,
    measured_orm,
    sigma_w,
    measured_eta_x,
    measured_eta_y,
    output_dir: Path,
    stage1_response_matrix_calculator: str | None = None,
    stage2_response_matrix_calculator: str | None = None,
):
    """
    Reproduce the historical two-stage PETRA IV LOCO workflow using the
    current production pyLOCO.pyloco API.
    """

    # The historical wrapper explicitly used a 4D model for LOCO.
    model_ring = model_ring.deepcopy() if hasattr(model_ring, "deepcopy") else model_ring
    model_ring.disable_6d()

    stage1_orm_calculator = (
        STAGE1_RESPONSE_MATRIX_CALCULATOR
        if stage1_response_matrix_calculator is None
        else stage1_response_matrix_calculator
    )
    stage2_orm_calculator = (
        STAGE2_RESPONSE_MATRIX_CALCULATOR
        if stage2_response_matrix_calculator is None
        else stage2_response_matrix_calculator
    )

    n_hcor = len(used_cor_ords[0])
    n_vcor = len(used_cor_ords[1])
    n_hbpm = n_vbpm = len(used_bpms_ords)

    constraint_cfg = make_constraint_config(
        model_ring, quad_indices, skew_quad_indices
    )

    def capture_iteration_timing(destination):
        def capture(metrics):
            destination.append({
                "iteration": int(metrics["iteration"]),
                "chi2_before": float(metrics["chi2_before"]),
                "chi2_after": float(metrics["chi2_after"]),
                "timings": dict(metrics["timings"]),
            })
        return capture

    common_kwargs = dict(
        used_bpms_ords=used_bpms_ords,
        used_cor_ords=used_cor_ords,
        quads_ords=quad_indices,
        skew_ords=skew_quad_indices,
        CAVords=CAVords,
        nHBPM=n_hbpm,
        nVBPM=n_vbpm,
        nHorCOR=n_hcor,
        nVerCOR=n_vcor,
        quads_tilt_ind=quad_indices,
        inetial_fit_parameters=None,

        # IMPORTANT: measured_orm is the PURE ORM here.
        orm_measured=measured_orm,
        weights=sigma_w,
        includeDispersion=INCLUDE_DISPERSION,
        measured_eta_x=measured_eta_x,
        measured_eta_y=measured_eta_y,
        hor_dispersion_weight=H_DISPERSION_WEIGHT,
        ver_dispersion_weight=V_DISPERSION_WEIGHT,

        CMstep=CMstep,
        rfStep=RF_STEP_HZ,
        Frequency=PETRA_IV_RF_FREQUENCY_HZ,

        quad_individuals=True,
        skew_individuals=True,
        tilt_individuals=True,

        outlier_rejection=OUTLIER_REJECTION,
        sigma_outlier=SIGMA_OUTLIER,
        apply_normalization=APPLY_NORMALIZATION,
        normalization_mode=NORMALIZATION_MODE,

        svd_selection_method=SVD_SELECTION_METHOD,
        svd_threshold=SVD_THRESHOLD,
        show_svd_plot=False,

        nLMIter=N_LM_ITER,
        Starting_Lambda=STARTING_LAMBDA,
        max_lm_lambda=MAX_LM_LAMBDA,
        scaled=SCALED_LM,

        plot_fit_parameters=False,
        auto_correct_delta=AUTO_CORRECT_DELTA,
        fixedpathlength=FIXED_PATH_LENGTH,
        fixedmomentum=FIXED_MOMENTUM,

        quad_jacobian_file=None,
        skew_jacobian_file=None,
        quads_tilt_jacobian_file=None,
        quad_jacobian_calculator=QUAD_JACOBIAN_CALCULATOR,
        skew_jacobian_calculator=SKEW_JACOBIAN_CALCULATOR,
        analytical_implementation=ANALYTICAL_IMPLEMENTATION,
        analytical_use_mp=ANALYTICAL_USE_MP,
        analytical_dispersion_calculator=ANALYTICAL_DISPERSION_CALCULATOR,
        analytical_skew_use_mp=ANALYTICAL_USE_MP,
        skew_analytical_dispersion_calculator=SKEW_ANALYTICAL_DISPERSION_CALCULATOR,
        skew_analytical_dispersion_worker=SKEW_ANALYTICAL_DISPERSION_WORKER,
        force_recompute=FORCE_RECOMPUTE_JACOBIANS,
        save_jacobians=SAVE_JACOBIANS,

        constraint_cfg=constraint_cfg,
        calculate_delta_chi2=False,
    )

    # -------------------------------------------------------------------------
    # Stage 1: normal optics / gains / corrector calibration
    # -------------------------------------------------------------------------
    stage1_dir = output_dir / "stage_1_normal"
    stage1_dir.mkdir(parents=True, exist_ok=True)

    stage1_fit_list = [
        "quads",
        "hbpm_gain",
        "vbpm_gain",
        "hcor_cal",
        "vcor_cal",
    ]
    stage1_fit_cfg = FitInitConfig(
        fit_list=stage1_fit_list,
        CMstep=CMstep,
        rfStep=RF_STEP_HZ,
        individuals=True,
    )

    print("\n" + "=" * 78)
    print("LOCO STAGE 1 — NORMAL OPTICS / BPM GAINS / CORRECTOR CALIBRATION")
    print("=" * 78)
    print(f"Iterations              : {STAGE1_NITER}")
    print(f"SVD user cut            : {STAGE1_SVD_CUT}")
    print(f"Dispersion weights H/V  : {H_DISPERSION_WEIGHT}/{V_DISPERSION_WEIGHT}")
    print(f"ORM calculator          : {stage1_orm_calculator}")
    print(f"Output                   : {stage1_dir}")

    stage1_analytical_timing = []
    stage1_iteration_timing = []

    stage1 = pyloco(
        model_ring,
        algorithm=ALGORITHM,
        nIter=STAGE1_NITER,
        fit_list=stage1_fit_list,
        remove_coupling_=True,
        svd_selection_method=SVD_SELECTION_METHOD,
        cut_=STAGE1_SVD_CUT,
        fit_cfg=stage1_fit_cfg,
        response_matrix_calculator=stage1_orm_calculator,
        analytical_timing_callback=stage1_analytical_timing.append,
        iteration_metrics_callback=capture_iteration_timing(stage1_iteration_timing),
        output_dir=str(stage1_dir),
        **{
            k: v for k, v in common_kwargs.items()
            if k not in {"svd_selection_method"}
        },
    )

    (
        fit_results1,
        fit_dict1,
        ring_stage1,
        fitted_orm1,
        c_bpms_after1,
        chi2_history1,
        delta_chi2_history1,
        blocks1,
    ) = stage1

    save_stage_returns(
        stage1_dir,
        fit_results=fit_results1,
        fit_dict=fit_dict1,
        ring_fit=ring_stage1,
        fitted_orm=fitted_orm1,
        c_bpms_after=c_bpms_after1,
        chi2_history=chi2_history1,
        delta_chi2_history=delta_chi2_history1,
        blocks=blocks1,
    )
    save_json(stage1_dir / "runtime_breakdown.json", {
        "analytical_jacobian_events": stage1_analytical_timing,
        "iterations": stage1_iteration_timing,
    })

    # -------------------------------------------------------------------------
    # Stage 2: skew + coupling, continuing from Stage 1
    # -------------------------------------------------------------------------
    stage2_dir = output_dir / "stage_2_coupling"
    stage2_dir.mkdir(parents=True, exist_ok=True)

    stage2_fit_list = [
        "skew_quads",
        "hbpm_coupling",
        "vbpm_coupling",
        "hcor_coupling",
        "vcor_coupling",
    ]
    stage2_fit_cfg = FitInitConfig(
        fit_list=stage2_fit_list,
        CMstep=CMstep,
        rfStep=RF_STEP_HZ,
        individuals=True,
    )

    print("\n" + "=" * 78)
    print("LOCO STAGE 2 — SKEW / BPM COUPLING / CORRECTOR COUPLING")
    print("=" * 78)
    print(f"Iterations              : {STAGE2_NITER}")
    print(f"SVD user cut            : {STAGE2_SVD_CUT}")
    print(f"Continue from Stage 1   : True")
    print(f"ORM calculator          : {stage2_orm_calculator}")
    print(f"Output                   : {stage2_dir}")

    stage2_analytical_timing = []
    stage2_skew_analytical_timing = []
    stage2_iteration_timing = []

    stage2 = pyloco(
        ring_stage1,
        algorithm=ALGORITHM,
        nIter=STAGE2_NITER,
        fit_list=stage2_fit_list,
        remove_coupling_=False,
        svd_selection_method=SVD_SELECTION_METHOD,
        cut_=STAGE2_SVD_CUT,
        fit_cfg=stage2_fit_cfg,
        response_matrix_calculator=stage2_orm_calculator,
        analytical_timing_callback=stage2_analytical_timing.append,
        skew_analytical_timing_callback=stage2_skew_analytical_timing.append,
        iteration_metrics_callback=capture_iteration_timing(stage2_iteration_timing),
        output_dir=str(stage2_dir),

        continue_from_previous=True,
        previous_fit_results=fit_results1,
        previous_fit_dict=fit_dict1,
        previous_ring=ring_stage1,

        **{
            k: v for k, v in common_kwargs.items()
            if k not in {"svd_selection_method"}
        },
    )

    save_json(stage2_dir / "runtime_breakdown.json", {
        "analytical_jacobian_events": stage2_analytical_timing,
        "skew_analytical_jacobian_events": stage2_skew_analytical_timing,
        "iterations": stage2_iteration_timing,
    })

    (
        fit_results2,
        fit_dict2,
        ring_stage2,
        fitted_orm2,
        c_bpms_after2,
        chi2_history2,
        delta_chi2_history2,
        blocks2,
    ) = stage2

    save_stage_returns(
        stage2_dir,
        fit_results=fit_results2,
        fit_dict=fit_dict2,
        ring_fit=ring_stage2,
        fitted_orm=fitted_orm2,
        c_bpms_after=c_bpms_after2,
        chi2_history=chi2_history2,
        delta_chi2_history=delta_chi2_history2,
        blocks=blocks2,
    )

    return (
        fit_results2,
        fit_dict2,
        ring_stage2,
        fitted_orm2,
        c_bpms_after2,
        {
            "stage1": {
                "fit_results": fit_results1,
                "fit_dict": fit_dict1,
                "ring": ring_stage1,
                "chi2_history": chi2_history1,
                "response_matrix_calculator": stage1_orm_calculator,
            },
            "stage2": {
                "fit_results": fit_results2,
                "fit_dict": fit_dict2,
                "ring": ring_stage2,
                "chi2_history": chi2_history2,
                "response_matrix_calculator": stage2_orm_calculator,
            },
        },
    )


# =============================================================================
# MAIN PETRA IV COMMISSIONING WORKFLOW
# =============================================================================

def main() -> int:
    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            "  python improve_pyLOCO_latest.py <seed_index OR seed_filename OR seed_stem>"
        )
        return 2

    disable_pySC_rich()
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    seed_arg = sys.argv[1]
    seed, seed_index = resolve_seed(seed_arg)

    if not LATTICE_FILE.exists():
        raise FileNotFoundError(f"Missing lattice: {LATTICE_FILE}")
    if not CORRECTOR_FILE.exists():
        raise FileNotFoundError(f"Missing corrector selection: {CORRECTOR_FILE}")

    seed_dir = OUTPUT_DIR / seed.stem
    seed_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("PETRA IV — LATEST PRODUCTION pyLOCO")
    print("=" * 78)
    print(f"Requested seed argument : {seed_arg}")
    print(f"Resolved seed index     : {seed_index}")
    print(f"Resolved seed file      : {seed.resolve()}")
    print(f"Output directory        : {seed_dir.resolve()}")
    print(f"LOCAL_TEST              : {LOCAL_TEST}")
    print(f"Outer correction cycles : {CORRECTION_CYCLES}")
    print(f"Stage 1 / Stage 2 iter  : {STAGE1_NITER} / {STAGE2_NITER}")
    print()

    t0 = time.perf_counter()

    try:
        # ---------------------------------------------------------------------
        # Load pySC machine
        # ---------------------------------------------------------------------
        SC = SimulatedCommissioning.from_json(
            seed,
            lattice_file=str(LATTICE_FILE),
        )

        SC.lattice.ring.enable_6d()
        SC.lattice.design.enable_6d()

        x0, y0 = SC.bpm_system.capture_orbit()
        SC.bpm_system.reference_x = x0
        SC.bpm_system.reference_y = y0

        sd_b3 = [
            control for control in SC.control_arrays["sd"]
            if control.endswith("B3")
        ]
        sf_b3 = [
            control for control in SC.control_arrays["sf"]
            if control.endswith("B3")
        ]
        SC.tuning.chromaticity.controls_1 = sd_b3
        SC.tuning.chromaticity.controls_2 = sf_b3

        used_bpms_ords = np.asarray(SC.bpm_system.indices, dtype=int)

        # ---------------------------------------------------------------------
        # Corrector selection
        # ---------------------------------------------------------------------
        data = np.load(CORRECTOR_FILE)
        hcor_inds = data["hcor_inds"].astype(int).tolist()
        vcor_inds = data["vcor_inds"].astype(int).tolist()

        if not hcor_inds:
            raise RuntimeError("No horizontal correctors loaded.")
        if not vcor_inds:
            raise RuntimeError("No vertical correctors loaded.")

        used_cor_ords = [hcor_inds, vcor_inds]

        CMstep = [
            np.full(len(hcor_inds), CORRECTOR_KICK_RAD, dtype=float),
            np.full(len(vcor_inds), CORRECTOR_KICK_RAD, dtype=float),
        ]

        quad_indices = np.asarray(
            get_refpts(SC.lattice.ring, at.elements.Quadrupole),
            dtype=int,
        )

        magnet_names = (
            SC.magnet_arrays["cysf"]
            + SC.magnet_arrays["cxysf"]
            + SC.magnet_arrays["cxsf"]
            + SC.magnet_arrays["cxysf2"]
        )
        skew_quad_indices = np.asarray(
            sorted(
                SC.magnet_settings.magnets[name].sim_index
                for name in magnet_names
            ),
            dtype=int,
        )

        CAVords = np.asarray(
            get_refpts(SC.lattice.ring, at.elements.RFCavity),
            dtype=int,
        )

        BPM_09_DW = get_refpts(SC.lattice.ring, "BPM_09_DW*")
        BPM_01_DW = get_refpts(SC.lattice.ring, "BPM_01_DW*")
        bpms_id = np.sort(
            np.concatenate([BPM_09_DW, BPM_01_DW])
        ).astype(int)

        print("Machine selection:")
        print(f"  BPMs              : {len(used_bpms_ords)}")
        print(f"  H correctors      : {len(hcor_inds)}")
        print(f"  V correctors      : {len(vcor_inds)}")
        print(f"  normal quads      : {len(quad_indices)}")
        print(f"  skew quads        : {len(skew_quad_indices)}")
        print(f"  RF cavities       : {len(CAVords)}")

        # =====================================================================
        # OUTER MACHINE CORRECTION CYCLES
        # =====================================================================
        for outer_it in range(CORRECTION_CYCLES):
            iter_tag = f"machine_cycle_{outer_it + 1:02d}"
            cycle_dir = seed_dir / iter_tag
            cycle_dir.mkdir(parents=True, exist_ok=True)

            print("\n" + "#" * 78)
            print(
                f"MACHINE CORRECTION CYCLE "
                f"{outer_it + 1}/{CORRECTION_CYCLES}"
            )
            print("#" * 78)

            optics_before = analyze_ring(
                SC,
                elements_indices=used_bpms_ords,
                special_elements=bpms_id,
                useIdealRing=False,
                makeplot=False,
                return_dict=True,
            )
            save_json(
                cycle_dir / "optics_before_loco.json",
                optics_before,
            )

            # -----------------------------------------------------------------
            # Measurements
            # -----------------------------------------------------------------
            print("\n[measurement] ORM ...")
            index_mapping = SC.magnet_settings.index_mapping
            HCORR = [index_mapping[cor] + "/B1L" for cor in hcor_inds]
            VCORR = [index_mapping[cor] + "/A1L" for cor in vcor_inds]

            measured_orm = measure_OrbitResponseMatrix(
                SC,
                HCORR,
                VCORR,
                dkick=CORRECTOR_KICK_RAD,
                normalize=False,
                bipolar=True,
            )
            measured_orm = np.asarray(measured_orm, dtype=float)

            # Preserve historical P4 sign workaround.
            nflip = min(FLIP_FIRST_N_ORM_COLUMNS, measured_orm.shape[1])
            measured_orm[:, :nflip] *= -1.0

            print("[measurement] RF-frequency orbit response ...")
            eta = np.asarray(
                measure_RFFrequencyOrbitResponse(
                    SC,
                    delta_frf=RF_STEP_HZ,
                    normalize=False,
                    bipolar=True,
                ),
                dtype=float,
            ).ravel()

            if eta.size % 2 != 0:
                raise RuntimeError(
                    f"RF response length must be even; received {eta.size}"
                )

            measured_eta_x = eta[: eta.size // 2]
            measured_eta_y = eta[eta.size // 2 :]

            print("[measurement] BPM repeatability/noise ...")
            _, _, x_std, y_std = get_average_orbit(SC, BPM_AVERAGES)
            sigma_w = np.concatenate(
                [
                    np.asarray(x_std, dtype=float).ravel(),
                    np.asarray(y_std, dtype=float).ravel(),
                ]
            )

            expected_rows = 2 * len(used_bpms_ords)
            expected_cols = len(hcor_inds) + len(vcor_inds)
            if measured_orm.shape != (expected_rows, expected_cols):
                raise RuntimeError(
                    "Unexpected ORM shape: "
                    f"{measured_orm.shape}, expected "
                    f"({expected_rows}, {expected_cols})"
                )
            if measured_eta_x.size != len(used_bpms_ords):
                raise RuntimeError(
                    "Horizontal RF-response length mismatch: "
                    f"{measured_eta_x.size} vs {len(used_bpms_ords)}"
                )
            if measured_eta_y.size != len(used_bpms_ords):
                raise RuntimeError(
                    "Vertical RF-response length mismatch: "
                    f"{measured_eta_y.size} vs {len(used_bpms_ords)}"
                )

            # Save exactly what was sent to pyLOCO.
            np.savez_compressed(
                cycle_dir / "measured_inputs.npz",
                measured_orm=measured_orm,
                measured_eta_x=measured_eta_x,
                measured_eta_y=measured_eta_y,
                sigma_w=sigma_w,
                hcor_inds=np.asarray(hcor_inds),
                vcor_inds=np.asarray(vcor_inds),
                used_bpms_ords=used_bpms_ords,
                CMstep_h=CMstep[0],
                CMstep_v=CMstep[1],
                rf_step_hz=RF_STEP_HZ,
                rf_frequency_hz=PETRA_IV_RF_FREQUENCY_HZ,
            )

            # -----------------------------------------------------------------
            # Latest production pyLOCO, two stages
            # -----------------------------------------------------------------
            (
                fit_results,
                fit_dict,
                ring_pyloco,
                fitted_orm,
                c_bpms_after,
                stage_details,
            ) = run_latest_pyloco_two_stage(
                model_ring=SC.lattice.design,
                CMstep=CMstep,
                CAVords=CAVords,
                quad_indices=quad_indices,
                skew_quad_indices=skew_quad_indices,
                used_cor_ords=used_cor_ords,
                used_bpms_ords=used_bpms_ords,
                measured_orm=measured_orm,
                sigma_w=sigma_w,
                measured_eta_x=measured_eta_x,
                measured_eta_y=measured_eta_y,
                output_dir=cycle_dir / "pyloco",
            )

            # -----------------------------------------------------------------
            # Extract and apply normal-quadrupole correction
            # -----------------------------------------------------------------
            quads = np.asarray(
                get_fit_param_block(fit_dict, "quads"),
                dtype=float,
            ).ravel()
            if quads.size != len(quad_indices):
                raise RuntimeError(
                    f"Fitted quad block has {quads.size} values; "
                    f"expected {len(quad_indices)}"
                )

            design_quad_k = np.asarray(
                [SC.lattice.design[i].K for i in quad_indices],
                dtype=float,
            )
            delta_q = quads - design_quad_k

            np.savez_compressed(
                cycle_dir / "quad_corrections.npz",
                delta_q=delta_q,
                quad_indices=quad_indices,
                fitted_quads=quads,
                design_quads=design_quad_k,
            )

            print("\n[correction] Applying reconstructed normal-quadrupole correction")
            print(f"  RMS(delta_q) = {np.sqrt(np.mean(delta_q**2)):.6e}")
            print(f"  max|delta_q| = {np.max(np.abs(delta_q)):.6e}")

            set_correction(
                SC,
                -delta_q,
                quad_indices,
                individuals=True,
                skewness=False,
            )

            SC.lattice.ring.enable_6d()
            SC.lattice.design.enable_6d()

            optics_after_quads = analyze_ring(
                SC,
                elements_indices=used_bpms_ords,
                special_elements=bpms_id,
                useIdealRing=False,
                makeplot=False,
                return_dict=True,
            )
            save_json(
                cycle_dir / "optics_after_quads_loco.json",
                optics_after_quads,
            )

            # -----------------------------------------------------------------
            # Extract and apply skew-quadrupole correction
            # -----------------------------------------------------------------
            skew_quads = np.asarray(
                get_fit_param_block(fit_dict, "skew_quads"),
                dtype=float,
            ).ravel()
            if skew_quads.size != len(skew_quad_indices):
                raise RuntimeError(
                    f"Fitted skew block has {skew_quads.size} values; "
                    f"expected {len(skew_quad_indices)}"
                )

            # Modern pyLOCO fits skew strength using PolynomA[1].
            design_skew = np.asarray(
                [
                    current_skew_strength(SC.lattice.design[i])
                    for i in skew_quad_indices
                ],
                dtype=float,
            )
            delta_skew = skew_quads - design_skew

            np.savez_compressed(
                cycle_dir / "skew_corrections.npz",
                delta_skew=delta_skew,
                skew_indices=skew_quad_indices,
                fitted_skews=skew_quads,
                design_skews=design_skew,
            )

            print("\n[correction] Applying reconstructed skew-quadrupole correction")
            print(f"  RMS(delta_skew) = {np.sqrt(np.mean(delta_skew**2)):.6e}")
            print(f"  max|delta_skew| = {np.max(np.abs(delta_skew)):.6e}")

            set_correction(
                SC,
                -delta_skew,
                skew_quad_indices,
                individuals=True,
                skewness=True,
            )

            SC.lattice.ring.enable_6d()
            SC.lattice.design.enable_6d()

            optics_after_skew = analyze_ring(
                SC,
                elements_indices=used_bpms_ords,
                special_elements=bpms_id,
                useIdealRing=False,
                makeplot=False,
                return_dict=True,
            )
            save_json(
                cycle_dir / "optics_after_skew_loco.json",
                optics_after_skew,
            )

            # -----------------------------------------------------------------
            # Standard commissioning corrections
            # -----------------------------------------------------------------
            print("\n[commissioning] Orbit correction ...")
            SC.tuning.correct_orbit(parameter=20)

            optics_after_orbit = analyze_ring(
                SC,
                elements_indices=used_bpms_ords,
                special_elements=bpms_id,
                useIdealRing=False,
                makeplot=False,
                return_dict=True,
            )
            save_json(
                cycle_dir / "optics_after_orbit.json",
                optics_after_orbit,
            )

            print("[commissioning] Tune correction ...")
            SC.tuning.tune.correct(measurement_method="cheat")

            optics_after_tune = analyze_ring(
                SC,
                elements_indices=used_bpms_ords,
                special_elements=bpms_id,
                useIdealRing=False,
                makeplot=False,
                return_dict=True,
            )
            save_json(
                cycle_dir / "optics_after_tune.json",
                optics_after_tune,
            )

            print("[commissioning] Chromaticity correction ...")
            SC.tuning.chromaticity.correct(gain=0.8)

            optics_after_chroma = analyze_ring(
                SC,
                elements_indices=used_bpms_ords,
                special_elements=bpms_id,
                useIdealRing=False,
                makeplot=False,
                return_dict=True,
            )
            save_json(
                cycle_dir / "optics_after_chroma.json",
                optics_after_chroma,
            )
            save_json(
                cycle_dir / "optics_after_loco_final.json",
                optics_after_chroma,
            )

        # =====================================================================
        # FINAL SAVE
        # =====================================================================
        final_optics = analyze_ring(
            SC,
            elements_indices=used_bpms_ords,
            special_elements=bpms_id,
            useIdealRing=False,
            makeplot=False,
            return_dict=True,
        )
        save_json(seed_dir / "optics_after_loco_final.json", final_optics)

        saved_path = save_corrected_lattice(SC, seed, OUTPUT_DIR)
        SC.to_json(OUTPUT_DIR / f"pySC_after_loco_{seed.stem}.json")

        elapsed = time.perf_counter() - t0
        summary = {
            "seed_argument": seed_arg,
            "seed_index": seed_index,
            "seed_file": str(seed),
            "output_directory": str(seed_dir),
            "local_test": LOCAL_TEST,
            "correction_cycles": CORRECTION_CYCLES,
            "stage1_iterations": STAGE1_NITER,
            "stage2_iterations": STAGE2_NITER,
            "rf_step_hz": RF_STEP_HZ,
            "rf_frequency_hz": PETRA_IV_RF_FREQUENCY_HZ,
            "stage1_response_matrix_calculator": STAGE1_RESPONSE_MATRIX_CALCULATOR,
            "stage2_response_matrix_calculator": STAGE2_RESPONSE_MATRIX_CALCULATOR,
            "quad_jacobian_calculator": QUAD_JACOBIAN_CALCULATOR,
            "skew_jacobian_calculator": SKEW_JACOBIAN_CALCULATOR,
            "analytical_implementation": ANALYTICAL_IMPLEMENTATION,
            "analytical_use_mp": ANALYTICAL_USE_MP,
            "analytical_dispersion_calculator": ANALYTICAL_DISPERSION_CALCULATOR,
            "skew_analytical_dispersion_calculator": SKEW_ANALYTICAL_DISPERSION_CALCULATOR,
            "skew_analytical_dispersion_worker": SKEW_ANALYTICAL_DISPERSION_WORKER,
            "save_jacobians": SAVE_JACOBIANS,
            "saved_corrected_lattice": str(saved_path),
            "elapsed_seconds": elapsed,
            "status": "completed",
        }
        save_json(seed_dir / "run_summary.json", summary)

        print("\n" + "=" * 78)
        print("SEED COMPLETED")
        print("=" * 78)
        print(f"Saved corrected lattice : {Path(saved_path).resolve()}")
        print(f"Elapsed                 : {elapsed:.2f} s")
        print(f"Output                  : {seed_dir.resolve()}")
        return 0

    except Exception as exc:
        tb_text = traceback.format_exc()
        print(f"\n!! Failed on seed {seed.name}: {exc}")
        print(tb_text)
        (seed_dir / "fatal_error.traceback.log").write_text(tb_text)

        save_json(
            seed_dir / "run_summary.json",
            {
                "seed_argument": seed_arg,
                "seed_index": seed_index,
                "seed_file": str(seed),
                "status": "failed",
                "error": str(exc),
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
