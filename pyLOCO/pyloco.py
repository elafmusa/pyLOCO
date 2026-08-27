import numpy as np
from numpy.linalg import svd
import logging
import matplotlib.pyplot as plt
from pathlib import Path
LOGGER = logging.getLogger(__name__)
import time
import json
from .initial_fit import build_initial_fit_parameters
from .set_parameters import set_correction, set_correction_tilt, _get_attr_scalar, _initial_values_for_block, \
    _resolve_attr_for_block_read
import os
import h5py
import multiprocessing as mp
from multiprocessing import shared_memory
from .config import RMConfig, FitInitConfig, get_mcf, fixed_parameters
from .response_matrix import response_matrix
import time
fit_cfg = FitInitConfig()
import warnings

from .analytic_orm_with_normal_quad_errors import analytic_orm_variation_with_normal_quadrupole
from .analytic_orm_with_skew_quad_errors import analytic_orm_variation_with_skew_quadrupole


warnings.filterwarnings("ignore", category=RuntimeWarning)

#SAVE_JACOBIANS = False

# ============================================================================== #
#                                    ORMs                                        #
# ============================================================================== #


def weight_matrix(W, include_dispersion=False,
                  hor_dispersion_weight=1, ver_dispersion_weight=1,
                  nHBPM=None, nVBPM=None, nHorCOR=None, nVerCOR=None):
    """
    Constructs the full weight matrix including orbit and optional dispersion terms.
    Parameters
    ----------
    W : 1D array of BPM stds (nHBPM + nVBPM,)
    SC : SimulatedCommissioning object (not used here but kept for interface consistency)
    include_dispersion : bool, whether to include dispersion weights
    hor_dispersion_weight : float, horizontal dispersion weight
    ver_dispersion_weight : float, vertical dispersion weight
    nHBPM, nVBPM : int, number of horizontal and vertical BPMs
    nHorCOR, nVerCOR : int, number of horizontal and vertical correctors

    Returns
    -------
    W_matrix : 2D array of weights, shape = (nHBPM + nVBPM [+ dispersion], nHorCOR + nVerCOR [+ 1])
    """

    if nHBPM is None or nVBPM is None or nHorCOR is None or nVerCOR is None:
        raise ValueError("Must provide nHBPM, nVBPM, nHorCOR, and nVerCOR.")

    # Orbit weight matrix: repeat std vector across all correctors
    W_matrix = np.outer(W, np.ones(nHorCOR + nVerCOR))  # shape: (nBPMs, nCorrectors)
    W_matrix_chi = np.outer(W, np.ones(nHorCOR + nVerCOR))

    if include_dispersion == True:
        # Split orbit BPM stds into horizontal and vertical parts
        W_H = W[:nHBPM]
        W_V = W[nHBPM:]

        dispersion_std = np.concatenate([
            W_H / hor_dispersion_weight,
            W_V / ver_dispersion_weight
        ]).reshape(-1, 1)  # column vector (nHBPM + nVBPM, 1)

        W_matrix = np.hstack((W_matrix, dispersion_std))  # shape: (nBPMs, nCORs + 1)

        dispersion_std_chi = np.concatenate([
            W_H / 1,
            W_V / 1
        ]).reshape(-1, 1)  # column vector (nHBPM + nVBPM, 1)

        W_matrix_chi = np.hstack((W_matrix_chi, dispersion_std_chi))  # shape: (nBPMs, nCORs + 1)

    W_flat = W_matrix.reshape(-1, 1, order='F')
    W_flat_chi = W_matrix_chi.reshape(-1, 1, order='F')

    return W_flat, W_flat_chi


def remove_coupling(orm1, orm2, W=None, Jacobian=None,
                    nHBPM=None, nVBPM=None, nHorCOR=None, nVerCOR=None,
                    include_dispersion=False, for_chi_squared=False):
    """
    Remove coupling-related rows from ORMs, Jacobian, and weight matrix.

    Parameters
    ----------
    orm1, orm2 : np.ndarray
        ORM-related arrays (e.g., measured and model ORMs), shape: (nBPM * nCOR [+disp], 1)
    W : np.ndarray, shape (nBPM * nCOR [+disp], 1)
        Weight vector or matrix
    Jacobian : np.ndarray, shape (nBPM * nCOR [+disp], nParams)
        Full Jacobian matrix
    nHBPM, nVBPM, nHorCOR, nVerCOR : int
        Number of horizontal/vertical BPMs and correctors
    include_dispersion : bool
        Whether dispersion terms are included

    Returns
    -------
    orm1_filtered : np.ndarray
    orm2_filtered : np.ndarray
    W_filtered : np.ndarray or None
    Jacobian_filtered : np.ndarray or None
    iNoCoupling : np.ndarray of indices kept
    """
    if None in [nHBPM, nVBPM, nHorCOR, nVerCOR]:
        raise ValueError("Must provide all BPM and corrector counts.")

    nBPM = nHBPM + nVBPM
    nCOR = nHorCOR + nVerCOR

    # Build base coupling filter matrix
    CF = np.block([
        [np.ones((nHBPM, nHorCOR)), np.zeros((nHBPM, nVerCOR))],
        [np.zeros((nVBPM, nHorCOR)), np.ones((nVBPM, nVerCOR))]
    ])

    if include_dispersion:
        dispersion_column_chi = np.concatenate([
            2 * np.ones((nHBPM, 1)),
            3 * np.ones((nVBPM, 1))
        ])

        dispersion_column = np.concatenate([
            2 * np.ones((nHBPM, 1)),
            np.zeros((nVBPM, 1))
        ])

        CF_chi = np.hstack((CF, dispersion_column_chi))
        CF_flat_chi = CF_chi.flatten(order='F')
        iNoCoupling_chi = np.where(CF_flat_chi > 0)[0]

        CF = np.hstack((CF, dispersion_column))
        CF_flat = CF.flatten(order='F')
        iNoCoupling = np.where(CF_flat > 0)[0]

    else:
        CF_flat = CF.flatten(order='F')
        iNoCoupling = np.where(CF_flat > 0)[0]
        iNoCoupling_chi = iNoCoupling

    # Apply filtering
    orm1_filtered = orm1[iNoCoupling]
    orm2_filtered = orm2[iNoCoupling]
    W_filtered = W[iNoCoupling] if W is not None else None
    Jacobian_filtered = Jacobian[iNoCoupling, :] if Jacobian is not None else None

    return orm1_filtered, orm2_filtered, W_filtered, Jacobian_filtered, iNoCoupling, iNoCoupling_chi


def build_iNoCoupling(nHBPM, nVBPM, nHorCOR, nVerCOR, includeDispersion):
    nBPM = nHBPM + nVBPM

    CF = np.block([
        [np.ones((nHBPM, nHorCOR)), np.zeros((nHBPM, nVerCOR))],
        [np.zeros((nVBPM, nHorCOR)), np.ones((nVBPM, nVerCOR))]
    ])

    if includeDispersion:
        # fit mask: keep horizontal dispersion only (match MATLAB CF=[...; zeros(VBPM,1)])
        disp_fit = np.concatenate([2 * np.ones((nHBPM, 1)), np.zeros((nVBPM, 1))])
        CF_fit = np.hstack((CF, disp_fit))
        iNoCoupling = np.where(CF_fit.flatten(order="F") > 0)[0]

        # chi mask (your choice): often keep both planes for chi² (or match MATLAB if you prefer)
        disp_chi = np.concatenate([2 * np.ones((nHBPM, 1)), 3 * np.ones((nVBPM, 1))])
        CF_chi = np.hstack((CF, disp_chi))
        iNoCoupling_chi = np.where(CF_chi.flatten(order="F") > 0)[0]
    else:
        iNoCoupling = np.where(CF.flatten(order="F") > 0)[0]
        iNoCoupling_chi = iNoCoupling.copy()

    return iNoCoupling, iNoCoupling_chi, nBPM


def select_equally_spaced_elements(total_elements, num_elements):
    step = len(total_elements) // (num_elements - 1)
    return total_elements[::step]


def remove_bad_bpms(data, bad_bpms, total_bpms, axis=0, input_type="positions"):
    """
    Remove bad BPMs from ORM or Jacobian.

    Parameters
    ----------
    data : np.ndarray
        2D ORM (axis=0) or Jacobian (axis=1).
    bad_bpms : array-like
        Bad BPMs given either as:
          - 'positions': BPM indices in BPM list (0-based, horizontal only)
          - 'indices': measurement indices in full ORM/Jacobian (both planes)
    total_bpms : int
        Number of BPMs in one plane.
    axis : int
        0 for ORM (rows), 1 for Jacobian (columns).
    input_type : str
        "positions" (default) → `bad_bpms` are BPM positions in BPM list
        "indices"   → `bad_bpms` are already measurement indices (e.g. 1605, 1659)

    Returns
    -------
    cleaned : np.ndarray
        Data array with bad BPM measurements removed.
    bad_rows : np.ndarray
        The row/column indices that were removed.
    """
    bad_bpms = np.array(bad_bpms, dtype=int)

    if input_type == "positions":

        bad_rows_h = bad_bpms
        bad_rows_v = bad_bpms + total_bpms
        bad_rows = np.concatenate([bad_rows_h, bad_rows_v])

    elif input_type == "indices":

        positions = np.unique(bad_bpms % total_bpms)
        bad_rows_h = positions
        bad_rows_v = positions + total_bpms
        bad_rows = np.concatenate([bad_rows_h, bad_rows_v])
    else:
        raise ValueError("input_type must be 'positions' or 'indices'")

    cleaned = np.delete(data, bad_rows, axis=axis)
    return cleaned, bad_rows


# ============================================================================== #
#                                    For chi Squared                                      #
# ============================================================================== #


def reduced_outliers_to_coupled(iOutliers_reduced, iNoCoupling, n_total):
    """
    Map outliers from reduced (no-coupling) space back to coupled space
    """
    tmp = np.zeros(n_total, dtype=bool)
    tmp[iNoCoupling[iOutliers_reduced]] = True
    return np.where(tmp)[0]


def split_orm_eta_outliers(iOutliers_coupled,
                           nHBPM, nVBPM, nHorCOR, nVerCOR,
                           includeDispersion):
    nORM = (nHBPM + nVBPM) * (nHorCOR + nVerCOR)

    if not includeDispersion:
        return iOutliers_coupled, np.array([], dtype=int)

    # Python 0-based: ORM part is [0 .. nORM-1]
    is_orm = iOutliers_coupled < nORM
    is_eta = iOutliers_coupled >= nORM

    orm_out = iOutliers_coupled[is_orm]
    eta_out = iOutliers_coupled[is_eta] - nORM  # eta-local indices

    return orm_out, eta_out


def rebuild_chi2_outliers(orm_outliers, eta_outliers,
                          nHBPM, nVBPM, nHorCOR, nVerCOR,
                          includeDispersion):
    nORM = (nHBPM + nVBPM) * (nHorCOR + nVerCOR)

    if not includeDispersion:
        return orm_outliers

    return np.concatenate([orm_outliers, eta_outliers + nORM])


def build_chi2_keep_mask(n_total, chi2_outliers):
    mask = np.ones(n_total, dtype=bool)
    mask[chi2_outliers] = False
    return mask


def compute_chi_squared_(
        Mmeas,
        Mmodel,
        Mstd,
        *,
        nHBPM, nVBPM, nHorCOR, nVerCOR,
        include_dispersion,
        remove_coupling_, iNoCoupling,
        iOutliers,
        n_fit_parameters
):
    """
    Faithful Python port of MATLAB lococalcchi2

    Parameters
    ----------
    Mmeas, Mmodel, Mstd : (N,1) arrays
        FULL coupled vectors (before remove_coupling)
    iOutliers : 1D int array
        Outlier indices in FULL coupled space (0-based)
    n_fit_parameters : ndarray
        Jacobian used for the fit (after remove_coupling & outliers)
    """

    Mmeas = Mmeas.copy()
    Mmodel = Mmodel.copy()
    Mstd = Mstd.copy()

    # Mark outliers as NaN (still COUPLED)
    if iOutliers is not None and len(iOutliers) > 0:
        Mmeas[iOutliers] = np.nan
        Mmodel[iOutliers] = np.nan
        Mstd[iOutliers] = np.nan

    # Remove coupling (MATLAB does this AFTER outliers)
    if remove_coupling_:
        # Mmeas, Mmodel, Mstd, _, _ = remove_coupling(
        #    Mmeas, Mmodel, Mstd, None,
        #    nHBPM, nVBPM, nHorCOR, nVerCOR,
        #    include_dispersion,
        #    for_chi_squared=True
        # )

        Mmeas = Mmeas[iNoCoupling]
        Mmodel = Mmodel[iNoCoupling]
        Mstd = Mstd[iNoCoupling]

    # ---- 4) Drop NaNs
    mask = ~np.isnan(Mmeas).ravel()
    Mmeas = Mmeas[mask]
    Mmodel = Mmodel[mask]
    Mstd = Mstd[mask]

    # ---- 5) Chi²
    residuals = (Mmeas - Mmodel) / Mstd
    chi2 = np.sum(residuals ** 2)

    # ---- 6) Degrees of freedom
    dof = len(Mstd) - n_fit_parameters.shape[1]
    if dof <= 0:
        raise ValueError(f"Invalid DOF: {dof}")
    return chi2 / dof


def compute_delta_chi2(
    ring,
    p_final,
    p_initial,
    *,
    fit_list,
    nHBPM, nVBPM, nHorCOR, nVerCOR,
    quads_ords, quads_tilt_ind, skew_ords,
    quad_individuals,
    skew_individuals,
    tilt_individuals, fit_cfg,
    used_bpms_ords, used_cor_ords,
    CMstep, rfStep,
    HCMCoupling, VCMCoupling,
    hbpm_gain, hbpm_coupling,
    vbpm_coupling, vbpm_gain,
    HCMEnergyShift, VCMEnergyShift,
    orm_measured, weights_flat_chi_,
    includeDispersion,
    iNoCoupling_chi, iOut_coupled,
    J_,
    response_matrix_calculator="Linear",
):
    """
    MATLAB-equivalent Δχ² per fit parameter
    """

    n_params = len(p_final)
    delta_chi2 = np.zeros(n_params)

    # --- Nominal χ² ---
    cfg = RMConfig(
        dkick=CMstep,
        bpm_ords=used_bpms_ords,
        cm_ords=used_cor_ords,
        HCMCoupling=HCMCoupling,
        VCMCoupling=VCMCoupling,
        rfStep=rfStep,
        includeDispersion=includeDispersion,
        calculator=response_matrix_calculator,
    )

    orm_model = response_matrix(ring, config=cfg)
    Cmat = _build_C_matrix(hbpm_gain, hbpm_coupling, vbpm_coupling, vbpm_gain)
    orm_model = Cmat @ orm_model

    chi2_nominal = compute_chi_squared_(
        Mmeas=orm_measured.reshape(-1, 1, order="F"),
        Mmodel=orm_model.reshape(-1, 1, order="F"),
        Mstd=weights_flat_chi_,
        nHBPM=nHBPM, nVBPM=nVBPM,
        nHorCOR=nHorCOR, nVerCOR=nVerCOR,
        include_dispersion=includeDispersion,
        remove_coupling_=True,
        iNoCoupling=iNoCoupling_chi,
        iOutliers=iOut_coupled,
        n_fit_parameters=J_
    )

    # --- Loop over parameters ---
    for j in range(n_params):

        # Copy final parameters
        p_test = p_final.copy()

        # Reset ONE parameter to initial
        p_test[j] = p_initial[j]

        # Build temporary ring
        ring_tmp, cfg2, Cmat2, Hshift2, Vshift2, prop_dict = _prepare_ring_and_rmconfig(
            ring, p_test,
            fit_list=fit_list,
            nHBPM=nHBPM, nVBPM=nVBPM,
            nHorCOR=nHorCOR, nVerCOR=nVerCOR,
            quads_ords=quads_ords,
            quads_tilt_ind=quads_tilt_ind,
            skew_ords=skew_ords,
            quad_individuals=quad_individuals,
            skew_individuals=skew_individuals,
            tilt_individuals=tilt_individuals, 
            fit_cfg=fit_cfg,
            used_bpms_ords=used_bpms_ords,
            used_cor_ords=used_cor_ords,
            CMstep=CMstep,
            rfStep=rfStep,
            HCMCoupling=HCMCoupling,
            VCMCoupling=VCMCoupling,
            hbpm_gain=hbpm_gain,
            hbpm_coupling=hbpm_coupling,
            vbpm_coupling=vbpm_coupling,
            vbpm_gain=vbpm_gain,
            HCMEnergyShift=HCMEnergyShift,
            VCMEnergyShift=VCMEnergyShift,
            includeDispersion=includeDispersion,
            response_matrix_calculator=response_matrix_calculator,
        )

        # Compute ORM
        orm_test = response_matrix(ring_tmp, config=cfg2)
        orm_test = Cmat2 @ orm_test

        # Compute χ²
        chi2_test = compute_chi_squared_(
            Mmeas=orm_measured.reshape(-1, 1, order="F"),
            Mmodel=orm_test.reshape(-1, 1, order="F"),
            Mstd=weights_flat_chi_,
            nHBPM=nHBPM, nVBPM=nVBPM,
            nHorCOR=nHorCOR, nVerCOR=nVerCOR,
            include_dispersion=includeDispersion,
            remove_coupling_=True,
            iNoCoupling=iNoCoupling_chi,
            iOutliers=iOut_coupled,
            n_fit_parameters=J_
        )

        delta_chi2[j] = chi2_test - chi2_nominal

        if j % 50 == 0:
            print(f"Δχ² progress: {j}/{n_params}")

    return delta_chi2, chi2_nominal

def compute_group_delta_chi2(delta_chi2, blocks):
    group_contributions = {}

    for name, sl in blocks.items():
        values = delta_chi2[sl]

        group_contributions[name] = {
            "sum": np.sum(values),
            "abs_sum": np.sum(np.abs(values)),
            "rms": np.sqrt(np.mean(values**2)),
            "max": np.max(np.abs(values)),
        }

    return group_contributions

# ============================================================================== #
#                           Compute Jacobians of fit parameters
# ============================================================================== #


def compute_jacobian(
    ring, C_model, dkick, dk, bpm_indexes, CMords, quads_ind,
    nHorCOR, nVerCOR, nHBPM, nVBPM, C, CAVords,
    skew_ind=None,
    includeDispersion=False,
    delta_coupling=1e-6,
    delta_skew_=1e-3,
    delta_q_tilt=1e-6,
    include_quads=True,
    include_skew=False,
    include_quads_tilt=False,
    include_bpm_gain=False,
    include_cor_kick=False,
    include_cor_coupling=False,
    include_bpm_coupling=False,
    quads_tilt_ind=None,
    include_delta_RF_frequency=False,
    include_HCMEnergyShift=False,
    include_VCMEnergyShift=False,
    rf_step=fixed_parameters.rfstep,
    quad_individuals=False,
    skew_individuals=True,
    tilt_individuals=True,
    auto_correct_delta=True,
    HCMCoupling=None,
    VCMCoupling=None,
    measured_eta_x=None,
    measured_eta_y=None,
    quads_tilt_fit=None,
    Frequency=fixed_parameters.Frequency,
    fit_cfg=None,
    iteration=1,
    quad_jacobian_file=None,
    skew_jacobian_file=None,
    quads_tilt_jacobian_file=None,
    force_recompute=True,
    output_dir='output',
    save_jacobians=False,

    # NEW
    quad_jacobian_calculator="Numerical",
    skew_jacobian_calculator="Numerical",
    analytical_thick_quadrupole=True,
    analytical_thick_steerers=False,
    analytical_verbose=False,
    analytical_use_mp=False,
    analytical_thick_skew=True,
    analytical_skew_thick_steerers=False,
    analytical_skew_verbose=False,
    analytical_skew_use_mp=False,
    response_matrix_calculator="Linear",
    calculator_trace_callback=None,
):
    """
    Master function to compute full LOCO Jacobian including:
    - Quadrupole strengths
    - BPM gains/coupling
    - Corrector gains/coupling
    - etc.
    """
    from pathlib import Path
    output_dir = Path(output_dir)
    calculator_plan = _calculator_execution_plan(
        response_matrix_calculator, quad_jacobian_calculator
    )
    response_matrix_calculator = calculator_plan["response_matrix_calculator"]

    nCOR = nHorCOR + nVerCOR
    C_inv = np.linalg.inv(C)

    if HCMCoupling is None:
        HCMCoupling = np.zeros(nHorCOR, dtype=float)
    else:
        HCMCoupling = np.asarray(HCMCoupling, dtype=float).reshape(-1)
        assert HCMCoupling.size == nHorCOR, "HCMCoupling must have length nHorCOR"

    if VCMCoupling is None:
        VCMCoupling = np.zeros(nVerCOR, dtype=float)
    else:
        VCMCoupling = np.asarray(VCMCoupling, dtype=float).reshape(-1)
        assert VCMCoupling.size == nVerCOR, "VCMCoupling must have length nVerCOR"

    # --- QUADS ---
    J_quad, delta = None, None

    if include_quads:
        method = str(quad_jacobian_calculator).strip().lower()
        quad_elapsed = None
        user_provided = quad_jacobian_file is not None

        quad_dir = output_dir / "jacobians" / "quads"

        # Only create an output directory when we actually intend
        # to save a newly calculated Jacobian.
        if save_jacobians and not user_provided:
            quad_dir.mkdir(parents=True, exist_ok=True)

        J_path = (
            Path(quad_jacobian_file)
            if user_provided
            else quad_dir / (
                f"J_quads_{quad_jacobian_calculator.lower()}_"
                f"iter{iteration}_"
                f"{len(quads_ind)}params_"
                f"{dkick[0][0]}urad_"
                f"{fixed_parameters.rfstep}Hz.h5"
            )
        )
  
        #J_path = quad_jacobian_file if user_provided else f"output/jacobians/quads/J_quads_iter{iteration}_{dkick[0][0]}urad_{fixed_parameters.rfstep}Hz.h5"

        # --- logic ---
        if J_path.exists() and not force_recompute and user_provided and iteration == 1:

            print(f"[Jacobian] Loading user-specified normal-quadrupole Jacobian from {J_path}")
            with h5py.File(J_path, "r") as f:
                J_quad = np.array(f["J_quads"])
                delta = None
        else:
            if J_path.exists() and force_recompute:
                print(f"[Jacobian] File exists, but recomputing as requested (force_recompute=True).")
            elif J_path.exists() and not user_provided:
                print(
                    f"[Jacobian] Ignoring existing auto file; computing new normal-quadrupole Jacobian (iteration {iteration}).")
            else:
                print(f"[Jacobian] Computing normal-quadrupole Jacobian (iteration {iteration})...")
            t = time.perf_counter()
            jacobian_reference_model = C_model
            orm_calculator_used = None

            # ============================================================
            # NUMERICAL NORMAL - QUADRUPOLE JACOBIAN
            # ============================================================

            if method == "numerical":

                _trace_calculator(
                    calculator_trace_callback,
                    "normal_quad_numerical_perturbation_orm",
                    response_matrix_calculator,
                )

                # Reuse the existing central finite-difference implementation
                # with the selected existing ORM implementation.
                orm_calculator = response_matrix_calculator
                # C_model is the main ORM already evaluated with that selected
                # calculator; use it as the finite-difference reference.
                jacobian_model = C_model
                jacobian_reference_model = jacobian_model
                orm_calculator_used = orm_calculator

                J_quad, delta = calculate_quads_jacobian(
                    ring,
                    jacobian_model,
                    dkick,
                    CMords,
                    bpm_indexes,
                    quads_ind,
                    dk,
                    C,
                    quad_individuals,
                    HCMCoupling,
                    VCMCoupling,
                    rf_step,
                    block="quads",
                    auto_correct_delta=auto_correct_delta,
                    fit_cfg=fit_cfg,
                    includeDispersion=includeDispersion,
                    output_dir=output_dir,
                    log_filename="quad_jacobian_logs2.txt",
                    orm_calculator=orm_calculator,
                )


            # ============================================================
            # ANALYTICAL NORMAL-QUADRUPOLE JACOBIAN
            # ============================================================

            elif method == "analytical":

                _trace_calculator(
                    calculator_trace_callback,
                    "normal_quad_analytical_derivative",
                    "Analytical",
                )

                # --------------------------------------------------------
                # The analytical formula currently calculates ONLY
                #
                #       d ORM / dK
                #
                # Therefore, when dispersion is included in C_model,
                # remove the last column before passing C_model to the
                # analytical function.
                # --------------------------------------------------------

                if includeDispersion:

                    expected_cols = nHorCOR + nVerCOR + 1

                    if C_model.shape[1] != expected_cols:
                        raise ValueError(
                            "includeDispersion=True, but C_model does not "
                            "have the expected dispersion column.\n"
                            f"C_model.shape = {C_model.shape}\n"
                            f"Expected columns = {expected_cols}"
                        )

                    C_model_orm = C_model[:, :-1]

                else:

                    C_model_orm = C_model


                # --------------------------------------------------------
                # 1. Analytical ORM derivative
                #
                # Shape:
                #
                #     (n_parameters,
                #      nHBPM+nVBPM,
                #      nHorCOR+nVerCOR)
                #
                # calculate_quads_jacobian_analytical() itself remains
                # completely unchanged.
                # --------------------------------------------------------

                J_quad, delta = calculate_quads_jacobian_analytical(
                    ring=ring,
                    C_model=C_model_orm,
                    dkick=dkick,
                    used_cor_ind=CMords,
                    bpm_indexes=bpm_indexes,
                    quads_ind=quads_ind,
                    C=C,

                    # IMPORTANT:
                    # analytical function receives ORM only
                    includeDispersion=False,

                    analytical_thick_quadrupole=analytical_thick_quadrupole,
                    analytical_thick_steerers=analytical_thick_steerers,
                    analytical_verbose=analytical_verbose,
                    analytical_use_mp=analytical_use_mp,
                )


                # --------------------------------------------------------
                # 2. Add numerical dispersion derivative, if requested
                # --------------------------------------------------------

                if includeDispersion:

                    J_eta, delta_eta = calculate_quads_dispersion_jacobian(
                        ring=ring,
                        C_model=C_model,
                        dkick=dkick,
                        used_cor_ind=CMords,
                        bpm_indexes=bpm_indexes,
                        quads_ind=quads_ind,
                        dk=dk,
                        C=C,
                        individuals=quad_individuals,
                        HCMCoupling=HCMCoupling,
                        VCMCoupling=VCMCoupling,
                        rf_step=rf_step,
                        auto_correct_delta=auto_correct_delta,
                        fit_cfg=fit_cfg,
                        orm_calculator=response_matrix_calculator,
                    )


                    # ----------------------------------------------------
                    # Sanity checks before concatenation
                    # ----------------------------------------------------

                    if J_quad.ndim != 3:
                        raise ValueError(
                            "Analytical quadrupole Jacobian must be 3D; "
                            f"got shape {J_quad.shape}"
                        )

                    if J_eta.ndim != 2:
                        raise ValueError(
                            "Dispersion quadrupole Jacobian must be 2D; "
                            f"got shape {J_eta.shape}"
                        )

                    if J_quad.shape[0] != J_eta.shape[0]:
                        raise ValueError(
                            "Number of fitted quadrupole parameters differs "
                            "between ORM and dispersion Jacobians:\n"
                            f"J_quad.shape = {J_quad.shape}\n"
                            f"J_eta.shape  = {J_eta.shape}"
                        )

                    if J_quad.shape[1] != J_eta.shape[1]:
                        raise ValueError(
                            "Number of BPM rows differs between ORM and "
                            "dispersion Jacobians:\n"
                            f"J_quad.shape = {J_quad.shape}\n"
                            f"J_eta.shape  = {J_eta.shape}"
                        )


                    # ----------------------------------------------------
                    # Append dispersion as LAST response-matrix column
                    #
                    # Before:
                    #
                    #   J_quad:
                    #   (P, nBPM_total, nCOR_total)
                    #
                    # J_eta:
                    #   (P, nBPM_total)
                    #
                    # After:
                    #
                    #   (P, nBPM_total, nCOR_total + 1)
                    #
                    # exactly matching response_matrix(...,
                    # includeDispersion=True)
                    # ----------------------------------------------------

                    J_quad = np.concatenate(
                        (
                            J_quad,
                            J_eta[:, :, np.newaxis],
                        ),
                        axis=2,
                    )


                    # ----------------------------------------------------
                    # The analytical ORM itself has no finite-difference
                    # delta.
                    #
                    # However, the hybrid analytical+dispersion Jacobian
                    # does use finite differences for dispersion.
                    #
                    # Return those steps so the caller can inspect them.
                    # ----------------------------------------------------

                    delta = delta_eta


                    # ----------------------------------------------------
                    # Final shape check
                    # ----------------------------------------------------

                    expected_shape = (
                        len(quads_ind),
                        nHBPM + nVBPM,
                        nHorCOR + nVerCOR + 1,
                    )

                    if J_quad.shape != expected_shape:
                        raise ValueError(
                            "Unexpected hybrid analytical quadrupole "
                            "Jacobian shape.\n"
                            f"Got      : {J_quad.shape}\n"
                            f"Expected : {expected_shape}"
                        )


                    print(
                        "[Analytical Jacobian] Added numerical "
                        "dispersion derivative"
                    )

                    print(
                        "[Analytical Jacobian] Final hybrid shape:",
                        J_quad.shape,
                    )


            # ============================================================
            # UNKNOWN CALCULATOR
            # ============================================================

            else:

                raise ValueError(
                    f"Unknown quad_jacobian_calculator="
                    f"{quad_jacobian_calculator!r}. "
                    "Choose 'Numerical' or 'Analytical'."
                )
            quad_elapsed = time.perf_counter() - t
            print(f"Normal quad Jacobian: {quad_elapsed:.1f} s")

        # Save each freshly computed, iteration-specific Jacobian. Never
        # overwrite a user-supplied file that was loaded for iteration 1.
        if save_jacobians and quad_elapsed is not None:
            with h5py.File(J_path, "w") as f:
                f.create_dataset("J_quads", data=J_quad)
                f.create_dataset("C_model", data=jacobian_reference_model)
                if isinstance(dkick, (list, tuple)):
                    f.create_dataset("correctors_kick_h", data=np.asarray(dkick[0]))
                    f.create_dataset("correctors_kick_v", data=np.asarray(dkick[1]))
                else:
                    f.create_dataset("correctors_dkick", data=np.asarray(dkick))
                f.attrs.update({
                "iteration": iteration,
                "jacobian_calculator": method,
                "orm_calculator": (
                    "Tracking" if str(orm_calculator_used).strip().lower() == "numerical"
                    else (orm_calculator_used or "Analytical derivative")
                ),
                "orm_calculator_backend": orm_calculator_used or "Analytical derivative",
                "nHBPM": nHBPM,
                "nVBPM": nVBPM,
                "nHorCOR": nHorCOR,
                "nVerCOR": nVerCOR,
                "includeDispersion": includeDispersion,
                "HCMCoupling": json.dumps(np.asarray(HCMCoupling).tolist()),
                "VCMCoupling": json.dumps(np.asarray(VCMCoupling).tolist()),
                "date": time.ctime(),
                "computation_seconds": quad_elapsed,

                # analytical Jacobian settings
                "analytical_thick_quadrupole": analytical_thick_quadrupole,
                "analytical_thick_steerers": analytical_thick_steerers,
                "analytical_verbose": analytical_verbose,
                "analytical_use_mp": analytical_use_mp,
            })

            print(f"[Jacobian] Saved normal-quadrupole Jacobian to {J_path}")

    # --- SKEW QUADS ---
    J_skew, delta_skew = None, None

    if include_skew:
        skew_elapsed = None
        user_provided = skew_jacobian_file is not None

        skew_dir = output_dir / "jacobians" / "skew"

        # Only create the output directory when a newly calculated
        # skew Jacobian will actually be saved.
        if save_jacobians and not user_provided:
            skew_dir.mkdir(parents=True, exist_ok=True)

        J_path_skew = (
            Path(skew_jacobian_file)
            if user_provided
            else skew_dir / (
                f"J_skew_{skew_jacobian_calculator.lower()}_"
                f"iter{iteration}_"
                f"{len(skew_ind)}params_"
                f"{dkick[0][0]}urad_"
                f"{fixed_parameters.rfstep}Hz.h5"
            )
        )


        # --- logic ---
        if J_path_skew.exists() and not force_recompute and user_provided and iteration == 1:
            print(f"[Jacobian] Loading user-specified skew-quadrupole Jacobian from {J_path_skew}")
            with h5py.File(J_path_skew, "r") as f:
                J_skew = np.array(f["J_skew"])
                delta_skew = None
        else:
            if J_path_skew.exists() and force_recompute:
                print(f"[Jacobian] File exists, but recomputing as requested (force_recompute=True).")
            elif J_path_skew.exists() and not user_provided:
                print(
                    f"[Jacobian] Ignoring existing auto file; computing new skew-quadrupole Jacobian (iteration {iteration}).")
            else:
                print(f"[Jacobian] Computing skew-quadrupole Jacobian (iteration {iteration})...")
            t = time.perf_counter()
            skew_method = str(skew_jacobian_calculator).strip().lower()
            if skew_method == "numerical":
                J_skew, delta_skew = calculate_quads_jacobian(
                    ring, C_model, dkick, CMords, bpm_indexes, skew_ind, delta_skew_, C,
                    skew_individuals, HCMCoupling, VCMCoupling, rf_step, block="skew_quads",
                    auto_correct_delta=auto_correct_delta,
                    fit_cfg=fit_cfg, includeDispersion=includeDispersion, output_dir=output_dir,
                    log_filename="skew_jacobian_logs.txt",
                    orm_calculator=response_matrix_calculator,
                )
            elif skew_method == "analytical":
                C_model_orm = C_model[:, :-1] if includeDispersion else C_model
                J_skew, delta_skew = calculate_skew_jacobian_analytical(
                    ring=ring,
                    C_model=C_model_orm,
                    dkick=dkick,
                    used_cor_ind=CMords,
                    bpm_indexes=bpm_indexes,
                    skew_ind=skew_ind,
                    C=C,
                    fit_cfg=fit_cfg,
                    analytical_thick_skew=analytical_thick_skew,
                    analytical_thick_steerers=analytical_skew_thick_steerers,
                    analytical_verbose=analytical_skew_verbose,
                    analytical_use_mp=analytical_skew_use_mp,
                )
                if includeDispersion:
                    # Reuse the exact numerical skew perturbation and central-
                    # difference implementation, retaining only d(eta)/dKs.
                    J_skew_numerical, delta_skew = calculate_quads_jacobian(
                        ring, C_model, dkick, CMords, bpm_indexes, skew_ind,
                        delta_skew_, C, skew_individuals, HCMCoupling,
                        VCMCoupling, rf_step, block="skew_quads",
                        auto_correct_delta=auto_correct_delta, fit_cfg=fit_cfg,
                        includeDispersion=True, output_dir=output_dir,
                        log_filename="skew_dispersion_jacobian_logs.txt",
                        orm_calculator=response_matrix_calculator,
                    )
                    J_skew = np.concatenate(
                        (J_skew, J_skew_numerical[:, :, -1, np.newaxis]), axis=2
                    )
                    print("[Analytical skew Jacobian] Added numerical dispersion derivative")
            else:
                raise ValueError(
                    f"Unknown skew_jacobian_calculator={skew_jacobian_calculator!r}. "
                    "Choose 'Numerical' or 'Analytical'."
                )
            skew_elapsed = time.perf_counter() - t
            print(f"Skew quad Jacobian: {skew_elapsed:.1f} s")

            if save_jacobians:
                with h5py.File(J_path_skew, "w") as f:
                    f.create_dataset("J_skew", data=J_skew)
                    f.create_dataset("C_model", data=C_model)

                    if isinstance(dkick, (list, tuple)):
                        f.create_dataset(
                            "correctors_kick_h",
                            data=np.asarray(dkick[0])
                        )
                        f.create_dataset(
                            "correctors_kick_v",
                            data=np.asarray(dkick[1])
                        )
                    else:
                        f.create_dataset(
                            "correctors_dkick",
                            data=np.asarray(dkick)
                        )

                    f.attrs["iteration"] = iteration
                    f.attrs["nHBPM"] = nHBPM
                    f.attrs["nVBPM"] = nVBPM
                    f.attrs["nHorCOR"] = nHorCOR
                    f.attrs["nVerCOR"] = nVerCOR
                    f.attrs["includeDispersion"] = includeDispersion
                    f.attrs["calculator"] = str(skew_jacobian_calculator)
                    f.attrs["analytical_thick_skew"] = analytical_thick_skew
                    f.attrs["analytical_thick_steerers"] = analytical_skew_thick_steerers
                    f.attrs["analytical_use_mp"] = analytical_skew_use_mp
                    f.attrs["HCMCoupling"] = json.dumps(
                        np.asarray(HCMCoupling).tolist()
                    )
                    f.attrs["VCMCoupling"] = json.dumps(
                        np.asarray(VCMCoupling).tolist()
                    )
                    f.attrs["date"] = time.ctime()
                    f.attrs["computation_seconds"] = skew_elapsed

                print(
                    f"[Jacobian] Saved skew-quadrupole Jacobian to {J_path_skew}"
                )

    # --- QUAD TILT ---
    J_quad_tilt, delta_quads_tilt = None, None

    if include_quads_tilt:
        user_provided = tilt_jacobian_file is not None

        tilt_dir = output_dir / "jacobians" / "tilt"

        # Only create the directory when a newly calculated
        # tilt Jacobian will actually be saved.
        if save_jacobians and not user_provided:
            tilt_dir.mkdir(parents=True, exist_ok=True)

        J_path_tilt = (
            Path(tilt_jacobian_file)
            if user_provided
            else tilt_dir / (
                f"J_tilt_iter{iteration}_"
                f"{dkick[0][0]}urad_"
                f"{fixed_parameters.rfstep}Hz.h5"
            )
        )

        # --- logic ---
        if J_path_tilt.exists() and not force_recompute and user_provided and iteration == 1:
            print(f"[Jacobian] Loading user-specified quadrupole-tilt Jacobian from {J_path_tilt}")
            with h5py.File(J_path_tilt, "r") as f:
                J_quad_tilt = np.array(f["J_quads_tilt"])
                delta_quads_tilt = None
        else:
            if J_path_tilt.exists() and force_recompute:
                print(f"[Jacobian] File exists, but recomputing as requested (force_recompute=True).")
            elif J_path_tilt.exists() and not user_provided:
                print(
                    f"[Jacobian] Ignoring existing auto file; computing new quadrupole-tilt Jacobian (iteration {iteration}).")
            else:
                print(f"[Jacobian] Computing quadrupole-tilt Jacobian (iteration {iteration})...")
            t = time.perf_counter()
            J_quad_tilt, delta_quads_tilt = calculate_quads_tilt_jacobian(
                ring, C_model, dkick, CMords, bpm_indexes, quads_tilt_ind, delta_q_tilt, C, tilt_individuals,
                HCMCoupling, VCMCoupling, rf_step, auto_correct_delta=auto_correct_delta,
                includeDispersion=includeDispersion, output_dir=output_dir,
                log_filename="tilt_quad_jacobian_logs.txt", quads_tilt_fit=quads_tilt_fit, fit_cfg=fit_cfg
            )
            print(f"Quad tilt Jacobian: {time.perf_counter()-t:.1f} s")

        # --- Save the computed Jacobian ---
        if save_jacobians:
            with h5py.File(J_path_tilt, "w") as f:
                f.create_dataset(
                    "J_quads_tilt",
                    data=J_quad_tilt
                )
                f.create_dataset(
                    "C_model",
                    data=C_model
                )

                if isinstance(dkick, (list, tuple)):
                    f.create_dataset(
                        "correctors_kick_h",
                        data=np.asarray(dkick[0])
                    )
                    f.create_dataset(
                        "correctors_kick_v",
                        data=np.asarray(dkick[1])
                    )
                else:
                    f.create_dataset(
                        "correctors_dkick",
                        data=np.asarray(dkick)
                    )

                f.attrs["iteration"] = iteration
                f.attrs["nHBPM"] = nHBPM
                f.attrs["nVBPM"] = nVBPM
                f.attrs["nHorCOR"] = nHorCOR
                f.attrs["nVerCOR"] = nVerCOR
                f.attrs["includeDispersion"] = includeDispersion
                f.attrs["HCMCoupling"] = json.dumps(
                    np.asarray(HCMCoupling).tolist()
                )
                f.attrs["VCMCoupling"] = json.dumps(
                    np.asarray(VCMCoupling).tolist()
                )
                f.attrs["date"] = time.ctime()

            print(
                f"[Jacobian] Saved quadrupole-tilt Jacobian to {J_path_tilt}"
            )

    J_bpm_gain = calculate_bpm_gain_jacobian(
        C_inv @ C_model, nHBPM, nVBPM, nHorCOR, nVerCOR, includeDispersion, include_bpm_coupling
    ) if include_bpm_gain == True else None

    if include_bpm_gain == False and include_bpm_coupling == True:
        J_bpm_gain = calculate_bpm_coupling_jacobian(
            C_inv @ C_model, nHBPM, nVBPM, includeDispersion
        )

    J_cor_gain = calculate_corrector_kick_jacobian(
        C_model, dkick, nHorCOR, nVerCOR, includeDispersion
    ) if include_cor_kick == True else None

    J_cor_coupling = calculate_corrector_coupling_jacobian(ring,
                                                           bpm_indexes,
                                                           CMords, C_model, dkick, nHBPM, nVBPM, nHorCOR, nVerCOR,
                                                           includeDispersion, C, HCMCoupling, VCMCoupling, rf_step,
                                                           delta_coupling,
                                                           ) if include_cor_coupling == True else None

    J_delta_RF_frequency = calculate_delta_RF_frequency_jacobian(C_model, nHBPM, nVBPM, nHorCOR, nVerCOR, rf_step
                                                                 ) if include_delta_RF_frequency == True else None

    J_HCMEnergyShift = calculate_HCMEnergyShift_jacobian(ring, C_model, nHBPM, nVBPM, nHorCOR, nVerCOR, rf_step,
                                                         measured_eta_x, measured_eta_y, Frequency
                                                         ) if include_HCMEnergyShift == True else None

    J_VCMEnergyShift = calculate_VCMEnergyShift_jacobian(ring, C_model, nHBPM, nVBPM, nHorCOR, nVerCOR, rf_step,
                                                         measured_eta_x, measured_eta_y, Frequency
                                                         ) if include_VCMEnergyShift == True else None

    return full_jacobian_(J_quad=J_quad, J_quad_tilt=J_quad_tilt, J_skew=J_skew, J_bpm=J_bpm_gain, J_cor=J_cor_gain,
                          J_cor_coupling=J_cor_coupling, J_delta_RF_frequency=J_delta_RF_frequency,
                          J_HCMEnergyShift=J_HCMEnergyShift,
                          J_VCMEnergyShift=J_VCMEnergyShift), delta, delta_skew, delta_quads_tilt


def full_jacobian_(
        J_quad=None,
        J_quad_tilt=None,
        J_skew=None,
        J_bpm=None,
        J_cor=None,
        J_cor_coupling=None,
        J_delta_RF_frequency=None,
        J_HCMEnergyShift=None,
        J_VCMEnergyShift=None,
        *,
        order=None,  # e.g. ("J_bpm","J_cor","J_cor_coupling", ... )
        allow_2d=True,  # auto-upgrade 2D (R,C) -> (1,R,C)
        strict=True  # if True, raise when (R,C) mismatch
):
    """
    Vertically concatenate Jacobian components (param-first layout).
    Each component should be shaped (P, R, C). If allow_2d=True, a (R, C)
    will be upgraded to (1, R, C).

    Parameters
    ----------
    order : tuple/list of str, optional
        Names of components in the exact order to append. Valid names:
        "J_bpm","J_cor","J_cor_coupling","J_HCMEnergyShift","J_VCMEnergyShift",
        "J_delta_RF_frequency","J_quad","J_skew","J_quad_tilt".
        If None, a sensible default order is used.

    Returns
    -------
    ndarray
        Concatenated array with shape (P_total, R, C). If no components are
        provided, returns an empty (0,0,0) array.
    """

    # Map string keys to the passed arrays
    pool = {
        "J_bpm": J_bpm,
        "J_cor": J_cor,
        "J_cor_coupling": J_cor_coupling,
        "J_HCMEnergyShift": J_HCMEnergyShift,
        "J_VCMEnergyShift": J_VCMEnergyShift,
        "J_delta_RF_frequency": J_delta_RF_frequency,
        "J_quad": J_quad,
        "J_skew": J_skew,
        "J_quad_tilt": J_quad_tilt,
    }

    # Default order
    if order is None:
        order = (
            "J_bpm",
            "J_cor",
            "J_cor_coupling",
            "J_HCMEnergyShift",
            "J_VCMEnergyShift",
            "J_delta_RF_frequency",
            "J_quad",
            "J_skew",
            "J_quad_tilt",
        )

    mats = []
    for key in order:
        if key not in pool:
            raise KeyError(f"Unknown Jacobian key in order: '{key}'")
        arr = pool[key]
        if arr is None:
            continue

        arr = np.asarray(arr)
        if arr.ndim == 2 and allow_2d:
            arr = arr[None, ...]
        elif arr.ndim != 3:
            raise ValueError(f"{key} must be 2D or 3D, got shape {arr.shape}")

        mats.append((key, arr))

    if not mats:
        return np.empty((0, 0, 0), dtype=float)

    _, first = mats[0]
    _, R, C = first.shape
    for key, arr in mats[1:]:
        if arr.shape[1] != R or arr.shape[2] != C:
            msg = (f"Incompatible shapes: '{mats[0][0]}' has (R,C)=({R},{C}) "
                   f"but '{key}' has (R,C)=({arr.shape[1]},{arr.shape[2]})")
            if strict:
                raise ValueError(msg)
            else:
                raise ValueError(msg)

    # Concatenate along parameter axis (P)
    return np.concatenate([arr for _, arr in mats], axis=0)


def calculate_quads_jacobian(
        ring, C_model, dkick, used_cor_ind, bpm_indexes, quads_ind, dk, C,
        individuals, HCMCoupling, VCMCoupling, rf_step, block,
        auto_correct_delta=True,
        fit_cfg=None, output_dir="output",
        log_filename="quad_jacobian_logs.txt", processes=None, includeDispersion=False,
        orm_calculator="Linear",
):
    from pathlib import Path

    output_dir = Path(output_dir)
    # Shared matrices (read-only)
    shm_C = shared_memory.SharedMemory(create=True, size=C.nbytes)
    C_sh = np.ndarray(C.shape, dtype=C.dtype, buffer=shm_C.buf);
    C_sh[:] = C
    shm_Cm = shared_memory.SharedMemory(create=True, size=C_model.nbytes)
    Cmodel_sh = np.ndarray(C_model.shape, dtype=C_model.dtype, buffer=shm_Cm.buf);
    Cmodel_sh[:] = C_model

    all_logs = []
    ctx = mp.get_context("spawn")

    try:
        quad_args = []
        fit_cfg_dict = fit_cfg.__dict__.copy()
        for quad_index in quads_ind:
            quad_args.append((
                quad_index, ring, dkick, used_cor_ind, bpm_indexes, dk,
                individuals, HCMCoupling, VCMCoupling, rf_step,
                auto_correct_delta,
                block, fit_cfg_dict, includeDispersion, orm_calculator
            ))

        with ctx.Pool(
                processes=processes,
                initializer=_init_shared,
                initargs=(shm_C.name, C.shape, C.dtype.str,
                          shm_Cm.name, C_model.shape, C_model.dtype.str),
                maxtasksperchild=64,
        ) as pool:
            results = pool.starmap(generating_quads_response_matrices, quad_args, chunksize=1)

        if results:
            J_blocks, deltas, logs_lists = zip(*results)
            for _logs in logs_lists:
                if _logs:
                    all_logs.extend(_logs)
            J_blocks = [np.asarray(blk) for blk in J_blocks]
            J_quad = np.stack(J_blocks, axis=0)  # (P, rows, cols)
            delta_vec = np.concatenate([np.atleast_1d(d) for d in deltas])
        else:
            J_quad = np.empty((0, C.shape[0], C.shape[1]))
            delta_vec = np.empty((0,))

        if all_logs:
            try:
                log_dir = output_dir / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)

                log_path = log_dir / log_filename

                with open(log_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(all_logs) + "\n")

                print(f"[calculate_quads_jacobian] Logs saved to '{log_path.resolve()}'")

            except Exception as e:
                print(f"[calculate_quads_jacobian] Could not write logs: {e}")

        return J_quad, delta_vec

    finally:
        for shm in (shm_C, shm_Cm):
            try:
                shm.close(); shm.unlink()
            except Exception:
                pass

def calculate_quads_dispersion_jacobian(
    ring,
    C_model,
    dkick,
    used_cor_ind,
    bpm_indexes,
    quads_ind,
    dk,
    C,
    individuals,
    HCMCoupling,
    VCMCoupling,
    rf_step,
    auto_correct_delta=True,
    fit_cfg=None,
    orm_calculator="Linear",
):
    """
    Calculate only the dispersion-column derivative with respect
    to normal-quadrupole fit parameters.

    This function is intended to complement the analytical ORM
    quadrupole Jacobian:

        analytical:
            d(ORM) / dK

        this function:
            d(eta) / dK

    The final hybrid Jacobian is therefore

        J = [ d(ORM)/dK | d(eta)/dK ]

    Dispersion is obtained through the existing response_matrix()
    implementation with includeDispersion=True.

    The quadrupole derivative is evaluated with a CENTRAL finite
    difference:

        d(eta)/dK =
            [eta(K + dK) - eta(K - dK)] / (2*dK)

    For family fitting, all physical quadrupoles belonging to the
    family are changed simultaneously by the same dK.

    Parameters
    ----------
    ring
        AT lattice.

    C_model : ndarray
        Nominal full response matrix INCLUDING the dispersion
        column. Expected shape:

            (nBPM_total, nCOR_total + 1)

    dkick
        Corrector kick values used by response_matrix().

    used_cor_ind
        [horizontal_corrector_indices, vertical_corrector_indices]

    bpm_indexes
        BPM lattice indices.

    quads_ind
        Fitted quadrupole parameters.

        individuals=True:
            [q1, q2, q3, ...]

        individuals=False:
            [[QF family], [QD family], ...]

    dk
        Quadrupole finite-difference step. If None, the same
        automatic step-selection strategy used by the numerical
        quadrupole Jacobian is applied.

    C : ndarray
        BPM calibration/coupling matrix.

    individuals : bool
        True for individual quadrupoles, False for families.

    HCMCoupling, VCMCoupling
        Corrector coupling arrays.

    rf_step
        RF-frequency step used by the dispersion calculation.

    auto_correct_delta : bool
        Automatically adapt dK using the ORM change.

    fit_cfg
        FitInitConfig defining the quadrupole attribute.

    Returns
    -------
    J_eta : ndarray
        Shape:

            (n_parameters, nBPM_total)

        Each row contains d(eta)/dK for one quadrupole fit
        parameter.

    delta_vec : ndarray
        Final numerical dK used for each fit parameter.
    """

    # ============================================================
    # 0. Basic checks
    # ============================================================

    if fit_cfg is None:
        fit_cfg = FitInitConfig()

    bpm_indexes = np.asarray(
        bpm_indexes,
        dtype=int,
    )

    hcor = np.asarray(
        used_cor_ind[0],
        dtype=int,
    )

    vcor = np.asarray(
        used_cor_ind[1],
        dtype=int,
    )

    n_bpm_total = C.shape[0]

    n_cor_total = (
        len(hcor)
        +
        len(vcor)
    )

    expected_shape = (
        n_bpm_total,
        n_cor_total + 1,
    )

    if C_model.shape != expected_shape:
        raise ValueError(
            "calculate_quads_dispersion_jacobian expects "
            "C_model to contain the dispersion column.\n"
            f"Got C_model.shape = {C_model.shape}\n"
            f"Expected          = {expected_shape}"
        )

    # ============================================================
    # 1. Resolve quadrupole fitted attribute
    #
    # Same mechanism used by generating_quads_response_matrices()
    # ============================================================

    attr_name, attr_idx = _resolve_attr_for_block_read(
        "quads",
        fit_cfg,
    )

    # ============================================================
    # 2. Response-matrix configuration
    #
    # IMPORTANT:
    # includeDispersion=True because we need the final column.
    # ============================================================

    cfg = RMConfig(
        dkick=dkick,
        bpm_ords=bpm_indexes,
        cm_ords=used_cor_ind,
        HCMCoupling=HCMCoupling,
        VCMCoupling=VCMCoupling,
        includeDispersion=True,
        rfStep=rf_step,
        calculator=orm_calculator,
    )

    # ============================================================
    # 3. Storage
    # ============================================================

    J_eta = np.zeros(
        (
            len(quads_ind),
            n_bpm_total,
        ),
        dtype=float,
    )

    delta_used = []

    # Same step-selection targets as the existing numerical
    # quadrupole Jacobian.
    RMSGoal = 1e-6
    RMSTol = 10.0

    # ============================================================
    # 4. Loop over fitted quadrupole parameters
    # ============================================================

    for p, quad_parameter in enumerate(quads_ind):

        # --------------------------------------------------------
        # Build physical quadrupole group
        # --------------------------------------------------------

        group = (
            [int(quad_parameter)]
            if np.isscalar(quad_parameter)
            else [
                int(q)
                for q in quad_parameter
            ]
        )

        # --------------------------------------------------------
        # Nominal strengths of physical magnets
        # --------------------------------------------------------

        k0_each = np.asarray(
            [
                _get_attr_scalar(
                    ring[q],
                    attr_name,
                    attr_idx,
                )
                for q in group
            ],
            dtype=float,
        )

        # ========================================================
        # 5. Individual versus family parameter
        # ========================================================

        if individuals:

            # Normally group contains one physical quadrupole.
            correction_indices = group

            nominal_values = (
                k0_each.copy()
            )

            if dk is None:

                delta_local = (
                    1e-3
                    *
                    nominal_values
                )

                delta_local[
                    delta_local == 0.0
                ] = 1e-3

            else:

                dk_array = np.atleast_1d(
                    dk
                ).astype(float)

                # If dk was supplied as one scalar, use it.
                # If a vector was supplied, use parameter p.
                if dk_array.size == 1:

                    delta_local = np.full(
                        len(group),
                        float(dk_array[0]),
                        dtype=float,
                    )

                elif dk_array.size == len(quads_ind):

                    delta_local = np.full(
                        len(group),
                        float(dk_array[p]),
                        dtype=float,
                    )

                else:

                    raise ValueError(
                        "Unexpected dk shape for individual "
                        "dispersion Jacobian: "
                        f"{dk_array.shape}"
                    )

        else:

            # ----------------------------------------------------
            # FAMILY PARAMETER
            #
            # One fitted value changes all physical magnets in
            # the family simultaneously.
            # ----------------------------------------------------

            correction_indices = [
                group
            ]

            # Family fitting assumes one common nominal strength.
            if not np.allclose(
                k0_each,
                k0_each[0],
                rtol=1e-10,
                atol=1e-14,
            ):
                raise ValueError(
                    f"Family {group} contains different "
                    f"nominal strengths:\n{k0_each}"
                )

            nominal_values = np.asarray(
                [
                    k0_each[0]
                ],
                dtype=float,
            )

            if dk is None:

                delta_value = (
                    1e-3
                    *
                    nominal_values[0]
                )

                if delta_value == 0.0:
                    delta_value = 1e-3

            else:

                dk_array = np.atleast_1d(
                    dk
                ).astype(float)

                if dk_array.size == 1:

                    delta_value = float(
                        dk_array[0]
                    )

                elif dk_array.size == len(quads_ind):

                    delta_value = float(
                        dk_array[p]
                    )

                else:

                    raise ValueError(
                        "Unexpected dk shape for family "
                        "dispersion Jacobian: "
                        f"{dk_array.shape}"
                    )

            delta_local = np.asarray(
                [
                    delta_value
                ],
                dtype=float,
            )

        # ========================================================
        # Everything below temporarily modifies the lattice.
        #
        # Always restore it, even if something fails.
        # ========================================================

        try:

            # ====================================================
            # 6. Select dK
            #
            # Match the numerical quadrupole Jacobian:
            #
            # step selection is based ONLY on the ORM columns,
            # not on dispersion.
            # ====================================================

            while True:

                plus_test_values = (
                    nominal_values
                    +
                    delta_local
                )

                set_correction(
                    ring,
                    plus_test_values,
                    correction_indices,
                    individuals=individuals,
                    block="quads",
                    config=fit_cfg,
                )

                C_plus_step_test = response_matrix(
                    ring,
                    config=cfg,
                )

                # Apply BPM calibration/coupling exactly as in the
                # numerical quadrupole Jacobian.
                C_plus_step_test = (
                    C
                    @
                    C_plus_step_test
                )

                # Restore immediately after test evaluation.
                set_correction(
                    ring,
                    nominal_values,
                    correction_indices,
                    individuals=individuals,
                    block="quads",
                    config=fit_cfg,
                )

                # ------------------------------------------------
                # IMPORTANT:
                #
                # Exclude the final dispersion column from the
                # automatic dK selection.
                #
                # This reproduces the existing numerical worker.
                # ------------------------------------------------

                difference = (
                    C_plus_step_test[:, :-1]
                    -
                    C_model[:, :-1]
                ).ravel(
                    order="F"
                )

                RMSDelta = float(
                    np.sqrt(
                        np.sum(
                            difference**2
                        )
                        /
                        max(
                            1,
                            difference.size,
                        )
                    )
                )

                if (
                    not np.isfinite(RMSDelta)
                    or
                    RMSDelta == 0.0
                ):
                    raise ValueError(
                        "Invalid RMS change while selecting dK "
                        f"for quadrupole group {group}: "
                        f"{RMSDelta}"
                    )

                # ------------------------------------------------
                # Fixed user-specified step
                # ------------------------------------------------

                if not auto_correct_delta:
                    break

                # ------------------------------------------------
                # dK too small
                # ------------------------------------------------

                if RMSDelta < RMSGoal / RMSTol:

                    delta_local *= (
                        RMSGoal
                        /
                        RMSDelta
                    )

                # ------------------------------------------------
                # dK too large
                # ------------------------------------------------

                elif (
                    RMSDelta
                    >
                    RMSGoal * RMSTol / 3.0
                ):

                    delta_local *= (
                        RMSGoal
                        /
                        RMSDelta
                    )

                # ------------------------------------------------
                # dK accepted
                # ------------------------------------------------

                else:
                    break

            # ====================================================
            # 7. Final scalar finite-difference step
            # ====================================================

            step = float(
                delta_local[0]
            )

            if step == 0.0:
                raise ValueError(
                    f"Zero quadrupole step for group {group}"
                )

            # ====================================================
            # 8. Positive perturbation
            #
            # eta_plus = eta(K + dK)
            # ====================================================

            plus_values = (
                nominal_values
                +
                delta_local
            )

            set_correction(
                ring,
                plus_values,
                correction_indices,
                individuals=individuals,
                block="quads",
                config=fit_cfg,
            )

            C_plus = response_matrix(
                ring,
                config=cfg,
            )

            C_plus = (
                C
                @
                C_plus
            )

            eta_plus = np.asarray(
                C_plus[:, -1],
                dtype=float,
            ).copy()

            # ====================================================
            # 9. Restore nominal before negative perturbation
            # ====================================================

            set_correction(
                ring,
                nominal_values,
                correction_indices,
                individuals=individuals,
                block="quads",
                config=fit_cfg,
            )

            # ====================================================
            # 10. Negative perturbation
            #
            # eta_minus = eta(K - dK)
            # ====================================================

            minus_values = (
                nominal_values
                -
                delta_local
            )

            set_correction(
                ring,
                minus_values,
                correction_indices,
                individuals=individuals,
                block="quads",
                config=fit_cfg,
            )

            C_minus = response_matrix(
                ring,
                config=cfg,
            )

            C_minus = (
                C
                @
                C_minus
            )

            eta_minus = np.asarray(
                C_minus[:, -1],
                dtype=float,
            ).copy()

            # ====================================================
            # 11. CENTRAL FINITE DIFFERENCE
            #
            #        eta(K+dK) - eta(K-dK)
            # dη/dK = ----------------------
            #                 2 dK
            # ====================================================

            J_eta[p, :] = (
                eta_plus
                -
                eta_minus
            ) / (
                2.0
                *
                step
            )

            delta_used.append(
                step
            )

        finally:

            # ====================================================
            # 12. Always restore nominal lattice
            # ====================================================

            set_correction(
                ring,
                nominal_values,
                correction_indices,
                individuals=individuals,
                block="quads",
                config=fit_cfg,
            )

    # ============================================================
    # 13. Final checks
    # ============================================================

    delta_vec = np.asarray(
        delta_used,
        dtype=float,
    )

    print(
        "[Dispersion Jacobian] "
        f"{len(quads_ind)} parameters"
    )

    print(
        "[Dispersion Jacobian] shape:",
        J_eta.shape,
    )

    print(
        "[Dispersion Jacobian] dK:",
        delta_vec,
    )

    return (
        J_eta,
        delta_vec,
    )


def calculate_quads_jacobian_analytical(
    ring,
    C_model,
    dkick,
    used_cor_ind,
    bpm_indexes,
    quads_ind,
    C,
    includeDispersion=False,
    analytical_thick_quadrupole=True,
    analytical_thick_steerers=False,
    analytical_verbose=False,
    analytical_use_mp=False,
):
    """
    # Analytical ORM derivative formulas based on:
    #
    # A. Franchi and Z. Marti,
    # "Analytic formulas for the rapid evaluation of the orbit response
    # matrix and chromatic functions from lattice parameters in circular
    # accelerators", arXiv:1711.06589.
    #
    # Integrated into pyLOCO by Ahmed Eldeeb,
    # DESY Summer Student, August 2026.

    Analytical normal-quadrupole ORM Jacobian.

    The analytical formulas return derivatives with respect to
    integrated quadrupole strength KL.

    Output
    ------
    J_quad : ndarray
        Shape (n_parameters, n_bpm_total, n_orm_columns)

    delta : None
        No finite-difference step is required.
    """

    # ----------------------------------------------------------
    # Current limitation
    # ----------------------------------------------------------

    if includeDispersion:
        raise NotImplementedError(
            "Analytical normal-quadrupole Jacobian currently "
            "supports includeDispersion=False only. "
            "The derivative of the dispersion column with "
            "respect to KL has not yet been implemented."
        )

    bpm_indexes = np.asarray(bpm_indexes, dtype=int)

    hcor = np.asarray(used_cor_ind[0], dtype=int)
    vcor = np.asarray(used_cor_ind[1], dtype=int)

    nHBPM = len(bpm_indexes)
    nVBPM = len(bpm_indexes)

    nHorCOR = len(hcor)
    nVerCOR = len(vcor)

    n_rows = nHBPM + nVBPM
    n_cols = nHorCOR + nVerCOR

    # ----------------------------------------------------------
    # Check C_model
    # ----------------------------------------------------------

    if C_model.shape != (n_rows, n_cols):
        raise ValueError(
            "Unexpected C_model shape for analytical Jacobian: "
            f"{C_model.shape}; expected {(n_rows, n_cols)}."
        )

    # ----------------------------------------------------------
    # Corrector kicks
    # ----------------------------------------------------------

    kick_h = np.asarray(dkick[0], dtype=float).reshape(-1)
    kick_v = np.asarray(dkick[1], dtype=float).reshape(-1)

    if kick_h.size != nHorCOR:
        raise ValueError(
            f"Horizontal dkick length {kick_h.size} "
            f"!= number of H correctors {nHorCOR}"
        )

    if kick_v.size != nVerCOR:
        raise ValueError(
            f"Vertical dkick length {kick_v.size} "
            f"!= number of V correctors {nVerCOR}"
        )

    # ----------------------------------------------------------
    # Build fit parameter groups
    # ----------------------------------------------------------

    groups = []

    for q in quads_ind:

        if np.isscalar(q):
            groups.append([int(q)])

        else:
            groups.append([int(i) for i in q])

    # ----------------------------------------------------------
    # Collect all physical quadrupoles needed by analytical code
    # ----------------------------------------------------------

    physical_quads = sorted({
        q
        for group in groups
        for q in group
    })

    print(
        "[Analytical Jacobian] "
        f"{len(groups)} fit parameters, "
        f"{len(physical_quads)} physical quadrupoles"
    )

    # ----------------------------------------------------------
    # Analytical derivatives
    #
    # IMPORTANT:
    # H and V corrector lists may be different.
    # Therefore calculate them separately.
    # ----------------------------------------------------------

    dMH, _ = analytic_orm_variation_with_normal_quadrupole(
        ring,
        ind_bpms=bpm_indexes,
        ind_cors=hcor,
        ind_quads=physical_quads,
        thick_quadrupole=analytical_thick_quadrupole,
        thick_steerers=analytical_thick_steerers,
        verbose=analytical_verbose,
        use_mp=analytical_use_mp,
    )

    _, dMV = analytic_orm_variation_with_normal_quadrupole(
        ring,
        ind_bpms=bpm_indexes,
        ind_cors=vcor,
        ind_quads=physical_quads,
        thick_quadrupole=analytical_thick_quadrupole,
        thick_steerers=analytical_thick_steerers,
        verbose=analytical_verbose,
        use_mp=analytical_use_mp,
    )

    # physical quad index -> analytical-array index
    q_to_pos = {
        q: i
        for i, q in enumerate(physical_quads)
    }

    # ----------------------------------------------------------
    # Allocate pyLOCO Jacobian
    # ----------------------------------------------------------

    J_quad = np.zeros(
        (
            len(groups),
            n_rows,
            n_cols,
        ),
        dtype=float,
    )

    # ----------------------------------------------------------
    # Construct each fit-parameter derivative
    # ----------------------------------------------------------

    for p, group in enumerate(groups):

        # One family parameter changes all magnets in that family.
        #
        # Therefore:
        #
        # dR/dK_family =
        #     sum_m dR/dK_m
        #
        # The analytical formula is dR/d(KL), so conversion to
        # pyLOCO's fitted K parameter requires multiplication by L.
        #
        # dR/dK = L * dR/d(KL)

        for q in group:

            aq = q_to_pos[q]

            Lq = float(ring[q].Length)

            # --------------------------------------------------
            # XX block
            #
            # Sign follows the numerical-vs-analytical
            # validation already performed.
            # --------------------------------------------------

            J_quad[
                p,
                :nHBPM,
                :nHorCOR,
            ] += (
                -dMH[:, :, aq]
                * Lq
                * kick_h[np.newaxis, :]
            )

            # --------------------------------------------------
            # YY block
            # --------------------------------------------------

            J_quad[
                p,
                nHBPM:,
                nHorCOR:,
            ] += (
                dMV[:, :, aq]
                * Lq
                * kick_v[np.newaxis, :]
            )

    # ----------------------------------------------------------
    # Apply BPM calibration/coupling matrix
    # ----------------------------------------------------------

    for p in range(J_quad.shape[0]):
        J_quad[p] = C @ J_quad[p]

    print(
        "[Analytical Jacobian] shape:",
        J_quad.shape,
    )

    return J_quad, None


def calculate_skew_jacobian_analytical(
    ring,
    C_model,
    dkick,
    used_cor_ind,
    bpm_indexes,
    skew_ind,
    C,
    fit_cfg=None,
    analytical_thick_skew=True,
    analytical_thick_steerers=False,
    analytical_verbose=False,
    analytical_use_mp=False,
):
    """Return the analytical skew ORM Jacobian in pyLOCO layout.

    ``analytic_orm_variation_with_skew_quadrupole`` differentiates with
    respect to integrated skew strength ``Ks*L``.  pyLOCO's skew fit parameter
    is the element coefficient ``PolynomA[1]`` (``Ks``), so each physical
    magnet contribution is multiplied by its length.  A family parameter is
    the simultaneous change of all its physical magnets and is consequently
    the sum of those converted contributions.

    The complete output layout is retained explicitly::

                         H correctors    V correctors
        horizontal BPM       XX              XY
        vertical BPM         YX              YY

    The first-order formula supplies XY and YX.  XX and YY remain allocated
    as zero blocks; applying the BPM calibration/coupling matrix may mix these
    physical blocks in the final calibrated response.
    """
    bpm_indexes = np.asarray(bpm_indexes, dtype=int)
    hcor = np.asarray(used_cor_ind[0], dtype=int)
    vcor = np.asarray(used_cor_ind[1], dtype=int)
    n_bpm = len(bpm_indexes)
    n_hcor = len(hcor)
    n_vcor = len(vcor)
    expected_shape = (2 * n_bpm, n_hcor + n_vcor)
    if C_model.shape != expected_shape:
        raise ValueError(
            f"Unexpected C_model shape for analytical skew Jacobian: "
            f"{C_model.shape}; expected {expected_shape}."
        )

    attr_name, attr_idx = _resolve_attr_for_block_read("skew_quads", fit_cfg)
    if attr_name != "PolynomA" or attr_idx != 1:
        raise ValueError(
            "Analytical skew Jacobian supports a fitted PolynomA[1] "
            f"coefficient; got {attr_name}[{attr_idx}]."
        )

    kick_h = np.asarray(dkick[0], dtype=float).reshape(-1)
    kick_v = np.asarray(dkick[1], dtype=float).reshape(-1)
    if kick_h.size != n_hcor or kick_v.size != n_vcor:
        raise ValueError("Corrector kick arrays do not match the selected correctors")

    groups = [
        [int(item)] if np.isscalar(item) else [int(q) for q in item]
        for item in skew_ind
    ]
    physical_skews = sorted({q for group in groups for q in group})
    skew_to_pos = {q: pos for pos, q in enumerate(physical_skews)}

    # The formula accepts a common corrector list.  Evaluate the horizontal
    # and vertical selections separately because pyLOCO permits them to differ.
    d_yx, _ = analytic_orm_variation_with_skew_quadrupole(
        ring, ind_bpms=bpm_indexes, ind_cors=hcor,
        ind_skews=physical_skews, verbose=analytical_verbose,
        thick_skew=analytical_thick_skew,
        thick_steerer=analytical_thick_steerers,
        use_mp=analytical_use_mp,
    )
    _, d_xy = analytic_orm_variation_with_skew_quadrupole(
        ring, ind_bpms=bpm_indexes, ind_cors=vcor,
        ind_skews=physical_skews, verbose=analytical_verbose,
        thick_skew=analytical_thick_skew,
        thick_steerer=analytical_thick_steerers,
        use_mp=analytical_use_mp,
    )

    jacobian = np.zeros((len(groups), *expected_shape), dtype=float)
    for parameter, group in enumerate(groups):
        for skew in group:
            formula_index = skew_to_pos[skew]
            length = float(ring[skew].Length)
            # pyLOCO's horizontal-corrector response convention is opposite
            # to MH2V's convention; keep this adapter sign outside the
            # analytical skew-response equations.
            jacobian[parameter, n_bpm:, :n_hcor] += (
                -d_yx[:, :, formula_index] * length * kick_h[np.newaxis, :]
            )
            jacobian[parameter, :n_bpm, n_hcor:] += (
                d_xy[:, :, formula_index] * length * kick_v[np.newaxis, :]
            )

    for parameter in range(jacobian.shape[0]):
        jacobian[parameter] = C @ jacobian[parameter]

    print(
        "[Analytical skew Jacobian] "
        f"{len(groups)} fit parameters, {len(physical_skews)} physical magnets, "
        f"shape={jacobian.shape}"
    )
    return jacobian, None

# ---------- worker globals ----------
G_C = None
G_CMODEL = None
def calculate_quads_tilt_jacobian(
    ring,
    C_model,
    dkick,
    used_cor_ind,
    bpm_indexes,
    quads_ind,
    dk,
    C,
    individuals,
    HCMCoupling,
    VCMCoupling,
    rf_step,
    auto_correct_delta=True,
    processes=None,
    includeDispersion=False,
    output_dir="output",
    log_filename="quads_tilt_jacobian_logs.txt",
    quads_tilt_fit=None,
    fit_cfg=None,
):

    from pathlib import Path

    output_dir = Path(output_dir)

    # ============================================================
    # Shared memory
    # ============================================================

    shm_C = shared_memory.SharedMemory(
        create=True,
        size=C.nbytes,
    )

    C_sh = np.ndarray(
        C.shape,
        dtype=C.dtype,
        buffer=shm_C.buf,
    )
    C_sh[:] = C

    shm_Cm = shared_memory.SharedMemory(
        create=True,
        size=C_model.nbytes,
    )

    Cmodel_sh = np.ndarray(
        C_model.shape,
        dtype=C_model.dtype,
        buffer=shm_Cm.buf,
    )
    Cmodel_sh[:] = C_model

    all_logs = []

    ctx = mp.get_context("spawn")

    try:

        # ========================================================
        # Sanity check
        # ========================================================

        assert len(quads_tilt_fit) == len(quads_ind), (
            f"Length mismatch: "
            f"{len(quads_tilt_fit)=} vs {len(quads_ind)=}"
        )

        # ========================================================
        # Compact debugging
        # ========================================================

        zero_positions = [
            i
            for i, x in enumerate(quads_ind)
            if np.isscalar(x) and int(x) == 0
        ]

        print(
            "\n========== TILT JACOBIAN START ==========",
            flush=True,
        )

        print(
            "individuals       :",
            individuals,
            flush=True,
        )

        print(
            "N tilt parameters :",
            len(quads_ind),
            flush=True,
        )

        print(
            "first 5 indices   :",
            quads_ind[:5],
            flush=True,
        )

        print(
            "last 5 indices    :",
            quads_ind[-5:],
            flush=True,
        )

        print(
            "scalar-0 positions:",
            zero_positions,
            flush=True,
        )

        if len(quads_tilt_fit) > 0:
            print(
                "first tilt value  :",
                quads_tilt_fit[0],
                flush=True,
            )

        print(
            "=========================================\n",
            flush=True,
        )

        # ========================================================
        # Prepare multiprocessing arguments
        # ========================================================

        quad_args = []

        fit_cfg_dict = fit_cfg.__dict__.copy()

        for i, quad_index in enumerate(quads_ind):

            tilt_fit_i = quads_tilt_fit[i]

            quad_args.append(
                (
                    quad_index,
                    ring,
                    dkick,
                    bpm_indexes,
                    used_cor_ind,
                    dk,
                    individuals,
                    auto_correct_delta,
                    HCMCoupling,
                    VCMCoupling,
                    rf_step,
                    tilt_fit_i,
                    fit_cfg_dict,
                    includeDispersion,
                )
            )

        # ========================================================
        # Calculate tilt Jacobian
        # ========================================================

        with ctx.Pool(
            processes=processes,
            initializer=_init_shared,
            initargs=(
                shm_C.name,
                C.shape,
                C.dtype.str,
                shm_Cm.name,
                C_model.shape,
                C_model.dtype.str,
            ),
            maxtasksperchild=64,
        ) as pool:

            results = pool.starmap(
                generating_quads_tilt_response_matrices,
                quad_args,
                chunksize=1,
            )

        # ========================================================
        # Collect results
        # ========================================================

        if results:

            J_blocks, deltas, logs_lists = zip(*results)

            for _logs in logs_lists:
                if _logs:
                    all_logs.extend(_logs)

            J_blocks = [
                np.asarray(blk)
                for blk in J_blocks
            ]

            J_quad = np.stack(
                J_blocks,
                axis=0,
            )

            delta_vec = np.concatenate(
                [
                    np.atleast_1d(d)
                    for d in deltas
                ]
            )

        else:

            J_quad = np.empty(
                (
                    0,
                    C.shape[0],
                    C.shape[1],
                )
            )

            delta_vec = np.empty((0,))

        # ========================================================
        # Save logs
        # ========================================================

        if all_logs:

            try:

                output_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                log_dir = output_dir / "logs"

                log_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                log_path = (
                    log_dir / log_filename
                )

                with open(
                    log_path,
                    "w",
                    encoding="utf-8",
                ) as f:

                    f.write(
                        "\n".join(all_logs)
                        + "\n"
                    )

                print(
                    "[calculate_quads_tilt_jacobian] "
                    f"Logs saved to "
                    f"'{log_path.resolve()}'"
                )

            except Exception as e:

                print(
                    "[calculate_quads_tilt_jacobian] "
                    f"Could not write logs: {e}"
                )

        # ========================================================
        # Finished
        # ========================================================

        print(
            "[TILT JACOBIAN] completed successfully "
            f"for {len(quads_ind)} parameters.",
            flush=True,
        )

        return J_quad, delta_vec

    finally:

        # ========================================================
        # Clean shared memory
        # ========================================================

        try:
            shm_C.close()
            shm_C.unlink()
        except Exception:
            pass

        try:
            shm_Cm.close()
            shm_Cm.unlink()
        except Exception:
            pass

def _init_shared(shm_name_C, shape_C, dtype_C, shm_name_Cm, shape_Cm, dtype_Cm):
    global G_C, G_CMODEL, _shm_C, _shm_Cm
    _shm_C = shared_memory.SharedMemory(name=shm_name_C)
    _shm_Cm = shared_memory.SharedMemory(name=shm_name_Cm)
    G_C = np.ndarray(shape_C, dtype=np.dtype(dtype_C), buffer=_shm_C.buf)
    G_CMODEL = np.ndarray(shape_Cm, dtype=np.dtype(dtype_Cm), buffer=_shm_Cm.buf)

def generating_quads_response_matrices(
        quad_index, ring, dkick, cor_indexes, bpm_indexes,
        delta_init, individuals, HCMCoupling, VCMCoupling,
        rf_step, auto_correct_delta, block, fit_cfg,
        includeDispersion, orm_calculator="Linear"
):
    """
    Generate the numerical quadrupole Jacobian for one fitted
    quadrupole parameter.

    The parameter may correspond to:

        individuals=True
            one physical quadrupole

        individuals=False
            one quadrupole family

    The finite-difference step can optionally be adjusted using
    the RMS ORM change.

    Once the step has been selected, the final Jacobian is
    evaluated using a CENTRAL finite difference:

        J = [C(K + dK) - C(K - dK)] / (2 dK)

    rather than the previous forward difference:

        J = [C(K + dK) - C(K)] / dK
    """

    logs = []

    # ============================================================
    # 1. Resolve fitted quadrupole attribute
    # ============================================================

    attr_name, attr_idx = _resolve_attr_for_block_read(
        block,
        fit_cfg,
    )

    # quad_index may be:
    #
    #   int
    #       individual quadrupole
    #
    #   list[int]
    #       quadrupole family
    #
    group = (
        [int(quad_index)]
        if np.isscalar(quad_index)
        else [int(q) for q in quad_index]
    )

    # ============================================================
    # 2. Read nominal quadrupole strengths
    # ============================================================

    k0_each = np.asarray(
        [
            _get_attr_scalar(
                ring[q],
                attr_name,
                attr_idx,
            )
            for q in group
        ],
        dtype=float,
    )

    # ============================================================
    # 3. Individual or family parameter
    # ============================================================

    if individuals:

        # --------------------------------------------------------
        # Individual parameter
        # --------------------------------------------------------

        correction_indices = group

        nominal_values = k0_each.copy()

        if delta_init is None:

            delta_local = (
                1e-3 * nominal_values
            )

            delta_local[
                delta_local == 0.0
            ] = 1e-3

        else:

            delta_local = np.atleast_1d(
                delta_init
            ).astype(float)[:len(group)]

    else:

        # --------------------------------------------------------
        # Family parameter
        # --------------------------------------------------------

        correction_indices = [
            group
        ]

        nominal_values = np.asarray(
            [k0_each[0]],
            dtype=float,
        )

        # Family fitting assumes that all magnets belonging
        # to the family have the same nominal strength.
        if not np.allclose(
            k0_each,
            k0_each[0],
            rtol=1e-10,
            atol=1e-14,
        ):
            raise ValueError(
                f"Family {group} contains different "
                f"nominal strengths: {k0_each}"
            )

        if delta_init is None:

            delta_value = (
                1e-3 * nominal_values[0]
            )

            if delta_value == 0.0:
                delta_value = 1e-3

        else:

            delta_value = float(
                np.atleast_1d(
                    delta_init
                ).ravel()[0]
            )

        delta_local = np.asarray(
            [delta_value],
            dtype=float,
        )

    # ============================================================
    # 4. Step-size selection settings
    # ============================================================

    RMSGoal = 1e-6
    RMSTol = 10.0

    # ============================================================
    # 5. Response-matrix configuration
    #
    # Create it once. The lattice itself is modified below.
    # ============================================================

    cfg = RMConfig(
        dkick=dkick,
        bpm_ords=bpm_indexes,
        cm_ords=cor_indexes,
        HCMCoupling=HCMCoupling,
        VCMCoupling=VCMCoupling,
        includeDispersion=includeDispersion,
        rfStep=rf_step,
        calculator=orm_calculator,
    )

    # ============================================================
    # 6. Automatically choose a suitable finite-difference step
    #
    # We preserve the previous pyLOCO logic here.
    #
    # The +dK response is compared with the nominal response only
    # to determine whether dK is sufficiently large/small.
    #
    # This response is NOT used as the final Jacobian.
    # ============================================================

    while True:

        plus_values = (
            nominal_values
            +
            delta_local
        )

        # --------------------------------------------------------
        # Apply +dK
        # --------------------------------------------------------

        set_correction(
            ring,
            plus_values,
            correction_indices,
            individuals=individuals,
            block=block,
            config=fit_cfg,
        )

        # --------------------------------------------------------
        # Calculate ORM at K + dK
        # --------------------------------------------------------

        C_plus_step_test = response_matrix(
            ring,
            config=cfg,
        )

        C_plus_step_test = (
            G_C
            @
            C_plus_step_test
        )

        # --------------------------------------------------------
        # Difference from nominal ORM
        # --------------------------------------------------------

        if includeDispersion:

            difference = (
                C_plus_step_test[:, :-1]
                -
                G_CMODEL[:, :-1]
            ).ravel(
                order="F"
            )

        else:

            difference = (
                C_plus_step_test
                -
                G_CMODEL
            ).ravel(
                order="F"
            )

        RMSDelta = float(
            np.sqrt(
                np.sum(
                    difference**2
                )
                /
                max(
                    1,
                    difference.size,
                )
            )
        )

        if (
        not np.isfinite(RMSDelta)
        or
        RMSDelta == 0.0):

            print(
                "\n========== INVALID QUAD/SKEW RMS ==========\n"
                f"group          : {group}\n"
                f"block          : {block}\n"
                f"individuals    : {individuals}\n"
                f"attr_name      : {attr_name}\n"
                f"attr_idx       : {attr_idx}\n"
                f"nominal_values : {np.asarray(nominal_values).tolist()}\n"
                f"delta_local    : {np.asarray(delta_local).tolist()}\n"
                f"RMSDelta       : {RMSDelta!r}\n"
                f"diff size      : {difference.size}\n"
                f"diff finite    : {np.all(np.isfinite(difference))}\n"
                f"diff min       : {np.nanmin(difference):.12e}\n"
                f"diff max       : {np.nanmax(difference):.12e}\n"
                "============================================\n",
                flush=True,
            )

            # Restore nominal lattice before raising.
            set_correction(
                ring,
                nominal_values,
                correction_indices,
                individuals=individuals,
                block=block,
                config=fit_cfg,
            )

            raise ValueError(
                f"LOCO error: RMS difference invalid "
                f"for group {group}; "
                f"block={block}; "
                f"RMSDelta={RMSDelta!r}; "
                f"nominal={np.asarray(nominal_values).tolist()}; "
                f"delta={np.asarray(delta_local).tolist()}"
            )

        # --------------------------------------------------------
        # User requested fixed dK
        # --------------------------------------------------------

        if not auto_correct_delta:

            logs.append(
                f"Group {group}: fixed delta used; "
                f"RMS={1000 * RMSDelta:0.5g} mm"
            )

            break

        # --------------------------------------------------------
        # dK too small
        # --------------------------------------------------------

        if RMSDelta < RMSGoal / RMSTol:

            logs.append(
                f"Group {group}: delta too small; "
                f"RMS={1000 * RMSDelta:0.5g} mm"
            )

            # Restore nominal lattice
            set_correction(
                ring,
                nominal_values,
                correction_indices,
                individuals=individuals,
                block=block,
                config=fit_cfg,
            )

            delta_local *= (
                RMSGoal
                /
                RMSDelta
            )

        # --------------------------------------------------------
        # dK too large
        # --------------------------------------------------------

        elif RMSDelta > RMSGoal * RMSTol / 3.0:

            logs.append(
                f"Group {group}: delta too large; "
                f"RMS={1000 * RMSDelta:0.5g} mm"
            )

            # Restore nominal lattice
            set_correction(
                ring,
                nominal_values,
                correction_indices,
                individuals=individuals,
                block=block,
                config=fit_cfg,
            )

            delta_local *= (
                RMSGoal
                /
                RMSDelta
            )

        # --------------------------------------------------------
        # dK accepted
        # --------------------------------------------------------

        else:

            logs.append(
                f"Group {group}: delta OK; "
                f"RMS={1000 * RMSDelta:0.5g} mm"
            )

            break

    # ============================================================
    # 7. Restore nominal lattice before final derivative
    # ============================================================

    set_correction(
        ring,
        nominal_values,
        correction_indices,
        individuals=individuals,
        block=block,
        config=fit_cfg,
    )

    # ============================================================
    # 8. Check finite-difference step
    # ============================================================

    step = float(
        delta_local[0]
    )

    if step == 0.0:

        raise ValueError(
            f"Zero Jacobian step "
            f"for group {group}"
        )

    # ============================================================
    # 9. POSITIVE perturbation
    #
    #                 C_plus = C(K + dK)
    # ============================================================

    plus_values = (
        nominal_values
        +
        delta_local
    )

    set_correction(
        ring,
        plus_values,
        correction_indices,
        individuals=individuals,
        block=block,
        config=fit_cfg,
    )

    C_plus = response_matrix(
        ring,
        config=cfg,
    )

    C_plus = (
        G_C
        @
        C_plus
    )

    # ============================================================
    # 10. Restore nominal before negative perturbation
    # ============================================================

    set_correction(
        ring,
        nominal_values,
        correction_indices,
        individuals=individuals,
        block=block,
        config=fit_cfg,
    )

    # ============================================================
    # 11. NEGATIVE perturbation
    #
    #                 C_minus = C(K - dK)
    # ============================================================

    minus_values = (
        nominal_values
        -
        delta_local
    )

    set_correction(
        ring,
        minus_values,
        correction_indices,
        individuals=individuals,
        block=block,
        config=fit_cfg,
    )

    C_minus = response_matrix(
        ring,
        config=cfg,
    )

    C_minus = (
        G_C
        @
        C_minus
    )

    # ============================================================
    # 12. Restore nominal lattice
    #
    # Very important: leave the worker lattice exactly as it
    # entered the derivative calculation.
    # ============================================================

    set_correction(
        ring,
        nominal_values,
        correction_indices,
        individuals=individuals,
        block=block,
        config=fit_cfg,
    )

    # ============================================================
    # 13. CENTRAL FINITE DIFFERENCE
    #
    #             C(K+dK) - C(K-dK)
    #     J = ---------------------------
    #                       2 dK
    # ============================================================

    J_block = (
        C_plus
        -
        C_minus
    ) / (
        2.0 * step
    )

    # ============================================================
    # 14. Log final step
    # ============================================================

    logs.append(
        f"Group {group}: central finite difference; "
        f"step={step:.12e}"
    )

    # ============================================================
    # 15. Return
    # ============================================================

    return (
        J_block,
        delta_local,
        logs,
    )
def generating_quads_tilt_response_matrices(
        quad_index,
        ring,
        dkick,
        bpm_indexes,
        cor_indexes,
        delta_init,
        individuals,
        auto_correct_delta,
        HCMCoupling,
        VCMCoupling,
        rf_step,
        quads_tilt_fit,
        fit_cfg,
        includeDispersion,
):
    logs = []

    # -------------------------------------------------------------
    # Build group
    # -------------------------------------------------------------
    group = (
        [int(quad_index)]
        if np.isscalar(quad_index)
        else [int(q) for q in quad_index]
    )

    # -------------------------------------------------------------
    # Individual vs family handling
    # -------------------------------------------------------------
    if individuals:

        correction_indices = group
        nominal_values = np.asarray(
            quads_tilt_fit,
            dtype=float,
        )

        if delta_init is None:
            delta_local = np.full(
                len(group),
                1e-6,
                dtype=float,
            )
        else:
            delta_local = np.atleast_1d(
                delta_init
            ).astype(float)[:len(group)]

    else:

        correction_indices = [group]

        nominal_values = np.asarray(
            [np.asarray(quads_tilt_fit).ravel()[0]],
            dtype=float,
        )

        if delta_init is None:
            delta_local = np.asarray(
                [1e-6],
                dtype=float,
            )
        else:
            delta_local = np.asarray(
                [
                    float(
                        np.atleast_1d(
                            delta_init
                        ).ravel()[0]
                    )
                ],
                dtype=float,
            )

    # -------------------------------------------------------------
    # Auto-step settings
    # -------------------------------------------------------------
    RMSGoal = 1e-6
    RMSTol = 10.0

    attempt = 0
    max_attempts = 20


   
    # -------------------------------------------------------------
    # Find suitable finite-difference step
    # -------------------------------------------------------------
    while True:

        attempt += 1

        # ---------------------------------------------------------
        # Apply positive tilt perturbation
        # ---------------------------------------------------------
        set_correction_tilt(
            ring,
            psi_values=nominal_values + delta_local,
            elem_ind=correction_indices,
            individuals=individuals,
            config=fit_cfg,
        )

        # ---------------------------------------------------------
        # Calculate perturbed response matrix
        # ---------------------------------------------------------
        cfg = RMConfig(
            dkick=dkick,
            bpm_ords=bpm_indexes,
            cm_ords=cor_indexes,
            HCMCoupling=HCMCoupling,
            VCMCoupling=VCMCoupling,
            includeDispersion=includeDispersion,
            rfStep=rf_step,
        )

        C_measured = response_matrix(
            ring,
            config=cfg,
        )

        C_measured = G_C @ C_measured

        # ---------------------------------------------------------
        # ORM difference used for automatic step selection
        # ---------------------------------------------------------
        if includeDispersion:

            diff = (
                C_measured[:, :-1]
                - G_CMODEL[:, :-1]
            ).ravel(order="F")

        else:

            diff = (
                C_measured
                - G_CMODEL
            ).ravel(order="F")

        RMSDelta = np.sqrt(
            np.mean(diff ** 2)
        )

        # ---------------------------------------------------------
        # Invalid numerical result
        # ---------------------------------------------------------
        if (
            not np.isfinite(RMSDelta)
            or RMSDelta == 0
        ):
            raise ValueError(
                f"Invalid RMS difference for tilt group {group}: "
                f"RMSDelta={RMSDelta}, "
                f"delta={delta_local.tolist()}, "
                f"attempt={attempt}"
            )

        
        # ---------------------------------------------------------
        # Safety against an infinite auto-step loop
        # ---------------------------------------------------------
        if attempt >= max_attempts:

            # Restore nominal tilt before failing
            set_correction_tilt(
                ring,
                psi_values=nominal_values,
                elem_ind=correction_indices,
                individuals=individuals,
                config=fit_cfg,
            )

            raise RuntimeError(
                "Tilt auto-delta failed to converge: "
                f"group={group}, "
                f"attempts={attempt}, "
                f"RMSDelta={RMSDelta:.6e}, "
                f"delta={delta_local.tolist()}"
            )

        # ---------------------------------------------------------
        # No automatic step correction requested
        # ---------------------------------------------------------
        if not auto_correct_delta:
            break

        # ---------------------------------------------------------
        # Step too small
        # ---------------------------------------------------------
        if RMSDelta < RMSGoal / RMSTol:

            logs.append(
                f"Group {group}: delta too small; "
                f"attempt={attempt}; "
                f"RMS={1000 * RMSDelta:.5g} mm; "
                f"delta={delta_local.tolist()}"
            )

            # Restore nominal lattice before changing step
            set_correction_tilt(
                ring,
                psi_values=nominal_values,
                elem_ind=correction_indices,
                individuals=individuals,
                config=fit_cfg,
            )

            delta_local *= (
                RMSGoal / RMSDelta
            )

        # ---------------------------------------------------------
        # Step too large
        # ---------------------------------------------------------
        elif RMSDelta > RMSGoal * RMSTol / 3:

            logs.append(
                f"Group {group}: delta too large; "
                f"attempt={attempt}; "
                f"RMS={1000 * RMSDelta:.5g} mm; "
                f"delta={delta_local.tolist()}"
            )

            # Restore nominal lattice before changing step
            set_correction_tilt(
                ring,
                psi_values=nominal_values,
                elem_ind=correction_indices,
                individuals=individuals,
                config=fit_cfg,
            )

            delta_local *= (
                RMSGoal / RMSDelta
            )

        # ---------------------------------------------------------
        # Step accepted
        # ---------------------------------------------------------
        else:

            logs.append(
                f"Group {group}: delta OK; "
                f"attempt={attempt}; "
                f"RMS={1000 * RMSDelta:.5g} mm; "
                f"delta={delta_local.tolist()}"
            )

            break

    # -------------------------------------------------------------
    # Restore nominal lattice
    # -------------------------------------------------------------
    set_correction_tilt(
        ring,
        psi_values=nominal_values,
        elem_ind=correction_indices,
        individuals=individuals,
        config=fit_cfg,
    )

    # -------------------------------------------------------------
    # Final finite-difference step
    # -------------------------------------------------------------
    step = delta_local.item()

    # -------------------------------------------------------------
    # Return Jacobian block
    # -------------------------------------------------------------
    return (
        (C_measured - G_CMODEL) / step,
        delta_local,
        logs,
    )


def calculate_bpm_gain_jacobian(C_model, nHBPM, nVBPM, nHorCOR, nVerCOR, includeDispersion, fit_bpms_coupling):
    nBPM, nCOR = C_model.shape

    if fit_bpms_coupling == True:
        J_bpm = np.zeros((2 * nBPM, nBPM, nCOR))
    else:
        J_bpm = np.zeros((nBPM, nBPM, nCOR))

    if fit_bpms_coupling == False:

        for i in range(nHBPM):
            J_bpm[i, i, :] = C_model[i, :]

        for i in range(nVBPM):
            idx = i + nHBPM
            J_bpm[idx, idx, :] = C_model[idx, :]

    if fit_bpms_coupling == True:

        for i in range(nHBPM):
            J_bpm[i, i, :] = C_model[i, :]

        # 1. XY Coupling : Horizontal BPMs coupling

        for i in range(nHBPM):
            idx = i + nVBPM
            J_bpm[i + nHBPM, i, :] = C_model[idx, :]

        # 2. YX Coupling : Vertical BPMs coupling

        for i in range(nVBPM):
            idx = i + nHBPM
            J_bpm[i + nHBPM + nVBPM, idx, :] = C_model[i, :]

        for i in range(nVBPM):
            idx = i + nHBPM
            J_bpm[i + nHBPM + nHBPM + nVBPM, idx, :] = C_model[idx, :]

    return J_bpm


def calculate_bpm_coupling_jacobian(
        C_model, nHBPM, nVBPM, includeDispersion
):
    nBPM, nCOR = C_model.shape
    J_bpm = np.zeros((nBPM, nBPM, nCOR))

    # 1. XY Coupling

    for i in range(nHBPM):
        idx = i + nVBPM
        J_bpm[i, i, :] = C_model[idx, :]  #####

    # 1. YX Coupling

    for i in range(nVBPM):
        idx = i + nHBPM
        J_bpm[idx, idx, :] = C_model[i, :]  ###

    return J_bpm


def calculate_corrector_kick_jacobian(C_model, cor_kicks, nHorCOR, nVerCOR, includeDispersion):
    nBPM, nCols = C_model.shape
    nCOR = nHorCOR + nVerCOR
    has_disp = nCols == nCOR + 1

    if has_disp:
        C_model_scaled = C_model[:, :nCOR]  # / cor_kicks[np.newaxis, :]
    else:
        C_model_scaled = C_model  # / cor_kicks[np.newaxis, :]

    J_cor = np.zeros((nCOR, nBPM, nCols))

    for i in range(nHorCOR):
        J_cor[i, :, i] = C_model_scaled[:, i] / cor_kicks[0][i]

    for i in range(nVerCOR):
        idx = i + nHorCOR
        J_cor[idx, :, idx] = C_model_scaled[:, idx] / cor_kicks[1][i]

    if includeDispersion == True and has_disp:
        for i in range(nCOR):
            J_cor[i, :, -1] = 0  # last column in each 2D matrix

    return J_cor


def calculate_corrector_coupling_jacobian(
        ring,
        bpm_ords,
        cm_ords,
        C_model,
        cor_kicks,
        nHBPM,
        nVBPM,
        nHorCOR,
        nVerCOR,
        includeDispersion, C, HCMCoupling, VCMCoupling, rf_step,
        delta_coupling=1e-6
):
    nBPM_total = nHBPM + nVBPM
    nCOR_total = nHorCOR + nVerCOR
    nCols = C_model.shape[1]
    has_disp = nCols == nCOR_total + 1

    HCMCoupling = HCMCoupling + delta_coupling * np.ones(len(HCMCoupling))
    VCMCoupling = VCMCoupling + delta_coupling * np.ones(len(VCMCoupling))

    cfg = RMConfig(dkick=cor_kicks, bpm_ords=bpm_ords, cm_ords=cm_ords, HCMCoupling=HCMCoupling,
                   VCMCoupling=VCMCoupling, includeDispersion=includeDispersion, rfStep=rf_step)
    GR = response_matrix(ring, config=cfg)

    GR = C @ GR

    nParams_total = nHorCOR + nVerCOR
    J_cor = np.zeros((nParams_total, nBPM_total, nCols))

    for i in range(nHorCOR):
        dC = (GR[:, i] - C_model[:, i]) / delta_coupling
        J_cor[i, :, i] = dC

    for k in range(nVerCOR):
        j = nHorCOR + k
        p = nHorCOR + k
        dC = (GR[:, j] - C_model[:, j]) / delta_coupling
        J_cor[p, :, j] = dC

    if includeDispersion == True or has_disp:
        J_cor[:, :, -1] = 0.0

    return J_cor


def calculate_HCMEnergyShift_jacobian(ring, C_model, nHBPM, nVBPM, nHorCOR, nVerCOR, rf_step, measured_eta_x,
                                      measured_eta_y, Frequency):
    nBPM_total = nHBPM + nVBPM
    nCOR_total = nHorCOR + nVerCOR
    nCols = C_model.shape[1]
    has_disp = nCols == nCOR_total + 1

    if has_disp:
        print("Error: Better to either include dispersion on ORM or fit the energy shift at correctors.")
        # return None

    nParams_total = nHorCOR
    alpha_mc = get_mcf(ring)

    eta_x_mcf = -alpha_mc * Frequency * measured_eta_x / rf_step
    eta_y_mcf = -alpha_mc * Frequency * measured_eta_y / rf_step
    J_HCMEnergyShift = np.zeros((nParams_total, nBPM_total, nCols))

    for i in range(nHorCOR):
        J_HCMEnergyShift[i, :nHBPM, i] = eta_x_mcf
        J_HCMEnergyShift[i, nHBPM:, i] = eta_y_mcf

    return J_HCMEnergyShift


def calculate_VCMEnergyShift_jacobian(ring, C_model, nHBPM, nVBPM, nHorCOR, nVerCOR, rf_step, measured_eta_x,
                                      measured_eta_y, Frequency):
    nBPM_total = nHBPM + nVBPM
    nCOR_total = nHorCOR + nVerCOR
    nCols = C_model.shape[1]
    has_disp = nCols == nCOR_total + 1

    if has_disp:
        print("Error: Better to either include dispersion on ORM or fit the energy shift at correctors.")
        # return None

    nParams_total = nVerCOR
    alpha_mc = get_mcf(ring)

    eta_x_mcf = -alpha_mc * Frequency * measured_eta_x / rf_step
    eta_y_mcf = -alpha_mc * Frequency * measured_eta_y / rf_step

    J_VCMEnergyShift = np.zeros((nParams_total, nBPM_total, nCols))

    for i in range(nVerCOR):
        J_VCMEnergyShift[i, :nHBPM, i + nHorCOR] = eta_x_mcf
        J_VCMEnergyShift[i, nHBPM:, i + nHorCOR] = eta_y_mcf

    return J_VCMEnergyShift


def calculate_delta_RF_frequency_jacobian(C_model, nHBPM, nVBPM, nHorCOR, nVerCOR, rf_step):
    """
    Calculate the Jacobian column corresponding to a delta RF frequency perturbation,
    assuming the last column of the ORM includes the effect of RF frequency variation.

    Parameters
    ----------
    C_model : ndarray
        Full ORM matrix including RF frequency sensitivity (last column).
    nHBPM, nVBPM : int
        Number of horizontal and vertical BPMs.
    nHorCOR, nVerCOR : int
        Number of horizontal and vertical correctors.
    rf_step : float
        RF frequency step used when computing the dispersion ORM.

    Returns
    -------
    J_delta_RF_frequency : ndarray
        Normalized Jacobian matrix for RF frequency parameter.
    """
    nBPM_total = nHBPM + nVBPM
    nCOR_total = nHorCOR + nVerCOR
    nCols = C_model.shape[1]
    has_disp = nCols == nCOR_total + 1

    if not has_disp:
        print("Error: Cannot fit delta RF frequency without including dispersion in the ORM.")
        return None

    J_delta_RF_frequency = np.zeros_like(C_model)
    J_delta_RF_frequency[:, -1] = C_model[:, -1] / rf_step

    # Apply normalization factor to increase fitting weight of RF frequency
    # normalization_factor = 1 / rf_step / 10
    # J_delta_RF_frequency = J_delta_RF_frequency / normalization_factor

    # print("The RF frequency parameter is normalized by 1 / rf_step / 10 to to get a better fit.")

    J_delta_RF_frequency = J_delta_RF_frequency[np.newaxis, :, :]  # convert it to 3 d

    return J_delta_RF_frequency


# ============================================================================== #
#                               NORMALIZATION OPTION
# ============================================================================== #


def normalize_jacobian_global(J_flat, model_orm_flat, weights_flat):
    """
    Normalize each column of J_flat by sqrt(sum(J[:,i]^2) / Mmodelsq)
    """

    Mmodelsq = np.sum((model_orm_flat / weights_flat) ** 2)
    norm_factors = np.sqrt(np.sum((J_flat / weights_flat) ** 2, axis=0) / Mmodelsq)
    J_normalized = J_flat / norm_factors[np.newaxis, :]
    return J_normalized, norm_factors.reshape(-1, 1)


def normalize_jacobian_componentwise(ring,
                                     J_flat, model_orm_flat, weights_flat,
                                     nHBPM, nVBPM, nHorCOR, nVerCOR, cor_kicks,
                                     fit_list, quads_ind, quads_tilt_ind, skew_ords, rf_step
                                     ):
    Mmodelsq = np.sum((model_orm_flat / weights_flat) ** 2)

    norm_factors = np.ones(J_flat.shape[1])  # default = no normalization
    J_flat_normalized = np.zeros_like(J_flat)
    idx = 0

    if 'hbpm_gain' in fit_list or 'vbpm_gain' in fit_list:
        J_bpm_gain = J_flat[:, idx:idx + nHBPM + nVBPM]
        J_flat_normalized[:, idx:idx + nHBPM + nVBPM] = J_bpm_gain  # no normalization
        idx += nHBPM + nVBPM

    if 'hbpm_coupling' in fit_list or 'vbpm_coupling' in fit_list:
        J_bpm_coupling = J_flat[:, idx:idx + nHBPM + nVBPM]
        J_flat_normalized[:, idx:idx + nHBPM + nVBPM] = J_bpm_coupling  # no normalization
        idx += nHBPM + nVBPM

    if 'hcor_cal' in fit_list or 'vcor_cal' in fit_list:
        n = nHorCOR + nVerCOR
        J_cor_cal = J_flat[:, idx:idx + n]
        cor_kicks_ = np.concatenate((cor_kicks[0], cor_kicks[1]))
        norm = 1 / cor_kicks_
        J_flat_normalized[:, idx:idx + n] = J_cor_cal / norm
        norm_factors[idx:idx + n] = norm
        idx += n

    if 'hcor_coupling' in fit_list or 'vcor_coupling' in fit_list:
        n = nHorCOR + nVerCOR
        J_cor_coupling = J_flat[:, idx:idx + n]
        norm = np.sqrt(np.sum((J_cor_coupling) ** 2, axis=0) / Mmodelsq)
        J_flat_normalized[:, idx:idx + n] = J_cor_coupling / norm[np.newaxis, :]
        norm_factors[idx:idx + n] = norm
        idx += n

    if 'HCMEnergyShift' in fit_list:
        n = nHorCOR
        J_HCMEnergyShift = J_flat[:, idx:idx + n]
        alpha_mc = get_mcf(ring)
        Frequency = fixed_parameters.Frequency
        norm = abs(alpha_mc * Frequency / rf_step)
        J_flat_normalized[:, idx:idx + n] = J_HCMEnergyShift / norm
        norm_factors[idx:idx + n] = norm
        idx += n

    if 'VCMEnergyShift' in fit_list:
        n = nVerCOR
        J_VCMEnergyShift = J_flat[:, idx:idx + n]
        alpha_mc = get_mcf(ring)
        Frequency = fixed_parameters.Frequency
        norm = abs(alpha_mc * Frequency / rf_step)

        J_flat_normalized[:, idx:idx + n] = J_VCMEnergyShift / norm
        norm_factors[idx:idx + n] = norm
        idx += n

    if 'delta_rf' in fit_list:
        J_delta_rf = J_flat[:, idx:idx + 1]
        J_flat_normalized[:, idx:idx + 1] = J_delta_rf  # already normalized
        norm_factors[idx:idx + 1] = 1
        idx += 1

    if 'quads' in fit_list:
        n = len(quads_ind)
        J_quads = J_flat[:, idx:idx + n]
        norm = np.sqrt(np.sum((J_quads) ** 2, axis=0) / Mmodelsq)
        J_flat_normalized[:, idx:idx + n] = J_quads / norm[np.newaxis, :]
        norm_factors[idx:idx + n] = norm
        idx += n

    if 'skew_quads' in fit_list:
        n = len(skew_ords)
        J_quads = J_flat[:, idx:idx + n]
        norm = np.sqrt(np.sum((J_quads) ** 2, axis=0) / Mmodelsq)
        J_flat_normalized[:, idx:idx + n] = J_quads / norm[np.newaxis, :]
        norm_factors[idx:idx + n] = norm
        idx += n

    if 'quads_tilt' in fit_list:
        n = len(quads_tilt_ind)
        J_quads = J_flat[:, idx:idx + n]
        norm = np.sqrt(np.sum((J_quads) ** 2, axis=0) / Mmodelsq)
        J_flat_normalized[:, idx:idx + n] = J_quads / norm[np.newaxis, :]
        norm_factors[idx:idx + n] = norm
        idx += n

    return J_flat_normalized, norm_factors.reshape(-1, 1)


def remove_rf_normalization(fit_list, rf_step, fit_result, nHBPM, nVBPM, nHorCOR, nVerCOR, quads_ind, quads_tilt_ind,
                            skew_ords):
    norm_factors_rf = 1 / rf_step / 10
    nf = np.asarray(norm_factors_rf).ravel()
    fit_result_unnormalized = np.zeros_like(fit_result)
    idx = 0
    if 'hbpm_gain' in fit_list or 'vbpm_gain' in fit_list:
        J_bpm_gain = fit_result[idx:idx + nHBPM + nVBPM]
        fit_result_unnormalized[idx:idx + nHBPM + nVBPM] = J_bpm_gain  # no normalization
        idx += nHBPM + nVBPM

    if 'hbpm_coupling' in fit_list or 'vbpm_coupling' in fit_list:
        J_bpm_coupling = fit_result[idx:idx + nHBPM + nVBPM]
        fit_result_unnormalized[idx:idx + nHBPM + nVBPM] = J_bpm_coupling  # no normalization
        idx += nHBPM + nVBPM

    if 'hcor_cal' in fit_list or 'vcor_cal' in fit_list:
        n = nHorCOR + nVerCOR
        J_cor_cal = fit_result[idx:idx + n]
        fit_result_unnormalized[idx:idx + n] = J_cor_cal
        idx += n

    if 'hcor_coupling' in fit_list or 'vcor_coupling' in fit_list:
        n = nHorCOR + nVerCOR
        J_cor_coupling = fit_result[idx:idx + n]
        fit_result_unnormalized[idx:idx + n] = J_cor_coupling
        idx += n

    if 'HCMEnergyShift' in fit_list:
        n = nHorCOR
        J_HCMEnergyShift = fit_result[idx:idx + n]
        fit_result_unnormalized[idx:idx + n] = J_HCMEnergyShift
        idx += n

    if 'VCMEnergyShift' in fit_list:
        n = nVerCOR
        J_VCMEnergyShift = fit_result[idx:idx + n]
        fit_result_unnormalized[idx:idx + n] = J_VCMEnergyShift
        norm_factors[idx:idx + n] = norm
        idx += n

    if 'delta_rf' in fit_list:
        J_delta_rf = fit_result[idx:idx + 1]
        nf = 1 / rf_step / 10
        fit_result_unnormalized[idx:idx + 1] = J_delta_rf / nf  # already normalized
        idx += 1

    if 'quads' in fit_list:
        n = len(quads_ind)
        J_quads = fit_result[idx:idx + n]
        fit_result_unnormalized[idx:idx + n] = J_quads
        idx += n

    if 'skew_quads' in fit_list:
        n = len(skew_ords)
        J_quads = fit_result[idx:idx + n]
        fit_result_unnormalized[idx:idx + n] = J_quads
        idx += n

    if 'quads_tilt' in fit_list:
        n = len(quads_tilt_ind)
        J_quads = fit_result[idx:idx + n]
        fit_result_unnormalized[idx:idx + n] = J_quads
        idx += n

    return fit_result_unnormalized


def rf_normalization(ring,
                     J_flat, model_orm_flat, weights_flat,
                     nHBPM, nVBPM, nHorCOR, nVerCOR, cor_kicks,
                     fit_list, quads_ind, quads_tilt_ind, skew_ords, rf_step
                     ):
    norm_factors = np.ones(J_flat.shape[1])  # default = no normalization
    J_flat_normalized = np.zeros_like(J_flat)
    idx = 0

    if 'hbpm_gain' in fit_list or 'vbpm_gain' in fit_list:
        J_bpm_gain = J_flat[:, idx:idx + nHBPM + nVBPM]
        J_flat_normalized[:, idx:idx + nHBPM + nVBPM] = J_bpm_gain  # no normalization
        idx += nHBPM + nVBPM

    if 'hbpm_coupling' in fit_list or 'vbpm_coupling' in fit_list:
        J_bpm_coupling = J_flat[:, idx:idx + nHBPM + nVBPM]
        J_flat_normalized[:, idx:idx + nHBPM + nVBPM] = J_bpm_coupling  # no normalization
        idx += nHBPM + nVBPM

    if 'hcor_cal' in fit_list or 'vcor_cal' in fit_list:
        n = nHorCOR + nVerCOR
        J_cor_cal = J_flat[:, idx:idx + n]
        cor_kicks_ = np.concatenate((cor_kicks[0], cor_kicks[1]))
        J_flat_normalized[:, idx:idx + n] = J_cor_cal  # / norm
        idx += n

    if 'hcor_coupling' in fit_list or 'vcor_coupling' in fit_list:
        n = nHorCOR + nVerCOR
        J_cor_coupling = J_flat[:, idx:idx + n]
        J_flat_normalized[:, idx:idx + n] = J_cor_coupling  # / norm[np.newaxis, :]
        idx += n

    if 'HCMEnergyShift' in fit_list:
        n = nHorCOR
        J_HCMEnergyShift = J_flat[:, idx:idx + n]
        alpha_mc = get_mcf(ring)
        Frequency = fixed_parameters.Frequency
        J_flat_normalized[:, idx:idx + n] = J_HCMEnergyShift  # / norm
        idx += n

    if 'VCMEnergyShift' in fit_list:
        n = nVerCOR
        J_VCMEnergyShift = J_flat[:, idx:idx + n]
        alpha_mc = get_mcf(ring)
        Frequency = fixed_parameters.Frequency
        J_flat_normalized[:, idx:idx + n] = J_VCMEnergyShift  # / norm
        idx += n

    if 'delta_rf' in fit_list:
        J_delta_rf = J_flat[:, idx:idx + 1]

        # Apply normalization factor to increase fitting weight of RF frequency
        normalization_factor = 1 / rf_step / 10
        print("The RF frequency parameter is normalized by 1 / rf_step / 10 to to get a better fit.")

        J_flat_normalized[:, idx:idx + 1] = J_delta_rf / normalization_factor
        norm_factors[idx:idx + 1] = normalization_factor
        idx += 1

    if 'quads' in fit_list:
        n = len(quads_ind)
        J_quads = J_flat[:, idx:idx + n]
        J_flat_normalized[:, idx:idx + n] = J_quads  # / norm[np.newaxis, :]
        idx += n

    if 'skew_quads' in fit_list:
        n = len(skew_ords)
        J_quads = J_flat[:, idx:idx + n]
        J_flat_normalized[:, idx:idx + n] = J_quads  # / norm[np.newaxis, :]
        idx += n

    if 'quads_tilt' in fit_list:
        n = len(quads_tilt_ind)
        J_quads = J_flat[:, idx:idx + n]
        J_flat_normalized[:, idx:idx + n] = J_quads  # / norm[np.newaxis, :]
        idx += n

    return J_flat_normalized, norm_factors.reshape(-1, 1)




# ============================================================================== #
#                               Constrained Quads
# ============================================================================== #


def _as_1d_scale(scale, n):
    if scale is None:
        return np.ones(n)
    arr = np.asarray(scale).ravel()
    if arr.size != n:
        raise ValueError(f"scale has size {arr.size}, expected {n}")
    return arr

def build_quad_constraint_rows(
    n_params,
    quad_slice,
    *,
    quad_sigma,
    quad_weights=None,
    quad_mask=None,
):
    """
    Build G and yc for penalty rows:
        || G * z - 0 ||^2
    where z is the SOLVER variable.

    If your solver uses normalized variables z and later converts back via
        delta_p = z / param_scale
    then the penalty on physical delta_p requires:
        G(row, col_j) = (w_k / quad_sigma) / param_scale[j]
    """
    quad_cols = np.arange(quad_slice.start, quad_slice.stop)

    if quad_mask is not None:
        quad_mask = np.asarray(quad_mask, dtype=bool)
        if quad_mask.size != quad_cols.size:
            raise ValueError(
                f"quad_mask has size {quad_mask.size}, expected {quad_cols.size}"
            )
        quad_cols = quad_cols[quad_mask]

    n_constrained = len(quad_cols)
    if n_constrained == 0:
        return None, None

    if quad_weights is None:
        wk = np.ones(n_constrained)
    else:
        quad_weights = np.asarray(quad_weights).ravel()
        if quad_mask is not None:
            if quad_weights.size != (quad_slice.stop - quad_slice.start):
                raise ValueError(
                    "quad_weights must have length equal to len(quads_ords) "
                    "before masking"
                )
            wk = quad_weights[quad_mask]
        else:
            if quad_weights.size != n_constrained:
                raise ValueError(
                    f"quad_weights has size {quad_weights.size}, expected {n_constrained}"
                )
            wk = quad_weights

    quad_sigma = np.asarray(quad_sigma).ravel()

    # --- case 1: scalar sigma ---
    if quad_sigma.size == 1:
        if quad_sigma[0] <= 0:
            raise ValueError("quad_sigma must be > 0")
        sigma_vec = np.full(n_constrained, quad_sigma[0])

    # --- case 2: per-quad sigma ---
    else:
        if quad_sigma.size != (quad_slice.stop - quad_slice.start):
            raise ValueError(
                f"quad_sigma has size {quad_sigma.size}, expected {(quad_slice.stop - quad_slice.start)}"
            )

        if quad_mask is not None:
            sigma_vec = quad_sigma[quad_mask]
        else:
            sigma_vec = quad_sigma

        if np.any(sigma_vec <= 0):
            raise ValueError("All quad_sigma values must be > 0")

    # --- build constraint ---
    diag_vals = wk / sigma_vec

    G = np.zeros((n_constrained, n_params), dtype=float)
    G[np.arange(n_constrained), quad_cols] = diag_vals

    yc = np.zeros((n_constrained, 1), dtype=float)


    return G, yc


def build_constraint_rows(
    n_params,
    param_slice,
    *,
    sigma,
    weights=None,
    mask=None,
    param_scale=None,
):
    cols_all = np.arange(param_slice.start, param_slice.stop)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)

        if mask.size != len(cols_all):
            raise ValueError(
                f"mask has size {mask.size}, expected {len(cols_all)}"
            )

        cols = cols_all[mask]
    else:
        cols = cols_all

    n = len(cols)

    if n == 0:
        return None, None

    # --------------------------------------------------
    # Weights
    # --------------------------------------------------
    if weights is None:
        wk = np.ones(n)
    else:
        weights = np.asarray(weights, dtype=float).ravel()

        if weights.size != len(cols_all):
            raise ValueError(
                f"weights has size {weights.size}, "
                f"expected {len(cols_all)}"
            )

        wk = weights[mask] if mask is not None else weights

    # --------------------------------------------------
    # Sigma in PHYSICAL parameter units
    # --------------------------------------------------
    sigma = np.asarray(sigma, dtype=float).ravel()

    if sigma.size == 1:
        sigma_vec = np.full(n, sigma[0])
    else:
        if sigma.size != len(cols_all):
            raise ValueError(
                f"sigma has size {sigma.size}, "
                f"expected {len(cols_all)}"
            )

        sigma_vec = sigma[mask] if mask is not None else sigma

    if np.any(sigma_vec <= 0):
        raise ValueError("All constraint sigma values must be > 0")

    # --------------------------------------------------
    # Solver -> physical scaling
    #
    # physical_delta = solver_delta / param_scale
    # --------------------------------------------------
    if param_scale is None:
        scale_vec = np.ones(n)
    else:
        param_scale = np.asarray(param_scale, dtype=float).ravel()

        if param_scale.size != n_params:
            raise ValueError(
                f"param_scale has size {param_scale.size}, "
                f"expected {n_params}"
            )

        scale_vec = param_scale[cols]

        if np.any(scale_vec == 0):
            raise ValueError("Constraint param_scale contains zero")

    # --------------------------------------------------
    # Constraint in SOLVER coordinates
    #
    # physical_delta = z / scale
    #
    # (w/sigma) * physical_delta
    # = w/(sigma*scale) * z
    # --------------------------------------------------
    diag_vals = wk / (sigma_vec * scale_vec)

    G = np.zeros((n, n_params), dtype=float)
    G[np.arange(n), cols] = diag_vals

    yc = np.zeros((n, 1), dtype=float)

    return G, yc

def augment_system_with_constraints(
    Jw,
    y,
    blocks,
    fit_list,
    *,
    constraint_cfg=None,
    param_scale=None,
):
    """
    Build constraint rows for the fitted parameter blocks.

    Supported constraints:
        - quads
        - skew_quads
        - hbpm_gain
        - vbpm_gain
        - hcor_cal
        - vcor_cal

    The BPM-gain and corrector-calibration constraints are only added
    when their corresponding weights in ConstraintConfig are not None.
    """

    # ------------------------------------------------------------
    # Constraints disabled
    # ------------------------------------------------------------
    if constraint_cfg is None or not constraint_cfg.enable:
        return Jw, y, None, None

    G_list = []
    yc_list = []

    # ============================================================
    # Normal quadrupoles
    # ============================================================
    if (
        "quads" in fit_list
        and "quads" in blocks
    ):
        Gq, yq = build_constraint_rows(
            n_params=Jw.shape[1],
            param_slice=blocks["quads"],
            sigma=constraint_cfg.quad_sigma,
            weights=constraint_cfg.quad_weights,
            mask=constraint_cfg.quad_mask,
            param_scale=param_scale,
        )

        if Gq is not None:
            G_list.append(Gq)
            yc_list.append(yq)

    # ============================================================
    # Skew quadrupoles
    # ============================================================
    if (
        "skew_quads" in fit_list
        and "skew_quads" in blocks
    ):
        Gs, ys = build_constraint_rows(
            n_params=Jw.shape[1],
            param_slice=blocks["skew_quads"],
            sigma=constraint_cfg.skew_sigma,
            weights=constraint_cfg.skew_weights,
            mask=constraint_cfg.skew_mask,
            param_scale=param_scale,
        )

        if Gs is not None:
            G_list.append(Gs)
            yc_list.append(ys)

    # ============================================================
    # Horizontal BPM gains
    # ============================================================
    if (
        "hbpm_gain" in fit_list
        and "hbpm_gain" in blocks
        and constraint_cfg.hbpm_gain_weights is not None
    ):
        Gh, yh = build_constraint_rows(
            n_params=Jw.shape[1],
            param_slice=blocks["hbpm_gain"],
            sigma=constraint_cfg.hbpm_gain_sigma,
            weights=constraint_cfg.hbpm_gain_weights,
            mask=constraint_cfg.hbpm_gain_mask,
            param_scale=param_scale,
        )

        if Gh is not None:
            G_list.append(Gh)
            yc_list.append(yh)

    # ============================================================
    # Vertical BPM gains
    # ============================================================
    if (
        "vbpm_gain" in fit_list
        and "vbpm_gain" in blocks
        and constraint_cfg.vbpm_gain_weights is not None
    ):
        Gv, yv = build_constraint_rows(
            n_params=Jw.shape[1],
            param_slice=blocks["vbpm_gain"],
            sigma=constraint_cfg.vbpm_gain_sigma,
            weights=constraint_cfg.vbpm_gain_weights,
            mask=constraint_cfg.vbpm_gain_mask,
            param_scale=param_scale,
        )

        if Gv is not None:
            G_list.append(Gv)
            yc_list.append(yv)

    # ============================================================
    # Horizontal corrector calibration
    # ============================================================
    if (
        "hcor_cal" in fit_list
        and "hcor_cal" in blocks
        and constraint_cfg.hcor_cal_weights is not None
    ):
        Ghc, yhc = build_constraint_rows(
            n_params=Jw.shape[1],
            param_slice=blocks["hcor_cal"],
            sigma=constraint_cfg.hcor_cal_sigma,
            weights=constraint_cfg.hcor_cal_weights,
            mask=constraint_cfg.hcor_cal_mask,
            param_scale=param_scale,
        )

        if Ghc is not None:
            G_list.append(Ghc)
            yc_list.append(yhc)

    # ============================================================
    # Vertical corrector calibration
    # ============================================================
    if (
        "vcor_cal" in fit_list
        and "vcor_cal" in blocks
        and constraint_cfg.vcor_cal_weights is not None
    ):
        Gvc, yvc = build_constraint_rows(
            n_params=Jw.shape[1],
            param_slice=blocks["vcor_cal"],
            sigma=constraint_cfg.vcor_cal_sigma,
            weights=constraint_cfg.vcor_cal_weights,
            mask=constraint_cfg.vcor_cal_mask,
            param_scale=param_scale,
        )

        if Gvc is not None:
            G_list.append(Gvc)
            yc_list.append(yvc)

    # ------------------------------------------------------------
    # No active constraints
    # ------------------------------------------------------------
    if not G_list:
        return Jw, y, None, None

    # ------------------------------------------------------------
    # Combine all constraint rows
    # ------------------------------------------------------------
    G = np.vstack(G_list)
    yc = np.vstack(yc_list)

    print(
        f"[Constraints] G={G.shape} | "
        f"active rows={G.shape[0]} | "
        f"solver parameters={G.shape[1]}"
    )


    return Jw, y, G, yc



# ============================================================================== #
#                               LOCO Minimization
# ============================================================================== #


def _build_C_matrix(hbpm_gain, hbpm_coupling, vbpm_coupling, vbpm_gain):
    C11 = np.diag(np.asarray(hbpm_gain).ravel())
    C12 = np.diag(np.asarray(hbpm_coupling).ravel())
    C21 = np.diag(np.asarray(vbpm_coupling).ravel())
    C22 = np.diag(np.asarray(vbpm_gain).ravel())
    return np.block([[C11, C12], [C21, C22]])


def _svd_select_indices(
        S,
        U,
        Vh,
        y,
        J_weighted,
        weights_flat,
        model_orm_flat,
        measured_orm_flat,
        method="threshold",
        svd_threshold=1e-7,
        cut_=None,
        interactive=False,
        show_plot=False,
        iteration_tag=""
):
    """Return indices Ivec of singular values to keep."""
    if method == "threshold":
        Ivec = np.where(S > svd_threshold * np.max(S))[0]
        print(
        f"[SVD] threshold={svd_threshold:.3e} | "
        f"kept={len(Ivec)}/{len(S)} | "
        f"cut={len(S)-len(Ivec)} | "
        f"smallest kept={S[Ivec[-1]]/S[0]:.3e}"
    )
    elif method == "user_input" and cut_ is not None:
        Ivec = np.arange(min(cut_, len(S)))
    elif method == "interactive" or interactive:
        sv_indices = np.arange(len(S))
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1);
        plt.semilogy(sv_indices, S, '.-');
        plt.xlabel("SV idx");
        plt.ylabel("SV")
        plt.subplot(1, 2, 2);
        plt.plot(sv_indices, S / np.max(S), '.-');
        plt.xlabel("SV idx");
        plt.ylabel("SV/max")
        plt.tight_layout();
        plt.show();
        time.sleep(0.5)
        user = input("Enter indices (e.g. 0:20 or 0,1,2): ")
        if ':' in user:
            a, b = user.split(':');
            Ivec = np.arange(int(a), int(b))
        else:
            Ivec = np.array([int(x.strip()) for x in user.split(',')])
        Ivec = Ivec[Ivec < len(S)]


    elif method == "rank":

        print("Performing rank-based singular value selection...")
        ChiSquareVector = np.full(len(S), np.nan)
        Ivec = None

        for i in reversed(range(1, len(S) + 1)):
            try:
                Amod = U[:, :i] @ np.diag(S[:i])
                b = np.linalg.lstsq(Amod, y, rcond=None)[0]
                b = Vh.T[:, :i] @ b

                Mfit = weights_flat * (J_weighted @ b)
                Mmodelnew = model_orm_flat + Mfit

                chi2 = np.sum(((measured_orm_flat - Mmodelnew) / weights_flat) ** 2) / len(weights_flat)
                ChiSquareVector[i - 1] = chi2

                np.linalg.inv(Amod.T @ Amod)

                Ivec = np.arange(i)
                print(f"Rank-based SVD selected rank is {i}")
                break

            except np.linalg.LinAlgError:
                continue

        if Ivec is None:
            raise RuntimeError("Rank-based selection failed: no stable solution found.")

    if show_plot:
        sv_indices = np.arange(len(S))
        unused = np.setdiff1d(sv_indices, Ivec)

        plt.figure(figsize=(10, 3))

        # used singular values
        plt.semilogy(Ivec, S[Ivec], 'o', color="green", label="Used", markersize=6)

        # cut singular values
        if len(unused):
            plt.semilogy(unused, S[unused], 'o', color="red", label="Cut", markersize=6)

        plt.title("SVD Spectrum", fontsize=12)
        plt.xlabel("Index")
        plt.ylabel("Singular Value")
        plt.legend(fontsize=10)
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)
        plt.tight_layout()
        plt.show()

    return Ivec


def _prepare_ring_and_rmconfig(
        base_ring, fit_vec, *, fit_list, nHBPM, nVBPM, nHorCOR, nVerCOR,
        quads_ords, quads_tilt_ind, skew_ords, quad_individuals,
        skew_individuals, tilt_individuals, fit_cfg,
        used_bpms_ords, used_cor_ords, CMstep, rfStep,
        HCMCoupling, VCMCoupling,
        hbpm_gain, hbpm_coupling, vbpm_coupling, vbpm_gain,
        HCMEnergyShift, VCMEnergyShift, includeDispersion,
        response_matrix_calculator="Linear",
):
    """
    Build a *temporary* ring with the trial fit applied, and an RMConfig
    """

    prop = _pack_fit_dict(
        fit_vec, fit_list, nHBPM, nVBPM, nHorCOR, nVerCOR,
        n_quads=len(quads_ords) if quads_ords is not None else 0,
        n_skew=len(skew_ords) if skew_ords is not None else 0,
        n_quads_tilt=len(quads_tilt_ind) if quads_tilt_ind is not None else 0
    )

    # Copy ring and apply trial lattice changes
    import copy
    ring_tmp = copy.deepcopy(base_ring)
    _apply_fit_to_ring(ring_tmp, prop, quads_ords, quads_tilt_ind, skew_ords, quad_individuals,
    skew_individuals,
    tilt_individuals, fit_cfg)

    dkick_H = np.asarray(prop.get('hcor_cal', CMstep[0]), dtype=float).ravel()
    dkick_V = np.asarray(prop.get('vcor_cal', CMstep[1]), dtype=float).ravel()
    dkick = [dkick_H, dkick_V]

    hbpm_gain = prop.get('hbpm_gain', hbpm_gain)
    vbpm_gain = prop.get('vbpm_gain', vbpm_gain)
    hbpm_coupling = prop.get('hbpm_coupling', hbpm_coupling)
    vbpm_coupling = prop.get('vbpm_coupling', vbpm_coupling)

    cfg = RMConfig(
        dkick=dkick,
        bpm_ords=used_bpms_ords,
        cm_ords=used_cor_ords,
        HCMCoupling=prop.get('hcor_coupling', HCMCoupling),
        VCMCoupling=prop.get('vcor_coupling', VCMCoupling),
        rfStep=float(np.asarray(prop.get('delta_rf', rfStep)).ravel()[0]),
        includeDispersion=includeDispersion,
        calculator=response_matrix_calculator,
    )

    Cmat = _build_C_matrix(
        prop.get('hbpm_gain', hbpm_gain),
        prop.get('hbpm_coupling', hbpm_coupling),
        prop.get('vbpm_coupling', vbpm_coupling),
        prop.get('vbpm_gain', vbpm_gain),
    )

    Hshift = np.asarray(prop.get('HCMEnergyShift', HCMEnergyShift), dtype=float).ravel()
    Vshift = np.asarray(prop.get('VCMEnergyShift', VCMEnergyShift), dtype=float).ravel()

    return ring_tmp, cfg, Cmat, Hshift, Vshift, prop


def _pack_fit_dict(vec, fit_list, nHBPM, nVBPM, nHorCOR, nVerCOR, n_quads, n_skew, n_quads_tilt):
    d = {};
    i = 0
    if 'hbpm_gain' in fit_list: d['hbpm_gain'] = vec[i:i + nHBPM]; i += nHBPM
    if 'hbpm_coupling' in fit_list: d['hbpm_coupling'] = vec[i:i + nHBPM]; i += nHBPM
    if 'vbpm_coupling' in fit_list: d['vbpm_coupling'] = vec[i:i + nVBPM]; i += nVBPM
    if 'vbpm_gain' in fit_list: d['vbpm_gain'] = vec[i:i + nVBPM]; i += nVBPM
    if 'hcor_cal' in fit_list: d['hcor_cal'] = vec[i:i + nHorCOR]; i += nHorCOR
    if 'vcor_cal' in fit_list: d['vcor_cal'] = vec[i:i + nVerCOR]; i += nVerCOR
    if 'hcor_coupling' in fit_list: d['hcor_coupling'] = vec[i:i + nHorCOR]; i += nHorCOR
    if 'vcor_coupling' in fit_list: d['vcor_coupling'] = vec[i:i + nVerCOR]; i += nVerCOR
    if 'HCMEnergyShift' in fit_list: d['HCMEnergyShift'] = vec[i:i + nHorCOR]; i += nHorCOR
    if 'VCMEnergyShift' in fit_list: d['VCMEnergyShift'] = vec[i:i + nVerCOR]; i += nVerCOR
    if 'delta_rf' in fit_list: d['delta_rf'] = vec[i:i + 1];       i += 1
    if 'quads' in fit_list: d['quads'] = vec[i:i + n_quads]; i += n_quads
    if 'skew_quads' in fit_list: d['skew_quads'] = vec[i:i + n_skew];  i += n_skew
    if 'quads_tilt' in fit_list: d['quads_tilt'] = vec[i:i + n_quads_tilt]; i += n_quads_tilt
    return d


def build_full_fit_state(
        *,
        hbpm_gain, vbpm_gain,
        hbpm_coupling, vbpm_coupling,
        HCMCoupling, VCMCoupling,
        HCMEnergyShift, VCMEnergyShift,
        rfStep,
        quads, skew_quads, quads_tilt,
        CMstep
):
    """
    Cumulative LOCO fit state (all parameters, all stages)
    """
    return {
        "hbpm_gain": np.asarray(hbpm_gain).copy(),
        "vbpm_gain": np.asarray(vbpm_gain).copy(),
        "hbpm_coupling": np.asarray(hbpm_coupling).copy(),
        "vbpm_coupling": np.asarray(vbpm_coupling).copy(),
        "hcor_coupling": np.asarray(HCMCoupling).copy(),
        "vcor_coupling": np.asarray(VCMCoupling).copy(),
        "HCMEnergyShift": np.asarray(HCMEnergyShift).copy(),
        "VCMEnergyShift": np.asarray(VCMEnergyShift).copy(),
        "delta_rf": float(rfStep),
        "quads": np.asarray(quads).copy() if quads is not None else None,
        "skew_quads": np.asarray(skew_quads).copy() if skew_quads is not None else None,
        "quads_tilt": np.asarray(quads_tilt).copy() if quads_tilt is not None else None,
        "hcor_cal": np.asarray(CMstep[0]).copy(),
        "vcor_cal": np.asarray(CMstep[1]).copy(),
    }


def _apply_fit_to_ring(ring, fit_dict, quads_ords, quads_tilt_ind, skew_ords, quad_individuals,skew_individuals,tilt_individuals, fit_cfg):
    if 'quads' in fit_dict:
        set_correction(ring, np.asarray(fit_dict['quads']).ravel(),
                       quads_ords, individuals=quad_individuals, block='quads', config=fit_cfg)
    if 'skew_quads' in fit_dict:
        set_correction(ring, np.asarray(fit_dict['skew_quads']).ravel(),
                       skew_ords, individuals=skew_individuals, block='skew_quads', config=fit_cfg)
    if 'quads_tilt' in fit_dict:
        set_correction_tilt(ring, np.asarray(fit_dict['quads_tilt']).ravel(),
                            quads_tilt_ind, individuals=tilt_individuals, config=fit_cfg)


# ----------------------- SOLVERS -----------------------

def solve_step_gn(
        J_weighted, y,
        svd_method, svd_threshold, cut_, show_plot, tag,
        weights_flat, model_orm_flat, measured_orm_flat
):
    # Reduced SVD is correct here
    U, S, Vh = np.linalg.svd(J_weighted, full_matrices=False)

    Ivec = _svd_select_indices(
        S,
        U=U,
        Vh=Vh,
        y=y,
        J_weighted=J_weighted,
        weights_flat=weights_flat,
        model_orm_flat=model_orm_flat,
        measured_orm_flat=measured_orm_flat,
        method=svd_method,
        svd_threshold=svd_threshold,
        cut_=cut_,
        show_plot=show_plot,
        iteration_tag=tag
    )

    # GN solution
    b = U[:, Ivec].T @ y
    b = np.diag(1.0 / S[Ivec]) @ b
    fit_results = Vh.T[:, Ivec] @ b

    return fit_results.ravel(), Ivec, S

def solve_step_lm(
        J_weighted, y, weights_flat,
        model_orm_flat, measured_orm_flat,
        scaled=True, Starting_Lambda=1e-3,
        max_lm_lambda=15,
        svd_method='threshold',
        svd_threshold=1e-3,
        cut_=None, show_plot=False, tag="",
        constraint_cfg=None,
        blocks=None,
        fit_list=None,
        param_scale=None,
):
    """
    Solve one Levenberg-Marquardt step.

    The unconstrained problem is

        min_delta ||J_weighted @ delta - y||^2

    and, when constraints are enabled,

        min_delta (
            ||J_weighted @ delta - y||^2
            + ||G @ delta - yc||^2
        )

    The corresponding normal equations are

        (J.T @ J + G.T @ G) delta
            = J.T @ y + G.T @ yc

    before LM damping is added.
    """

    # ============================================================
    # 1. Build constraints
    # ============================================================

    G = None
    yc = None

    if constraint_cfg is not None:
        _, _, G, yc = augment_system_with_constraints(
            J_weighted,
            y,
            blocks,
            fit_list,
            constraint_cfg=constraint_cfg,
            param_scale=param_scale,
        )

    # ============================================================
    # 2. Construct normal equations from measured data
    # ============================================================

    lam = Starting_Lambda

    C_data = J_weighted.T @ J_weighted
    ay = J_weighted.T @ y

    # Start with the data contribution
    C = C_data.copy()

    # ============================================================
    # 3. Add constraint contribution
    # ============================================================

    if G is not None:
        C_constraint = G.T @ G

        C += C_constraint
        ay += G.T @ yc
    else:
        C_constraint = None

    # ============================================================
    # 4. Add LM damping
    # ============================================================

    if scaled:
        C_lm = C + lam * np.diag(np.diag(C))
    else:
        C_lm = C + lam * np.eye(C.shape[0])

    # ============================================================
    # 5. SVD of LM system
    # ============================================================

    Uc, Sc, Vhc = np.linalg.svd(
        C_lm,
        full_matrices=False
    )

    Ivec = _svd_select_indices(
        Sc,
        U=Uc,
        Vh=Vhc,
        y=y,
        J_weighted=J_weighted,
        weights_flat=weights_flat,
        model_orm_flat=model_orm_flat,
        measured_orm_flat=measured_orm_flat,
        method=svd_method,
        svd_threshold=svd_threshold,
        cut_=cut_,
        show_plot=show_plot,
        iteration_tag=tag,
    )

   
    # ============================================================
    # 6. Solve constrained LM system
    # ============================================================

    b = Uc[:, Ivec].T @ ay

    b = np.diag(
        1.0 / Sc[Ivec]
    ) @ b

    fit_results = Vhc.T[:, Ivec] @ b

    return fit_results.ravel(), lam, Ivec, Sc

# ----------------------- MASTER pyloco FUNCTION -----------------------

def pyloco(
        ring,
        *,
        algorithm="lm",  # "lm" or "gn"
        nIter=3,
        # indices & number of elements
        used_bpms_ords=None, used_cor_ords=None, quads_ords=None, skew_ords=None, CAVords=None,
        nHBPM=None, nVBPM=None, nHorCOR=None, nVerCOR=None, quads_tilt_ind=None, inetial_fit_parameters=None,
        # measurment data
        orm_measured=None, weights=None, includeDispersion=False,
        measured_eta_x=None, measured_eta_y=None,
        hor_dispersion_weight=1.0, ver_dispersion_weight=1.0,
        # Correctors kicks & RF steps
        CMstep=None, rfStep=None, Frequency=fixed_parameters.Frequency,
        # features
        fit_list=(), quad_individuals=True,
            skew_individuals=True,
            tilt_individuals=True, remove_coupling_=True,
        # outliers & normalization
        outlier_rejection=False, sigma_outlier=10,
        apply_normalization=False, normalization_mode='global',
        # SVD selection
        svd_selection_method='threshold', svd_threshold=1e-7, cut_=None,
        show_svd_plot=False,
        constraint_cfg=None,
        # LM options
        nLMIter=10, Starting_Lambda=1e-3, max_lm_lambda=15, scaled=True,
        # more options
        plot_fit_parameters=False, auto_correct_delta=True, fixedpathlength=True, fixedmomentum=False,
        fit_cfg=None,
        # Jacopians files
        quad_jacobian_file=None,
        skew_jacobian_file=None,
        quads_tilt_jacobian_file=None,
        quad_jacobian_calculator="Numerical",
        skew_jacobian_calculator="Numerical",
        response_matrix_calculator="Linear",
        analytical_thick_quadrupole=True,
        analytical_thick_steerers=False,
        analytical_verbose=False,
        analytical_use_mp=False,
        analytical_thick_skew=True,
        analytical_skew_thick_steerers=False,
        analytical_skew_verbose=False,
        analytical_skew_use_mp=False,
        force_recompute=True,
        # Fit multi stage
        continue_from_previous=False,
        previous_fit_results=None,
        previous_fit_dict=None,
        previous_ring=None,
        calculate_delta_chi2=False,
        initial_model_orm_callback=None,
        initial_chi2_callback=None,
        iteration_metrics_callback=None,
        calculator_trace_callback=None,
        output_dir='output',
        save_jacobians=False,

):

    calculator_plan = _calculator_execution_plan(
        response_matrix_calculator, quad_jacobian_calculator
    )
    response_matrix_calculator = calculator_plan["response_matrix_calculator"]
    _trace_calculator(
        calculator_trace_callback, "calculator_configuration", calculator_plan
    )

    if fit_cfg is None:
        fit_cfg = FitInitConfig(fit_list=fit_list, CMstep=CMstep, rfStep=rfStep,
                                individuals=quad_individuals)
    if CMstep is not None:
        CMstep = [np.asarray(step, dtype=float).copy() for step in CMstep]
    if continue_from_previous:
        if previous_ring is None and previous_fit_dict is None and previous_fit_results is None:
            raise ValueError(
                "continue_from_previous requires previous_ring, previous_fit_dict, "
                "or previous_fit_results"
            )
        # Activate the fitted lattice before constructing lattice-backed
        # initial parameters (quadrupoles, skews, and tilts).
        if previous_ring is not None:
            ring = previous_ring
        if isinstance(previous_fit_dict, np.ndarray) and previous_fit_dict.shape == ():
            previous_fit_dict = previous_fit_dict.item()


    hbpm_gain = np.ones(nHBPM)
    vbpm_gain = np.ones(nVBPM)
    hbpm_coupling = np.zeros(nHBPM)
    vbpm_coupling = np.zeros(nVBPM)
    HCMEnergyShift = np.zeros(nHorCOR)
    VCMEnergyShift = np.zeros(nVerCOR)
    HCMCoupling = np.zeros(nHorCOR)
    VCMCoupling = np.zeros(nVerCOR)

    deltaqt = np.zeros(len(quads_tilt_ind)) if ('quads_tilt' in fit_list) else None


    quads_fit, _ = build_initial_fit_parameters(
    ring=ring,
    fit_list=["quads"],
    nHBPM=nHBPM,
    nVBPM=nVBPM,
    nHorCOR=nHorCOR,
    nVerCOR=nVerCOR,
    quads_ords=quads_ords,
    skew_ords=skew_ords,
    quads_tilt=quads_tilt_ind,
    CMstep=CMstep,
    rfStep=rfStep,
    quad_individuals=quad_individuals,
    skew_individuals=skew_individuals,
    tilt_individuals=tilt_individuals,
    config=fit_cfg,
    )

    skew_quads_fit, _ = build_initial_fit_parameters(
        ring=ring,
        fit_list=["skew_quads"],
        nHBPM=nHBPM,
        nVBPM=nVBPM,
        nHorCOR=nHorCOR,
        nVerCOR=nVerCOR,
        quads_ords=quads_ords,
        skew_ords=skew_ords,
        quads_tilt=quads_tilt_ind,
        CMstep=CMstep,
        rfStep=rfStep,
        quad_individuals=quad_individuals,
        skew_individuals=skew_individuals,
        tilt_individuals=tilt_individuals,
        config=fit_cfg,
    )

    iOut_coupled_persistent = np.array([], dtype=int)
    iNoCoupling_chi_persistent = np.array([], dtype=int)
    chi2_history = []

    # --- Resume from previous fit if requested ---
    if continue_from_previous:
        print("[pyloco] Continuing from previous iteration set...")
        if previous_fit_results is not None:
            current_fit_parameters = np.asarray(previous_fit_results[-1]).copy()
        if previous_fit_dict is not None and len(previous_fit_dict) > 0:
            last_fit = last_by_sorted_key(previous_fit_dict)
            # Restore key parameters
            hbpm_gain = last_fit.get("hbpm_gain", hbpm_gain)
            vbpm_gain = last_fit.get("vbpm_gain", vbpm_gain)
            hbpm_coupling = last_fit.get("hbpm_coupling", hbpm_coupling)
            vbpm_coupling = last_fit.get("vbpm_coupling", vbpm_coupling)
            HCMCoupling = last_fit.get("hcor_coupling", HCMCoupling)
            VCMCoupling = last_fit.get("vcor_coupling", VCMCoupling)
            HCMEnergyShift = last_fit.get("HCMEnergyShift", HCMEnergyShift)
            VCMEnergyShift = last_fit.get("VCMEnergyShift", VCMEnergyShift)
            if "delta_rf" in last_fit:
                rfStep = float(np.asarray(last_fit["delta_rf"]).ravel()[0])
            if "quads_tilt" in last_fit:
                deltaqt = np.asarray(last_fit["quads_tilt"]).ravel()
            if 'hcor_cal' in last_fit:
                CMstep[0] = np.asarray(last_fit['hcor_cal']).ravel()
            if 'vcor_cal' in last_fit:
                CMstep[1] = np.asarray(last_fit['vcor_cal']).ravel()
            if 'quads' in last_fit:
                quads_fit = np.asarray(last_fit['quads']).ravel()
            if 'skew_quads' in last_fit:
                skew_quads_fit = np.asarray(last_fit['skew_quads']).ravel()
            if previous_ring is None:
                _apply_fit_to_ring(
                    ring, last_fit, quads_ords, quads_tilt_ind, skew_ords,
                    quad_individuals, skew_individuals, tilt_individuals, fit_cfg,
                )

    if fixedmomentum and \
            'HCMEnergyShift' not in fit_list and \
            'VCMEnergyShift' not in fit_list:
        fixedmomentum = True
        # fixedpathlength = True

    if fixedpathlength == False or 'HCMEnergyShift' in fit_list or 'VCMEnergyShift' in fit_list:
        fixedmomentum = True

    if inetial_fit_parameters is None and (not continue_from_previous or not previous_fit_dict):
        inetial_fit_parameters, blocks = build_initial_fit_parameters(
            ring=ring,
            fit_list=fit_list,
            nHBPM=nHBPM, nVBPM=nVBPM, nHorCOR=nHorCOR, nVerCOR=nVerCOR,
            quads_ords=quads_ords, skew_ords=skew_ords, quads_tilt=quads_tilt_ind,
            CMstep=CMstep, rfStep=rfStep,
            quad_individuals=quad_individuals,
            skew_individuals=skew_individuals,
            tilt_individuals=tilt_individuals,
            config=fit_cfg,
            )
    # elif continue_from_previous and previous_fit_results is not None:
    #     inetial_fit_parameters = np.asarray(previous_fit_results[-1]).copy()
    elif continue_from_previous and previous_fit_dict is not None:
        print("[pyloco] Building initial vector from previous stage...")
        last_fit = last_by_sorted_key(previous_fit_dict)

        # Build new vector with the current fit_list structure
        inetial_fit_parameters, blocks = build_initial_fit_parameters(
            ring=ring,
            fit_list=fit_list,
            nHBPM=nHBPM, nVBPM=nVBPM, nHorCOR=nHorCOR, nVerCOR=nVerCOR,
            quads_ords=quads_ords, skew_ords=skew_ords, quads_tilt=quads_tilt_ind,
            CMstep=CMstep, rfStep=rfStep,
            quad_individuals=quad_individuals,
            skew_individuals=skew_individuals,
            tilt_individuals=tilt_individuals,
            config=fit_cfg,
        )

        # Overwrite any parameters that were fitted before
        for key in fit_list:
            if key in last_fit and key in blocks:
                print(f"[pyloco] Restoring previous values for {key}...")
                arr = np.asarray(last_fit[key]).ravel()
                sl = blocks[key]  # use the slice directly
                expected = sl.stop - sl.start
                if arr.size != expected:
                    raise ValueError(
                        f"Previous block {key!r} has {arr.size} values; expected {expected}"
                    )
                inetial_fit_parameters[sl] = arr

    if (continue_from_previous and previous_fit_results is not None
            and not previous_fit_dict and inetial_fit_parameters is not None):
        previous_vector = np.asarray(previous_fit_results[-1]).ravel()
        expected = np.asarray(inetial_fit_parameters).size
        if previous_vector.size != expected:
            raise ValueError(
                "The previous fit vector does not match the current fit list; "
                "provide previous_fit_dict to resume across different fit lists"
            )
        inetial_fit_parameters = previous_vector.copy()

    inetial_fit_parameters = np.asarray(inetial_fit_parameters).ravel()
    current_fit_parameters = inetial_fit_parameters.copy()

    p_initial = inetial_fit_parameters.copy()
    delta_chi2_history = []
    group_delta_history = []

    # histories
    fit_results_all = []
    fit_dict_all = {}

    # ------- Outer iterations -------
    iterations_started = time.perf_counter()
    for it in range(nIter):
        iteration_started = time.perf_counter()
        trial_orm_seconds = 0.0
        print(f"\n==== Iteration {it + 1}/{nIter} – {algorithm.upper()} ====")
        # --- 1) ORM model ---
        cfg = RMConfig(dkick=CMstep, bpm_ords=used_bpms_ords, cm_ords=used_cor_ords,
                       HCMCoupling=HCMCoupling, VCMCoupling=VCMCoupling, rfStep=rfStep,
                       includeDispersion=includeDispersion,
                       calculator=response_matrix_calculator)
        orm_started = time.perf_counter()
        _trace_calculator(calculator_trace_callback, "main_model_orm", cfg.calculator)
        orm_model = response_matrix(ring, config=cfg)
        model_orm_seconds = time.perf_counter() - orm_started

        Cmat = _build_C_matrix(hbpm_gain, hbpm_coupling, vbpm_coupling, vbpm_gain)
        orm_model = Cmat @ orm_model

        # --- 2) Jacobian ---
        include_quads = ('quads' in fit_list)
        include_skew = ('skew_quads' in fit_list)
        include_quads_tilt = ('quads_tilt' in fit_list)
        include_bpm_gain = ('hbpm_gain' in fit_list) or ('vbpm_gain' in fit_list)
        include_cor_kick = ('hcor_cal' in fit_list) or ('vcor_cal' in fit_list)
        include_cor_coupling = ('hcor_coupling' in fit_list) or ('vcor_coupling' in fit_list)
        include_bpm_coupling = ('hbpm_coupling' in fit_list) or ('vbpm_coupling' in fit_list)
        include_HCMEnergyShift = ('HCMEnergyShift' in fit_list)
        include_VCMEnergyShift = ('VCMEnergyShift' in fit_list)
        include_delta_RF = ('delta_rf' in fit_list)

        jacobian_started = time.perf_counter()
        Jfull, dq, dskew, dtilt = compute_jacobian(
            ring, C_model=orm_model, dkick=CMstep,
            bpm_indexes=used_bpms_ords, CMords=used_cor_ords, quads_ind=quads_ords,
            nHorCOR=nHorCOR, nVerCOR=nVerCOR, nHBPM=nHBPM, nVBPM=nVBPM,
            C=Cmat, CAVords=CAVords,
            dk=fixed_parameters.dk,
            skew_ind=skew_ords, quads_tilt_ind=quads_tilt_ind,
            includeDispersion=includeDispersion,
            quad_individuals=quad_individuals,
            skew_individuals=skew_individuals,
            tilt_individuals=tilt_individuals,
            delta_skew_=fixed_parameters.delta_skew,
            delta_q_tilt=getattr(fixed_parameters, "delta_q_tilt", 1e-6),
            include_quads=include_quads,
            include_skew=include_skew,
            include_bpm_gain=include_bpm_gain,
            include_cor_kick=include_cor_kick,
            include_cor_coupling=include_cor_coupling,
            include_quads_tilt=include_quads_tilt,
            include_bpm_coupling=include_bpm_coupling,
            include_delta_RF_frequency=include_delta_RF,
            include_HCMEnergyShift=include_HCMEnergyShift,
            include_VCMEnergyShift=include_VCMEnergyShift,
            rf_step=rfStep,
            auto_correct_delta=auto_correct_delta,
            VCMCoupling=VCMCoupling, HCMCoupling=HCMCoupling,
            measured_eta_x=measured_eta_x, measured_eta_y=measured_eta_y,
            quads_tilt_fit=(deltaqt if deltaqt is not None else None),
            fit_cfg=fit_cfg,
            Frequency=Frequency,
            iteration=it + 1,
            quad_jacobian_file=quad_jacobian_file,
            skew_jacobian_file=skew_jacobian_file,
            quad_jacobian_calculator=quad_jacobian_calculator,
            skew_jacobian_calculator=skew_jacobian_calculator,
            analytical_thick_quadrupole=analytical_thick_quadrupole,
            analytical_thick_steerers=analytical_thick_steerers,
            analytical_verbose=analytical_verbose,
            analytical_use_mp=analytical_use_mp,
            analytical_thick_skew=analytical_thick_skew,
            analytical_skew_thick_steerers=analytical_skew_thick_steerers,
            analytical_skew_verbose=analytical_skew_verbose,
            analytical_skew_use_mp=analytical_skew_use_mp,
            response_matrix_calculator=response_matrix_calculator,
            calculator_trace_callback=calculator_trace_callback,
            quads_tilt_jacobian_file=quads_tilt_jacobian_file,
            force_recompute=force_recompute,
            output_dir=output_dir,
            save_jacobians=save_jacobians,


        )
        jacobian_seconds = time.perf_counter() - jacobian_started


        if fixedmomentum == True:

            AlphaMCF = get_mcf(ring)

            eta_x_mcf = -AlphaMCF * Frequency * measured_eta_x / rfStep
            eta_y_mcf = -AlphaMCF * Frequency * measured_eta_y / rfStep
            # Modify ORM with HCMEnergyShift/VCMEnergyShift effects
            for i in range(nHorCOR):
                orm_model[:nHBPM, i] += HCMEnergyShift[i] * eta_x_mcf
                orm_model[nHBPM:, i] += HCMEnergyShift[i] * eta_y_mcf
            for i in range(nVerCOR):
                j = nHorCOR + i
                orm_model[:nHBPM, j] += VCMEnergyShift[i] * eta_x_mcf
                orm_model[nHBPM:, j] += VCMEnergyShift[i] * eta_y_mcf

        # --- 3) Flatten, weights, optional coupling removal/outliers ---
        weights_flat_, weights_flat_chi_ = weight_matrix(weights, includeDispersion,
                                                         hor_dispersion_weight, ver_dispersion_weight,
                                                         nHBPM, nVBPM, nHorCOR, nVerCOR)
        if it == 0 and initial_model_orm_callback is not None:
            initial_model_orm_callback(orm_model.copy())

        y_meas_ = orm_measured.reshape(-1, 1, order="F")
        y_model_ = orm_model.reshape(-1, 1, order="F")
        J_ = Jfull.transpose(1, 2, 0).reshape(-1, Jfull.shape[0], order="F")



        iNoCoupling, iNoCoupling_chi, nBPM = build_iNoCoupling(nHBPM, nVBPM, nHorCOR, nVerCOR, includeDispersion)
        if remove_coupling_ == True:
            y_meas, y_model, weights_flat, J, iNoCoupling, iNoCoupling_chi = remove_coupling(
                y_meas_, y_model_, weights_flat_, J_,
                nHBPM, nVBPM, nHorCOR, nVerCOR, includeDispersion
            )


        else:
            y_meas = y_meas_
            y_model = y_model_
            weights_flat = weights_flat_
            J = J_


        Jw = J / weights_flat
        y = (y_meas - y_model) / weights_flat


        # ------------------------------------------------------------
        # Consentrain
        # ------------------------------------------------------------

        #if constraint_cfg is not None:
        #    Jw, y = augment_system_with_constraints(
        #        Jw, y, blocks, fit_list,
        #        constraint_cfg=constraint_cfg,
        #    )

        # ------------------------------------------------------------
        # Apply outliers BEFORE computing chi2_before
        # ------------------------------------------------------------

        keep_mask_reduced = slice(None)
        iOut_coupled = np.array([], dtype=int)

        if outlier_rejection:
            r = (y_meas - y_model).ravel()

            # ----- First test -----
            m, s = np.mean(r), np.std(r, ddof=1)
            i1 = np.where(np.abs(r - m) > sigma_outlier * s)[0]
            j1 = np.where(np.abs(r - m) <= sigma_outlier * s)[0]

            # ----- Second test -----
            r2 = r[j1]
            m2, s2 = np.mean(r2), np.std(r2, ddof=1)
            i2 = np.where(np.abs(r2 - m2) > sigma_outlier * s2)[0]

            out_reduced = np.sort(np.concatenate([i1, j1[i2]]))

            if out_reduced.size == 0:
                print("   No outliers in the data set.")
            else:
                print(
                    f"   std(Model-Measurement) = {1000 * np.std(y_meas - y_model):.6f} mm "
                    "(with outliers)"
                )

                print(
                    f"   {out_reduced.size} outliers removed out of {r.size} points "
                    f"(> {sigma_outlier} sigma) "
                    f"({len(i1)} first test + {len(i2)} second test)."
                )

                # ----- Build keep mask -----
                keep = np.ones(r.size, dtype=bool)
                keep[out_reduced] = False

                # ----- Apply reduction -----
                y_meas = y_meas[keep]
                y_model = y_model[keep]
                weights_flat = weights_flat[keep, :]
                J = J[keep, :]
                # Map reduced outliers -> coupled indices for chi^2
                n_total = y_meas_.size  # full coupled length
                if remove_coupling_ == True:
                    iOut_coupled = reduced_outliers_to_coupled(out_reduced, iNoCoupling_chi, n_total)
                else:
                    iOut_coupled = out_reduced.copy()
                Jw = J / weights_flat
                y = (y_meas - y_model) / weights_flat





        if 'delta_rf' in fit_list:
            Jw, rf_norm_factors = rf_normalization(ring,
                                                   Jw, y_model, weights_flat,
                                                   nHBPM, nVBPM, nHorCOR, nVerCOR, CMstep,
                                                   fit_list, quads_ords, quads_tilt_ind, skew_ords, rfStep
                                                   )

        if apply_normalization == True:
            if normalization_mode == 'global':
                Jw, norm_factors = normalize_jacobian_global(Jw, y_model, weights_flat)
            else:
                Jw, norm_factors = normalize_jacobian_componentwise(
                    ring, Jw, y_model, weights_flat, nHBPM, nVBPM, nHorCOR, nVerCOR, CMstep,
                    fit_list, quads_ords, quads_tilt_ind, skew_ords, rfStep
                )
        else:
            norm_factors = None

        if it == 0:
            iOut_coupled_persistent = iOut_coupled.copy()
            iNoCoupling_chi_persistent = iNoCoupling_chi.copy()

        chi2_before = compute_chi_squared_(
            Mmeas=y_meas_,
            Mmodel=y_model_,
            Mstd=weights_flat_chi_,
            nHBPM=nHBPM, nVBPM=nVBPM, nHorCOR=nHorCOR, nVerCOR=nVerCOR,
            include_dispersion=includeDispersion,
            remove_coupling_=remove_coupling_, iNoCoupling=iNoCoupling_chi_persistent,
            iOutliers=iOut_coupled_persistent,
            n_fit_parameters=J_
        )
        print(f"Initial Chi²: {chi2_before:.4e}")
        if it == 0 and initial_chi2_callback is not None:
            initial_chi2_callback(float(chi2_before))


        if it == 0:
            full_path = Path(output_dir) / "jacobians" / "full"
            full_path.mkdir(parents=True, exist_ok=True)

            np.save(full_path / "J_fit_filtered.npy", J)
            np.save(full_path / "Jw_solver.npy", Jw)
            np.save(full_path / "weights_fit.npy", weights_flat)
            np.save(full_path / "residual_weighted.npy", y)

            if norm_factors is not None:
                np.save(
                    full_path / "normalization_factors.npy",
                    np.asarray(norm_factors)
                )

            print("\n========== FULL SOLVER JACOBIAN ==========")
            print("Jfull 3D :", Jfull.shape)
            print("J raw    :", J_.shape)
            print("J fit    :", J.shape)
            print("Jw solver:", Jw.shape)
            print("==========================================\n")
            import json

            blocks_serializable = {
                name: {
                    "start": int(sl.start),
                    "stop": int(sl.stop),
                    "size": int(sl.stop - sl.start),
                }
                for name, sl in blocks.items()
            }

            with open(full_path / "parameter_blocks.json", "w") as f:
                json.dump(blocks_serializable, f, indent=2)

            print("\nParameter blocks:")
            for name, info in blocks_serializable.items():
                print(
                    f"{name:20s} "
                    f"{info['start']:4d}:{info['stop']:4d} "
                    f"({info['size']:4d})"
                )

        if algorithm.lower() == "lm":
            # LM inner loop with accept/reject and lambda updates
            LMlambda = Starting_Lambda
            chi2_0 = chi2_before
            accepted = False

            for j in range(nLMIter):


                fit_results, lam_used, Ivec, S = solve_step_lm(
                Jw, y, weights_flat, y_model, y_meas,
                scaled=scaled,
                Starting_Lambda=LMlambda,
                max_lm_lambda=max_lm_lambda,
                svd_method=svd_selection_method,
                svd_threshold=svd_threshold,
                cut_=cut_,
                show_plot=show_svd_plot,
                tag=f"LM it{it + 1}/in{j + 1}",
                constraint_cfg=constraint_cfg,
                blocks=blocks,
                fit_list=fit_list,
                param_scale=norm_factors,
            )
                if 'delta_rf' in fit_list:
                    # fit_results = remove_rf_normalization(fit_list, rfStep, fit_results, nHBPM, nVBPM, nHorCOR, nVerCOR, quads_ords, quads_tilt_ind, skew_ords)
                    nf = np.asarray(rf_norm_factors).ravel()
                    fit_results = fit_results / nf

                if norm_factors is not None:
                    nf = np.asarray(norm_factors).ravel()
                    fit_results = fit_results / nf

                old_fit_parameters = current_fit_parameters.copy()
                new_vec = old_fit_parameters + fit_results

                # Build temp ring + config (trial)

                ring_tmp, cfg2, Cmat2, Hshift2, Vshift2, prop_dict = _prepare_ring_and_rmconfig(
                    ring, new_vec,
                    fit_list=fit_list, nHBPM=nHBPM, nVBPM=nVBPM, nHorCOR=nHorCOR, nVerCOR=nVerCOR,
                    quads_ords=quads_ords, quads_tilt_ind=quads_tilt_ind, skew_ords=skew_ords,
                    quad_individuals=quad_individuals,
                    skew_individuals=skew_individuals,
                    tilt_individuals=tilt_individuals, fit_cfg=fit_cfg,
                    used_bpms_ords=used_bpms_ords, used_cor_ords=used_cor_ords,
                    CMstep=CMstep, rfStep=rfStep,
                    HCMCoupling=HCMCoupling, VCMCoupling=VCMCoupling,
                    hbpm_gain=hbpm_gain, hbpm_coupling=hbpm_coupling,
                    vbpm_coupling=vbpm_coupling, vbpm_gain=vbpm_gain,
                    HCMEnergyShift=HCMEnergyShift, VCMEnergyShift=VCMEnergyShift, includeDispersion=includeDispersion,
                    response_matrix_calculator=response_matrix_calculator,
                )

                # Trial ORM on the *temp* ring
                trial_orm_started = time.perf_counter()
                _trace_calculator(calculator_trace_callback, "trial_model_orm", cfg2.calculator)
                orm_trial = response_matrix(ring_tmp, config=cfg2)
                trial_orm_seconds += time.perf_counter() - trial_orm_started
                orm_trial = Cmat2 @ orm_trial

                # Fixed path length adjustment (trial)
                if fixedmomentum == True:
                    AlphaMCF = get_mcf(ring_tmp)
                    rf_used = float(np.asarray(prop_dict.get('delta_rf', rfStep)).ravel()[0])
                    eta_x_mcf = -AlphaMCF * Frequency * measured_eta_x / rf_used
                    eta_y_mcf = -AlphaMCF * Frequency * measured_eta_y / rf_used
                    for iH in range(nHorCOR):
                        orm_trial[:nHBPM, iH] += Hshift2[iH] * eta_x_mcf
                        orm_trial[nHBPM:, iH] += Hshift2[iH] * eta_y_mcf
                    for iV in range(nVerCOR):
                        jcol = nHorCOR + iV
                        orm_trial[:nHBPM, jcol] += Vshift2[iV] * eta_x_mcf
                        orm_trial[nHBPM:, jcol] += Vshift2[iV] * eta_y_mcf

                y_model_trial_ = orm_trial.reshape(-1, 1, order="F")

                if remove_coupling_ == True:
                    _, y_model_trial, _, _, _, _ = remove_coupling(
                        orm_measured.reshape(-1, 1, order="F"), y_model_trial_, None, None, nHBPM, nVBPM, nHorCOR,
                        nVerCOR, includeDispersion
                    )

                # Flatten and compute trial chi² using the SAME keep_mask and weights

                chi2_new = compute_chi_squared_(
                    Mmeas=orm_measured.reshape(-1, 1, order="F"),
                    Mmodel=y_model_trial_,
                    Mstd=weights_flat_chi_,
                    nHBPM=nHBPM, nVBPM=nVBPM, nHorCOR=nHorCOR, nVerCOR=nVerCOR,
                    include_dispersion=includeDispersion,
                    remove_coupling_=remove_coupling_, iNoCoupling=iNoCoupling_chi_persistent,
                    iOutliers=iOut_coupled_persistent,
                    n_fit_parameters=J_
                )

                print(f"  LM inner {j + 1}: chi² {chi2_new:.4e} (previous {chi2_0:.4e}), λ={LMlambda:g}")

                if chi2_new < chi2_0:
                    # ACCEPT: commit everything from trial to live state
                    chi2_0 = chi2_new
                    LMlambda = LMlambda / 10.0
                    accepted = True

                    # Commit live vectors from prop_dict (if present)
                    upd = prop_dict
                    hbpm_gain = upd.get('hbpm_gain', hbpm_gain)
                    vbpm_gain = upd.get('vbpm_gain', vbpm_gain)
                    hbpm_coupling = upd.get('hbpm_coupling', hbpm_coupling)
                    vbpm_coupling = upd.get('vbpm_coupling', vbpm_coupling)
                    HCMCoupling = upd.get('hcor_coupling', HCMCoupling)
                    VCMCoupling = upd.get('vcor_coupling', VCMCoupling)
                    HCMEnergyShift = upd.get('HCMEnergyShift', HCMEnergyShift)
                    VCMEnergyShift = upd.get('VCMEnergyShift', VCMEnergyShift)
                    if 'delta_rf' in upd:
                        rfStep = float(np.asarray(upd['delta_rf']).ravel()[0])

                    if 'quads_tilt' in fit_list and ('quads_tilt' in upd):
                        deltaqt = np.asarray(upd['quads_tilt']).ravel()

                    # Commit CMstep base vectors if you track calibrated steps permanently:
                    # (optional; you can also keep base CMstep fixed and only pass gains via dkick each time)
                    if 'hcor_cal' in upd:
                        CMstep[0] = np.asarray(upd['hcor_cal']).ravel()
                    if 'vcor_cal' in upd:
                        CMstep[1] = np.asarray(upd['vcor_cal']).ravel()

                    if 'quads' in upd:
                        quads_fit = np.asarray(upd['quads']).ravel()

                    if 'skew_quads' in upd:
                        skew_quads_fit = np.asarray(upd['skew_quads']).ravel()

                    # MOST IMPORTANT: commit lattice by replacing live ring with the accepted temp ring
                    ring = ring_tmp
                    current_fit_parameters = new_vec

                    break
                else:
                    # REJECT: do not touch ring or state; increase lambda
                    LMlambda *= 10.0
                    if LMlambda > max_lm_lambda:
                        print("  λ exceeded maximum; stop inner loop.")
                        break

            if not accepted:
                fit_results = np.zeros(Jw.shape[1])


        # GN de-normalization and application (LM handled above)
        elif algorithm.lower() == "gn":

            fit_results, Ivec, S = solve_step_gn(
                Jw, y, svd_selection_method, svd_threshold, cut_,
                show_svd_plot, tag=f"GN it{it + 1}",weights_flat= weights_flat, model_orm_flat=y_model,measured_orm_flat= y_meas
            )

            if 'delta_rf' in fit_list:
                fit_results = remove_rf_normalization(fit_list, rfStep, fit_results, nHBPM, nVBPM, nHorCOR, nVerCOR,
                                                      quads_ords, quads_tilt_ind, skew_ords)

            # 2) De-normalize (if used)
            if apply_normalization and (norm_factors is not None):
                nf = np.asarray(norm_factors)
                nf = np.diag(nf) if nf.ndim == 2 and nf.shape[0] == nf.shape[1] else nf.ravel()
                fit_results = fit_results / nf

            # 6) Update the parameter vector (so history/returns are correct)
            current_fit_parameters = current_fit_parameters + np.asarray(fit_results).ravel()

            # 3) Unpack into a dict of physical knobs
            fit_dict = _pack_fit_dict(
                current_fit_parameters, fit_list, nHBPM, nVBPM, nHorCOR, nVerCOR,
                n_quads=len(quads_ords) if quads_ords is not None else 0,
                n_skew=len(skew_ords) if skew_ords is not None else 0,
                n_quads_tilt=len(quads_tilt_ind) if quads_tilt_ind is not None else 0
            )

            # 4) Update non-lattice state (keep conventions same as LM)
            if 'hbpm_gain' in fit_dict: hbpm_gain = fit_dict['hbpm_gain']
            if 'vbpm_gain' in fit_dict: vbpm_gain = fit_dict['vbpm_gain']
            if 'hbpm_coupling' in fit_dict: hbpm_coupling = fit_dict['hbpm_coupling']
            if 'vbpm_coupling' in fit_dict: vbpm_coupling = fit_dict['vbpm_coupling']
            if 'hcor_coupling' in fit_dict: HCMCoupling = fit_dict['hcor_coupling']
            if 'vcor_coupling' in fit_dict: VCMCoupling = fit_dict['vcor_coupling']
            if 'HCMEnergyShift' in fit_dict: HCMEnergyShift = fit_dict['HCMEnergyShift']
            if 'VCMEnergyShift' in fit_dict: VCMEnergyShift = fit_dict['VCMEnergyShift']
            if 'delta_rf' in fit_dict: rfStep = float(np.asarray(fit_dict['delta_rf']).ravel()[0])
            if 'quads_tilt' in fit_list and ('quads_tilt' in fit_dict):
                deltaqt = np.asarray(fit_dict['quads_tilt']).ravel()

            # If you use calibrated CM steps, update consistently (same convention as LM)
            if 'hcor_cal' in fit_dict:
                CMstep[0] = np.asarray(fit_dict['hcor_cal']).ravel()
            if 'vcor_cal' in fit_dict:
                CMstep[1] = np.asarray(fit_dict['vcor_cal']).ravel()

            if 'quads' in fit_dict:
                quads_fit = np.asarray(fit_dict['quads']).ravel()

            if 'skew_quads' in fit_dict:
                skew_quads_fit = np.asarray(fit_dict['skew_quads']).ravel()

            # 5) Apply lattice changes to ring
            _apply_fit_to_ring(ring, fit_dict, quads_ords, quads_tilt_ind, skew_ords, quad_individuals,
            skew_individuals, tilt_individuals,fit_cfg)




        else:
            raise ValueError("algorithm must be 'lm' or 'gn'")

        # --- Recompute chi²  ---
        cfg3 = RMConfig(dkick=[CMstep[0], CMstep[1]],
                        bpm_ords=used_bpms_ords, cm_ords=used_cor_ords,
                        HCMCoupling=HCMCoupling, VCMCoupling=VCMCoupling,
                        rfStep=rfStep, includeDispersion=includeDispersion,
                        calculator=response_matrix_calculator)
        final_orm_started = time.perf_counter()
        _trace_calculator(calculator_trace_callback, "final_model_orm", cfg3.calculator)
        orm_model_after = response_matrix(ring, config=cfg3)
        final_orm_seconds = time.perf_counter() - final_orm_started
        C_bpms_after = _build_C_matrix(hbpm_gain, hbpm_coupling, vbpm_coupling, vbpm_gain)
        orm_model_after = _build_C_matrix(hbpm_gain, hbpm_coupling, vbpm_coupling, vbpm_gain) @ orm_model_after

        if fixedmomentum == True:

            AlphaMCF = get_mcf(ring)
            eta_x_mcf = -AlphaMCF * Frequency * measured_eta_x / rfStep
            eta_y_mcf = -AlphaMCF * Frequency * measured_eta_y / rfStep
            for i in range(nHorCOR):
                orm_model_after[:nHBPM, i] += HCMEnergyShift[i] * eta_x_mcf
                orm_model_after[nHBPM:, i] += HCMEnergyShift[i] * eta_y_mcf
            for i in range(nVerCOR):
                jj = nHorCOR + i
                orm_model_after[:nHBPM, jj] += VCMEnergyShift[i] * eta_x_mcf
                orm_model_after[nHBPM:, jj] += VCMEnergyShift[i] * eta_y_mcf

        y_model_after_ = orm_model_after.reshape(-1, 1, order="F")
        if remove_coupling_ == True:
            _, y_model_after, _, _, _, _ = remove_coupling(
                orm_measured.reshape(-1, 1, order="F"), y_model_after_, None, None, nHBPM,
                nVBPM, nHorCOR, nVerCOR, includeDispersion
            )

        # chi2_after = compute_chi_squared(y_meas, y_model_after[keep_mask],
        #                                 J=Jw, bpm_noise=weights_flat)

        chi2_after = compute_chi_squared_(
            Mmeas=orm_measured.reshape(-1, 1, order="F"),
            Mmodel=y_model_after_,
            Mstd=weights_flat_chi_,
            nHBPM=nHBPM, nVBPM=nVBPM, nHorCOR=nHorCOR, nVerCOR=nVerCOR,
            include_dispersion=includeDispersion,
            remove_coupling_=remove_coupling_, iNoCoupling=iNoCoupling_chi,
            iOutliers=iOut_coupled,
            n_fit_parameters=J_
        )

        iOut_coupled_persistent = iOut_coupled
        iNoCoupling_chi_persistent = iNoCoupling_chi

        print(f"Chi² after correction: {chi2_after:.4e}")

        chi2_history.append(chi2_after)

        if iteration_metrics_callback is not None:
            residual_after = np.asarray(orm_measured) - np.asarray(orm_model_after)
            horizontal_residual = residual_after[:nHBPM, :]
            vertical_residual = residual_after[nHBPM:nHBPM + nVBPM, :]

            def _residual_metrics(values):
                finite = np.asarray(values, dtype=float)
                finite = finite[np.isfinite(finite)]
                if not finite.size:
                    return {"rms": None, "max_abs": None}
                return {
                    "rms": float(np.sqrt(np.mean(finite ** 2))),
                    "max_abs": float(np.max(np.abs(finite))),
                }

            iteration_metrics_callback({
                "iteration": int(it + 1),
                "chi2_before": float(chi2_before),
                "chi2_after": float(chi2_after),
                "orm_residual": _residual_metrics(residual_after),
                "horizontal_orm_residual": _residual_metrics(horizontal_residual),
                "vertical_orm_residual": _residual_metrics(vertical_residual),
                "timings": {
                    "model_orm_seconds": float(model_orm_seconds),
                    "jacobian_seconds": float(jacobian_seconds),
                    "trial_orm_seconds": float(trial_orm_seconds),
                    "final_orm_seconds": float(final_orm_seconds),
                    "total_orm_seconds": float(
                        model_orm_seconds + trial_orm_seconds + final_orm_seconds
                    ),
                    "iteration_seconds": float(time.perf_counter() - iteration_started),
                    "cumulative_seconds": float(time.perf_counter() - iterations_started),
                },
                "fit_parameters": np.asarray(current_fit_parameters, dtype=float).copy(),
                "ring": ring,
                "orm_model": np.asarray(orm_model_after, dtype=float).copy(),
            })

        if calculate_delta_chi2 ==True:

            print(f"Calculating delta Chi² for all fit paramaters from iteration 0 ...")
            p_initial_iter = current_fit_parameters.copy()  ########
            p_initial_global = inetial_fit_parameters.copy()


            delta_chi2_iter, chi2_nominal = compute_delta_chi2(
                ring=ring,
                p_final=current_fit_parameters,
                p_initial=p_initial_global, ####
                fit_list=fit_list,
                nHBPM=nHBPM, nVBPM=nVBPM,
                nHorCOR=nHorCOR, nVerCOR=nVerCOR,
                quads_ords=quads_ords,
                quads_tilt_ind=quads_tilt_ind,
                skew_ords=skew_ords,
                individuals=individuals,
                fit_cfg=fit_cfg,
                used_bpms_ords=used_bpms_ords,
                used_cor_ords=used_cor_ords,
                CMstep=CMstep,
                rfStep=rfStep,
                HCMCoupling=HCMCoupling,
                VCMCoupling=VCMCoupling,
                hbpm_gain=hbpm_gain,
                hbpm_coupling=hbpm_coupling,
                vbpm_coupling=vbpm_coupling,
                vbpm_gain=vbpm_gain,
                HCMEnergyShift=HCMEnergyShift,
                VCMEnergyShift=VCMEnergyShift,
                orm_measured=orm_measured,
                weights_flat_chi_=weights_flat_chi_,
                includeDispersion=includeDispersion,
                iNoCoupling_chi=iNoCoupling_chi_persistent,
                iOut_coupled=iOut_coupled_persistent,
                J_=J_,
                response_matrix_calculator=response_matrix_calculator,
            )

            delta_chi2_history.append(delta_chi2_iter)
            group_delta = compute_group_delta_chi2(delta_chi2_iter, blocks)

            print("\nΔχ² contribution per group:")
            for k, v in group_delta.items():
                print(f"{k:20s}  sum={v['sum']:.3e}   rms={v['rms']:.3e}   max={v['max']:.3e}")
            group_delta_history.append(group_delta)
            print(f"Calculating delta Chi² DONE ...")
        # Save iteration

        fit_results_all.append(current_fit_parameters.copy())
        # fit_dict_all[it] = _pack_fit_dict(
        #    current_fit_parameters,
        #    fit_list, nHBPM, nVBPM, nHorCOR, nVerCOR,
        #    n_quads=len(quads_ords) if quads_ords is not None else 0,
        #    n_skew=len(skew_ords) if skew_ords is not None else 0,
        #    n_quads_tilt=len(quads_tilt_ind) if quads_tilt_ind is not None else 0
        # )

        fit_dict_all[it] = build_full_fit_state(
            hbpm_gain=hbpm_gain,
            vbpm_gain=vbpm_gain,
            hbpm_coupling=hbpm_coupling,
            vbpm_coupling=vbpm_coupling,
            HCMCoupling=HCMCoupling,
            VCMCoupling=VCMCoupling,
            HCMEnergyShift=HCMEnergyShift,
            VCMEnergyShift=VCMEnergyShift,
            rfStep=rfStep,
            quads=quads_fit,
            skew_quads=skew_quads_fit,
            quads_tilt=deltaqt,
            CMstep=CMstep
        )

    print(f"LOCO {algorithm.upper()} completed! :).")

    # if continue_from_previous and previous_fit_results is not None:
    #    fit_results_all = previous_fit_results + fit_results_all
    #    fit_dict_all = {**previous_fit_dict, **fit_dict_all}
    return (
        fit_results_all,
        fit_dict_all,
        ring,
        orm_model_after,
        C_bpms_after,
        chi2_history,
        delta_chi2_history,
        blocks
    )


# ----------------------- SAVE DICTIONARY -----------------------

from pathlib import Path


def save_fit_dict(fit_dict, output_path: Path):
    """
    Save a LOCO fit_dict to JSON, converting NumPy and non-serializable types safely.
    Falls back to .npz if the data is mostly numeric arrays.
    """

    def _to_jsonable(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (list, tuple)):
            return [_to_jsonable(xx) for xx in x]
        if isinstance(x, dict):
            return {kk: _to_jsonable(vv) for kk, vv in x.items()}
        if isinstance(x, (np.bool_)):
            return bool(x)
        if x is None or isinstance(x, (str, bool, int, float)):
            return x
        # Fallback: represent unknown objects as string
        return str(x)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_path.with_suffix(".json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(fit_dict), f, ensure_ascii=False, indent=4)

    print(f"[info] Saved fit_dict to {json_path}")
    return json_path


# ----------------------- Extract Correction Value -----------------------

def last_by_sorted_key(d):
    def norm(k):
        # try numeric sort first; fallback to string
        try:
            return (0, float(k))
        except Exception:
            return (1, str(k))

    last_key = sorted(d.keys(), key=norm)[-1]
    return d[last_key]


def get_quads_block(fit_vec):
    """
    Extract (quads_fit, skew_fit) from the *last* entry of fit_vec.

    Accepted last-entry formats:
      - dict with keys 'quads' and optionally 'skew' or 'skew_quads'
      - tuple/list like (quads, skew)
      - plain ndarray/sequence of quads only
    Returns:
      (np.ndarray, np.ndarray)  # skew may be empty if not available
    Missing values -> np.nan.
    """
    # empty / None guard
    if fit_vec is None:
        return np.asarray(np.nan), np.asarray(np.nan)
    try:
        last = fit_vec[-1]  # works for list/tuple/ndarray
    except Exception:
        # fit_vec is scalar or not indexable
        last = fit_vec

    # Case 1: dict-like
    if isinstance(last, dict):
        quads = np.asarray(last.get('quads', np.nan), dtype=float)
        # accept several possible names for skew block
        skew = last.get('skew', last.get('skew_quads', np.nan))
        skew = np.asarray(skew, dtype=float)
        return quads, skew

    # Case 2: tuple/list: (quads, skew) or just [quads]
    if isinstance(last, (list, tuple)):
        if len(last) == 0:
            return np.asarray(np.nan), np.asarray(np.nan)
        if len(last) >= 2:
            quads = np.asarray(last[0], dtype=float)
            skew = np.asarray(last[1], dtype=float)
            return quads, skew
        # single-entry -> treat as quads only
        return np.asarray(last[0], dtype=float), np.asarray([])

    # Case 3: NumPy array
    if isinstance(last, np.ndarray):
        # If it's a structured array with named fields, try fields first
        if last.dtype.names:
            q = last[last.dtype.names[0]] if 'quads' not in last.dtype.names else last['quads']
            quads = np.asarray(q, dtype=float)
            if 'skew' in last.dtype.names:
                skew = np.asarray(last['skew'], dtype=float)
            elif 'skew_quads' in last.dtype.names:
                skew = np.asarray(last['skew_quads'], dtype=float)
            else:
                skew = np.asarray([])
            return quads, skew
        # plain numeric array -> assume it's quads only
        return np.asarray(last, dtype=float), np.asarray([])

    # Fallback: unknown type -> return NaNs
    return np.asarray(np.nan), np.asarray(np.nan)


def get_fit_param_block(fit_dict, name):
    """
    Return a fit-parameter block from the LAST iteration.
    """
    if not fit_dict:
        raise ValueError("Empty fit_dict")

    last_key = max(fit_dict)
    inner = fit_dict[last_key]

    if not isinstance(inner, dict):
        raise TypeError(f"Iteration {last_key} is not a dict")

    if name not in inner:
        raise KeyError(
            f"Parameter '{name}' not found in last iteration. "
            f"Available: {list(inner.keys())}"
        )

    return np.asarray(inner[name], dtype=float)
def _calculator_execution_plan(response_matrix_calculator, quad_jacobian_calculator):
    """Return canonical independent calculator choices used by the fit."""
    response_aliases = {
        "linear": "Linear", "analytical": "Analytical",
        "numerical": "Numerical", "tracking": "Numerical",
    }
    response = response_aliases.get(str(response_matrix_calculator).strip().lower())
    if response is None:
        raise ValueError(
            f"Unknown response_matrix_calculator={response_matrix_calculator!r}. "
            "Choose 'Linear', 'Analytical', or 'Tracking'."
        )
    jacobian = str(quad_jacobian_calculator).strip().capitalize()
    if jacobian not in {"Numerical", "Analytical"}:
        raise ValueError(
            f"Unknown quad_jacobian_calculator={quad_jacobian_calculator!r}. "
            "Choose 'Numerical' or 'Analytical'."
        )
    return {
        "response_matrix_calculator": response,
        "normal_quad_jacobian": jacobian,
        "numerical_jacobian_orm_calculator": response if jacobian == "Numerical" else None,
    }


def _trace_calculator(callback, stage, calculator):
    if callback is not None:
        callback({"stage": stage, "calculator": calculator})
