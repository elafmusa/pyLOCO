import numpy as np
from numpy.linalg import svd
import logging
import matplotlib.pyplot as plt
LOGGER = logging.getLogger(__name__)
import time
import json
from .initial_fit import build_initial_fit_parameters
from .set_parameters import set_correction, set_correction_tilt, _get_attr_scalar, _initial_values_for_block, _resolve_attr_for_block_read
import os
import h5py
import multiprocessing as mp
from multiprocessing import shared_memory
from pyloco_config import RMConfig, FitInitConfig, get_mcf, fixed_parameters
from .response_matrix import response_matrix
fit_cfg = FitInitConfig()
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

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
        disp_fit = np.concatenate([2*np.ones((nHBPM,1)), np.zeros((nVBPM,1))])
        CF_fit = np.hstack((CF, disp_fit))
        iNoCoupling = np.where(CF_fit.flatten(order="F") > 0)[0]

        # chi mask (your choice): often keep both planes for chi² (or match MATLAB if you prefer)
        disp_chi = np.concatenate([2*np.ones((nHBPM,1)), 3*np.ones((nVBPM,1))])
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

    # ---- 1) Copy (do NOT modify originals)
    Mmeas  = Mmeas.copy()
    Mmodel = Mmodel.copy()
    Mstd   = Mstd.copy()

    # ---- 2) Mark outliers as NaN (still COUPLED)
    if iOutliers is not None and len(iOutliers) > 0:
        Mmeas[iOutliers]  = np.nan
        Mmodel[iOutliers] = np.nan
        Mstd[iOutliers]   = np.nan

    # ---- 3) Remove coupling (MATLAB does this AFTER outliers)
    if remove_coupling_:
        #Mmeas, Mmodel, Mstd, _, _ = remove_coupling(
        #    Mmeas, Mmodel, Mstd, None,
        #    nHBPM, nVBPM, nHorCOR, nVerCOR,
        #    include_dispersion,
        #    for_chi_squared=True
        #)

        Mmeas = Mmeas[iNoCoupling]
        Mmodel = Mmodel[iNoCoupling]
        Mstd = Mstd[iNoCoupling]

    # ---- 4) Drop NaNs
    mask = ~np.isnan(Mmeas).ravel()
    Mmeas  = Mmeas[mask]
    Mmodel = Mmodel[mask]
    Mstd   = Mstd[mask]

    # ---- 5) Chi²
    residuals = (Mmeas - Mmodel) / Mstd
    chi2 = np.sum(residuals ** 2)

    # ---- 6) Degrees of freedom
    dof = len(Mstd) - n_fit_parameters.shape[1]
    if dof <= 0:
        raise ValueError(f"Invalid DOF: {dof}")
    return chi2 / dof




# ============================================================================== #
#                           Compute Jacobians of fit parameters
# ============================================================================== #



def compute_jacobian(ring, C_model, dkick, dk, bpm_indexes, CMords, quads_ind,
                     nHorCOR, nVerCOR, nHBPM, nVBPM, C,CAVords,
                     skew_ind=None, includeDispersion=False, delta_coupling=1e-6, delta_skew_=1e-3, delta_q_tilt=1e-6,
                     include_quads=True, include_skew=False,include_quads_tilt=False,  include_bpm_gain=False,
                     include_cor_kick=False, include_cor_coupling=False, include_bpm_coupling=False, quads_tilt_ind = None,
                     include_delta_RF_frequency=False, include_HCMEnergyShift=False, include_VCMEnergyShift=False,
                     rf_step=fixed_parameters.rfstep
                     ,individuals=False, auto_correct_delta=True,HCMCoupling = None, VCMCoupling = None, measured_eta_x=None, measured_eta_y=None,quads_tilt_fit=None, Frequency = fixed_parameters.Frequency,fit_cfg=None, iteration=1,  quad_jacobian_file=None,
    skew_jacobian_file=None, quads_tilt_jacobian_file=None,force_recompute=True):

    """
    Master function to compute full LOCO Jacobian including:
    - Quadrupole strengths
    - BPM gains/coupling
    - Corrector gains/coupling
    - etc.
    """
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
    # --- QUADS ---
    J_quad, delta = None, None
    if include_quads:

        user_provided = quad_jacobian_file is not None
        J_path = quad_jacobian_file if user_provided else f"output/jacobians/quads/J_quads_iter{iteration}_{dkick[0][0]}urad_{fixed_parameters.rfstep}Hz.h5"
        os.makedirs(os.path.dirname(J_path), exist_ok=True)

        # --- logic ---
        if os.path.exists(J_path) and not force_recompute and user_provided and iteration == 1:


            print(f"[Jacobian] Loading user-specified normal-quadrupole Jacobian from {J_path}")
            with h5py.File(J_path, "r") as f:
                J_quad = np.array(f["J_quads"])
                delta = None
        else:
            if os.path.exists(J_path) and force_recompute:
                print(f"[Jacobian] File exists, but recomputing as requested (force_recompute=True).")
            elif os.path.exists(J_path) and not user_provided:
                print(f"[Jacobian] Ignoring existing auto file; computing new normal-quadrupole Jacobian (iteration {iteration}).")
            else:
                print(f"[Jacobian] Computing normal-quadrupole Jacobian (iteration {iteration})...")

            J_quad, delta = calculate_quads_jacobian(
                ring, C_model, dkick, CMords, bpm_indexes, quads_ind, dk, C,
                individuals, HCMCoupling, VCMCoupling, rf_step, block="quads",
                auto_correct_delta=auto_correct_delta,
                fit_cfg=fit_cfg, includeDispersion=includeDispersion,
                log_filename="quad_jacobian_logs2.txt"
            )

            # Save
            if iteration == 1:
                with h5py.File(J_path, "w") as f:
                    f.create_dataset("J_quads", data=J_quad)
                    f.create_dataset("C_model", data=C_model)
                    if isinstance(dkick, (list, tuple)):
                        f.create_dataset("correctors_kick_h", data=np.asarray(dkick[0]))
                        f.create_dataset("correctors_kick_v", data=np.asarray(dkick[1]))
                    else:
                        f.create_dataset("correctors_dkick", data=np.asarray(dkick))
                    f.attrs.update({
                        "iteration": iteration,
                        "nHBPM": nHBPM, "nVBPM": nVBPM,
                        "nHorCOR": nHorCOR, "nVerCOR": nVerCOR,
                        "includeDispersion": includeDispersion,
                        "HCMCoupling": json.dumps(np.asarray(HCMCoupling).tolist()),
                        "VCMCoupling": json.dumps(np.asarray(VCMCoupling).tolist()),
                        "date": time.ctime(),
                    })

                print(f"[Jacobian] Saved normal-quadrupole Jacobian to {J_path}")


    # --- SKEW ---
    J_skew, delta_skew = None, None
    if include_skew:

        # Determine file path
        user_provided = skew_jacobian_file is not None
        J_path_skew = skew_jacobian_file if user_provided else f"output/jacobians/skew/J_skew_iter{iteration}_{dkick[0][0]}urad_{fixed_parameters.rfstep}Hz.h5"

        os.makedirs(os.path.dirname(J_path_skew), exist_ok=True)

        # --- logic ---
        if os.path.exists(J_path_skew) and not force_recompute and user_provided and iteration == 1:
            print(f"[Jacobian] Loading user-specified skew-quadrupole Jacobian from {J_path_skew}")
            with h5py.File(J_path_skew, "r") as f:
                J_skew = np.array(f["J_skew"])
                delta_skew = None
        else:
            if os.path.exists(J_path_skew) and force_recompute:
                print(f"[Jacobian] File exists, but recomputing as requested (force_recompute=True).")
            elif os.path.exists(J_path_skew) and not user_provided:
                print(f"[Jacobian] Ignoring existing auto file; computing new skew-quadrupole Jacobian (iteration {iteration}).")
            else:
                print(f"[Jacobian] Computing skew-quadrupole Jacobian (iteration {iteration})...")

            J_skew, delta_skew = calculate_quads_jacobian(
                ring, C_model, dkick, CMords, bpm_indexes, skew_ind, delta_skew_, C,
                individuals, HCMCoupling, VCMCoupling, rf_step, block="skew_quads",
                auto_correct_delta=auto_correct_delta,
                fit_cfg=fit_cfg, includeDispersion=includeDispersion,
                log_filename="skew_jacobian_logs.txt"
            )
            if iteration == 1:
                # --- Save the computed Jacobian ---
                with h5py.File(J_path_skew, "w") as f:
                    f.create_dataset("J_skew", data=J_skew)
                    f.create_dataset("C_model", data=C_model)
                    if isinstance(dkick, (list, tuple)):
                        f.create_dataset("correctors_kick_h", data=np.asarray(dkick[0]))
                        f.create_dataset("correctors_kick_v", data=np.asarray(dkick[1]))
                    else:
                        f.create_dataset("correctors_dkick", data=np.asarray(dkick))

                    f.attrs["iteration"] = iteration
                    f.attrs["nHBPM"] = nHBPM
                    f.attrs["nVBPM"] = nVBPM
                    f.attrs["nHorCOR"] = nHorCOR
                    f.attrs["nVerCOR"] = nVerCOR
                    f.attrs["includeDispersion"] = includeDispersion
                    f.attrs["HCMCoupling"] = json.dumps(np.asarray(HCMCoupling).tolist())
                    f.attrs["VCMCoupling"] = json.dumps(np.asarray(VCMCoupling).tolist())
                    f.attrs["date"] = time.ctime()

                print(f"[Jacobian] Saved skew-quadrupole Jacobian to {J_path_skew}")

    # --- QUAD TILT ---
    J_quad_tilt, delta_quads_tilt = None, None
    if include_quads_tilt:

        # Determine file path
        user_provided = quads_tilt_jacobian_file is not None
        J_path_tilt = quads_tilt_jacobian_file if user_provided else f"output/jacobians/tilt/J_tilt_iter{iteration}_{dkick[0][0]}urad_{fixed_parameters.rfstep}Hz.h5"

        os.makedirs(os.path.dirname(J_path_tilt), exist_ok=True)

        # --- logic ---
        if os.path.exists(J_path_tilt) and not force_recompute and user_provided and iteration == 1:
            print(f"[Jacobian] Loading user-specified quadrupole-tilt Jacobian from {J_path_tilt}")
            with h5py.File(J_path_tilt, "r") as f:
                J_quad_tilt = np.array(f["J_quads_tilt"])
                delta_quads_tilt = None
        else:
            if os.path.exists(J_path_tilt) and force_recompute:
                print(f"[Jacobian] File exists, but recomputing as requested (force_recompute=True).")
            elif os.path.exists(J_path_tilt) and not user_provided:
                print(f"[Jacobian] Ignoring existing auto file; computing new quadrupole-tilt Jacobian (iteration {iteration}).")
            else:
                print(f"[Jacobian] Computing quadrupole-tilt Jacobian (iteration {iteration})...")


            J_quad_tilt, delta_quads_tilt = calculate_quads_tilt_jacobian(
                ring, C_model, dkick, CMords, bpm_indexes, quads_tilt_ind, delta_q_tilt, C, individuals,
                HCMCoupling, VCMCoupling, rf_step, auto_correct_delta=auto_correct_delta, includeDispersion=includeDispersion,
                log_filename="tilt_quad_jacobian_logs.txt", quads_tilt_fit=quads_tilt_fit, fit_cfg=fit_cfg
            )

            # --- Save the computed Jacobian ---
            if iteration == 1:
                with h5py.File(J_path_tilt, "w") as f:
                    f.create_dataset("J_quads_tilt", data=J_quad_tilt)
                    f.create_dataset("C_model", data=C_model)
                    if isinstance(dkick, (list, tuple)):
                        f.create_dataset("correctors_kick_h", data=np.asarray(dkick[0]))
                        f.create_dataset("correctors_kick_v", data=np.asarray(dkick[1]))
                    else:
                        f.create_dataset("correctors_dkick", data=np.asarray(dkick))

                    f.attrs["iteration"] = iteration
                    f.attrs["nHBPM"] = nHBPM
                    f.attrs["nVBPM"] = nVBPM
                    f.attrs["nHorCOR"] = nHorCOR
                    f.attrs["nVerCOR"] = nVerCOR
                    f.attrs["includeDispersion"] = includeDispersion
                    f.attrs["HCMCoupling"] = json.dumps(np.asarray(HCMCoupling).tolist())
                    f.attrs["VCMCoupling"] = json.dumps(np.asarray(VCMCoupling).tolist())
                    f.attrs["date"] = time.ctime()

                print(f"[Jacobian] Saved quadrupole-tilt Jacobian to {J_path_tilt}")



    J_bpm_gain = calculate_bpm_gain_jacobian(
        C_inv @ C_model, nHBPM, nVBPM, nHorCOR, nVerCOR, includeDispersion, include_bpm_coupling
    ) if include_bpm_gain  == True else None

    if include_bpm_gain  == False and include_bpm_coupling == True:
        J_bpm_gain = calculate_bpm_coupling_jacobian(
            C_inv @ C_model, nHBPM, nVBPM, includeDispersion
        )

    J_cor_gain = calculate_corrector_kick_jacobian(
        C_model, dkick, nHorCOR, nVerCOR,includeDispersion
    ) if include_cor_kick == True else None



    J_cor_coupling = calculate_corrector_coupling_jacobian(ring,
                                                           bpm_indexes,
                                                           CMords, C_model, dkick, nHBPM, nVBPM, nHorCOR, nVerCOR,
                                                           includeDispersion, C, HCMCoupling, VCMCoupling,rf_step,
                                                           delta_coupling,
                                                           ) if include_cor_coupling == True else None


    J_delta_RF_frequency = calculate_delta_RF_frequency_jacobian(C_model, nHBPM, nVBPM, nHorCOR, nVerCOR, rf_step
                                                     ) if include_delta_RF_frequency  == True else None

    J_HCMEnergyShift = calculate_HCMEnergyShift_jacobian(ring, C_model, nHBPM, nVBPM, nHorCOR, nVerCOR, rf_step , measured_eta_x, measured_eta_y, Frequency
                                                                 ) if include_HCMEnergyShift  == True else None



    J_VCMEnergyShift = calculate_VCMEnergyShift_jacobian(ring, C_model, nHBPM, nVBPM, nHorCOR, nVerCOR, rf_step,   measured_eta_x, measured_eta_y, Frequency
                                                                 ) if include_VCMEnergyShift  == True else None

    return full_jacobian_(J_quad=J_quad, J_quad_tilt=J_quad_tilt, J_skew=J_skew, J_bpm=J_bpm_gain, J_cor=J_cor_gain, J_cor_coupling=J_cor_coupling, J_delta_RF_frequency =J_delta_RF_frequency, J_HCMEnergyShift=J_HCMEnergyShift, J_VCMEnergyShift=J_VCMEnergyShift) , delta, delta_skew, delta_quads_tilt



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
    order=None,            # e.g. ("J_bpm","J_cor","J_cor_coupling", ... )
    allow_2d=True,         # auto-upgrade 2D (R,C) -> (1,R,C)
    strict=True            # if True, raise when (R,C) mismatch
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
        "J_bpm":               J_bpm,
        "J_cor":               J_cor,
        "J_cor_coupling":      J_cor_coupling,
        "J_HCMEnergyShift":    J_HCMEnergyShift,
        "J_VCMEnergyShift":    J_VCMEnergyShift,
        "J_delta_RF_frequency":J_delta_RF_frequency,
        "J_quad":              J_quad,
        "J_skew":              J_skew,
        "J_quad_tilt":         J_quad_tilt,
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
    fit_cfg=None,
    log_filename="quad_jacobian_logs.txt",     processes = None, includeDispersion=False
):
    # Shared matrices (read-only)
    shm_C   = shared_memory.SharedMemory(create=True, size=C.nbytes)
    C_sh    = np.ndarray(C.shape, dtype=C.dtype, buffer=shm_C.buf);     C_sh[:]    = C
    shm_Cm  = shared_memory.SharedMemory(create=True, size=C_model.nbytes)
    Cmodel_sh = np.ndarray(C_model.shape, dtype=C_model.dtype, buffer=shm_Cm.buf); Cmodel_sh[:] = C_model


    all_logs = []
    ctx = mp.get_context("spawn")

    try:
        quad_args = []
        fit_cfg_dict = fit_cfg.__dict__.copy()
        for quad_index in quads_ind:
            quad_args.append((
                quad_index, ring, dkick,used_cor_ind,  bpm_indexes, dk,
                individuals,HCMCoupling, VCMCoupling,rf_step,
                auto_correct_delta,
                 block, fit_cfg_dict,includeDispersion
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
            J_quad   = np.stack(J_blocks, axis=0)     # (P, rows, cols)
            delta_vec = np.concatenate([np.atleast_1d(d) for d in deltas])
        else:
            J_quad    = np.empty((0, C.shape[0], C.shape[1]))
            delta_vec = np.empty((0,))



        if all_logs:
            try:
                os.makedirs("output", exist_ok=True)

                log_path = os.path.join("output", log_filename)

                with open(log_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(all_logs) + "\n")

                print(f"[calculate_quads_jacobian] Logs saved to '{os.path.abspath(log_path)}'")

            except Exception as e:
                print(f"[calculate_quads_jacobian] Could not write logs: {e}")

        return J_quad, delta_vec

    finally:
        for shm in (shm_C, shm_Cm):
            try: shm.close(); shm.unlink()
            except Exception: pass

# ---------- worker globals ----------
G_C = None
G_CMODEL = None

def _init_shared(shm_name_C, shape_C, dtype_C, shm_name_Cm, shape_Cm, dtype_Cm):
    global G_C, G_CMODEL, _shm_C, _shm_Cm
    _shm_C  = shared_memory.SharedMemory(name=shm_name_C)
    _shm_Cm = shared_memory.SharedMemory(name=shm_name_Cm)
    G_C      = np.ndarray(shape_C,  dtype=np.dtype(dtype_C),  buffer=_shm_C.buf)
    G_CMODEL = np.ndarray(shape_Cm, dtype=np.dtype(dtype_Cm), buffer=_shm_Cm.buf)




def generating_quads_response_matrices(
    quad_index, ring, dkick, cor_indexes,bpm_indexes, delta_init, individuals,
    HCMCoupling, VCMCoupling, rf_step, auto_correct_delta, block, fit_cfg, includeDispersion
):
    logs = []



    attr_name, attr_idx = _resolve_attr_for_block_read(block, fit_cfg)
    group = [int(quad_index)] if np.isscalar(quad_index) else [int(q) for q in quad_index]

    k0_list = np.fromiter(
        (_get_attr_scalar(ring[q], attr_name, attr_idx) for q in group),
        dtype=float,
        count=len(group)
    )

    # choose delta
    if delta_init is None:
        delta_local = 1e-3 * k0_list
        delta_local[delta_local == 0] = 1e-3
    else:
        delta_local = np.atleast_1d(delta_init)[:len(group)].astype(float)

    RMSGoal = 1e-6
    RMSTol  = 10.0
    while True:
        dk = k0_list + delta_local

        set_correction(ring, dk, group, block=block, config=fit_cfg)

        #  ORM with current dk
        cfg = RMConfig(dkick=dkick, bpm_ords=bpm_indexes, cm_ords=cor_indexes,
                       HCMCoupling=HCMCoupling, VCMCoupling=VCMCoupling, includeDispersion=includeDispersion, rfStep=rf_step)

        C_measured = response_matrix(ring, config=cfg)


        C_measured = G_C @ C_measured

        if includeDispersion == True:
            # exclude last column (dispersion)
            Mdiff_no_disp = (
                    C_measured[:, :-1] - G_CMODEL[:, :-1]
            ).ravel(order='F')

            RMSDelta = float(
                np.sqrt(np.sum(Mdiff_no_disp ** 2) / max(1, Mdiff_no_disp.size))
            )
        else:
            Mdiff = (C_measured - G_CMODEL).ravel(order='F')
            RMSDelta = float(np.sqrt(np.sum(Mdiff**2) / max(1, Mdiff.size)))

        if not np.isfinite(RMSDelta) or RMSDelta == 0:
            raise ValueError(f"LOCO error: RMS difference invalid for group {group}")

        if auto_correct_delta:
            if RMSDelta < RMSGoal / RMSTol:
                for idx, q in enumerate(group):
                    logs.append(f"Param #{q}: delta too small; RMS={1000*RMSDelta:0.5g} mm")
                # restore to nominal before changing step
                set_correction(ring, k0_list, group,individuals=individuals, block=block, config=fit_cfg)
                scale = (RMSGoal / RMSDelta)
                delta_local *= scale
            elif RMSDelta > RMSGoal * RMSTol / 3.0:
                for idx, q in enumerate(group):
                    logs.append(f"Param #{q}: delta too big; RMS={1000*RMSDelta:0.5g} mm")
                set_correction(ring, k0_list, group,individuals=individuals, block=block, config=fit_cfg)
                scale = (RMSGoal / RMSDelta)
                delta_local *= scale
            else:
                for idx, q in enumerate(group):
                    logs.append(f"Param #{q}: delta OK; RMS={1000*RMSDelta:0.5g} mm")
                # keep last delta_local used
                break
        else:
            # not auto-correcting; one pass only
            break

    # restore nominal lattice before returning
    set_correction(ring, k0_list, group, block=block, config=fit_cfg)

    step = float(delta_local[0]) if delta_local.size else 1.0
    if step == 0.0:
        step = 1.0  # avoid division by zero
    return (C_measured - G_CMODEL) / step, delta_local, logs



def calculate_quads_tilt_jacobian(

    ring, C_model, dkick, used_cor_ind, bpm_indexes, quads_ind, dk, C, individuals,
    HCMCoupling, VCMCoupling, rf_step, auto_correct_delta=True,
    processes=None, includeDispersion=False,
    log_filename="quads_tilt_jacobian_logs.txt", quads_tilt_fit=None, fit_cfg=None
):

    shm_C   = shared_memory.SharedMemory(create=True, size=C.nbytes)
    C_sh    = np.ndarray(C.shape, dtype=C.dtype, buffer=shm_C.buf)
    C_sh[:] = C
    shm_Cm   = shared_memory.SharedMemory(create=True, size=C_model.nbytes)
    Cmodel_sh = np.ndarray(C_model.shape, dtype=C_model.dtype, buffer=shm_Cm.buf)
    Cmodel_sh[:] = C_model

    all_logs = []

    ctx = mp.get_context("spawn")
    try:


        assert len(quads_tilt_fit) == len(quads_ind), \
            f"Length mismatch: {len(quads_tilt_fit)=} vs {len(quads_ind)=}"

        quad_args = []
        fit_cfg_dict = fit_cfg.__dict__.copy()
        for i, quad_index in enumerate(quads_ind):
            tilt_fit_i = quads_tilt_fit[i]
            quad_args.append((
                quad_index, ring, dkick, bpm_indexes, used_cor_ind, dk, individuals
                , auto_correct_delta,
                HCMCoupling, VCMCoupling, rf_step,
                tilt_fit_i,fit_cfg_dict,includeDispersion
            ))

        with ctx.Pool(
            processes=processes,
            initializer=_init_shared,
            initargs=(shm_C.name, C.shape, C.dtype.str,
                      shm_Cm.name, C_model.shape, C_model.dtype.str),
            maxtasksperchild=64,
        ) as pool:
            results = pool.starmap(generating_quads_tilt_response_matrices, quad_args, chunksize=1)

        if results:
            J_blocks, deltas, logs_lists = zip(*results)

            for _logs in logs_lists:
                if _logs:
                    all_logs.extend(_logs)

            J_blocks = [np.asarray(blk) for blk in J_blocks]
            J_quad   = np.stack(J_blocks, axis=0)

            delta_vec = np.concatenate([np.atleast_1d(d) for d in deltas])
        else:
            J_quad    = np.empty((0, C.shape[0], C.shape[1]))
            delta_vec = np.empty((0,))

        if all_logs:
            try:
                os.makedirs("output", exist_ok=True)

                log_path = os.path.join("output", log_filename)

                with open(log_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(all_logs) + "\n")

                print(f"[calculate_quads_jacobian] Logs saved to '{os.path.abspath(log_path)}'")

            except Exception as e:
                print(f"[calculate_quads_jacobian] Could not write logs: {e}")

        return J_quad, delta_vec

    finally:
        try:
            shm_C.close(); shm_C.unlink()
        except Exception:
            pass
        try:
            shm_Cm.close(); shm_Cm.unlink()
        except Exception:
            pass


def generating_quads_tilt_response_matrices(
    quad_index, ring, dkick, bpm_indexes, cor_indexes, delta_init, individuals,
    auto_correct_delta, HCMCoupling, VCMCoupling, rf_step, quads_tilt_fit,fit_cfg,includeDispersion
):
    logs = []

    group = [int(quad_index)] if isinstance(quad_index, (np.integer, int)) else [int(q) for q in quad_index]
    delta_local = np.atleast_1d(delta_init)[:len(group)].astype(float)

    RMSGoal = 1e-6
    RMSTol = 10.0
    DeltaCheckFlag = True

    while DeltaCheckFlag:


        set_correction_tilt(ring, psi_values=delta_local + quads_tilt_fit,
                            elem_ind=group, individuals=individuals, config=fit_cfg)

        cfg = RMConfig(dkick=dkick, bpm_ords=bpm_indexes, cm_ords=cor_indexes, HCMCoupling=HCMCoupling,
                       VCMCoupling=VCMCoupling,includeDispersion=includeDispersion, rfStep=rf_step)
        C_measured = response_matrix(ring, config=cfg)

        C_measured = G_C @ C_measured

        if includeDispersion == True:
            # exclude last column (dispersion)
            Mdiff_no_disp = (
                    C_measured[:, :-1] - G_CMODEL[:, :-1]
            ).ravel(order='F')

            RMSDelta = float(
                np.sqrt(np.sum(Mdiff_no_disp ** 2) / max(1, Mdiff_no_disp.size))
            )
        else:
            Mdiff = (C_measured - G_CMODEL).ravel(order='F')
            RMSDelta = float(np.sqrt(np.sum(Mdiff ** 2) / max(1, Mdiff.size)))

        logs.append(f"quads_tilt_fit #{quads_tilt_fit}")
        logs.append(f"delta_local #{delta_local}")
        logs.append(f"R1 #{ring[quad_index].R1}")
        logs.append(f"R2 #{ring[quad_index].R2}")



        if not np.isfinite(RMSDelta) or RMSDelta == 0:
            raise ValueError(f"LOCO error: RMS difference invalid for group {group}")

        if auto_correct_delta:
            if RMSDelta < RMSGoal / RMSTol:
                for idx, q in enumerate(group):
                    logs.append(f"Param #{q}: delta too small; RMS={1000*RMSDelta:0.5g} mm")
                # restore to nominal before changing step
                set_correction_tilt(ring, psi_values=quads_tilt_fit,
                                    elem_ind=group, individuals=individuals, config=fit_cfg)
                scale = (RMSGoal / RMSDelta)
                delta_local *= scale
            elif RMSDelta > RMSGoal * RMSTol / 3.0:
                for idx, q in enumerate(group):
                    logs.append(f"Param #{q}: delta too big; RMS={1000*RMSDelta:0.5g} mm")
                set_correction_tilt(ring, psi_values=quads_tilt_fit,
                                    elem_ind=group, individuals=individuals, config=fit_cfg)
                scale = (RMSGoal / RMSDelta)
                delta_local *= scale
            else:
                for idx, q in enumerate(group):
                    logs.append(f"Param #{q}: delta OK; RMS={1000*RMSDelta:0.5g} mm")
                # keep last delta_local used
                break
        else:
            # not auto-correcting; one pass only
            break



    set_correction_tilt(ring, psi_values=quads_tilt_fit,
                        elem_ind=group, individuals=individuals, config=fit_cfg)

    return (C_measured - G_CMODEL) / delta_local[0], delta_local, logs



def calculate_bpm_gain_jacobian(C_model, nHBPM, nVBPM, nHorCOR, nVerCOR, includeDispersion, fit_bpms_coupling):

    nBPM, nCOR = C_model.shape

    if fit_bpms_coupling == True:
        J_bpm = np.zeros((2* nBPM, nBPM, nCOR))
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
            J_bpm[i+nHBPM + nVBPM, idx, :] = C_model[i, :]


        for i in range(nVBPM):
            idx = i + nHBPM
            J_bpm[i+nHBPM+nHBPM+nVBPM, idx, :] = C_model[idx, :]


    return J_bpm


def calculate_bpm_coupling_jacobian(
    C_model, nHBPM, nVBPM, includeDispersion
):

    nBPM, nCOR = C_model.shape
    J_bpm = np.zeros((nBPM, nBPM, nCOR))

    # 1. XY Coupling

    for i in range(nHBPM):
        idx = i + nVBPM
        J_bpm[idx, i, :] = C_model[idx, :]

    # 1. YX Coupling

    for i in range(nVBPM):
        idx = i + nHBPM
        J_bpm[i, idx, :] = C_model[i, :]




    return J_bpm





def calculate_corrector_kick_jacobian(C_model, cor_kicks, nHorCOR, nVerCOR, includeDispersion):


    nBPM, nCols = C_model.shape
    nCOR = nHorCOR + nVerCOR
    has_disp = nCols == nCOR + 1

    if has_disp:
        C_model_scaled = C_model[:, :nCOR] #/ cor_kicks[np.newaxis, :]
    else:
        C_model_scaled = C_model #/ cor_kicks[np.newaxis, :]

    J_cor = np.zeros((nCOR, nBPM, nCols))

    for i in range(nHorCOR):
        J_cor[i, :, i] = C_model_scaled[:, i] / cor_kicks[0][i]

    for i in range(nVerCOR):
        idx = i + nHorCOR
        J_cor[idx, :, idx] = C_model_scaled[:, idx] /cor_kicks[1][i]

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
        includeDispersion, C, HCMCoupling, VCMCoupling,rf_step,
        delta_coupling=1e-6
):

    nBPM_total = nHBPM + nVBPM
    nCOR_total = nHorCOR + nVerCOR
    nCols = C_model.shape[1]
    has_disp = nCols == nCOR_total + 1

    HCMCoupling = HCMCoupling + delta_coupling * np.ones(len(HCMCoupling))
    VCMCoupling = VCMCoupling + delta_coupling * np.ones(len(VCMCoupling))



    cfg = RMConfig(dkick=cor_kicks, bpm_ords=bpm_ords, cm_ords=cm_ords, HCMCoupling=HCMCoupling,
                   VCMCoupling=VCMCoupling,includeDispersion=includeDispersion, rfStep=rf_step)
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




def calculate_HCMEnergyShift_jacobian(ring, C_model, nHBPM, nVBPM, nHorCOR, nVerCOR, rf_step, measured_eta_x, measured_eta_y,Frequency):


    nBPM_total = nHBPM + nVBPM
    nCOR_total = nHorCOR + nVerCOR
    nCols = C_model.shape[1]
    has_disp = nCols == nCOR_total + 1


    if has_disp:
        print("Error: Better to either include dispersion on ORM or fit the energy shift at correctors.")
        #return None

    nParams_total = nHorCOR
    alpha_mc = get_mcf(ring)


    eta_x_mcf = -alpha_mc * Frequency * measured_eta_x / rf_step
    eta_y_mcf = -alpha_mc * Frequency * measured_eta_y / rf_step
    J_HCMEnergyShift = np.zeros((nParams_total, nBPM_total, nCols))


    for i in range(nHorCOR):
        J_HCMEnergyShift[i, :nHBPM, i] = eta_x_mcf
        J_HCMEnergyShift[i, nHBPM:, i] = eta_y_mcf


    return J_HCMEnergyShift


def calculate_VCMEnergyShift_jacobian(ring, C_model, nHBPM, nVBPM, nHorCOR, nVerCOR, rf_step,  measured_eta_x, measured_eta_y,Frequency):

    nBPM_total = nHBPM + nVBPM
    nCOR_total = nHorCOR + nVerCOR
    nCols = C_model.shape[1]
    has_disp = nCols == nCOR_total + 1


    if has_disp:
        print("Error: Better to either include dispersion on ORM or fit the energy shift at correctors.")
        #return None

    nParams_total = nVerCOR
    alpha_mc = get_mcf(ring)

    eta_x_mcf = -alpha_mc * Frequency * measured_eta_x / rf_step
    eta_y_mcf = -alpha_mc * Frequency * measured_eta_y / rf_step

    J_VCMEnergyShift = np.zeros((nParams_total, nBPM_total, nCols))

    for i in range(nVerCOR):
        J_VCMEnergyShift[i , :nHBPM, i + nHorCOR] = eta_x_mcf
        J_VCMEnergyShift[i ,  nHBPM:,i + nHorCOR] = eta_y_mcf


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
    #normalization_factor = 1 / rf_step / 10
    #J_delta_RF_frequency = J_delta_RF_frequency / normalization_factor

    #print("The RF frequency parameter is normalized by 1 / rf_step / 10 to to get a better fit.")

    J_delta_RF_frequency = J_delta_RF_frequency[np.newaxis, :, :] # convert it to 3 d

    return J_delta_RF_frequency



# ============================================================================== #
#                               NORMALIZATION OPTION
# ============================================================================== #


def normalize_jacobian_global(J_flat, model_orm_flat, weights_flat):
    """
    Normalize each column of J_flat by sqrt(sum(J[:,i]^2) / Mmodelsq)
    """

    Mmodelsq = np.sum((model_orm_flat / weights_flat) ** 2)
    norm_factors = np.sqrt(np.sum((J_flat / weights_flat)**2, axis=0) / Mmodelsq)
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
        norm = np.sqrt(np.sum((J_cor_coupling)**2, axis=0) / Mmodelsq)
        J_flat_normalized[:, idx:idx + n] = J_cor_coupling / norm[np.newaxis, :]
        norm_factors[idx:idx + n] = norm
        idx += n



    if 'HCMEnergyShift' in fit_list:
        n = nHorCOR
        J_HCMEnergyShift = J_flat[:, idx:idx + n]
        alpha_mc = get_mcf(ring)
        Frequency = fixed_parameters.Frequency
        norm  = abs(alpha_mc * Frequency / rf_step)
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
        norm = np.sqrt(np.sum((J_quads)**2, axis=0) / Mmodelsq)
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



def remove_rf_normalization(fit_list, rf_step, fit_result, nHBPM, nVBPM, nHorCOR, nVerCOR, quads_ind, quads_tilt_ind, skew_ords):

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
        nf  = 1 / rf_step / 10
        fit_result_unnormalized[idx:idx + 1] = J_delta_rf / nf # already normalized
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
        J_quads =  fit_result[idx:idx + n]
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
        J_flat_normalized[:, idx:idx + n] = J_cor_cal #/ norm
        idx += n


    if 'hcor_coupling' in fit_list or 'vcor_coupling' in fit_list:
        n = nHorCOR + nVerCOR
        J_cor_coupling = J_flat[:, idx:idx + n]
        J_flat_normalized[:, idx:idx + n] = J_cor_coupling #/ norm[np.newaxis, :]
        idx += n



    if 'HCMEnergyShift' in fit_list:
        n = nHorCOR
        J_HCMEnergyShift = J_flat[:, idx:idx + n]
        alpha_mc = get_mcf(ring)
        Frequency = fixed_parameters.Frequency
        J_flat_normalized[:, idx:idx + n] = J_HCMEnergyShift #/ norm
        idx += n

    if 'VCMEnergyShift' in fit_list:
        n = nVerCOR
        J_VCMEnergyShift = J_flat[:, idx:idx + n]
        alpha_mc = get_mcf(ring)
        Frequency = fixed_parameters.Frequency
        J_flat_normalized[:, idx:idx + n] = J_VCMEnergyShift #/ norm
        idx += n

    if 'delta_rf' in fit_list:
        J_delta_rf = J_flat[:, idx:idx + 1]

        # Apply normalization factor to increase fitting weight of RF frequency
        normalization_factor = 1 / rf_step / 10
        print("The RF frequency parameter is normalized by 1 / rf_step / 10 to to get a better fit.")

        J_flat_normalized[:, idx:idx + 1] = J_delta_rf/ normalization_factor
        norm_factors[idx:idx + 1] = normalization_factor
        idx += 1

    if 'quads' in fit_list:
        n = len(quads_ind)
        J_quads = J_flat[:, idx:idx + n]
        J_flat_normalized[:, idx:idx + n] = J_quads #/ norm[np.newaxis, :]
        idx += n

    if 'skew_quads' in fit_list:
        n = len(skew_ords)
        J_quads = J_flat[:, idx:idx + n]
        J_flat_normalized[:, idx:idx + n] = J_quads #/ norm[np.newaxis, :]
        idx += n


    if 'quads_tilt' in fit_list:
        n = len(quads_tilt_ind)
        J_quads = J_flat[:, idx:idx + n]
        J_flat_normalized[:, idx:idx + n] = J_quads #/ norm[np.newaxis, :]
        idx += n


    return J_flat_normalized, norm_factors.reshape(-1, 1)

# ============================================================================== #
#                               LOCO Minimization
# ============================================================================== #


def _build_C_matrix(hbpm_gain, hbpm_coupling, vbpm_coupling, vbpm_gain):
    C11 = np.diag(np.asarray(hbpm_gain).ravel())
    C12 = np.diag(np.asarray(hbpm_coupling).ravel())
    C21 = np.diag(np.asarray(vbpm_coupling).ravel())
    C22 = np.diag(np.asarray(vbpm_gain).ravel())
    return np.block([[C11, C12], [C21, C22]])

def _svd_select_indices(S, method="threshold", svd_threshold=1e-7, cut_=None,
                        interactive=False, show_plot=False, iteration_tag=""):
    """Return indices Ivec of singular values to keep."""
    if method == "threshold":
        Ivec = np.where(S > svd_threshold * np.max(S))[0]
    elif method == "user_input" and cut_ is not None:
        Ivec = np.arange(min(cut_, len(S)))
    elif method == "interactive" or interactive:
        sv_indices = np.arange(len(S))
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1); plt.semilogy(sv_indices, S, '.-'); plt.xlabel("SV idx"); plt.ylabel("SV")
        plt.subplot(1, 2, 2); plt.plot(sv_indices, S/np.max(S), '.-'); plt.xlabel("SV idx"); plt.ylabel("SV/max")
        plt.tight_layout(); plt.show(); time.sleep(0.5)
        user = input("Enter indices (e.g. 0:20 or 0,1,2): ")
        if ':' in user:
            a,b = user.split(':'); Ivec = np.arange(int(a), int(b))
        else:
            Ivec = np.array([int(x.strip()) for x in user.split(',')])
        Ivec = Ivec[Ivec < len(S)]
    else:
        Ivec = np.where(S > svd_threshold * np.max(S))[0]

    if show_plot:
        sv_indices = np.arange(len(S))
        unused = np.setdiff1d(sv_indices, Ivec)

        plt.figure(figsize=(10, 3))
        plt.semilogy(Ivec, S[Ivec], '-', color="green", label="Used")
        if len(unused):
            plt.semilogy(unused, S[unused], '-', color="red", label="Cut")

        plt.title("SVD Spectrum", fontsize=12)
        plt.legend(fontsize=10)
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)
        plt.tight_layout()
        plt.show()

    return Ivec

def _prepare_ring_and_rmconfig(
    base_ring, fit_vec, *, fit_list, nHBPM, nVBPM, nHorCOR, nVerCOR,
    quads_ords, quads_tilt_ind, skew_ords, individuals, fit_cfg,
    used_bpms_ords, used_cor_ords, CMstep, rfStep,
    HCMCoupling, VCMCoupling,
    hbpm_gain, hbpm_coupling, vbpm_coupling, vbpm_gain,
    HCMEnergyShift, VCMEnergyShift,includeDispersion
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
    _apply_fit_to_ring(ring_tmp, prop, quads_ords, quads_tilt_ind, skew_ords, individuals, fit_cfg)


    dkick_H = np.asarray(prop.get('hcor_cal', CMstep[0]), dtype=float).ravel()
    dkick_V = np.asarray(prop.get('vcor_cal', CMstep[1]), dtype=float).ravel()
    dkick   = [dkick_H, dkick_V]

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
        rfStep=float(np.asarray(prop.get('delta_rf', rfStep)).ravel()[0]),includeDispersion=includeDispersion
    )

    Cmat = _build_C_matrix(
        prop.get('hbpm_gain',     hbpm_gain),
        prop.get('hbpm_coupling', hbpm_coupling),
        prop.get('vbpm_coupling', vbpm_coupling),
        prop.get('vbpm_gain',     vbpm_gain),
    )

    Hshift = np.asarray(prop.get('HCMEnergyShift', HCMEnergyShift), dtype=float).ravel()
    Vshift = np.asarray(prop.get('VCMEnergyShift', VCMEnergyShift), dtype=float).ravel()

    return ring_tmp, cfg, Cmat, Hshift, Vshift, prop



def _pack_fit_dict(vec, fit_list, nHBPM, nVBPM, nHorCOR, nVerCOR, n_quads, n_skew, n_quads_tilt):

    d = {}; i = 0
    if 'hbpm_gain'      in fit_list: d['hbpm_gain']      = vec[i:i+nHBPM]; i += nHBPM
    if 'hbpm_coupling'  in fit_list: d['hbpm_coupling']  = vec[i:i+nHBPM]; i += nHBPM
    if 'vbpm_coupling'  in fit_list: d['vbpm_coupling']  = vec[i:i+nVBPM]; i += nVBPM
    if 'vbpm_gain'      in fit_list: d['vbpm_gain']      = vec[i:i+nVBPM]; i += nVBPM
    if 'hcor_cal'       in fit_list: d['hcor_cal']       = vec[i:i+nHorCOR]; i += nHorCOR
    if 'vcor_cal'       in fit_list: d['vcor_cal']       = vec[i:i+nVerCOR]; i += nVerCOR
    if 'hcor_coupling'  in fit_list: d['hcor_coupling']  = vec[i:i+nHorCOR]; i += nHorCOR
    if 'vcor_coupling'  in fit_list: d['vcor_coupling']  = vec[i:i+nVerCOR]; i += nVerCOR
    if 'HCMEnergyShift' in fit_list: d['HCMEnergyShift'] = vec[i:i+nHorCOR]; i += nHorCOR
    if 'VCMEnergyShift' in fit_list: d['VCMEnergyShift'] = vec[i:i+nVerCOR]; i += nVerCOR
    if 'delta_rf'       in fit_list: d['delta_rf']       = vec[i:i+1];       i += 1
    if 'quads'          in fit_list: d['quads']          = vec[i:i+n_quads]; i += n_quads
    if 'skew_quads'     in fit_list: d['skew_quads']     = vec[i:i+n_skew];  i += n_skew
    if 'quads_tilt'     in fit_list: d['quads_tilt']     = vec[i:i+n_quads_tilt]; i += n_quads_tilt
    return d

def _apply_fit_to_ring(ring, fit_dict, quads_ords, quads_tilt_ind, skew_ords, individuals, fit_cfg):

    if 'quads' in fit_dict:
        set_correction(ring, np.asarray(fit_dict['quads']).ravel(),
                       quads_ords, individuals=individuals, block='quads', config=fit_cfg)
    if 'skew_quads' in fit_dict:
        set_correction(ring, np.asarray(fit_dict['skew_quads']).ravel(),
                       skew_ords, individuals=individuals, block='skew_quads', config=fit_cfg)
    if 'quads_tilt' in fit_dict:
        set_correction_tilt(ring, np.asarray(fit_dict['quads_tilt']).ravel(),
                            quads_tilt_ind, config=fit_cfg)

# ----------------------- SOLVERS -----------------------

def solve_step_gn(J_weighted, y, svd_method, svd_threshold, cut_, show_plot, tag):

    U, S, Vh = np.linalg.svd(J_weighted, full_matrices=False) # or True ?
    Ivec = _svd_select_indices(S, method=svd_method, svd_threshold=svd_threshold,
                               cut_=cut_, show_plot=show_plot, iteration_tag=tag)
    b = U[:, Ivec].T @ y
    b = np.diag(1.0 / S[Ivec]) @ b
    fit_results = Vh.T[:, Ivec] @ b
    return fit_results.ravel(), Ivec, S

def solve_step_lm(J_weighted, y, *, scaled=True, Starting_Lambda=1e-3,
                  max_lm_lambda=15, svd_method='threshold', svd_threshold=1e-3,
                  cut_=None, show_plot=False, tag=""):
    """
    One LM inner step (choose Lamda, compute update). Returns (fit_results, lambda_used, Ivec, S).
    Note: This computes update; caller decides accept/reject after recomputing chi².
    """

    C = J_weighted.T @ J_weighted
    ay = J_weighted.T @ y
    lam = Starting_Lambda

    Uc, Sc, Vhc = np.linalg.svd(C + (lam*np.diag(np.diag(C)) if scaled == True else lam*np.eye(C.shape[0])),
                                full_matrices=False)
    Ivec = _svd_select_indices(Sc, method=svd_method, svd_threshold=svd_threshold,
                               cut_=cut_, show_plot=show_plot, iteration_tag=tag)
    # Solve (Uc diag(Sc) Vc^T) b = ay   b = Vc diag(1/Sc) Uc^T ay
    b = Uc[:, Ivec].T @ ay
    b = np.diag(1.0 / Sc[Ivec]) @ b
    fit_results = Vhc.T[:, Ivec] @ b
    return fit_results.ravel(), lam, Ivec, Sc

# ----------------------- MASTER pyloco FUNCTION -----------------------

def pyloco(
    ring,
    *,
    algorithm="lm",                # "lm" or "gn"
    nIter=3,
    # indices & number of elements
    used_bpms_ords=None, used_cor_ords=None, quads_ords=None, skew_ords=None, CAVords=None,
    nHBPM=None, nVBPM=None, nHorCOR=None, nVerCOR=None, quads_tilt_ind=None, inetial_fit_parameters = None,
    # measurment data
    orm_measured=None, weights=None, includeDispersion=False,
    measured_eta_x=None, measured_eta_y=None,
    hor_dispersion_weight=1.0, ver_dispersion_weight=1.0,
    # Correctors kicks & RF steps
    CMstep=None, rfStep=None, Frequency = fixed_parameters.Frequency,
    # features
    fit_list=(), individuals=True, remove_coupling_=True,
    # outliers & normalization
    outlier_rejection=False, sigma_outlier=10,
    apply_normalization=False, normalization_mode='global',
    # SVD selection
    svd_selection_method='threshold', svd_threshold=1e-7, cut_=None,
    show_svd_plot=False,
    # LM options
    nLMIter=10, Starting_Lambda=1e-3, max_lm_lambda=15, scaled=True,
    # more options
    plot_fit_parameters=False, auto_correct_delta=True, fixedpathlength=True, fixedmomentum=False,
    fit_cfg=None,
    # Jacopians files
    quad_jacobian_file=None,
    skew_jacobian_file=None,
    quads_tilt_jacobian_file=None,
    force_recompute=False,
    # Fit multi stage
    continue_from_previous=False,
    previous_fit_results=None,
    previous_fit_dict=None,
    previous_ring=None,

):

    hbpm_gain      = np.ones(nHBPM)
    vbpm_gain      = np.ones(nVBPM)
    hbpm_coupling  = np.zeros(nHBPM)
    vbpm_coupling  = np.zeros(nVBPM)
    HCMEnergyShift = np.zeros(nHorCOR)
    VCMEnergyShift = np.zeros(nVerCOR)
    HCMCoupling    = np.zeros(nHorCOR)
    VCMCoupling    = np.zeros(nVerCOR)
    deltaqt        = np.zeros(len(quads_ords)) if ('quads_tilt' in fit_list) else None

    iOut_coupled_persistent = np.array([], dtype=int)
    iNoCoupling_chi_persistent = np.array([], dtype=int)

    # --- Resume from previous fit if requested ---
    if continue_from_previous:
        print("[pyloco] Continuing from previous iteration set...")
        if previous_ring is not None:
            ring = previous_ring
        if previous_fit_results is not None:
            current_fit_parameters = np.asarray(previous_fit_results[-1]).copy()
        if previous_fit_dict is not None and len(previous_fit_dict) > 0:
            last_fit = previous_fit_dict[max(previous_fit_dict.keys())] ## last by order
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

    if fixedmomentum and \
            'HCMEnergyShift' not in fit_list and \
            'VCMEnergyShift' not in fit_list:
        fixedmomentum = True
        #fixedpathlength = True

    if fixedpathlength == False or 'HCMEnergyShift' in fit_list or 'VCMEnergyShift' in fit_list:
        fixedmomentum = True


    if inetial_fit_parameters is None and not continue_from_previous:
        inetial_fit_parameters, blocks = build_initial_fit_parameters(
            ring=ring,
            fit_list=fit_list,
            nHBPM=nHBPM, nVBPM=nVBPM, nHorCOR=nHorCOR, nVerCOR=nVerCOR,
            quads_ords=quads_ords, skew_ords=skew_ords, quads_tilt=quads_tilt_ind,
            CMstep = CMstep, rfStep = rfStep,
            individuals = individuals)
    #elif continue_from_previous and previous_fit_results is not None:
    #     inetial_fit_parameters = np.asarray(previous_fit_results[-1]).copy()
    elif continue_from_previous and previous_fit_dict is not None:
        print("[pyloco] Building initial vector from previous stage...")
        last_fit = previous_fit_dict[max(previous_fit_dict.keys())]

        # Build new vector with the current fit_list structure
        inetial_fit_parameters, blocks = build_initial_fit_parameters(
            ring=ring,
            fit_list=fit_list,
            nHBPM=nHBPM, nVBPM=nVBPM, nHorCOR=nHorCOR, nVerCOR=nVerCOR,
            quads_ords=quads_ords, skew_ords=skew_ords, quads_tilt=quads_tilt_ind,
            CMstep=CMstep, rfStep=rfStep,
            individuals=individuals,
        )

        # Overwrite any parameters that were fitted before
        for key in fit_list:
            if key in last_fit and key in blocks:
                print(f"[pyloco] Restoring previous values for {key}...")
                arr = np.asarray(last_fit[key]).ravel()
                sl = blocks[key]  # use the slice directly
                inetial_fit_parameters[sl] = arr[: sl.stop - sl.start]

    inetial_fit_parameters = np.asarray(inetial_fit_parameters).ravel()
    current_fit_parameters = inetial_fit_parameters.copy()

    # histories
    fit_results_all = []
    fit_dict_all    = {}

    # ------- Outer iterations -------
    for it in range(nIter):
        print(f"\n==== Iteration {it+1}/{nIter} – {algorithm.upper()} ====")
        # --- 1) ORM model ---
        cfg = RMConfig(dkick=CMstep, bpm_ords=used_bpms_ords, cm_ords=used_cor_ords,
                       HCMCoupling=HCMCoupling, VCMCoupling=VCMCoupling, rfStep=rfStep,includeDispersion=includeDispersion)
        orm_model = response_matrix(ring, config=cfg)
        Cmat = _build_C_matrix(hbpm_gain, hbpm_coupling, vbpm_coupling, vbpm_gain)
        orm_model = Cmat @ orm_model

        # --- 2) Jacobian ---
        include_quads         = ('quads' in fit_list)
        include_skew          = ('skew_quads' in fit_list)
        include_quads_tilt    = ('quads_tilt' in fit_list)
        include_bpm_gain      = ('hbpm_gain' in fit_list) or ('vbpm_gain' in fit_list)
        include_cor_kick      = ('hcor_cal' in fit_list) or ('vcor_cal' in fit_list)
        include_cor_coupling  = ('hcor_coupling' in fit_list) or ('vcor_coupling' in fit_list)
        include_bpm_coupling  = ('hbpm_coupling' in fit_list) or ('vbpm_coupling' in fit_list)
        include_HCMEnergyShift= ('HCMEnergyShift' in fit_list)
        include_VCMEnergyShift= ('VCMEnergyShift' in fit_list)
        include_delta_RF      = ('delta_rf' in fit_list)


        Jfull, dq, dskew, dtilt = compute_jacobian(
            ring, C_model=orm_model, dkick=CMstep,
            bpm_indexes=used_bpms_ords, CMords=used_cor_ords, quads_ind=quads_ords,
            nHorCOR=nHorCOR, nVerCOR=nVerCOR, nHBPM=nHBPM, nVBPM=nVBPM,
            C=Cmat, CAVords=CAVords,
            dk=fixed_parameters.dk,
            skew_ind=skew_ords, quads_tilt_ind=quads_tilt_ind,
            includeDispersion=includeDispersion,
            individuals=individuals,
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
            Frequency = Frequency,
            iteration = it+1,
            quad_jacobian_file=quad_jacobian_file,
            skew_jacobian_file=skew_jacobian_file,
            quads_tilt_jacobian_file=quads_tilt_jacobian_file,
            force_recompute=force_recompute

        )
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
        y_meas_ = orm_measured.reshape(-1, 1, order="F")
        y_model_ = orm_model.reshape(-1, 1, order="F")
        J_ = Jfull.transpose(1, 2, 0).reshape(-1, Jfull.shape[0], order="F")

        iNoCoupling, iNoCoupling_chi, nBPM = build_iNoCoupling(nHBPM, nVBPM, nHorCOR, nVerCOR, includeDispersion)
        if remove_coupling_==True:
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
                    f"   std(Model-Measurement) = {1000*np.std(y_meas-y_model):.6f} mm "
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

        if apply_normalization==True:
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

        if algorithm.lower() == "lm":
            # LM inner loop with accept/reject and lambda updates
            LMlambda = Starting_Lambda
            chi2_0 = chi2_before
            accepted = False

            for j in range(nLMIter):
                fit_results, lam_used, Ivec, S = solve_step_lm(
                    Jw, y, scaled=scaled, Starting_Lambda=LMlambda, max_lm_lambda=max_lm_lambda,
                    svd_method=svd_selection_method, svd_threshold=svd_threshold,
                    cut_=cut_, show_plot=show_svd_plot, tag=f"LM it{it+1}/in{j+1}"
                )
                if 'delta_rf' in fit_list:

                    #fit_results = remove_rf_normalization(fit_list, rfStep, fit_results, nHBPM, nVBPM, nHorCOR, nVerCOR, quads_ords, quads_tilt_ind, skew_ords)
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
                    individuals=individuals, fit_cfg=fit_cfg,
                    used_bpms_ords=used_bpms_ords, used_cor_ords=used_cor_ords,
                    CMstep=CMstep, rfStep=rfStep,
                    HCMCoupling=HCMCoupling, VCMCoupling=VCMCoupling,
                    hbpm_gain=hbpm_gain, hbpm_coupling=hbpm_coupling,
                    vbpm_coupling=vbpm_coupling, vbpm_gain=vbpm_gain,
                    HCMEnergyShift=HCMEnergyShift, VCMEnergyShift=VCMEnergyShift, includeDispersion=includeDispersion,
                )

                # Trial ORM on the *temp* ring
                orm_trial = response_matrix(ring_tmp, config=cfg2)
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
                    _, y_model_trial, _, _, _ , _ = remove_coupling(
                        orm_measured.reshape(-1, 1, order="F"), y_model_trial_, None, None, nHBPM, nVBPM, nHorCOR, nVerCOR, includeDispersion
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
                show_svd_plot, tag=f"GN it{it + 1}"
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

            # 5) Apply lattice changes to ring
            _apply_fit_to_ring(ring, fit_dict, quads_ords, quads_tilt_ind, skew_ords, individuals, fit_cfg)




        else:
            raise ValueError("algorithm must be 'lm' or 'gn'")

        # --- Recompute chi²  ---
        cfg3 = RMConfig(dkick=[CMstep[0],CMstep[1]],
                        bpm_ords=used_bpms_ords, cm_ords=used_cor_ords,
                        HCMCoupling=HCMCoupling, VCMCoupling=VCMCoupling,
                        rfStep=rfStep,includeDispersion=includeDispersion)
        orm_model_after = response_matrix(ring, config=cfg3)
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


        #chi2_after = compute_chi_squared(y_meas, y_model_after[keep_mask],
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

        # Save iteration

        fit_results_all.append(current_fit_parameters.copy())
        fit_dict_all[it] = _pack_fit_dict(
            current_fit_parameters,
            fit_list, nHBPM, nVBPM, nHorCOR, nVerCOR,
            n_quads=len(quads_ords) if quads_ords is not None else 0,
            n_skew=len(skew_ords) if skew_ords is not None else 0,
            n_quads_tilt=len(quads_tilt_ind) if quads_tilt_ind is not None else 0
        )

    print(f"LOCO {algorithm.upper()} completed! :).")

    #if continue_from_previous and previous_fit_results is not None:
    #    fit_results_all = previous_fit_results + fit_results_all
    #    fit_dict_all = {**previous_fit_dict, **fit_dict_all}

    return fit_results_all, fit_dict_all, ring



def plot_data(s_pos, data, xlabel, ylabel, title):
    plt.figure(figsize=(7, 3))
    plt.plot(s_pos, data, color='navy')  # Deep blue color
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    # plt.grid(True, which='both', linestyle=':', color='gray')
    plt.tight_layout()
    plt.show()



def plot_matrices(*matrices, titles=None, cmap='viridis', plot_type='2d', save_path=None):
    n = len(matrices)
    if n == 0:
        raise ValueError("At least one matrix must be provided.")

    if titles is None:
        titles = [f"Matrix {i + 1}" for i in range(n)]
    elif len(titles) < n:
        titles += [f"Matrix {i + 1}" for i in range(len(titles), n)]

    fig = plt.figure(figsize=(6 * n, 5))

    for i, matrix in enumerate(matrices):
        if plot_type == '3d':
            ax = fig.add_subplot(1, n, i + 1, projection='3d')
            X, Y = np.meshgrid(np.arange(matrix.shape[1]), np.arange(matrix.shape[0]))
            surf = ax.plot_surface(X, Y, matrix, cmap=cmap, edgecolor='none')
            ax.set_title(titles[i])
            ax.set_xlabel('Correctors')
            ax.set_ylabel('BPMs')
            ax.set_zlabel('[m]')
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        else:
            ax = fig.add_subplot(1, n, i + 1)
            im = ax.imshow(matrix, aspect='auto', cmap=cmap)
            ax.set_title(titles[i])
            ax.set_xlabel('Correctors')
            ax.set_ylabel('BPMs')
            fig.colorbar(im, ax=ax)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()

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
            skew  = np.asarray(last[1], dtype=float)
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


