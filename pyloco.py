import numpy as np
from numpy.linalg import svd
import logging
import matplotlib.pyplot as plt
LOGGER = logging.getLogger(__name__)
import time
from initial_fit import build_initial_fit_parameters
from set_parameters import set_correction, set_correction_tilt, _get_attr_scalar, _initial_values_for_block, _resolve_attr_for_block_read
import os
import multiprocessing as mp
from multiprocessing import shared_memory
from pyloco_config import RMConfig, FitInitConfig, get_mcf, fixed_parameters
from response_matrix import response_matrix
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

    if include_dispersion:
        # Split orbit BPM stds into horizontal and vertical parts
        W_H = W[:nHBPM]
        W_V = W[nHBPM:]

        dispersion_std = np.concatenate([
            W_H / hor_dispersion_weight,
            W_V / ver_dispersion_weight
        ]).reshape(-1, 1)  # column vector (nHBPM + nVBPM, 1)

        W_matrix = np.hstack((W_matrix, dispersion_std))  # shape: (nBPMs, nCORs + 1)

    W_flat = W_matrix.reshape(-1, 1, order='F')

    return W_flat


def remove_coupling(orm1, orm2, W=None, Jacobian=None,
                    nHBPM=None, nVBPM=None, nHorCOR=None, nVerCOR=None,
                    include_dispersion=False):
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

    if include_dispersion==True:
        dispersion_column = np.concatenate([
            2 * np.ones((nHBPM, 1)),
            np.zeros((nVBPM, 1))
        ])
        CF = np.hstack((CF, dispersion_column))

    CF_flat = CF.flatten(order='F')
    iNoCoupling = np.where(CF_flat > 0)[0]

    # Apply filtering
    orm1_filtered = orm1[iNoCoupling]
    orm2_filtered = orm2[iNoCoupling]
    W_filtered = W[iNoCoupling] if W is not None else None
    Jacobian_filtered = Jacobian[iNoCoupling, :] if Jacobian is not None else None

    return orm1_filtered, orm2_filtered, W_filtered, Jacobian_filtered, iNoCoupling


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




def compute_chi_squared(measured_orm, model_orm, J, bpm_noise):


    residuals = (measured_orm - model_orm) / bpm_noise
    chi2 = np.sum(residuals ** 2)
    dof = len(bpm_noise) - J.shape[1]

    if dof <= 0:
        raise ValueError("Degrees of freedom must be positive.")

    return chi2 / dof




# ============================================================================== #
#                           Compute Jacobians of fit parameters
# ============================================================================== #



def compute_jacobian(ring, C_model, dkick, dk, bpm_indexes, CMords, quads_ind,
                     nHorCOR, nVerCOR, nHBPM, nVBPM, C,CAVords,
                     skew_ind=None, includeDispersion=False, delta_coupling=1e-6, delta_skew_=1e-3, delta_q_tilt=1e-6,
                     include_quads=True, include_skew=False,include_quads_tilt=False,  include_bpm_gain=False,
                     include_cor_kick=False, include_cor_coupling=False, include_bpm_coupling=False,
                     include_delta_RF_frequency=False, include_HCMEnergyShift=False, include_VCMEnergyShift=False,
                     rf_step=fixed_parameters.rfstep
                     ,individuals=False, auto_correct_delta=True,HCMCoupling = None, VCMCoupling = None, measured_eta_x=None, measured_eta_y=None,quads_tilt_fit=None, Frequency = fixed_parameters.Frequency,fit_cfg=None):

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
    if include_quads:
        J_quad, delta = calculate_quads_jacobian(
    ring, C_model, dkick, CMords, bpm_indexes, quads_ind, dk, C,
    individuals, HCMCoupling, VCMCoupling, block="quads",
    auto_correct_delta=auto_correct_delta,
    fit_cfg=fit_cfg,
    log_filename="quad_jacobian_logs.txt"
)
    else:
        J_quad, delta = None, None

    # --- SKEW ---
    if include_skew  == True:
        J_skew, delta_skew =calculate_quads_jacobian(
    ring, C_model, dkick, CMords, bpm_indexes, skew_ind, delta_skew_, C,
    individuals, HCMCoupling, VCMCoupling, block="skew_quads",
    auto_correct_delta=auto_correct_delta,
    fit_cfg=fit_cfg,
    log_filename="skew_jacobian_logs.txt",
)
    else:
        J_skew, delta_skew = None, None



    if include_quads_tilt == True:
        J_quad_tilt, delta_quads_tilt = calculate_quads_tilt_jacobian(
            ring, C_model, dkick, CMords, bpm_indexes, quads_ind, delta_q_tilt, C, individuals,
            HCMCoupling, VCMCoupling, auto_correct_delta=auto_correct_delta,
            log_filename="tilt_quad_jacobian_logs.txt", quads_tilt_fit=quads_tilt_fit, fit_cfg=fit_cfg
        )


    else:
        J_quad_tilt, delta_quads_tilt = None, None



    J_bpm_gain = calculate_bpm_gain_jacobian(
        C_inv @ C_model, nHBPM, nVBPM, nHorCOR, nVerCOR, includeDispersion, include_bpm_coupling
    ) if include_bpm_gain  == True else None

    J_cor_gain = calculate_corrector_kick_jacobian(
        C_model, dkick, nHorCOR, nVerCOR,includeDispersion
    ) if include_cor_kick == True else None



    J_cor_coupling = calculate_corrector_coupling_jacobian(ring,
                                                           bpm_indexes,
                                                           CMords, C_model, dkick, nHBPM, nVBPM, nHorCOR, nVerCOR,
                                                           includeDispersion, C, HCMCoupling, VCMCoupling,
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
    individuals, HCMCoupling, VCMCoupling, block,
    auto_correct_delta=True,
    fit_cfg=None,
    log_filename="quad_jacobian_logs.txt",     processes = None
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
        for quad_index in quads_ind:
            quad_args.append((
                quad_index, ring, dkick,used_cor_ind,  bpm_indexes, dk,
                individuals,HCMCoupling, VCMCoupling,
                auto_correct_delta,
                 block, fit_cfg
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
                with open(log_filename, "w", encoding="utf-8") as f:
                    f.write("\n".join(all_logs) + "\n")
                print(f"[calculate_quads_jacobian] Logs saved to '{os.path.abspath(log_filename)}'")
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
    HCMCoupling, VCMCoupling, auto_correct_delta, block, fit_cfg
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
                       HCMCoupling=HCMCoupling, VCMCoupling=VCMCoupling)

        C_measured = response_matrix(ring, config=cfg)


        C_measured = G_C @ C_measured
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
    HCMCoupling, VCMCoupling, auto_correct_delta=True,
    processes=None,
    log_filename="quads_tilt_jacobian_logs.txt", quads_tilt_fit=None, fit_cfg=None,rf_step=None
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
        for i, quad_index in enumerate(quads_ind):
            tilt_fit_i = quads_tilt_fit[i]
            quad_args.append((
                quad_index, ring, dkick, bpm_indexes, used_cor_ind, dk, individuals
                , auto_correct_delta,
                HCMCoupling, VCMCoupling,
                tilt_fit_i,fit_cfg
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
                with open(log_filename, "w", encoding="utf-8") as f:
                    f.write("\n".join(all_logs) + "\n")
                print(f"[calculate_quads_tilt_jacobian] Logs saved to '{os.path.abspath(log_filename)}'")
            except Exception as e:
                print(f"[calculate_quads_tilt_jacobian] Could not write logs: {e}")

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
    auto_correct_delta, HCMCoupling, VCMCoupling, quads_tilt_fit,fit_cfg
):
    logs = []

    group = [int(quad_index)] if isinstance(quad_index, (np.integer, int)) else [int(q) for q in quad_index]
    k0_list = np.array([0 for q in group], dtype=float)
    delta_local = np.atleast_1d(delta_init)[:len(group)].astype(float)

    RMSGoal = 1e-6
    RMSGoal = 1e-6
    RMSTol = 10.0
    DeltaCheckFlag = True

    while DeltaCheckFlag:


        set_correction_tilt(ring, psi_values=delta_local + quads_tilt_fit,
                            elem_ind=group, individuals=individuals, config=fit_cfg)

        cfg = RMConfig(dkick=dkick, bpm_ords=bpm_indexes, cm_ords=cor_indexes, HCMCoupling=HCMCoupling,
                       VCMCoupling=VCMCoupling)
        C_measured = response_matrix(ring, config=cfg)

        C_measured = G_C @ C_measured

        np.save('C_measured',C_measured)
        np.save('G_CMODEL', G_CMODEL)

        Mdiff = (C_measured - G_CMODEL).ravel(order='F')
        RMSDelta = np.sqrt(np.sum(Mdiff**2) / len(Mdiff))

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

    ring.save('lat_tilt_ppython.mat', mat_key='ring')

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
            J_bpm[i, i, :nHorCOR] = C_model[i, :nHorCOR]

        for i in range(nVBPM):
            idx = i + nHBPM
            J_bpm[idx, idx, nHorCOR:nHorCOR + nVerCOR] = C_model[idx, nHorCOR:nHorCOR + nVerCOR]

        if includeDispersion == True:
            for i in range(nBPM):
                J_bpm[i, :, -1] = 0.0

    if fit_bpms_coupling == True:

        for i in range(nHBPM):
            J_bpm[i, i, :nHorCOR] = C_model[i, :nHorCOR]

        # 1. YX Coupling

        for i in range(nVBPM):
            idx = i + nHBPM
            J_bpm[i+nHBPM, i, :] = C_model[idx, :]


        # 2. XY Coupling


        for i in range(nHBPM):
            idx = i + nVBPM
            J_bpm[i+nHBPM+nHBPM, idx, :] = C_model[i, :]


        for i in range(nVBPM):
            idx = i + nHBPM
            J_bpm[i+nHBPM+nHBPM+nVBPM, idx, nHorCOR:nHorCOR + nVerCOR] = C_model[idx, nHorCOR:nHorCOR + nVerCOR]


        if includeDispersion == True:
            for i in range(nBPM):
                J_bpm[i, :, -1] = 0.0

    return J_bpm


def calculate_bpm_coupling_jacobian(
    C_model, nHBPM, nVBPM, includeDispersion
):

    nBPM, nCOR = C_model.shape
    J_bpm = np.zeros((nBPM, nBPM, nCOR))

    # 1. YX Coupling

    for i in range(nVBPM):
        idx = i + nHBPM
        J_bpm[i, i, :] = C_model[idx, :]

    # 1. XY Coupling

    for i in range(nHBPM):
        idx = i + nVBPM
        J_bpm[idx, idx, :] = C_model[i, :]


    if includeDispersion == True:
        for i in range(nBPM):
            J_bpm[i, :, -1] = 0

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
        includeDispersion, C, HCMCoupling, VCMCoupling,
        delta_coupling=1e-6,rf_step = None
):

    nBPM_total = nHBPM + nVBPM
    nCOR_total = nHorCOR + nVerCOR
    nCols = C_model.shape[1]
    has_disp = nCols == nCOR_total + 1

    HCMCoupling = HCMCoupling + delta_coupling * np.ones(len(HCMCoupling))
    VCMCoupling = VCMCoupling + delta_coupling * np.ones(len(VCMCoupling))



    cfg = RMConfig(dkick=cor_kicks, bpm_ords=bpm_ords, cm_ords=cm_ords, HCMCoupling=HCMCoupling,
                   VCMCoupling=VCMCoupling)
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
        return None

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
        return None

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
    normalization_factor = 1 / rf_step / 10
    J_delta_RF_frequency = J_delta_RF_frequency / normalization_factor

    print("The RF frequency parameter is normalized by 1 / rf_step / 10 to to get a better fit.")

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
    fit_list, quads_ind, skew_ords, rf_step
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
        n = len(quads_ind)
        J_quads = J_flat[:, idx:idx + n]
        norm = np.sqrt(np.sum((J_quads) ** 2, axis=0) / Mmodelsq)
        J_flat_normalized[:, idx:idx + n] = J_quads / norm[np.newaxis, :]
        norm_factors[idx:idx + n] = norm
        idx += n


    return J_flat_normalized, norm_factors.reshape(-1, 1)



# ============================================================================== #
#                               LOCO Minimization
# ============================================================================== #

def loco_correction_gn(SC, used_bpms_ords, used_cor_ords, CMstep, nIter,
                       nHorCOR, nVerCOR, nHBPM, nVBPM, orm_measured,
                       cut_, weights, twiss0, elements_ind, quads_ords, skew_ords, fit_list, remove_coupling_, includeDispersion = False,hor_dispersion_weight = 1,
                       ver_dispersion_weight = 1, outlier_rejection= False, sigma_outlier = 10, quads_kick =None
                       , skewness=False,
                       show_svd_plot=False, plot_fit_parameters=False, n_orbits_avg=1, svd_selection_method='threshold',
svd_threshold=1e-3,  apply_normalization=False, normalization_mode='global', method = 'add', individuals=True, auto_correct_delta = True
):



    fit_results_all = []
    twiss_all = []
    etax_all = []
    etay_all = []
    fit_dict_all = {}



    for i in range(nIter):
        print(f"\n\n========== LOCO Iteration {i+1} / {nIter} GN Method ==========\n")

        print("Measuring ORM from simulation ...")



        cfg = RMConfig(dkick=CMstep, bpm_ords=used_bpms_ords, cm_ords=cm_ords, HCMCoupling=HCMCoupling,
                       VCMCoupling=VCMCoupling, rfStep=rf_step)
        orm_model = response_matrix(ring, config=cfg)


        include_quads = 'quads' in fit_list
        include_skew = 'skew_quads' in fit_list
        include_bpm_gain = 'hbpm_gain' in fit_list or 'vbpm_gain' in fit_list
        include_cor_kick = 'hcor_cal' in fit_list or 'vcor_cal' in fit_list
        include_cor_coupling = 'hcor_coupling' in fit_list or 'vcor_coupling' in fit_list
        include_bpm_coupling = 'hbpm_coupling' in fit_list or 'vbpm_coupling' in fit_list
        include_delta_RF_frequency = 'delta_rf' in fit_list
        include_HCMEnergyShift = 'HCMEnergyShift' in fit_list
        include_VCMEnergyShift = 'VCMEnergyShift' in fit_list

        full_jacobian, delta, delta_skew = compute_jacobian(
            ring, orm_model, CMstep, quads_kick,
            used_bpms_ords, used_cor_ords, quads_ords, skew_ords,
            nHorCOR, nVerCOR, nHBPM, nVBPM,
            skewness=skewness,  # ← REQUIRED
            includeDispersion=includeDispersion,

            # delta_coupling uses default unless you want to set it
            include_quads=include_quads,
            include_skew = include_skew,
            include_bpm_gain=include_bpm_gain,
            include_cor_kick=include_cor_kick,
            include_cor_coupling=include_cor_coupling,
            include_bpm_coupling=include_bpm_coupling,
            include_delta_RF_frequency=include_delta_RF_frequency,
            include_HCMEnergyShift=include_HCMEnergyShift,
            include_VCMEnergyShift=include_VCMEnergyShift,
            bidirectional=bidirectional,
            rm_calculator=rm_calculator,
            rf_step=rfStep,
            # only pass method if it’s defined; otherwise remove this line
            method=method,
            n_orbits_avg=n_orbits_avg, HCMCoupling=HCMCoupling, VCMCoupling=VCMCoupling
        )



        # Flatten weights if needed

        weights_flat = weight_matrix(weights, includeDispersion,hor_dispersion_weight, ver_dispersion_weight, nHBPM, nVBPM, nHorCOR, nVerCOR)


        measured_orm_flat = orm_measured.reshape(-1, 1, order='F')
        model_orm_flat = orm_model.reshape(-1, 1, order='F')
        J_flat = full_jacobian.transpose(1, 2, 0).reshape(-1, full_jacobian.shape[0], order='F')


        if remove_coupling_==True:
            print("Removing XY coupling from ORM and Jacobian and weight ...")
            measured_orm_flat, model_orm_flat, weights_flat, J_flat,iNoCoupling = remove_coupling(measured_orm_flat, model_orm_flat, weights_flat, J_flat,
                            nHBPM, nVBPM, nHorCOR, nVerCOR,
                            includeDispersion)

        J_weighted = J_flat / weights_flat

        if outlier_rejection == True :

            y = (measured_orm_flat - model_orm_flat)  # / sigma_w2

            i1 = np.where(np.abs(y - np.mean(y)) >  sigma_outlier * np.std(y))[0]
            j1 = np.where(np.abs(y - np.mean(y)) <= sigma_outlier * np.std(y))[0]

            y_j1 = y[j1]
            i2 = np.where(np.abs(y_j1 - np.mean(y_j1)) > sigma_outlier * np.std(y_j1))[0]
            iOutliers = np.sort(np.concatenate((i1, j1[i2])))

            if iOutliers.size == 0:
                print("   No outliers in the data set.")
            else:
                print(
                    f"   std(Model-Measurement) = {1000 * np.std(measured_orm_flat - model_orm_flat):.6f} mm (with outliers)")



            J_weighted = np.delete(J_weighted, iOutliers, axis=0)
            orm_measured_flat = np.delete(measured_orm_flat, iOutliers, axis=0)
            orm_model_corr_flat = np.delete(model_orm_flat, iOutliers, axis=0)
            J_flat = np.delete(J_flat, iOutliers, axis=0)
            weights_flat = np.delete(weights_flat, iOutliers, axis=0)

        if apply_normalization == True:
            fit_flags = {
                'fit_cor_cal': 'hcor_cal' in fit_list or 'vcor_cal' in fit_list,
                'fit_cor_coupling': 'hcor_coupling' in fit_list or 'vcor_coupling' in fit_list,
                'fit_RF_energy_shift': 'delta_rf' in fit_list,
            }

            if normalization_mode == 'global':
                J_weighted, norm_factors = normalize_jacobian_global(J_weighted, model_orm_flat, weights_flat)
            elif normalization_mode == 'component':
                J_weighted, norm_factors = normalize_jacobian_componentwise(SC,
                    J_weighted, model_orm_flat, weights_flat,
                    nHBPM, nVBPM, nHorCOR, nVerCOR, CMstep,
                    fit_flags, fit_list, fixed_parameters.rfstep)


        chi2_before = compute_chi_squared(
            orm_measured_flat,
            orm_model_corr_flat,
            J=J_weighted,
            bpm_noise=weights_flat
        )

        y = (measured_orm_flat - model_orm_flat) / weights_flat




        print(f"Initial Chi^2 before correction: {chi2_before:.4e}")

        print("Performing SVD of weighted Jacobian ...")

        U, S, Vh = svd(J_weighted, full_matrices=False)

        V = Vh.T

        if svd_selection_method == 'threshold':
            # Keep SVs greater than threshold * max(S)
            Ivec = np.where(S > svd_threshold * np.max(S))[0]

        #elif svd_selection_method == 'rank':
        #    print("  Performing rank-based singular value selection ...")
        #    Ivec = []
        #    for i in reversed(range(1, len(S) + 1)):
        #        try:
        #            Amod = U[:, :i] @ np.diag(S[:i])
        #            np.linalg.lstsq(Amod, y, rcond=None)  # Check stability
        #            Ivec = np.arange(i)
        #            break
        #        except np.linalg.LinAlgError:
        #            continue
        #    if len(Ivec) == 0:
        #        raise RuntimeError("Rank-based selection failed: no stable solution found.")


        elif svd_selection_method == 'rank':

            print("  Performing rank-based singular value selection ...")
            ChiSquareVector = np.full(len(S), np.nan)
            LastGoodSvalue = 0

            for i in reversed(range(1, len(S) + 1)):
                try:
                    # Step 1: reconstruct Amod = U_i * S_i
                    Amod = U[:, :i] @ np.diag(S[:i])

                    # Step 2: solve b = Amod \ y
                    b = np.linalg.lstsq(Amod, y, rcond=None)[0]

                    # Step 3: back-transform using V (to match MATLAB logic)
                    b = V[:, :i] @ b

                    # Step 4: compute model response update
                    Mfit = weights_flat * (J_flat @ b)
                    Mmodelnew = model_orm_flat + Mfit

                    # Step 5: compute chi² for these SVs
                    chi2 = np.sum(((measured_orm_flat - Mmodelnew) / weights_flat) ** 2) / len(weights_flat)
                    ChiSquareVector[i - 1] = chi2

                    # Step 6: test covariance matrix
                    Cmod = Amod.T @ Amod
                    np.linalg.inv(Cmod)  # will raise if ill-conditioned

                    LastGoodSvalue = i
                    Ivec = np.arange(i)
                    break  # found the largest stable SV count
                except np.linalg.LinAlgError:
                    continue

            if LastGoodSvalue == 0:
                raise RuntimeError("Rank-based selection failed: no stable solution found.")



        elif svd_selection_method == 'user_input':

            Ivec = np.arange(min(cut_, len(S)))


        elif svd_selection_method == 'interactive':

            sv_indices = np.arange(len(S))

            # ---------- Plot 1: Raw SVD spectrum BEFORE input ----------

            print("\n>>> SVD plots displayed to help you choose :)")

            print("You can enter a range like 0:20 for example")

            plt.figure(figsize=(10, 5))

            plt.subplot(1, 2, 1)
            plt.semilogy(sv_indices, S, 'b.-')
            plt.xlabel("Singular Value Index")
            plt.ylabel("Magnitude")


            plt.subplot(1, 2, 2)
            plt.plot(sv_indices, S / np.max(S), 'b.-')
            plt.xlabel("Singular Value Index")
            plt.ylabel("Magnitude / max(SV)")


            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
            plt.pause(0.1)
            time.sleep(1)

            # ---------- User input ----------



            user_input = input("Please enter the singulare values indices to be used (e.g. 0:20): ")

            try:

                if ':' in user_input:

                    parts = user_input.split(':')

                    start = int(parts[0])

                    end = int(parts[1])

                    Ivec = np.arange(start, end)

                else:

                    Ivec = np.array([int(x.strip()) for x in user_input.split(',')])


                Ivec = Ivec[Ivec < len(S)]

            except Exception as e:

                raise ValueError("Invalid input format for singular value indices.") from e




        if show_svd_plot:
            plt.figure(figsize=(10, 3))
            sv_indices = np.arange(len(S))

            plt.semilogy(Ivec, S[Ivec], color='green', linewidth=1.0, label='Used')
            unused_indices = np.setdiff1d(sv_indices, Ivec)
            plt.semilogy(unused_indices, S[unused_indices], color='red', linewidth=1.0, label='Cut Off')

            plt.xlabel("Singular Value Index")
            plt.ylabel("Magnitude")
            plt.title(f"SVD Spectrum (Iteration {i + 1})")
            plt.legend()
            plt.tight_layout()
            plt.show()

        print(f"SVD retained: {len(Ivec)} / {len(S)} singular values")

        V = Vh.T
        #Ivec = np.arange(cut_)
        b = U[:, Ivec].T @ y
        b = np.diag(1.0 / S[Ivec]) @ b
        fit_results = V[:, Ivec] @ b



        if apply_normalization == True:

                fit_results = fit_results/norm_factors


        fit_results_all.append(fit_results)

        fit_dict = {}
        idx = 0




        if 'hbpm_gain' in fit_list:
            fit_dict['hbpm_gain'] = fit_results[idx:idx + nHBPM]

            idx += nHBPM

        if 'vbpm_gain' in fit_list:
            fit_dict['vbpm_gain'] = fit_results[idx:idx + nVBPM]
            idx += nVBPM

        if 'hbpm_coupling' in fit_list:
            fit_dict['hbpm_coupling'] = fit_results[idx:idx + nHBPM]
            idx += nHBPM

        if 'vbpm_coupling' in fit_list:
            fit_dict['vbpm_coupling'] = fit_results[idx:idx + nVBPM]
            idx += nVBPM

        if 'hcor_cal' in fit_list:
            fit_dict['hcor_cal'] = fit_results[idx:idx + nHorCOR]

            new_hcor_kick = np.array(fit_results[idx:idx + nHorCOR]).ravel()
            CMstep[0] = CMstep[0].ravel() + new_hcor_kick
            idx += nHorCOR

        if 'vcor_cal' in fit_list:
            fit_dict['vcor_cal'] = fit_results[idx:idx + nVerCOR]

            new_vcor_kick = np.array(fit_results[idx:idx + nVerCOR]).ravel()
            CMstep[1] = CMstep[1].ravel() + new_vcor_kick
            idx += nVerCOR

        if 'hcor_coupling' in fit_list:
            fit_dict['hcor_coupling'] = fit_results[idx:idx + nHorCOR]
            idx += nHorCOR

        if 'vcor_coupling' in fit_list:
            fit_dict['vcor_coupling'] = fit_results[idx:idx + nVerCOR]
            idx += nVerCOR

        if 'HCMEnergyShift' in fit_list:
            fit_dict['HCMEnergyShift'] = fit_results[idx:idx + nHorCOR]
            idx += nHorCOR

        if 'VCMEnergyShift' in fit_list:
            fit_dict['VCMEnergyShift'] = fit_results[idx:idx + nVerCOR]
            idx += nVerCOR

        if 'delta_rf' in fit_list:
            fit_dict['delta_rf'] = fit_results[idx:idx + 1]
            idx += 1

        if 'quads' in fit_list:
            fit_dict['quads'] = fit_results[idx:]

        fit_dict_all[i] = fit_dict.copy()

        if plot_fit_parameters:
            print("Plotting fit parameters ...")
            n_plots = len(fit_dict)
            fig, axs = plt.subplots(n_plots, 1, figsize=(5, 6))
            if n_plots == 1:
                axs = [axs]

            for ax, (key, values) in zip(axs, fit_dict.items()):
                ax.plot(values, marker='o', linestyle='-', markersize=3)
                ax.set_title(key.replace('_', ' ').title())
                ax.set_ylabel("Fit Value")
                ax.grid(True, linestyle=':', alpha=0.7)

            axs[-1].set_xlabel("Index")
            plt.suptitle(f"LOCO Fit Parameters – Iteration {i + 1}", fontsize=10)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()



        ring = set_correction(ring, fit_dict['quads'], quads_ords , individuals=individuals, skewness=skewness)



        print("Computing chi-squared after correction ...")


        cfg = RMConfig(dkick=cor_kicks, bpm_ords=bpm_ords, cm_ords=cm_ords, HCMCoupling=HCMCoupling,
                       VCMCoupling=VCMCoupling, rfStep=rf_step)
        orm_model_ = response_matrix(ring, config=cfg)


        weights_flat = weight_matrix(weights, includeDispersion,hor_dispersion_weight, ver_dispersion_weight, nHBPM, nVBPM, nHorCOR, nVerCOR)


        measured_orm_flat = orm_measured.reshape(-1, 1, order='F')
        model_orm_flat = orm_model_.reshape(-1, 1, order='F')
        J_flat = full_jacobian.transpose(1, 2, 0).reshape(-1, full_jacobian.shape[0], order='F')

        if remove_coupling_==True:
            print("Removing XY coupling from ORM and Jacobian and weight ...")
            measured_orm_flat, model_orm_flat, weights_flat, J_flat,iNoCoupling = remove_coupling(measured_orm_flat, model_orm_flat,
                                                                                      weights_flat, J_flat,
                                                                                      nHBPM, nVBPM, nHorCOR, nVerCOR,
                                                                                      includeDispersion)

        #J_weighted = J_flat / weights_flat

        # If outlier rejection is active, reuse filtered vectors
        if outlier_rejection == True:

            y = (measured_orm_flat - model_orm_flat)  # / sigma_w2

            i1 = np.where(np.abs(y - np.mean(y)) >  sigma_outlier * np.std(y))[0]
            j1 = np.where(np.abs(y - np.mean(y)) <= sigma_outlier * np.std(y))[0]

            y_j1 = y[j1]
            i2 = np.where(np.abs(y_j1 - np.mean(y_j1)) > sigma_outlier * np.std(y_j1))[0]
            iOutliers = np.sort(np.concatenate((i1, j1[i2])))

            if iOutliers.size == 0:
                print("   No outliers in the data set.")
            else:
                print(
                    f"   std(Model-Measurement) = {1000 * np.std(measured_orm_flat - model_orm_flat):.6f} mm (with outliers)")


            orm_measured_flat = np.delete(measured_orm_flat, iOutliers, axis=0)
            orm_model_corr_flat = np.delete(model_orm_flat, iOutliers, axis=0)

        chi2_after = compute_chi_squared(
            orm_measured_flat,
            orm_model_corr_flat,
            J=J_weighted,
            bpm_noise=weights_flat
        )

        print(f"Chi-squared after correction: {chi2_after:.3f}")

    print("LOCO GN Correction Completed! :)")

    return fit_results_all, fit_dict_all, twiss_all, etax_all, etay_all, SC

def loco_correction_lm(ring, used_bpms_ords, used_cor_ords, CMstep, nIter,
                       nHorCOR, nVerCOR, nHBPM, nVBPM, orm_measured,
                       cut_, weights , quads_ords, fit_list, CAVords, includeDispersion=False
                       , fixedpathlength = False, quads_tilt = None, skew_ords = None,
                       rfStep= 40, hor_dispersion_weight=1,
                       ver_dispersion_weight=1, outlier_rejection= False, sigma_outlier = 10
                       , nLMIter=10, max_lm_lambda=15, remove_coupling_=True,
                       show_svd_plot=False, plot_fit_parameters=False, scaled = True, Starting_Lambda = 1e-3, svd_selection_method='threshold',
svd_threshold=1e-3,  apply_normalization=False,  normalization_mode='global', individuals = True, auto_correct_delta = True, dk =None, inetial_fit_parameters = None, measured_eta_x=None, measured_eta_y=None, fit_cfg=None):


    current_dk = dk
    current_dk_skew = 1e-3
    current_dk_q_tilt = 1e-6
    dk_history = []
    dk_skew_history = []
    dk_q_tilt_history = []

    fit_results_all = []
    twiss_all = []
    etax_all = []
    etay_all = []
    fit_dict_all = {}
    fit_dict_all_ = {}
    fit_results_all_ = []

    keep_mask = slice(None)  # keep everything when used as [keep_mask]

    rfStep = fixed_parameters.rfstep

    hbpm_gain = np.array(np.ones((nHBPM, 1))).ravel()
    vbpm_gain = np.array(np.ones((nVBPM, 1))).ravel()
    hbpm_coupling = np.array(np.zeros((nHBPM, 1))).ravel()
    vbpm_coupling = np.array(np.zeros((nVBPM, 1))).ravel()
    HCMEnergyShift = np.array(np.zeros((nHorCOR, 1))).ravel()
    VCMEnergyShift = np.array(np.zeros((nVerCOR, 1))).ravel()
    HCMCoupling = np.array(np.zeros((nHorCOR, 1))).ravel()
    VCMCoupling = np.array(np.zeros((nVerCOR, 1))).ravel()
    deltaqt = np.array(np.zeros((len(quads_ords), 1))).ravel()

    if inetial_fit_parameters is None:

        inetial_fit_parameters, blocks = build_initial_fit_parameters(
            ring=ring,
            fit_list=fit_list,
            nHBPM=nHBPM, nVBPM=nVBPM, nHorCOR=nHorCOR, nVerCOR=nVerCOR,
            quads_ords=quads_ords, skew_ords=skew_ords, quads_tilt=quads_tilt,
            CMstep = CMstep, rfStep = rfStep,
            individuals = individuals)

    inetial_fit_parameters = np.asarray(inetial_fit_parameters).ravel()
    current_fit_parameters = inetial_fit_parameters.copy()

    fit_parameters_history = [current_fit_parameters.copy()]


    if fixedpathlength == False or 'HCMEnergyShift' in fit_list or 'VCMEnergyShift' in fit_list:
        Fixedmomentum = True
    else:
        Fixedmomentum = False


    for i in range(nIter):
        print(f"\n\n========== LOCO Iteration {i+1} / {nIter} LM Method ==========")

        print("Measuring ORM from simulation ...")

        LMlambda = Starting_Lambda



        cfg = RMConfig(dkick=CMstep, bpm_ords=used_bpms_ords, cm_ords=used_cor_ords, HCMCoupling=HCMCoupling,
                       VCMCoupling=VCMCoupling, rfStep=rfStep)
        orm_model = response_matrix(ring, config=cfg)


        print("Measuring ORM from simulation Done ...")
     

        C11 = hbpm_gain
        C12 = hbpm_coupling
        C21 = vbpm_coupling
        C22 = vbpm_gain

        C = np.block([[np.diag(C11), np.diag(C12)], [np.diag(C21), np.diag(C22)]])

        orm_model = C @ orm_model

        rfStep = rfStep

        include_quads = 'quads' in fit_list
        include_skew = 'skew_quads' in fit_list
        include_quads_tilt = 'quads_tilt' in fit_list
        include_bpm_gain = ('hbpm_gain' in fit_list) or ('vbpm_gain' in fit_list)
        include_cor_kick = ('hcor_cal' in fit_list) or ('vcor_cal' in fit_list)
        include_cor_coupling = ('hcor_coupling' in fit_list) or ('vcor_coupling' in fit_list)
        include_bpm_coupling = ('hbpm_coupling' in fit_list) or ('vbpm_coupling' in fit_list)
        include_HCMEnergyShift = ('HCMEnergyShift' in fit_list)
        include_VCMEnergyShift = ('VCMEnergyShift' in fit_list)
        include_delta_RF_frequency = ('delta_RF' in fit_list)

        print("Compute Jacopian ...")
        full_jacobian, delta_q, delta_skew, delta_quads_tilt = compute_jacobian(
            ring, C_model=orm_model, dkick = CMstep,
            bpm_indexes= used_bpms_ords, CMords = used_cor_ords, quads_ind = quads_ords,
            nHorCOR =  nHorCOR, nVerCOR= nVerCOR, nHBPM=nHBPM, nVBPM= nVBPM,
            C=C, CAVords=CAVords,
            dk=fixed_parameters.dk,
            skew_ind=skew_ords,
            includeDispersion=includeDispersion,
            delta_skew_=fixed_parameters.delta_skew,
            delta_q_tilt = fixed_parameters.delta_q_tilt,
            include_quads=include_quads,
            include_skew=include_skew,
            include_bpm_gain=include_bpm_gain,
            include_cor_kick=include_cor_kick,
            include_cor_coupling=include_cor_coupling,
            include_quads_tilt=include_quads_tilt,
            include_bpm_coupling=include_bpm_coupling,
            include_delta_RF_frequency=include_delta_RF_frequency,
            include_HCMEnergyShift=include_HCMEnergyShift,
            include_VCMEnergyShift=include_VCMEnergyShift,
            rf_step=fixed_parameters.rfstep,
            individuals=individuals,
            auto_correct_delta=auto_correct_delta,
            VCMCoupling = VCMCoupling,
            HCMCoupling = HCMCoupling,
            measured_eta_x=measured_eta_x,
            measured_eta_y=measured_eta_y,
            quads_tilt_fit=deltaqt,
            fit_cfg =fit_cfg
        )

        print("Compute Jacopian Done...")

        if delta_q is not None:
            current_dk = np.copy(delta_q)
        dk_history.append(current_dk)

        if delta_skew is not None:
            current_dk_skew = np.copy(delta_skew)
        dk_skew_history.append(current_dk_skew)


        if delta_quads_tilt is not None:
            current_dk_q_tilt = np.copy(delta_quads_tilt)
        dk_q_tilt_history.append(current_dk_q_tilt)

        if Fixedmomentum == True:

            print('Fixedmomentum == True')

            AlphaMCF = get_mcf(ring)

            Frequency = fixed_parameters.Frequency
            eta_x_mcf = -AlphaMCF * Frequency * measured_eta_x / rfStep
            eta_y_mcf = -AlphaMCF * Frequency * measured_eta_y / rfStep

            for i in range(nHorCOR):

                orm_model[:nHBPM, i] = orm_model[:nHBPM, i]  + HCMEnergyShift[i] *  eta_x_mcf
                orm_model[nHBPM:, i] = orm_model[nHBPM:, i] + HCMEnergyShift[i] * eta_y_mcf

            for i in range(nVerCOR):

                orm_model[:nHBPM, i + nHorCOR] = orm_model[:nHBPM, i + nHorCOR]  + VCMEnergyShift[i] *  eta_x_mcf
                orm_model[nHBPM:, i + nHorCOR] = orm_model[nHBPM:, i + nHorCOR] + VCMEnergyShift[i] * eta_y_mcf

        weights_flat = weight_matrix(weights, includeDispersion,hor_dispersion_weight, ver_dispersion_weight, nHBPM, nVBPM, nHorCOR, nVerCOR)
        measured_orm_flat = orm_measured.reshape(-1, 1, order='F')
        model_orm_flat = orm_model.reshape(-1, 1, order='F')
        J_flat = full_jacobian.transpose(1, 2, 0).reshape(-1, full_jacobian.shape[0], order='F')




        if remove_coupling_==True:


            print("Removing XY coupling from ORM and Jacobian and weight ...")
            measured_orm_flat, model_orm_flat, weights_flat, J_flat, iNoCoupling = remove_coupling(measured_orm_flat, model_orm_flat,
                                                                                      weights_flat, J_flat,
                                                                                      nHBPM, nVBPM, nHorCOR, nVerCOR,
                                                                                      includeDispersion)
        J_weighted = J_flat / weights_flat

        if outlier_rejection == True:
            y = (measured_orm_flat - model_orm_flat).ravel()
            y_mean = np.mean(y)
            y_std = np.std(y, ddof=1)
            i1 = np.where(np.abs(y - y_mean) > sigma_outlier * y_std)[0]
            j1 = np.where(np.abs(y - y_mean) <= sigma_outlier * y_std)[0]

            y_j1 = y[j1]
            yj_mean = np.mean(y_j1)
            yj_std = np.std(y_j1, ddof=1)
            i2 = np.where(np.abs(y_j1 - yj_mean) > sigma_outlier * yj_std)[0]

            iOutliers = np.sort(np.concatenate((i1, j1[i2])))

            std_with_outliers_mm = 1000.0 * np.std(measured_orm_flat - model_orm_flat, ddof=1)
            print(f"   std(Model-Measurement) = {std_with_outliers_mm:.6f} mm (with outliers)")

            if iOutliers.size == 0:
                print("   No outliers in the data set.")
            else:
                print(f"   {iOutliers.size} outliers removed out of {len(y)} points (> {sigma_outlier} sigma) "
                      f"({len(i1)} first test + {len(i2)} second test).")
                keep = np.ones(len(y), dtype=bool)
                keep[iOutliers] = False
                keep_mask = keep
                measured_orm_flat = measured_orm_flat[keep_mask]
                model_orm_flat = model_orm_flat[keep_mask]
                weights_flat = weights_flat[keep_mask, :]
                J_flat = J_flat[keep_mask, :]
                J_weighted = J_flat / weights_flat

                std_after_mm = 1000.0 * np.std(measured_orm_flat - model_orm_flat, ddof=1)
                print(f"   std(Model-Measurement) = {std_after_mm:.6f} mm")

        if apply_normalization == True:


            if normalization_mode == 'global':
                J_weighted, norm_factors = normalize_jacobian_global(J_weighted, model_orm_flat, weights_flat)
            elif normalization_mode == 'component':
                J_weighted, norm_factors = normalize_jacobian_componentwise(ring,
                    J_weighted, model_orm_flat, weights_flat,
                    nHBPM, nVBPM, nHorCOR, nVerCOR,CMstep,
                    fit_list, quads_ords, skew_ords, fixed_parameters.rfstep
                )



        chi2_0 = compute_chi_squared(
            measured_orm_flat,
            model_orm_flat,
            J=J_weighted,
            bpm_noise=weights_flat
        )

        print(f"Initial Chi^2 before inner loop: {chi2_0:.4e}")
        y = (measured_orm_flat - model_orm_flat) / weights_flat

        for j in range(nLMIter):

            print(f"\n========== Inner Iteration {j+1} / {nLMIter} – LM ==========")

            ay = J_weighted.T @ y
            C = J_weighted.T @ J_weighted
            C += (LMlambda * np.diag(np.diag(C)) if scaled else LMlambda * np.eye(C.shape[0]))

            print("Performing SVD of weighted Jacobian ...")

            U, S, Vh = np.linalg.svd(C, full_matrices=False)


            if svd_selection_method == 'threshold':
                # Keep SVs greater than threshold * max(S)
                Ivec = np.where(S > svd_threshold * np.max(S))[0]


            elif svd_selection_method == 'rank':

                print("  Performing rank-based singular value selection ...")

                ChiSquareVector = np.full(len(S), np.nan)

                LastGoodSvalue = 0

                for i in reversed(range(1, len(S) + 1)):

                    try:

                        # reconstruct Amod = U_i * S_i

                        Amod = U[:, :i] @ np.diag(S[:i])

                        # solve b = Amod \ y

                        b = np.linalg.lstsq(Amod, y, rcond=None)[0]

                        # back-transform using V (to match MATLAB logic)

                        b = V[:, :i] @ b

                        #  compute model response update

                        Mfit = weights_flat * (J_flat @ b)

                        Mmodelnew = model_orm_flat + Mfit

                        #  compute chi² for these SVs

                        chi2 = np.sum(((measured_orm_flat - Mmodelnew) / weights_flat) ** 2) / len(weights_flat)

                        ChiSquareVector[i - 1] = chi2

                        # test covariance matrix

                        Cmod = Amod.T @ Amod

                        np.linalg.inv(Cmod)  # will raise if ill-conditioned

                        LastGoodSvalue = i

                        Ivec = np.arange(i)

                        break  # found the largest stable SV count

                    except np.linalg.LinAlgError:

                        continue



            elif svd_selection_method == 'user_input':

                Ivec = np.arange(min(cut_, len(S)))


            elif svd_selection_method == 'interactive':

                sv_indices = np.arange(len(S))

                # ---------- Plot 1: Raw SVD spectrum BEFORE input ----------

                print("\n>>> SVD plots displayed to help you choose :)")

                print("You can enter a range like 0:20 for example")

                plt.figure(figsize=(10, 5))

                plt.subplot(1, 2, 1)
                plt.semilogy(sv_indices, S, 'b.-')
                plt.xlabel("Singular Value Index")
                plt.ylabel("Magnitude")
                # plt.title("Singular Value Spectrum (log scale)")
                # plt.grid(True, linestyle=':', alpha=0.6)

                plt.subplot(1, 2, 2)
                plt.plot(sv_indices, S / np.max(S), 'b.-')
                plt.xlabel("Singular Value Index")
                plt.ylabel("Magnitude / max(SV)")
                # plt.title("Normalized Singular Value Spectrum")
                # plt.grid(True, linestyle=':', alpha=0.6)

                # plt.suptitle("Inspect SVD Spectrum Before Selecting Indices", fontsize=12)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.show()
                plt.pause(0.1)  # <- This forces the plot to render before input box appears
                time.sleep(1)  # <- Add short sleep if still racing with input popup

                # ---------- User input ----------

                user_input = input("Please enter the singulare values indices to be used (e.g. 0:20): ")

                try:

                    if ':' in user_input:

                        parts = user_input.split(':')

                        start = int(parts[0])

                        end = int(parts[1])

                        Ivec = np.arange(start, end)

                    else:

                        Ivec = np.array([int(x.strip()) for x in user_input.split(',')])

                    # Safety clip

                    Ivec = Ivec[Ivec < len(S)]

                except Exception as e:

                    raise ValueError("Invalid input format for singular value indices.") from e

            if show_svd_plot:
                print("\n SVD plots displayed ..")
                plt.figure(figsize=(10, 3))
                sv_indices = np.arange(len(S))

                plt.semilogy(Ivec, S[Ivec], color='green', linewidth=1.0, label='Used')
                unused_indices = np.setdiff1d(sv_indices, Ivec)
                plt.semilogy(unused_indices, S[unused_indices], color='red', linewidth=1.0, label='Cut Off')

                plt.xlabel("Singular Value Index")
                plt.ylabel("Magnitude")
                plt.title(f"SVD Spectrum")
                plt.legend()
                plt.tight_layout()
                plt.show()

            print(f"SVD retained: {len(Ivec)} / {len(S)} singular values")



            b = U[:, Ivec].T @ ay
            b = np.diag(1.0 / S[Ivec]) @ b
            b = Vh.T[:, Ivec] @ b

            old_fit_parameters = current_fit_parameters.copy()

            fit_results = np.asarray(b).reshape(-1)  # (nParams,)



            if apply_normalization == True:

                nf = np.asarray(norm_factors)
                # reduce nf to a 1-D vector
                if nf.ndim == 2:
                    if nf.shape[0] == nf.shape[1]:
                        # take per-parameter factors from diagonal
                        nf = np.diag(nf)
                    else:
                        # unexpected 2D shape: fall back to flatten
                        nf = nf.ravel()
                else:
                    nf = nf.ravel()

                if nf.size != fit_results.size:
                    raise ValueError(
                        f"norm_factors size mismatch: expected {fit_results.size}, got {nf.size}"
                    )

                fit_results = fit_results / nf


            # Sanity checks
            if fit_results.ndim != 1:
                raise ValueError(f"fit_results must be 1-D, got shape {fit_results.shape}")
            if current_fit_parameters.ndim != 1:
                current_fit_parameters = np.asarray(current_fit_parameters).ravel()
            if inetial_fit_parameters.ndim != 1:
                inetial_fit_parameters = np.asarray(inetial_fit_parameters).ravel()

            # Combine with previous parameters


            idx = 0

            if 'hbpm_gain' in fit_list:

                idx += nHBPM

            if 'hbpm_coupling' in fit_list:

                idx += nHBPM

            if 'vbpm_coupling' in fit_list:

                idx += nVBPM

            if 'vbpm_gain' in fit_list:

                idx += nVBPM

            if 'hcor_cal' in fit_list:

                idx += nHorCOR

            if 'vcor_cal' in fit_list:

                idx += nVerCOR

            if 'hcor_coupling' in fit_list:


                idx += nHorCOR

            if 'vcor_coupling' in fit_list:


                idx += nVerCOR

            if 'HCMEnergyShift' in fit_list:
                idx += nHorCOR

            if 'VCMEnergyShift' in fit_list:

                idx += nVerCOR

            if 'delta_rf' in fit_list:
                idx += 1

            if 'quads' in fit_list:
                idx += len(quads_ords)

            if 'skew_quads' in fit_list:
                idx += len(skew_ords)

            if 'quads_tilt' in fit_list:
                deltaqt0 = old_fit_parameters[idx:idx + len(quads_ords)]
                idx += len(quads_ords)


            new_fit_parameters = old_fit_parameters + fit_results



            idx = 0
            fit_dict = {}

            if 'hbpm_gain' in fit_list:
                fit_dict['hbpm_gain'] = new_fit_parameters[idx:idx + nHBPM]

                hbpm_gain = new_fit_parameters[idx:idx + nHBPM]
                idx += nHBPM

            if 'hbpm_coupling' in fit_list:
                fit_dict['hbpm_coupling'] = new_fit_parameters[idx:idx + nHBPM]
                hbpm_coupling = new_fit_parameters[idx:idx + nHBPM]
                idx += nHBPM


            if 'vbpm_coupling' in fit_list:
                fit_dict['vbpm_coupling'] = new_fit_parameters[idx:idx + nVBPM]
                vbpm_coupling = new_fit_parameters[idx:idx + nVBPM]
                idx += nVBPM

            if 'vbpm_gain' in fit_list:
                fit_dict['vbpm_gain'] = new_fit_parameters[idx:idx + nVBPM]
                vbpm_gain = new_fit_parameters[idx:idx + nVBPM]
                idx += nVBPM

            if 'hcor_cal' in fit_list:
                fit_dict['hcor_cal'] = new_fit_parameters[idx:idx + nHorCOR]

                CMstep[0] = new_fit_parameters[idx:idx + nHorCOR]
                idx += nHorCOR

            if 'vcor_cal' in fit_list:
                fit_dict['vcor_cal'] = new_fit_parameters[idx:idx + nVerCOR]
                CMstep[1] = new_fit_parameters[idx:idx + nVerCOR]
                idx += nVerCOR

            if 'hcor_coupling' in fit_list:
                fit_dict['hcor_coupling'] = new_fit_parameters[idx:idx + nHorCOR]
                HCMCoupling = new_fit_parameters[idx:idx + nHorCOR]

                idx += nHorCOR

            if 'vcor_coupling' in fit_list:
                fit_dict['vcor_coupling'] = new_fit_parameters[idx:idx + nVerCOR]
                VCMCoupling = new_fit_parameters[idx:idx + nVerCOR]

                idx += nVerCOR

            if 'HCMEnergyShift' in fit_list:
                fit_dict['HCMEnergyShift'] = new_fit_parameters[idx:idx + nHorCOR]
                HCMEnergyShift = new_fit_parameters[idx:idx + nHorCOR]
                idx += nHorCOR

            if 'VCMEnergyShift' in fit_list:
                fit_dict['VCMEnergyShift'] = new_fit_parameters[idx:idx + nVerCOR]
                VCMEnergyShift = new_fit_parameters[idx:idx + nVerCOR]

                idx += nVerCOR

            if 'delta_rf' in fit_list:
                fit_dict['delta_rf'] = new_fit_parameters[idx:idx + 1]
                rfStep = new_fit_parameters[idx:idx + 1]
                idx += 1


            if 'quads' in fit_list:
                fit_dict['quads'] = new_fit_parameters[idx:idx + len(quads_ords)]
                idx += len(quads_ords)


            if 'skew_quads' in fit_list:
                fit_dict['skew_quads'] = new_fit_parameters[idx:idx + len(skew_ords)]
                idx += len(skew_ords)


            if 'quads_tilt' in fit_list:
                fit_dict['quads_tilt'] = new_fit_parameters[idx:idx + len(quads_ords)]
                deltaqt = new_fit_parameters[idx:idx + len(quads_ords)]
                idx += len(quads_ords)

            fit_dict_all_[j] = fit_dict.copy()
            fit_results_all_.append(fit_results)


            # quads initial values
            if 'quads' in fit_list:

                delta0q = _initial_values_for_block(
                    ring, quads_ords,
                    block="quads",
                    individuals=individuals,
                    config=fit_cfg,  # your FitInitConfig instance
                )

            if 'skew_quads' in fit_list:

                delta0s = _initial_values_for_block(
                    ring, skew_ords,
                    block="skew_quads",
                    individuals=individuals,
                    config=fit_cfg,
                )

            if 'quads' in fit_list:

               deltaq = np.array(fit_dict['quads']).ravel()

            if 'skew_quads' in fit_list:
                deltas = np.array(fit_dict['skew_quads']).ravel()

            if 'quads_tilt' in fit_list:
                deltaqt = np.array(fit_dict['quads_tilt']).ravel()

            if 'quads' in fit_list:
                set_correction(ring, deltaq, quads_ords, individuals=individuals, block='quads',config=fit_cfg)

            if 'skew_quads' in fit_list:
                set_correction(ring, deltas, skew_ords, individuals=individuals, block='skew_quads',config=fit_cfg)

            if 'quads_tilt' in fit_list:
                set_correction_tilt(ring, deltaqt, quads_ords,config=fit_cfg)



            cfg = RMConfig(dkick=CMstep, bpm_ords=used_bpms_ords, cm_ords=used_cor_ords, HCMCoupling=HCMCoupling,
                           VCMCoupling=VCMCoupling, rfStep=rfStep)
            orm_model_LM = response_matrix(ring, config=cfg)



            C11 = hbpm_gain # shape (239,)
            C12 = hbpm_coupling
            C21 = vbpm_coupling
            C22 = vbpm_gain


            C = np.block([[np.diag(C11), np.diag(C12)], [np.diag(C21), np.diag(C22)]])


            orm_model_LM = C @ orm_model_LM

            if Fixedmomentum == True:

                AlphaMCF = get_mcf(ring)
                Frequency = fixed_parameters.Frequency


                eta_x_mcf = -AlphaMCF * Frequency * measured_eta_x / rfStep
                eta_y_mcf = -AlphaMCF * Frequency * measured_eta_y / rfStep


                for i in range(nHorCOR):
                    orm_model_LM[:nHBPM, i] = orm_model_LM[:nHBPM, i] + HCMEnergyShift[i] * eta_x_mcf
                    orm_model_LM[nHBPM:, i] = orm_model_LM[nHBPM:, i] + HCMEnergyShift[i] * eta_y_mcf

                for i in range(nVerCOR):
                    orm_model_LM[:nHBPM, i + nHorCOR] = orm_model_LM[:nHBPM, i + nHorCOR] + VCMEnergyShift[i] * eta_x_mcf
                    orm_model_LM[nHBPM:, i + nHorCOR] = orm_model_LM[nHBPM:, i + nHorCOR] + VCMEnergyShift[i] * eta_y_mcf


            full_jacobian_LM = full_jacobian

            weights_flat_LM = weight_matrix(weights, includeDispersion, hor_dispersion_weight, ver_dispersion_weight,
                                         nHBPM, nVBPM, nHorCOR, nVerCOR)

            measured_orm_flat = orm_measured.reshape(-1, 1, order='F')
            model_orm_flat_LM = orm_model_LM.reshape(-1, 1, order='F')
            J_flat_LM = full_jacobian_LM.transpose(1, 2, 0).reshape(-1, full_jacobian_LM.shape[0], order='F')



            if remove_coupling_ == True:
                #print("Removing XY coupling from ORM and Jacobian and weight ...")
                measured_orm_flat, model_orm_flat_LM, weights_flat_LM, J_flat_LM, iNoCoupling = remove_coupling(
                    measured_orm_flat,
                    model_orm_flat_LM, weights_flat_LM,
                    J_flat_LM,
                    nHBPM, nVBPM, nHorCOR,
                    nVerCOR,
                    includeDispersion)

            J_weighted_LM = J_flat_LM / weights_flat_LM

            # If outlier rejection is active, reuse filtered vectors
            if outlier_rejection == True:

                    measured_orm_flat = measured_orm_flat[keep_mask]
                    model_orm_flat_LM = model_orm_flat_LM[keep_mask]
                    weights_flat_LM = weights_flat_LM[keep_mask, :]
                    J_flat_LM = J_flat_LM[keep_mask, :]

                    J_weighted_LM = J_flat_LM / weights_flat_LM


            y_LM = (measured_orm_flat - model_orm_flat_LM) / weights_flat_LM

            chi2_new = compute_chi_squared(

                measured_orm_flat,
                model_orm_flat_LM,
                J=J_weighted_LM,
                bpm_noise=weights_flat_LM
            )


            # Apply corrections

            print(f"Chi² after correction: {chi2_new:.4e} (previous: {chi2_0:.4e})")

            if chi2_new < chi2_0:
                print("Chi² improved — reducing lambda")
                chi2_0 = chi2_new

                LMlambda = LMlambda / 10
                current_fit_parameters = new_fit_parameters.copy()
                fit_parameters_history.append(current_fit_parameters.copy())

                break

            else:
                print("Chi² did not improve — increasing lambda")


                LMlambda *= 10.0

                if 'quads' in fit_list:
                    deltaq = delta0q
                    set_correction(ring, deltaq, quads_ords, individuals=individuals, block='quads',config=fit_cfg)

                if 'skew_quads' in fit_list:
                    deltas = delta0s
                    set_correction(ring, deltas, skew_ords, individuals=individuals, block='skew_quads',config=fit_cfg)

                if 'quads_tilt' in fit_list:
                    deltaqt = deltaqt0
                    set_correction_tilt(ring, deltaqt, quads_ords,config=fit_cfg)




                current_fit_parameters = old_fit_parameters.copy()

                if LMlambda > max_lm_lambda:
                    print("Lambda exceeded maximum. Stopping.")
                    break


            idx = 0
            fit_dict = {}

            if 'hbpm_gain' in fit_list:

                fit_dict['hbpm_gain'] = current_fit_parameters[idx:idx + nHBPM]

                hbpm_gain = current_fit_parameters[idx:idx + nHBPM]
                idx += nHBPM


            if 'hbpm_coupling' in fit_list:

                fit_dict['hbpm_coupling'] = current_fit_parameters[idx:idx + nHBPM]
                hbpm_coupling = current_fit_parameters[idx:idx + nHBPM]
                idx += nHBPM

            if 'vbpm_coupling' in fit_list:

                fit_dict['vbpm_coupling'] = current_fit_parameters[idx:idx + nVBPM]
                vbpm_coupling = current_fit_parameters[idx:idx + nVBPM]
                idx += nVBPM


            if 'vbpm_gain' in fit_list:

                fit_dict['vbpm_gain'] = current_fit_parameters[idx:idx + nVBPM]
                vbpm_gain = current_fit_parameters[idx:idx + nVBPM]
                idx += nVBPM





            if 'hcor_cal' in fit_list:

                fit_dict['hcor_cal'] = current_fit_parameters[idx:idx + nHorCOR]

                CMstep[0] = current_fit_parameters[idx:idx + nHorCOR]
                idx += nHorCOR

            if 'vcor_cal' in fit_list:
                fit_dict['vcor_cal'] = current_fit_parameters[idx:idx + nVerCOR]
                CMstep[1] = current_fit_parameters[idx:idx + nVerCOR]
                idx += nVerCOR

            if 'hcor_coupling' in fit_list:
                fit_dict['hcor_coupling'] = current_fit_parameters[idx:idx + nHorCOR]
                HCMCoupling  = current_fit_parameters[idx:idx + nHorCOR]

                idx += nHorCOR

            if 'vcor_coupling' in fit_list:
                fit_dict['vcor_coupling'] = current_fit_parameters[idx:idx + nVerCOR]
                VCMCoupling = current_fit_parameters[idx:idx + nVerCOR]

                idx += nVerCOR

            if 'HCMEnergyShift' in fit_list:
                fit_dict['HCMEnergyShift'] = current_fit_parameters[idx:idx + nHorCOR]
                HCMEnergyShift = current_fit_parameters[idx:idx + nHorCOR]
                idx += nHorCOR

            if 'VCMEnergyShift' in fit_list:
                fit_dict['VCMEnergyShift'] = current_fit_parameters[idx:idx + nVerCOR]
                VCMEnergyShift = current_fit_parameters[idx:idx + nVerCOR]

                idx += nVerCOR

            if 'delta_rf' in fit_list:
                fit_dict['delta_rf'] = current_fit_parameters[idx:idx + 1]
                rfStep = new_fit_parameters[idx:idx + 1]
                idx += 1

            if 'quads' in fit_list:
                fit_dict['quads'] = current_fit_parameters[idx:idx + len(quads_ords)]
                idx += len(quads_ords)

            if 'skew_quads' in fit_list:
                fit_dict['skew_quads'] = current_fit_parameters[idx:idx + len(skew_ords)]
                idx += len(skew_ords)

            if 'quads_tilt' in fit_list:
                fit_dict['quads_tilt'] = current_fit_parameters[idx:idx + len(quads_ords)]
                deltaqt = current_fit_parameters[idx:idx + len(quads_ords)]
                idx += len(quads_ords)

            fit_dict_all_[j] = fit_dict.copy()
            fit_results_all_.append(fit_results)




            j = j+1
            if j < nLMIter:
                C = J_weighted.T @ J_weighted
                U, S, Vh = np.linalg.svd(C, full_matrices=False)




        fit_dict = {}
        idx = 0

        if 'hbpm_gain' in fit_list:
            fit_dict['hbpm_gain'] = fit_results[idx:idx + nHBPM]
            idx += nHBPM



        if 'hbpm_coupling' in fit_list:
            fit_dict['hbpm_coupling'] = fit_results[idx:idx + nHBPM]
            idx += nHBPM


        if 'vbpm_coupling' in fit_list:
            fit_dict['vbpm_coupling'] = fit_results[idx:idx + nVBPM]
            idx += nVBPM


        if 'vbpm_gain' in fit_list:
            fit_dict['vbpm_gain'] = fit_results[idx:idx + nVBPM]
            idx += nVBPM





        if 'hcor_cal' in fit_list:
            fit_dict['hcor_cal'] = fit_results[idx:idx + nHorCOR]
            idx += nHorCOR
            #new_hcor_kick = fit_results[idx:idx + nHorCOR]
            #CMstep[0] = CMstep[0] + new_hcor_kick,

        if 'vcor_cal' in fit_list:
            fit_dict['vcor_cal'] = fit_results[idx:idx + nVerCOR]
            idx += nVerCOR
            #new_vcor_kick = fit_results[idx:idx + nVerCOR]
            #CMstep[1] = CMstep[1] + new_vcor_kick,

        if 'hcor_coupling' in fit_list:
            fit_dict['hcor_coupling'] = fit_results[idx:idx + nHorCOR]
            idx += nHorCOR

        if 'vcor_coupling' in fit_list:
            fit_dict['vcor_coupling'] = fit_results[idx:idx + nVerCOR]
            idx += nVerCOR

        if 'HCMEnergyShift' in fit_list:
            fit_dict['HCMEnergyShift'] = fit_results[idx:idx + nHorCOR]
            idx += nHorCOR

        if 'VCMEnergyShift' in fit_list:
            fit_dict['VCMEnergyShift'] = fit_results[idx:idx + nVerCOR]
            idx += nVerCOR

        if 'delta_rf' in fit_list:
            fit_dict['delta_rf'] = fit_results[idx:idx + 1]
            idx += 1

        if 'quads' in fit_list:
            fit_dict['quads'] = fit_results[idx:idx + len(quads_ords)]
            idx += len(quads_ords)

        if 'skew_quads' in fit_list:
            fit_dict['skew_quads'] = fit_results[idx:idx + len(skew_ords)]
            idx += len(skew_ords)

        if 'quads_tilt' in fit_list:
            fit_dict['quads_tilt'] = fit_results[idx:idx + len(quads_ords)]
            idx += len(quads_ords)


        fit_dict_all[i] = fit_dict_all_.copy()
        fit_results_all.append(fit_results_all_)


        print("Computing chi-squared after correction ...")


        cfg = RMConfig(dkick=CMstep, bpm_ords=used_bpms_ords, cm_ords=used_cor_ords, HCMCoupling=HCMCoupling,
                       VCMCoupling=VCMCoupling, rfStep=rfStep)
        orm_model_ = response_matrix(ring, config=cfg)

        C11 = hbpm_gain  # shape (239,)
        C12 = hbpm_coupling
        C21 = vbpm_coupling
        C22 = vbpm_gain

        C = np.block([[np.diag(C11), np.diag(C12)], [np.diag(C21), np.diag(C22)]])

        orm_model_ =  C @ orm_model_

        if Fixedmomentum == True:


            AlphaMCF = get_mcf(ring)

            Frequency = fixed_parameters.Frequency
            eta_x_mcf = -AlphaMCF * Frequency * measured_eta_x / rfStep
            eta_y_mcf = -AlphaMCF * Frequency * measured_eta_y / rfStep

        for i in range(nHorCOR):

                orm_model_[:nHBPM, i] = orm_model_[:nHBPM, i]  + HCMEnergyShift[i] *  eta_x_mcf
                orm_model_[nHBPM:, i] = orm_model_[nHBPM:, i] + HCMEnergyShift[i] * eta_y_mcf

        for i in range(nVerCOR):

            orm_model_[:nHBPM, i + nHorCOR] = orm_model_[:nHBPM, i + nHorCOR]  + VCMEnergyShift[i] *  eta_x_mcf
            orm_model_[nHBPM:, i + nHorCOR] = orm_model_[nHBPM:, i + nHorCOR] + VCMEnergyShift[i] * eta_y_mcf


        # If outlier rejection is active, reuse filtered vectors

        weights_flat = weight_matrix(weights, includeDispersion,hor_dispersion_weight, ver_dispersion_weight, nHBPM, nVBPM, nHorCOR, nVerCOR)
        measured_orm_flat = orm_measured.reshape(-1, 1, order='F')
        model_orm_flat = orm_model_.reshape(-1, 1, order='F')
        J_flat = full_jacobian.transpose(1, 2, 0).reshape(-1, full_jacobian.shape[0], order='F')

        if remove_coupling_==True:
            #print("Removing XY coupling from ORM and Jacobian and weight ...")
            measured_orm_flat, model_orm_flat, weights_flat, J_flat,iNoCoupling = remove_coupling(measured_orm_flat, model_orm_flat,
                                                                                      weights_flat, J_flat,
                                                                                      nHBPM, nVBPM, nHorCOR, nVerCOR,
                                                                                      includeDispersion)

        J_weighted = J_flat / weights_flat

        if outlier_rejection == True:
            measured_orm_flat = measured_orm_flat[keep_mask]
            model_orm_flat = model_orm_flat[keep_mask]
            weights_flat = weights_flat[keep_mask, :]
            J_flat = J_flat[keep_mask, :]

            J_weighted = J_flat / weights_flat


        chi2_after= compute_chi_squared(
            measured_orm_flat,
            model_orm_flat,
            J=J_weighted,
            bpm_noise=weights_flat
        )


        print(f"Chi-squared after correction: {chi2_after:.3f}")

        if plot_fit_parameters:
            print("Plotting fit parameters ...")
            n_plots = len(fit_dict)
            fig, axs = plt.subplots(n_plots, 1, figsize=(5, 6))
            if n_plots == 1:
                axs = [axs]

            for ax, (key, values) in zip(axs, fit_dict.items()):
                ax.plot(values, marker='o', linestyle='-', markersize=3)
                ax.set_title(key.replace('_', ' ').title())
                ax.set_ylabel("Fit Value")
                ax.grid(True, linestyle=':', alpha=0.7)

            axs[-1].set_xlabel("Index")
            plt.suptitle(f"LOCO Fit Parameters – Iteration {i + 1}", fontsize=10)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

    print("LOCO LM Correction Completed! :)")
    return current_fit_parameters, ring





def plot_data(s_pos, data, xlabel, ylabel, title):
    plt.figure(figsize=(7, 3))
    plt.plot(s_pos, data, color='navy')  # Deep blue color
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    # plt.grid(True, which='both', linestyle=':', color='gray')
    plt.tight_layout()
    plt.show()
