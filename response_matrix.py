import at
import logging
from pyloco_config import RMConfig as config
from pyloco_config import _cfg_get, get_mcf
from typing import Optional, Union, List, Tuple
import numpy as np



def response_matrix(

    ring,
    bpm_ords =None,
    cm_ords=None,
    cav_ords=None,
    dkick=1e-5,
    bidirectional=True,
    includeDispersion=False,
    rfStep = 40,
    delta_coupling=1e-6,
    coupling_orm=False,
    calculator = 'Linear',
    NewVectorizedMethod = True,
    fixedpathlength = False,
    log_info=False,
    HCMCoupling=None,
    VCMCoupling=None, Frequency = None, HarmNumber = None,

    config=None,
):
    bpm_ords = _cfg_get(config, "bpm_ords", bpm_ords)
    cm_ords = _cfg_get(config, "cm_ords", cm_ords)
    cav_ords = _cfg_get(config, "cav_ords", cav_ords)
    dkick = _cfg_get(config, "dkick", dkick)
    bidirectional = _cfg_get(config, "bidirectional", bidirectional)
    includeDispersion = _cfg_get(config, "includeDispersion", includeDispersion)
    rfStep = _cfg_get(config, "rfStep", rfStep)
    delta_coupling = _cfg_get(config, "delta_coupling", delta_coupling)
    coupling_orm = _cfg_get(config, "coupling_orm", coupling_orm)
    calculator = _cfg_get(config, "calculator", calculator)
    NewVectorizedMethod = _cfg_get(config, "NewVectorizedMethod", NewVectorizedMethod)
    fixedpathlength = _cfg_get(config, "fixedpathlength", fixedpathlength)
    log_info = _cfg_get(config, "log_info", log_info)
    HCMCoupling = _cfg_get(config, "HCMCoupling", HCMCoupling)
    VCMCoupling = _cfg_get(config, "VCMCoupling", VCMCoupling)
    Frequency = _cfg_get(config, "Frequency", Frequency)
    HarmNumber = _cfg_get(config, "HarmNumber", HarmNumber)
    RFAttr = _cfg_get(config, "RFAttr", "Frequency")

    # ---------------------------------------------------------------

    if log_info:
        try:
            LOGGER.info(
                "Calculating ORM: %s, %s",
                calculator,

            )
        except Exception:
            pass  # don't let logging break the computation

    n_bpm = len(bpm_ords)
    n_hcor, n_vcor = len(cm_ords[0]), len(cm_ords[1])
    n_cm = n_hcor + n_vcor
    response_matrix = np.full((2 * n_bpm, n_cm), np.nan)

    if HCMCoupling is None:
        HCMCoupling = np.zeros(n_hcor)

    if VCMCoupling is None:
        VCMCoupling = np.zeros(n_vcor)

    if calculator == 'Linear':

        if HCMCoupling is None:
            HCMCoupling = np.zeros(n_hcor)

        if VCMCoupling is None:
            VCMCoupling = np.zeros(n_vcor)

        n_bpm = len(bpm_ords)
        n_hcor, n_vcor = len(cm_ords[0]), len(cm_ords[1])
        n_cm = n_hcor + n_vcor
        response_matrix = np.full((2 * n_bpm, n_cm), np.nan)
        NE = len(ring)
        M44, T = at.find_m44(ring, 0, np.arange(0, NE + 1))

        ClosedOrbit = at.find_orbit4(ring, 0, np.arange(0, NE + 1))
        DP = 1e-5
        ClosedOrbitDP = at.find_orbit4(ring, DP, np.arange(0, NE + 1))
        ClosedOrbit = ClosedOrbit[1].T
        ClosedOrbitDP = ClosedOrbitDP[1].T
        T = np.transpose(T, (1, 2, 0))

        Dispersion0 = (ClosedOrbitDP - ClosedOrbit) / DP
        Dispersion = Dispersion0[:4,:]
        L0 = at.get_s_pos(ring, NE)
        MCF= get_mcf(ring)

        M44HCOR = [None] * n_hcor
        M44VCOR = [None] * n_vcor

        for i in range(n_hcor):
            idx = cm_ords[0][i]
            M44HCOR[i] = findelemm44(ring, idx, np.concatenate([ClosedOrbit[:, idx]]), dt=None)

        for i in range(n_vcor):
            idx_v = cm_ords[1][i]
            matches = np.where(cm_ords[0] == idx_v)[0]
            if matches.size > 0:
                M44VCOR[i] = M44HCOR[matches[0]]
            else:
                M44VCOR[i] = findelemm44(ring, idx_v, np.concatenate([ClosedOrbit[:, idx_v]]), dt=None)

        HCORTheta = np.zeros((4, n_hcor))
        VCORTheta = np.zeros((4, n_vcor))

        HCORTheta[1, :] = dkick[0][:]
        HCORTheta[3, :] = HCMCoupling   *   dkick[0]

        VCORTheta[1, :] = VCMCoupling   *   dkick[1]
        VCORTheta[3, :] = dkick[1][:]

        for i in range(n_hcor):
            CI = cm_ords[0][i]

            InverseT = np.linalg.inv(T[:, :, CI])

            OrbitEntrance = np.linalg.inv(
                np.eye(4) - T[:, :, CI] @ M44 @ InverseT
            ) @ T[:, :, CI] @ M44 @ InverseT @ (
                np.eye(4) + np.linalg.inv(M44HCOR[i])
            ) @ (HCORTheta[:, i] / 2.0)

            OrbitExit = HCORTheta[:, i] / 2.0 + M44HCOR[i] @ (
                OrbitEntrance + HCORTheta[:, i] / 2.0
            )

            R0 = np.linalg.inv(T[:, :, CI + 1]) @ OrbitExit

            if NewVectorizedMethod == True:

                vectind = bpm_ords[:n_bpm]
                T3 = T[[0, 2], :, :][:, :, vectind]
                T2 = np.transpose(T3, (0, 2, 1)).reshape(n_bpm * 2, 4, order='F')

                bgtc = np.where(vectind > cm_ords[0][i] - 1)[0]
                bltc = np.where(vectind <= cm_ords[0][i] - 1)[0]

                bgtc = np.concatenate(((bgtc) * 2, (bgtc) * 2 + 1))
                bltc = np.concatenate(((bltc) * 2, (bltc) * 2 + 1))

                R0 = np.atleast_2d(R0)
                if R0.shape[0] != 4:
                    R0 = R0.T

                Tout1 = T2 @ R0
                Tout2 = T2 @ M44 @ R0
                Tout = np.zeros_like(Tout1)

                Tout[bgtc, :] = Tout1[bgtc, :]
                Tout[bltc, :] = Tout2[bltc, :]

                jjj = np.zeros((2, n_bpm), dtype=int)
                jjj[0, :] = np.arange(n_bpm)
                jjj[1, :] = np.arange(n_bpm, n_bpm * 2)

                response_matrix[jjj.ravel(order='F'), i] = Tout.ravel(order='F')

            else:
                for j in range(n_bpm):
                    bpm_idx = bpm_ords[j]
                    if bpm_idx > cm_ords[0][i]:
                        response_matrix[[j, j + n_bpm], i] = T[[0, 2], :, bpm_idx] @ R0
                    else:
                        response_matrix[[j, j + n_bpm], i] = T[[0, 2], :, bpm_idx] @ M44 @ R0

            if fixedpathlength == True:
                D = (
                    HCORTheta[1, i]
                    * (Dispersion[0, cm_ords[0][i]] + Dispersion[0, cm_ords[0][i] + 1])
                    * Dispersion[np.ix_([0, 2], bpm_ords)] / L0 / MCF / 2.0
                )
                response_matrix[:n_bpm, i] -= D[0, :].T
                response_matrix[n_bpm:, i] -= D[1, :].T

        for i in range(n_vcor):
            CI = cm_ords[1][i]

            InverseT = np.linalg.inv(T[:, :, CI])

            OrbitEntrance = np.linalg.inv(
                np.eye(4) - T[:, :, CI] @ M44 @ InverseT
            ) @ T[:, :, CI] @ M44 @ InverseT @ (
                np.eye(4) + np.linalg.inv(M44VCOR[i])
            ) @ (VCORTheta[:, i] / 2.0)

            OrbitExit = VCORTheta[:, i] / 2.0 + M44VCOR[i] @ (
                OrbitEntrance + VCORTheta[:, i] / 2.0
            )

            R0 = np.linalg.inv(T[:, :, CI + 1]) @ OrbitExit  # column 4-vector

            if NewVectorizedMethod:
                vectind = bpm_ords[:n_bpm]
                T3 = T[[0, 2], :, :][:, :, vectind]
                T2 = np.transpose(T3, (0, 2, 1)).reshape(n_bpm * 2, 4, order='F')

                bgtc = np.where(vectind > cm_ords[1][i] - 1)[0]
                bltc = np.where(vectind <= cm_ords[1][i] - 1)[0]

                bgtc = np.concatenate(((bgtc) * 2, (bgtc) * 2 + 1))
                bltc = np.concatenate(((bltc) * 2, (bltc) * 2 + 1))

                R0 = np.atleast_2d(R0)
                if R0.shape[0] != 4:
                    R0 = R0.T

                Tout1 = T2 @ R0
                Tout2 = T2 @ M44 @ R0
                Tout = np.zeros_like(Tout1)

                Tout[bgtc, :] = Tout1[bgtc, :]
                Tout[bltc, :] = Tout2[bltc, :]

                jjj = np.zeros((2, n_bpm), dtype=int)
                jjj[0, :] = np.arange(n_bpm)
                jjj[1, :] = np.arange(n_bpm, n_bpm * 2)

                response_matrix[jjj.ravel(order='F'), i+n_hcor] = Tout.ravel(order='F')

            else:
                for j in range(n_bpm):
                    bpm_idx = bpm_ords[j]
                    if bpm_idx > CI:
                        response_matrix[[j, j + n_bpm], n_hcor + i] = T[[0, 2], :, bpm_idx] @ R0
                    else:
                        response_matrix[[j, j + n_bpm], n_hcor + i] = T[[0, 2], :, bpm_idx] @ M44 @ R0

            if fixedpathlength:
                D = (
                    VCORTheta[1, i]
                    * (Dispersion[0, CI] + Dispersion[0, CI + 1])
                    * Dispersion[np.ix_([0, 2], bpm_ords)] / L0 / MCF / 2.0
                )
                response_matrix[:n_bpm, n_hcor + i] -= D[0, :]
                response_matrix[n_bpm:, n_hcor + i] -= D[1, :]

    else:

        _, orbit0 = at.find_orbit4(ring, 0, bpm_ords)
        orbit0_x = orbit0[:,0]
        orbit0_y = orbit0[:,2]

        cnt = 0
        for n_dim in range(2):  # 0 = H, 1 = V
            other_dim = 1 - n_dim
            for j, cm_ord in enumerate(cm_ords[n_dim]):
                if isinstance(dkick, (list, tuple, np.ndarray)):
                    try:
                        this_dkick = dkick[n_dim][j]
                    except Exception:
                        this_dkick = dkick[j] if isinstance(dkick[j], (int, float)) else float(dkick[j])
                else:
                    this_dkick = float(dkick)

                kick0 = ring[cm_ord].KickAngle[n_dim]
                kick1 = ring[cm_ord].KickAngle[other_dim]

                if bidirectional == True:

                    ring[cm_ord].KickAngle[n_dim] = kick0 + this_dkick / 2
                    if coupling_orm and delta_coupling:
                        ring[cm_ord].KickAngle[other_dim] = kick1 + this_dkick * delta_coupling

                    _,orbit = at.find_orbit4(ring, 0, bpm_ords)
                    orbit_plus_x = orbit[:,0]
                    orbit_plus_y = orbit[:,2]

                    ring[cm_ord].KickAngle[n_dim] = kick0
                    ring[cm_ord].KickAngle[n_dim] = kick0 - this_dkick / 2

                    if coupling_orm and delta_coupling:
                        ring[cm_ord].KickAngle[other_dim] = kick1 - this_dkick * delta_coupling

                    _,orbit = at.find_orbit4(ring, 0, bpm_ords)
                    orbit_minus_x = orbit[:,0]
                    orbit_minus_y = orbit[:,2]

                    # Reset
                    ring[cm_ord].KickAngle[n_dim] = kick0
                    ring[cm_ord].KickAngle[other_dim] = kick1

                    dx = orbit_plus_x - orbit_minus_x
                    dy = orbit_plus_y - orbit_minus_y

                else:

                    ring[cm_ord].KickAngle[n_dim] = kick0 + this_dkick
                    if coupling_orm and delta_coupling:
                        ring[cm_ord].KickAngle[other_dim] = kick1 + this_dkick * delta_coupling

                    _,orbit = at.find_orbit4(ring, 0, bpm_ords)
                    orbit_new_x = orbit[:,0]
                    orbit_new_y = orbit[:,2]

                    ring[cm_ord].KickAngle[n_dim] = kick0
                    ring[cm_ord].KickAngle[other_dim] = kick1

                    dx = orbit_new_x - orbit0_x
                    dy = orbit_new_y - orbit0_y

                response_matrix[:, cnt] = np.concatenate((dx, dy))
                cnt += 1



    if includeDispersion:

        C = 2.99792458e8

        if calculator == 'Linear':

            f_rf =Frequency
            h_rf = HarmNumber

            _, ORBITPLUS = at.find_sync_orbit(
                ring,
                (-C * rfStep * h_rf / f_rf ** 2) / 2,
                refpts=bpm_ords
            )

            dx = ORBITPLUS[:, 0]
            dy = ORBITPLUS[:, 2]

            _, ORBIT0 = at.find_sync_orbit(
                ring,
                (C * rfStep * h_rf / f_rf ** 2) / 2,
                refpts=bpm_ords
            )

            dx0 = ORBIT0[:, 0]
            dy0 = ORBIT0[:, 2]

            dispersion_meas = np.concatenate((dx - dx0, dy - dy0))
            response_matrix = np.hstack((response_matrix, dispersion_meas.reshape(-1, 1)))

        else:

            if bidirectional == True:

                shift_rf(ring, cav_ords,+rfStep / 2,attr=RFAttr)

                _,orbit = at.find_orbit4(ring, 0, bpm_ords)
                orbit_plus_x = orbit[:,0]
                orbit_plus_y = orbit[:,2]

                shift_rf(ring, cav_ords,-rfStep/2, attr=RFAttr)
                shift_rf(ring, cav_ords,-rfStep / 2, attr=RFAttr)

                _,orbit = at.find_orbit4(ring, 0, bpm_ords)
                orbit_minus_x = orbit[:,0]
                orbit_minus_y = orbit[:,2]

                shift_rf(ring, cav_ords,+rfStep / 2, attr=RFAttr)  # Restore

                dx = orbit_plus_x - orbit_minus_x - orbit0_x
                dy = orbit_plus_y - orbit_minus_y - orbit0_y

            else:
                shift_rf(ring, cav_ords,+rfStep, attr=RFAttr)

                _,orbit = at.find_orbit4(ring, 0, bpm_ords)
                orbit_new_x = orbit[:, 0]
                orbit_new_y = orbit[:,2]

                shift_rf(ring, cav_ords, -rfStep, attr=RFAttr)  # Restore

                dx = orbit_new_x - orbit0_x
                dy = orbit_new_y - orbit0_y

            dispersion_meas = np.concatenate((dx, dy))
            response_matrix = np.hstack((response_matrix, dispersion_meas.reshape(-1, 1)))

    return response_matrix


def findelemm44(ring, ELEM, orbit_in, dt=None):
    from at import element_pass
    if dt is None:
        dt = 1e-7  # default step

    orbit_in = np.asarray(orbit_in).reshape(6, 1)

    D4 = np.vstack((dt * np.eye(4), np.zeros((2, 4))))
    RIN = np.hstack((orbit_in + D4, orbit_in - D4))

    RIN6_F = np.asfortranarray(RIN)
    ROUT = element_pass(ring[ELEM], RIN6_F)

    M44 = (ROUT[0:4, 0:4] - ROUT[0:4, 4:8]) / (2 * dt)

    return M44


import numpy as np

def shift_rf(ring, cav_ords, freq_delta, attr="Frequency"):
    freq_delta = float(freq_delta)
    if isinstance(attr, (list, tuple, np.ndarray)):
        if len(attr) != len(cav_ords):
            raise ValueError("Length of 'attr' must match length of 'cav_ords'.")
        for idx, name in zip(cav_ords, attr):
            elem = ring[int(idx)]
            setattr(elem, name, getattr(elem, name) + freq_delta)
    else:
        for idx in cav_ords:
            elem = ring[int(idx)]
            setattr(elem, attr, getattr(elem, attr) + freq_delta)
