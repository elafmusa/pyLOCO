import at
import logging
from .config import RMConfig as config
from .config import _cfg_get, get_mcf, fixed_parameters
from typing import Optional, Union, List, Tuple
import numpy as np


LOGGER = logging.getLogger(__name__)


def _corrector_kick_component(corrector, plane):
    """Return ``(array, index, scale)`` for an integrated AT kick.

    ``dkick`` is public API and is always an integrated physical angle. AT's
    dedicated ``KickAngle`` storage uses that convention directly. Multipole
    dipole coefficients follow AT's canonical signs,

        dpx = -Length * dPolynomB[0]
        dpy = +Length * dPolynomA[0],

    while ``ThinMultipole`` coefficients are already integrated. A
    zero-length element using a thick multipole pass cannot generate a kick
    through its polynomial coefficients and is therefore rejected.
    """
    if plane not in (0, 1):
        raise ValueError(f"Corrector plane must be 0 or 1, got {plane!r}")

    if hasattr(corrector, "KickAngle"):
        return corrector.KickAngle, plane, 1.0

    attribute = "PolynomB" if plane == 0 else "PolynomA"
    if not hasattr(corrector, attribute):
        raise TypeError(
            f"Corrector {getattr(corrector, 'FamName', '')!r} must provide "
            "KickAngle or PolynomA/PolynomB."
        )

    coefficients = getattr(corrector, attribute)
    length = float(getattr(corrector, "Length", 0.0) or 0.0)
    pass_method = str(getattr(corrector, "PassMethod", ""))
    # AT's finite ``Multipole`` inherits from ``ThinMultipole`` in some
    # releases, so class membership alone does not identify the storage
    # convention. The pass method and zero length do.
    is_thin = length == 0.0 and "ThinMPole" in pass_method
    sign = -1.0 if plane == 0 else 1.0

    if is_thin:
        return coefficients, 0, sign
    if length > 0.0:
        return coefficients, 0, sign / length
    raise TypeError(
        f"Zero-length corrector {getattr(corrector, 'FamName', '')!r} uses "
        f"{pass_method or 'an unspecified thick pass method'} and has no "
        "integrated-kick representation. Use KickAngle or ThinMultipole."
    )


def calculate_rf_response(
        ring, bpm_ords, cav_ords, rf_step, *, calculator="Linear",
        bidirectional=True, frequency=None, harm_number=None,
        rf_attr="Frequency", orbit0=None):
    """Calculate only the RF-response column used by ``response_matrix``."""
    bpm_ords = np.atleast_1d(np.asarray(bpm_ords, dtype=int))
    calculator = "Numerical" if str(calculator).strip().lower() == "tracking" else calculator
    frequency = fixed_parameters.Frequency if frequency is None else frequency
    harm_number = fixed_parameters.HarmNumber if harm_number is None else harm_number

    if calculator in ("Linear", "Analytical"):
        speed_of_light = 2.99792458e8
        dp = (-speed_of_light * rf_step * harm_number / frequency ** 2) / 2.0
        _, plus = at.find_sync_orbit(ring, dp, refpts=bpm_ords)
        _, minus = at.find_sync_orbit(ring, -dp, refpts=bpm_ords)
        return np.concatenate((plus[:, 0] - minus[:, 0], plus[:, 2] - minus[:, 2]))

    if calculator != "Numerical":
        raise ValueError(f"Unknown calculator={calculator!r} for RF response")

    if orbit0 is None:
        _, orbit0 = at.find_orbit4(ring, 0, bpm_ords)
    orbit0_x = orbit0[:, 0]
    orbit0_y = orbit0[:, 2]
    if bidirectional:
        shift_rf(ring, cav_ords, +rf_step / 2, attr=rf_attr)
        _, plus = at.find_orbit4(ring, 0, bpm_ords)
        shift_rf(ring, cav_ords, -rf_step, attr=rf_attr)
        _, minus = at.find_orbit4(ring, 0, bpm_ords)
        shift_rf(ring, cav_ords, +rf_step / 2, attr=rf_attr)
        return np.concatenate((
            plus[:, 0] - minus[:, 0] - orbit0_x,
            plus[:, 2] - minus[:, 2] - orbit0_y,
        ))

    shift_rf(ring, cav_ords, +rf_step, attr=rf_attr)
    _, shifted = at.find_orbit4(ring, 0, bpm_ords)
    shift_rf(ring, cav_ords, -rf_step, attr=rf_attr)
    return np.concatenate((shifted[:, 0] - orbit0_x, shifted[:, 2] - orbit0_y))


def response_matrix(
    ring,
    bpm_ords=None,
    cm_ords=None,
    cav_ords=None,
    dkick=1e-5,
    bidirectional=True,
    includeDispersion=False,
    rfStep=40,
    delta_coupling=1e-6,
    coupling_orm=False,
    calculator='Linear',
    NewVectorizedMethod=True,
    fixedpathlength=True,
    log_info=False,
    HCMCoupling=None,
    VCMCoupling=None,
    Frequency=None,
    HarmNumber=None,
    config=None,
):

    # ======================================================================
    # READ CONFIGURATION
    # ======================================================================

    bpm_ords = _cfg_get(config, "bpm_ords", bpm_ords)
    cm_ords = _cfg_get(config, "cm_ords", cm_ords)
    cav_ords = _cfg_get(config, "cav_ords", cav_ords)
    dkick = _cfg_get(config, "dkick", dkick)
    bidirectional = _cfg_get(config, "bidirectional", bidirectional)
    includeDispersion = _cfg_get(
        config, "includeDispersion", includeDispersion
    )
    rfStep = _cfg_get(config, "rfStep", rfStep)
    delta_coupling = _cfg_get(
        config, "delta_coupling", delta_coupling
    )
    coupling_orm = _cfg_get(
        config, "coupling_orm", coupling_orm
    )
    calculator = _cfg_get(
        config, "calculator", calculator
    )
    # "Tracking" was emitted by older GUI/project files for the numerical
    # closed-orbit tracking implementation. Keep it as an input alias while
    # using the API's established canonical name internally.
    if isinstance(calculator, str) and calculator.strip().lower() == "tracking":
        calculator = "Numerical"
    NewVectorizedMethod = _cfg_get(
        config, "NewVectorizedMethod", NewVectorizedMethod
    )
    fixedpathlength = _cfg_get(
        config, "fixedpathlength", fixedpathlength
    )
    log_info = _cfg_get(
        config, "log_info", log_info
    )
    HCMCoupling = _cfg_get(
        config, "HCMCoupling", HCMCoupling
    )
    VCMCoupling = _cfg_get(
        config, "VCMCoupling", VCMCoupling
    )
    Frequency = _cfg_get(
        config, "Frequency", Frequency
    )
    HarmNumber = _cfg_get(
        config, "HarmNumber", HarmNumber
    )

    Frequency = (
        fixed_parameters.Frequency
        if Frequency is None
        else Frequency
    )

    HarmNumber = (
        fixed_parameters.HarmNumber
        if HarmNumber is None
        else HarmNumber
    )

    RFAttr = _cfg_get(
        config, "RFAttr", "Frequency"
    )

    # ======================================================================
    # BASIC CHECKS
    # ======================================================================

    if bpm_ords is None:
        raise ValueError("bpm_ords must be provided.")

    if cm_ords is None:
        raise ValueError("cm_ords must be provided.")

    bpm_ords = np.atleast_1d(
        np.asarray(bpm_ords, dtype=int)
    )

    if (
        isinstance(cm_ords, np.ndarray)
        and cm_ords.dtype == object
    ):
        cm_ords = cm_ords.item()

    cm_ords = (
        np.atleast_1d(
            np.asarray(cm_ords[0], dtype=int)
        ),
        np.atleast_1d(
            np.asarray(cm_ords[1], dtype=int)
        ),
    )

    n_bpm = len(bpm_ords)
    n_hcor = len(cm_ords[0])
    n_vcor = len(cm_ords[1])
    n_cm = n_hcor + n_vcor

    if HCMCoupling is None:
        HCMCoupling = np.zeros(
            n_hcor,
            dtype=float
        )
    else:
        HCMCoupling = np.asarray(
            HCMCoupling,
            dtype=float
        ).reshape(-1)

    if VCMCoupling is None:
        VCMCoupling = np.zeros(
            n_vcor,
            dtype=float
        )
    else:
        VCMCoupling = np.asarray(
            VCMCoupling,
            dtype=float
        ).reshape(-1)

    if HCMCoupling.size != n_hcor:
        raise ValueError(
            "HCMCoupling must have one value "
            "per horizontal corrector."
        )

    if VCMCoupling.size != n_vcor:
        raise ValueError(
            "VCMCoupling must have one value "
            "per vertical corrector."
        )

    # ======================================================================
    # LOGGING
    # ======================================================================

    if log_info:

        try:
            LOGGER.info(
                "Calculating ORM: calculator=%s",
                calculator
            )
        except Exception:
            pass

    response_matrix = np.full(
        (2 * n_bpm, n_cm),
        np.nan
    )

    # ======================================================================
    #
    #                         LINEAR CALCULATOR
    #
    # ======================================================================

    if calculator == 'Linear':

        NE = len(ring)

        M44, T = at.find_m44(
            ring,
            0,
            np.arange(0, NE + 1)
        )

        ClosedOrbit = at.find_orbit4(
            ring,
            0,
            np.arange(0, NE + 1)
        )

        DP = 1e-5

        ClosedOrbitDP = at.find_orbit4(
            ring,
            DP,
            np.arange(0, NE + 1)
        )

        ClosedOrbit = ClosedOrbit[1].T
        ClosedOrbitDP = ClosedOrbitDP[1].T

        T = np.transpose(
            T,
            (1, 2, 0)
        )

        Dispersion0 = (
            ClosedOrbitDP -
            ClosedOrbit
        ) / DP

        Dispersion = Dispersion0[:4, :]

        L0 = at.get_s_pos(
            ring,
            NE
        )

        MCF = get_mcf(ring)

        # --------------------------------------------------------------
        # Element transfer matrices
        # --------------------------------------------------------------

        M44HCOR = [None] * n_hcor
        M44VCOR = [None] * n_vcor

        for i in range(n_hcor):

            idx = cm_ords[0][i]

            M44HCOR[i] = findelemm44(
                ring,
                idx,
                np.concatenate(
                    [ClosedOrbit[:, idx]]
                ),
                dt=None
            )

        for i in range(n_vcor):

            idx_v = cm_ords[1][i]

            matches = np.where(
                cm_ords[0] == idx_v
            )[0]

            if matches.size > 0:

                M44VCOR[i] = (
                    M44HCOR[matches[0]]
                )

            else:

                M44VCOR[i] = findelemm44(
                    ring,
                    idx_v,
                    np.concatenate(
                        [ClosedOrbit[:, idx_v]]
                    ),
                    dt=None
                )

        # --------------------------------------------------------------
        # Corrector kicks
        # --------------------------------------------------------------

        HCORTheta = np.zeros(
            (4, n_hcor)
        )

        VCORTheta = np.zeros(
            (4, n_vcor)
        )

        HCORTheta[1, :] = dkick[0][:]
        HCORTheta[3, :] = (
            HCMCoupling *
            dkick[0]
        )

        VCORTheta[1, :] = (
            VCMCoupling *
            dkick[1]
        )

        VCORTheta[3, :] = dkick[1][:]

        # ==============================================================
        # HORIZONTAL CORRECTORS
        # ==============================================================

        for i in range(n_hcor):

            CI = cm_ords[0][i]

            InverseT = np.linalg.inv(
                T[:, :, CI]
            )

            OrbitEntrance = (
                np.linalg.inv(
                    np.eye(4)
                    -
                    T[:, :, CI]
                    @ M44
                    @ InverseT
                )
                @ T[:, :, CI]
                @ M44
                @ InverseT
                @ (
                    np.eye(4)
                    +
                    np.linalg.inv(
                        M44HCOR[i]
                    )
                )
                @ (
                    HCORTheta[:, i] /
                    2.0
                )
            )

            OrbitExit = (
                HCORTheta[:, i] /
                2.0
                +
                M44HCOR[i]
                @ (
                    OrbitEntrance
                    +
                    HCORTheta[:, i] /
                    2.0
                )
            )

            R0 = (
                np.linalg.inv(
                    T[:, :, CI + 1]
                )
                @ OrbitExit
            )

            if NewVectorizedMethod:

                vectind = bpm_ords[:n_bpm]

                T3 = T[
                    [0, 2],
                    :,
                    :
                ][:, :, vectind]

                T2 = np.transpose(
                    T3,
                    (0, 2, 1)
                ).reshape(
                    n_bpm * 2,
                    4,
                    order='F'
                )

                bgtc = np.where(
                    vectind >
                    cm_ords[0][i] - 1
                )[0]

                bltc = np.where(
                    vectind <=
                    cm_ords[0][i] - 1
                )[0]

                bgtc = np.concatenate(
                    (
                        bgtc * 2,
                        bgtc * 2 + 1
                    )
                )

                bltc = np.concatenate(
                    (
                        bltc * 2,
                        bltc * 2 + 1
                    )
                )

                R0 = np.atleast_2d(R0)

                if R0.shape[0] != 4:
                    R0 = R0.T

                Tout1 = T2 @ R0
                Tout2 = T2 @ M44 @ R0

                Tout = np.zeros_like(
                    Tout1
                )

                Tout[bgtc, :] = (
                    Tout1[bgtc, :]
                )

                Tout[bltc, :] = (
                    Tout2[bltc, :]
                )

                jjj = np.zeros(
                    (2, n_bpm),
                    dtype=int
                )

                jjj[0, :] = np.arange(
                    n_bpm
                )

                jjj[1, :] = np.arange(
                    n_bpm,
                    n_bpm * 2
                )

                response_matrix[
                    jjj.ravel(order='F'),
                    i
                ] = Tout.ravel(
                    order='F'
                )

            else:

                for j in range(n_bpm):

                    bpm_idx = bpm_ords[j]

                    if bpm_idx > cm_ords[0][i]:

                        response_matrix[
                            [j, j + n_bpm],
                            i
                        ] = (
                            T[
                                [0, 2],
                                :,
                                bpm_idx
                            ]
                            @ R0
                        )

                    else:

                        response_matrix[
                            [j, j + n_bpm],
                            i
                        ] = (
                            T[
                                [0, 2],
                                :,
                                bpm_idx
                            ]
                            @ M44
                            @ R0
                        )

            # ----------------------------------------------------------
            # Fixed path length correction
            # ----------------------------------------------------------

            if fixedpathlength:

                D = (
                    HCORTheta[1, i]
                    *
                    (
                        Dispersion[
                            0,
                            cm_ords[0][i]
                        ]
                        +
                        Dispersion[
                            0,
                            cm_ords[0][i] + 1
                        ]
                    )
                    *
                    Dispersion[
                        np.ix_(
                            [0, 2],
                            bpm_ords
                        )
                    ]
                    /
                    L0
                    /
                    MCF
                    /
                    2.0
                )

                response_matrix[
                    :n_bpm,
                    i
                ] -= D[0, :].T

                response_matrix[
                    n_bpm:,
                    i
                ] -= D[1, :].T

        # ==============================================================
        # VERTICAL CORRECTORS
        # ==============================================================

        for i in range(n_vcor):

            CI = cm_ords[1][i]

            InverseT = np.linalg.inv(
                T[:, :, CI]
            )

            OrbitEntrance = (
                np.linalg.inv(
                    np.eye(4)
                    -
                    T[:, :, CI]
                    @ M44
                    @ InverseT
                )
                @ T[:, :, CI]
                @ M44
                @ InverseT
                @ (
                    np.eye(4)
                    +
                    np.linalg.inv(
                        M44VCOR[i]
                    )
                )
                @ (
                    VCORTheta[:, i] /
                    2.0
                )
            )

            OrbitExit = (
                VCORTheta[:, i] /
                2.0
                +
                M44VCOR[i]
                @ (
                    OrbitEntrance
                    +
                    VCORTheta[:, i] /
                    2.0
                )
            )

            R0 = (
                np.linalg.inv(
                    T[:, :, CI + 1]
                )
                @ OrbitExit
            )

            if NewVectorizedMethod:

                vectind = bpm_ords[:n_bpm]

                T3 = T[
                    [0, 2],
                    :,
                    :
                ][:, :, vectind]

                T2 = np.transpose(
                    T3,
                    (0, 2, 1)
                ).reshape(
                    n_bpm * 2,
                    4,
                    order='F'
                )

                bgtc = np.where(
                    vectind >
                    cm_ords[1][i] - 1
                )[0]

                bltc = np.where(
                    vectind <=
                    cm_ords[1][i] - 1
                )[0]

                bgtc = np.concatenate(
                    (
                        bgtc * 2,
                        bgtc * 2 + 1
                    )
                )

                bltc = np.concatenate(
                    (
                        bltc * 2,
                        bltc * 2 + 1
                    )
                )

                R0 = np.atleast_2d(R0)

                if R0.shape[0] != 4:
                    R0 = R0.T

                Tout1 = T2 @ R0
                Tout2 = T2 @ M44 @ R0

                Tout = np.zeros_like(
                    Tout1
                )

                Tout[bgtc, :] = (
                    Tout1[bgtc, :]
                )

                Tout[bltc, :] = (
                    Tout2[bltc, :]
                )

                jjj = np.zeros(
                    (2, n_bpm),
                    dtype=int
                )

                jjj[0, :] = np.arange(
                    n_bpm
                )

                jjj[1, :] = np.arange(
                    n_bpm,
                    n_bpm * 2
                )

                response_matrix[
                    jjj.ravel(order='F'),
                    i + n_hcor
                ] = Tout.ravel(
                    order='F'
                )

            else:

                for j in range(n_bpm):

                    bpm_idx = bpm_ords[j]

                    if bpm_idx > CI:

                        response_matrix[
                            [j, j + n_bpm],
                            n_hcor + i
                        ] = (
                            T[
                                [0, 2],
                                :,
                                bpm_idx
                            ]
                            @ R0
                        )

                    else:

                        response_matrix[
                            [j, j + n_bpm],
                            n_hcor + i
                        ] = (
                            T[
                                [0, 2],
                                :,
                                bpm_idx
                            ]
                            @ M44
                            @ R0
                        )

            if fixedpathlength:

                D = (
                    VCORTheta[1, i]
                    *
                    (
                        Dispersion[
                            0,
                            CI
                        ]
                        +
                        Dispersion[
                            0,
                            CI + 1
                        ]
                    )
                    *
                    Dispersion[
                        np.ix_(
                            [0, 2],
                            bpm_ords
                        )
                    ]
                    /
                    L0
                    /
                    MCF
                    /
                    2.0
                )

                response_matrix[
                    :n_bpm,
                    n_hcor + i
                ] -= D[0, :]

                response_matrix[
                    n_bpm:,
                    n_hcor + i
                ] -= D[1, :]

    # ======================================================================
    #
    #                     ANALYTICAL CALCULATOR
    #
    # ======================================================================

    elif calculator == 'Analytical':

        # ==============================================================
        # ANALYTICAL ORM
        # ==============================================================
        #
        # Betatron closed-orbit response:
        #
        #                 sqrt(beta_i * beta_j)
        # R_beta(i,j) = --------------------------- *
        #                    2 sin(pi Q)
        #
        #                 cos(Delta_mu_ij - pi Q)
        #
        #
        # If fixedpathlength=True, the dispersion/path-length
        # contribution is also included:
        #
        #                       eta_cor_avg * eta_bpm
        # R_disp(i,j) = - --------------------------------
        #                              L0 * alpha_c
        #
        # where
        #
        #                   eta_cor(j) + eta_cor(j+1)
        # eta_cor_avg = --------------------------------
        #                                2
        #
        #
        # Therefore the horizontal response to an H corrector is
        #
        # dx_i = theta_xj * (R_beta_xx + R_disp_xx)
        #
        #
        # and the vertical response to an H corrector is
        #
        # dy_i = theta_xj * R_disp_yx
        #
        #
        # because the BETATRON cross-plane response is assumed zero,
        # but a vertical-dispersion contribution may still exist.
        #
        #
        # Similarly, for a V corrector:
        #
        #   - betatron Mxy = 0
        #   - betatron Myy is calculated analytically
        #   - the same fixed-path-length convention used by the
        #     Linear calculator is applied when fixedpathlength=True.
        #
        # IMPORTANT:
        #
        # This analytical implementation does NOT calculate transverse
        # betatron coupling. Therefore the betatron Mxy/Myx terms are zero.
        # Any non-zero cross-plane response produced by the
        # fixed-path-length correction comes from dispersion, not
        # betatron coupling.
        # ==============================================================
        # The Analytical ORM was implemented in pyLOCO by Ahmed Eldeeb
        # DESY Summer Student, August 2026
        # ==============================================================

        has_hcm_coupling = np.any(
            np.abs(HCMCoupling) > 0.0
        )

        has_vcm_coupling = np.any(
            np.abs(VCMCoupling) > 0.0
        )

        if (
            coupling_orm
            or has_hcm_coupling
            or has_vcm_coupling
        ):

            raise ValueError(
                "calculator='Analytical "
                "supports only uncoupled ORM calculation. "
                "Mxy and Myx are assumed to be zero. "
                "Use calculator='Linear' or 'Numerical' "
                "when coupling is included."
            )

        if log_info:

            try:
                LOGGER.warning(
                    "Analytical ORM assumes uncoupled "
                    "transverse optics: Mxy = Myx = 0."
                )
            except Exception:
                pass

        # Zero matrix is intentional:
        # off-diagonal coupling blocks remain zero.

        response_matrix = np.zeros(
            (2 * n_bpm, n_cm),
            dtype=float
        )

        # --------------------------------------------------------------
        # Optics
        # --------------------------------------------------------------

        NE = len(ring)

        all_refpts = np.arange(
            NE + 1
        )

        _, ringdata, twiss_all = at.get_optics(
            ring,
            refpts=all_refpts,
            method=at.linopt2
        )

        beta_x = twiss_all.beta[:, 0]
        beta_y = twiss_all.beta[:, 1]

        mu_x = twiss_all.mu[:, 0]
        mu_y = twiss_all.mu[:, 1]

        # Full accumulated phase after one turn

        mu_x_end = mu_x[-1]
        mu_y_end = mu_y[-1]

        Qx = (
            mu_x_end /
            (2.0 * np.pi)
        )

        Qy = (
            mu_y_end /
            (2.0 * np.pi)
        )

        sin_pi_Qx = np.sin(
            np.pi * Qx
        )

        sin_pi_Qy = np.sin(
            np.pi * Qy
        )

        if np.isclose(
            sin_pi_Qx,
            0.0,
            atol=1e-12
        ):
            raise ValueError(
                f"Qx={Qx:.12f} is too close "
                "to an integer resonance."
            )

        if np.isclose(
            sin_pi_Qy,
            0.0,
            atol=1e-12
        ):
            raise ValueError(
                f"Qy={Qy:.12f} is too close "
                "to an integer resonance."
            )

        # --------------------------------------------------------------
        # Forward phase advance
        # --------------------------------------------------------------

        def forward_phase_advance(
            mu_bpm,
            mu_cor,
            mu_end
        ):

            dmu = (
                mu_bpm -
                mu_cor
            )

            if dmu < 0.0:

                dmu += mu_end

            return dmu

        # --------------------------------------------------------------
        # Corrector kick arrays
        # --------------------------------------------------------------

        if isinstance(
            dkick,
            (list, tuple, np.ndarray)
        ):

            dkick_h = np.asarray(
                dkick[0],
                dtype=float
            ).reshape(-1)

            dkick_v = np.asarray(
                dkick[1],
                dtype=float
            ).reshape(-1)

        else:

            dkick_h = np.full(
                n_hcor,
                float(dkick)
            )

            dkick_v = np.full(
                n_vcor,
                float(dkick)
            )

        if dkick_h.size != n_hcor:

            raise ValueError(
                f"Horizontal dkick contains "
                f"{dkick_h.size} values, "
                f"but {n_hcor} H correctors exist."
            )

        if dkick_v.size != n_vcor:

            raise ValueError(
                f"Vertical dkick contains "
                f"{dkick_v.size} values, "
                f"but {n_vcor} V correctors exist."
            )

        # ==============================================================
        # HORIZONTAL CORRECTORS
        # ==============================================================

        for j, cor in enumerate(
            cm_ords[0]
        ):

            theta = dkick_h[j]

            for i, bpm in enumerate(
                bpm_ords
            ):

                dmu_x = (
                    forward_phase_advance(
                        mu_x[bpm],
                        mu_x[cor],
                        mu_x_end
                    )
                )

                Rxx = (
                    np.sqrt(
                        beta_x[bpm]
                        *
                        beta_x[cor]
                    )
                    /
                    (
                        2.0 *
                        sin_pi_Qx
                    )
                    *
                    np.cos(
                        dmu_x
                        -
                        np.pi * Qx
                    )
                )

                # pyLOCO stores the orbit change
                # produced by dkick.

                response_matrix[
                    i,
                    j
                ] = theta * Rxx

                # Myx = 0 for uncoupled formula

                response_matrix[
                    n_bpm + i,
                    j
                ] = 0.0

        # ==============================================================
        # VERTICAL CORRECTORS
        # ==============================================================

        for j, cor in enumerate(
            cm_ords[1]
        ):

            theta = dkick_v[j]

            col = (
                n_hcor +
                j
            )

            for i, bpm in enumerate(
                bpm_ords
            ):

                dmu_y = (
                    forward_phase_advance(
                        mu_y[bpm],
                        mu_y[cor],
                        mu_y_end
                    )
                )

                Ryy = (
                    np.sqrt(
                        beta_y[bpm]
                        *
                        beta_y[cor]
                    )
                    /
                    (
                        2.0 *
                        sin_pi_Qy
                    )
                    *
                    np.cos(
                        dmu_y
                        -
                        np.pi * Qy
                    )
                )

                # Mxy = 0

                response_matrix[
                    i,
                    col
                ] = 0.0

                response_matrix[
                    n_bpm + i,
                    col
                ] = theta * Ryy

        # ==============================================================
        # FIXED PATH LENGTH CORRECTION
        # ==============================================================
        #
        # Same convention as the existing Linear calculator.
        #
        # ==============================================================

        if fixedpathlength:

            ClosedOrbit = at.find_orbit4(
                ring,
                0,
                np.arange(NE + 1)
            )

            DP = 1e-5

            ClosedOrbitDP = at.find_orbit4(
                ring,
                DP,
                np.arange(NE + 1)
            )

            ClosedOrbit = (
                ClosedOrbit[1].T
            )

            ClosedOrbitDP = (
                ClosedOrbitDP[1].T
            )

            Dispersion0 = (
                ClosedOrbitDP -
                ClosedOrbit
            ) / DP

            Dispersion = (
                Dispersion0[:4, :]
            )

            L0 = at.get_s_pos(
                ring,
                NE
            )

            MCF = get_mcf(ring)

            # ----------------------------------------------------------
            # Horizontal correctors
            # ----------------------------------------------------------

            for j, cor in enumerate(
                cm_ords[0]
            ):

                D = (
                    dkick_h[j]
                    *
                    (
                        Dispersion[
                            0,
                            cor
                        ]
                        +
                        Dispersion[
                            0,
                            cor + 1
                        ]
                    )
                    *
                    Dispersion[
                        np.ix_(
                            [0, 2],
                            bpm_ords
                        )
                    ]
                    /
                    L0
                    /
                    MCF
                    /
                    2.0
                )

                response_matrix[
                    :n_bpm,
                    j
                ] -= D[0, :]

                response_matrix[
                    n_bpm:,
                    j
                ] -= D[1, :]

            # ----------------------------------------------------------
            # Vertical correctors
            # ----------------------------------------------------------

            for j, cor in enumerate(
                cm_ords[1]
            ):

                col = (
                    n_hcor +
                    j
                )

                D = (
                    dkick_v[j]
                    *
                    (
                        Dispersion[
                            0,
                            cor
                        ]
                        +
                        Dispersion[
                            0,
                            cor + 1
                        ]
                    )
                    *
                    Dispersion[
                        np.ix_(
                            [0, 2],
                            bpm_ords
                        )
                    ]
                    /
                    L0
                    /
                    MCF
                    /
                    2.0
                )

                response_matrix[
                    :n_bpm,
                    col
                ] -= D[0, :]

                response_matrix[
                    n_bpm:,
                    col
                ] -= D[1, :]

    # ======================================================================
    #
    #                      NUMERICAL CALCULATOR
    #
    # ======================================================================

    elif calculator == 'Numerical':

        _, orbit0 = at.find_orbit4(
            ring,
            0,
            bpm_ords
        )

        orbit0_x = orbit0[:, 0]
        orbit0_y = orbit0[:, 2]

        cnt = 0

        for n_dim in range(2):

            # 0 = horizontal
            # 1 = vertical

            other_dim = 1 - n_dim

            for j, cm_ord in enumerate(
                cm_ords[n_dim]
            ):

                # ------------------------------------------------------
                # Corrector-specific kick
                # ------------------------------------------------------

                if isinstance(
                    dkick,
                    (list, tuple, np.ndarray)
                ):

                    try:

                        this_dkick = (
                            dkick[n_dim][j]
                        )

                    except Exception:

                        this_dkick = (
                            dkick[j]
                            if isinstance(
                                dkick[j],
                                (int, float)
                            )
                            else float(
                                dkick[j]
                            )
                        )

                else:

                    this_dkick = float(
                        dkick
                    )

                corrector = ring[
                    cm_ord
                ]

                kick_array, kick_index, kick_scale = _corrector_kick_component(
                    corrector, n_dim
                )
                other_array, other_index, other_scale = _corrector_kick_component(
                    corrector, other_dim
                )
                base_kick = kick_array[kick_index]
                other_kick = other_array[other_index]

                # ======================================================
                # BIDIRECTIONAL
                # ======================================================

                if bidirectional:

                    # +delta / 2

                    kick_array[
                        kick_index
                    ] = (
                        base_kick
                        +
                        kick_scale * this_dkick /
                        2.0
                    )

                    _, orbit = (
                        at.find_orbit4(
                            ring,
                            0,
                            bpm_ords
                        )
                    )

                    orbit_plus_x = (
                        orbit[:, 0]
                    )

                    orbit_plus_y = (
                        orbit[:, 2]
                    )

                    # -delta / 2

                    kick_array[
                        kick_index
                    ] = (
                        base_kick
                        -
                        kick_scale * this_dkick /
                        2.0
                    )

                    _, orbit = (
                        at.find_orbit4(
                            ring,
                            0,
                            bpm_ords
                        )
                    )

                    orbit_minus_x = (
                        orbit[:, 0]
                    )

                    orbit_minus_y = (
                        orbit[:, 2]
                    )

                    # Restore

                    kick_array[
                        kick_index
                    ] = base_kick

                    dx = (
                        orbit_plus_x
                        -
                        orbit_minus_x
                    )

                    dy = (
                        orbit_plus_y
                        -
                        orbit_minus_y
                    )

                # ======================================================
                # ONE-SIDED
                # ======================================================

                else:

                    kick_array[
                        kick_index
                    ] = (
                        base_kick
                        +
                        kick_scale * this_dkick
                    )

                    if (
                        coupling_orm
                        and delta_coupling
                    ):

                        other_array[
                            other_index
                        ] = (
                            other_kick
                            +
                            other_scale * this_dkick * delta_coupling
                        )

                    _, orbit = (
                        at.find_orbit4(
                            ring,
                            0,
                            bpm_ords
                        )
                    )

                    orbit_new_x = (
                        orbit[:, 0]
                    )

                    orbit_new_y = (
                        orbit[:, 2]
                    )

                    # Restore

                    kick_array[
                        kick_index
                    ] = base_kick

                    other_array[
                        other_index
                    ] = other_kick

                    dx = (
                        orbit_new_x
                        -
                        orbit0_x
                    )

                    dy = (
                        orbit_new_y
                        -
                        orbit0_y
                    )

                response_matrix[
                    :,
                    cnt
                ] = np.concatenate(
                    (dx, dy)
                )

                cnt += 1

    # ======================================================================
    # UNKNOWN CALCULATOR
    # ======================================================================

    else:

        raise ValueError(
            f"Unknown calculator='{calculator}'. "
            "Available calculators are "
            "'Linear', 'Analytical', and 'Numerical'."
        )

    # ======================================================================
    #
    #                        DISPERSION COLUMN
    #
    # ======================================================================
    #
    # includeDispersion is different from fixedpathlength.
    #
    # fixedpathlength:
    #     modifies the corrector ORM columns.
    #
    # includeDispersion:
    #     adds the RF-dispersion measurement as an additional column.
    #
    # ======================================================================

    if includeDispersion:

        C = 2.99792458e8

        # --------------------------------------------------------------
        # Linear / Analytical
        # --------------------------------------------------------------

        if calculator in (
            'Linear',
            'Analytical'
        ):

            f_rf = Frequency
            h_rf = HarmNumber

            _, ORBITPLUS = (
                at.find_sync_orbit(
                    ring,
                    (
                        -C
                        *
                        rfStep
                        *
                        h_rf
                        /
                        f_rf ** 2
                    )
                    /
                    2.0,
                    refpts=bpm_ords
                )
            )

            dx = ORBITPLUS[:, 0]
            dy = ORBITPLUS[:, 2]

            _, ORBIT0 = (
                at.find_sync_orbit(
                    ring,
                    (
                        C
                        *
                        rfStep
                        *
                        h_rf
                        /
                        f_rf ** 2
                    )
                    /
                    2.0,
                    refpts=bpm_ords
                )
            )

            dx0 = ORBIT0[:, 0]
            dy0 = ORBIT0[:, 2]

            dispersion_meas = (
                np.concatenate(
                    (
                        dx - dx0,
                        dy - dy0
                    )
                )
            )

            response_matrix = np.hstack(
                (
                    response_matrix,
                    dispersion_meas.reshape(
                        -1,
                        1
                    )
                )
            )

        # --------------------------------------------------------------
        # Numerical
        # --------------------------------------------------------------

        elif calculator == 'Numerical':

            if bidirectional:

                shift_rf(
                    ring,
                    cav_ords,
                    +rfStep / 2,
                    attr=RFAttr
                )

                _, orbit = (
                    at.find_orbit4(
                        ring,
                        0,
                        bpm_ords
                    )
                )

                orbit_plus_x = (
                    orbit[:, 0]
                )

                orbit_plus_y = (
                    orbit[:, 2]
                )

                shift_rf(
                    ring,
                    cav_ords,
                    -rfStep / 2,
                    attr=RFAttr
                )

                shift_rf(
                    ring,
                    cav_ords,
                    -rfStep / 2,
                    attr=RFAttr
                )

                _, orbit = (
                    at.find_orbit4(
                        ring,
                        0,
                        bpm_ords
                    )
                )

                orbit_minus_x = (
                    orbit[:, 0]
                )

                orbit_minus_y = (
                    orbit[:, 2]
                )

                # Restore RF

                shift_rf(
                    ring,
                    cav_ords,
                    +rfStep / 2,
                    attr=RFAttr
                )

                dx = (
                    orbit_plus_x
                    -
                    orbit_minus_x
                    -
                    orbit0_x
                )

                dy = (
                    orbit_plus_y
                    -
                    orbit_minus_y
                    -
                    orbit0_y
                )

            else:

                shift_rf(
                    ring,
                    cav_ords,
                    +rfStep,
                    attr=RFAttr
                )

                _, orbit = (
                    at.find_orbit4(
                        ring,
                        0,
                        bpm_ords
                    )
                )

                orbit_new_x = (
                    orbit[:, 0]
                )

                orbit_new_y = (
                    orbit[:, 2]
                )

                # Restore RF

                shift_rf(
                    ring,
                    cav_ords,
                    -rfStep,
                    attr=RFAttr
                )

                dx = (
                    orbit_new_x
                    -
                    orbit0_x
                )

                dy = (
                    orbit_new_y
                    -
                    orbit0_y
                )

            dispersion_meas = (
                np.concatenate(
                    (dx, dy)
                )
            )

            response_matrix = np.hstack(
                (
                    response_matrix,
                    dispersion_meas.reshape(
                        -1,
                        1
                    )
                )
            )

    return response_matrix


# ==========================================================================
# ELEMENT 4x4 MATRIX
# ==========================================================================

def findelemm44(
    ring,
    ELEM,
    orbit_in,
    dt=None
):

    from at import element_pass

    if dt is None:
        dt = 1e-7

    orbit_in = np.asarray(
        orbit_in
    ).reshape(
        6,
        1
    )

    D4 = np.vstack(
        (
            dt * np.eye(4),
            np.zeros((2, 4))
        )
    )

    RIN = np.hstack(
        (
            orbit_in + D4,
            orbit_in - D4
        )
    )

    RIN6_F = np.asfortranarray(
        RIN
    )

    ROUT = element_pass(
        ring[ELEM],
        RIN6_F
    )

    M44 = (
        ROUT[0:4, 0:4]
        -
        ROUT[0:4, 4:8]
    ) / (
        2 * dt
    )

    return M44


# ==========================================================================
# RF SHIFT
# ==========================================================================

def shift_rf(
    ring,
    cav_ords,
    freq_delta,
    attr="Frequency"
):

    freq_delta = float(
        freq_delta
    )

    if isinstance(
        attr,
        (
            list,
            tuple,
            np.ndarray
        )
    ):

        if len(attr) != len(cav_ords):

            raise ValueError(
                "Length of 'attr' must match "
                "length of 'cav_ords'."
            )

        for idx, name in zip(
            cav_ords,
            attr
        ):

            elem = ring[
                int(idx)
            ]

            setattr(
                elem,
                name,
                getattr(
                    elem,
                    name
                )
                +
                freq_delta
            )

    else:

        for idx in cav_ords:

            elem = ring[
                int(idx)
            ]

            setattr(
                elem,
                attr,
                getattr(
                    elem,
                    attr
                )
                +
                freq_delta
            )
