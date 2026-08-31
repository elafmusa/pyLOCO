"""
formulas from
A.Franchi (ESRF), Z.Marti (CELLS),
"Analytic formulas for the rapid evaluation of the orbit response matrix and chromatic functions from lattice
parameters in circular accelerators" arXiv:1711:06589v2 17 Apr 2018
"""

import at
import math
import cmath
import numpy as np
import multiprocessing
import time
from itertools import repeat

__author__='Simone Maria Liuzzo, Andrea Franchi'

def analytic_orm_variation_with_skew_quadrupole(
        ring,
        ind_bpms=None,
        ind_cors=None,
        ind_skews=None,
        verbose=True,
        thick_skew=True,
        thick_steerer=True,
        opt_all_location=None,
        use_mp=False,
        cancel_callback=None,
        implementation="vectorized",
        progress_callback=None,
        timing_callback=None):
    """
    Computes the derivative of the orbit response matrix compared to skew quadrupoles

    A.Franchi (ESRF), Z.Marti (CELLS),
    "Analytic formulas for the rapid evaluation of the orbit response matrix and chromatic functions from lattice
    parameters in circular accelerators" arXiv:1711:06589v2 17 Apr 2018

    :param ring: AT lattice
    :param ind_bpms: numpy array of np.uint32. indexes of BPMs
    :param ind_cors: numpy array of np.uint32. indexes of correctors
    :param ind_skews: numpy array of np.uint32. indexes of skew quadrupoles
    :param verbose: True/False
    :param thick_skew: True/False
    :param thick_steerer: True/False
    :param opt_all_location: output of at.linopt6 at all ring elements
    :param use_mp: True/False use multiprocessing
    :return: two 3D lists of floats.
    ORM_HorSteererToVerBPM_over_Ks[BPMS][CORRECTORS][SKEW], ORM_VerSteererToHorBPM_over_Ks[BPMS][CORRECTORS][SKEW],
    """

    implementation = str(implementation).strip().lower()
    if implementation not in {"legacy", "vectorized"}:
        raise ValueError(
            f"Unknown skew analytical implementation {implementation!r}; "
            "choose 'legacy' or 'vectorized'."
        )
    if len(ind_skews) == 0:
        shape = (len(ind_bpms), len(ind_cors), 0)
        return np.zeros(shape), np.zeros(shape)

    started = time.perf_counter()
    worker = (
        _analytic_orm_variation_with_skew_quadrupole_legacy
        if implementation == "legacy"
        else _analytic_orm_variation_with_skew_quadrupole_vectorized
    )

    # loop quadrupoles
    if use_mp:

        n_cpu = multiprocessing.cpu_count()
        n_processes = min(n_cpu, len(ind_skews))
        skew_chunks = [
            chunk.tolist()
            for chunk in np.array_split(np.asarray(ind_skews, dtype=int), n_processes)
            if len(chunk)
        ]
        if verbose:
            print('parallel computation using {} cores'.format(n_processes))

        with multiprocessing.Pool(processes=n_processes) as p:
            pending = p.starmap_async(worker,
                                 zip(repeat(ring),
                                     repeat(ind_bpms),
                                     repeat(ind_cors),
                                     skew_chunks,
                                     repeat(verbose),
                                     repeat(thick_skew),
                                     repeat(thick_steerer),
                                     repeat(opt_all_location),
                                     repeat(None),
                                     )
                                 )
            while True:
                try:
                    results = pending.get(timeout=0.1)
                    break
                except multiprocessing.TimeoutError:
                    if cancel_callback is not None and cancel_callback():
                        p.terminate()
                        raise RuntimeError("LOCO run cancelled during analytical skew Jacobian calculation.")

        MH2V = np.concatenate([result[0] for result in results], axis=2)
        MV2H = np.concatenate([result[1] for result in results], axis=2)
        if progress_callback is not None:
            progress_callback(len(ind_skews), len(ind_skews))
        if timing_callback is not None:
            timing_callback({
                "skew_analytical_implementation": implementation,
                "multiprocessing_total_seconds": time.perf_counter() - started,
                "workers": n_processes,
                "chunks": len(skew_chunks),
            })

    else:  # sequential

        MH2V, MV2H = worker(
                        ring,
                        ind_bpms=ind_bpms,
                        ind_cors=ind_cors,
                        ind_skews=ind_skews,
                        verbose=verbose,
                        thick_skew=thick_skew,
                        thick_steerer=thick_steerer,
                        opt_all_location=opt_all_location,
                        cancel_callback=cancel_callback,
                        progress_callback=progress_callback,
                        timing_callback=timing_callback)
        if timing_callback is not None:
            timing_callback({
                "skew_analytical_implementation": implementation,
                "serial_total_seconds": time.perf_counter() - started,
            })

    return MH2V, MV2H


def _analytic_orm_variation_with_skew_quadrupole_legacy(
        ring,
        ind_bpms=None,
        ind_cors=None,
        ind_skews=None,
        verbose=True,
        thick_skew=True,
        thick_steerer=True,
        opt_all_location=None,
        cancel_callback=None,
        progress_callback=None,
        timing_callback=None):
    """
    analytic orbit response matrix derivative with integrated (KL) errors at skew quadrupoles


    :param ring:
    :param ind_bpms:
    :param ind_cors:
    :param verbose:
    :param filename_cod_response:
    :return:
    """
    # optics at correctors and BPMS
    if opt_all_location is None:
        _, _, opt_all_location = ring.linopt6(range(len(ring)))

    # ``Pool.starmap`` supplies a single lattice ordinal (normally a Python
    # int) to each worker.  Keep worker and sequential input contracts equal.
    if np.isscalar(ind_skews):
        ind_skews = [int(ind_skews)]

    bpm = opt_all_location[ind_bpms]
    cor = opt_all_location[ind_cors]
    qua = opt_all_location[ind_skews]

    # full tunes
    Q = at.get_tune(ring, get_integer=True)

    def tau(pl, a, b, idx_a=None, idx_b=None):
        # Edited: E.M 20 Aug 2026
        return (
            dphi(
                pl,
                a,
                b,
                idx_w=idx_a,
                idx_j=idx_b,
            )
            - math.pi * Q[pl]
        )

    def dphi(pl, w, j, idx_w=None, idx_j=None):
        """
        Forward phase advance from element w to element j.

        For elements at the same optical location, lattice indices
        are used to determine their physical ordering.

        Edited: E.M. 20 Aug 2026
        """

        d = j.mu[pl] - w.mu[pl]

        # Same optical location: lattice order breaks the tie
        if np.isclose(j.mu[pl], w.mu[pl], rtol=0.0, atol=1e-12):

            if idx_w is not None and idx_j is not None:
                if idx_j < idx_w:
                    d += 2.0 * math.pi * Q[pl]

        # Normal phase wrap-around
        elif j.mu[pl] < w.mu[pl]:
            d += 2.0 * math.pi * Q[pl]

        return d

    # formulas for response

    MH2V = np.zeros(shape=(len(bpm), len(cor), len(qua)))
    MV2H = np.zeros(shape=(len(bpm), len(cor), len(qua)))

    x = 0
    y = 1
    _sign = [+1, -1]
    h = 0
    v = 1

    def cosabc(ax, sign_by, by, sign_c, c, _GCm, _GSm,  _TCw, _TSw):
        # sby = True = +, False = -

        JSmjx = math.sin(ax) * _GCm[x] - math.cos(ax) * _GSm[x]
        JCmjx = math.cos(ax) * _GCm[x] + math.sin(ax) * _GSm[x]

        JSmjy = sign_by*(math.sin(by) * _GCm[y] - math.cos(by) * _GSm[y])
        JCmjy = math.cos(by) * _GCm[y] + math.sin(by) * _GSm[y]

        JSwj = sign_c*(math.sin(c) * _TCw - math.cos(c) * _TSw)
        JCwj = math.cos(c) * _TCw + math.sin(c) * _TSw

        # test (recover thin from thick formulas, _GCm is sqrt(beta)  GCm = [qua[m].beta[p]**0.5 for p in [x,y]]
        # JSmja = _GCm[x]*math.sin(ax)
        # JCmja = _GCm[x]*math.cos(ax)
        # JSmjb = _GCm[y]*math.sin(by)
        # JCmjb = _GCm[y]*math.cos(by)

        # cos(a + b + c) =
        # cos(a + b + c) = cos(a)cos(b + c) - sin(a)sin(b + c)
        # cos(b + c) = cos(b)cos(c) - sin(b)sin(c)
        # sin(b + c) = cos(c)sin(b) + cos(b)sin(c)

        # cos(a + b + c) =
        # cos(a)[cos(b)cos(c) - sin(b)sin(c)] - sin(a)[cos(c)sin(b) + cos(b)sin(c)] =
        # cos(a)[cos(b)cos(c) - sin(b)sin(c)] - sin(a)[sin(b)cos(c) + cos(b)sin(c)]
        #
        # cos(a + b + c) =
        # + cos(a)cos(b)cos(c)
        # - cos(a)sin(b)sin(c)
        # - sin(a)sin(b)cos(c)
        # - sin(a)cos(b)sin(c)

        #     + sb[x]*math.cos(a[x]) * sb[y]*math.cos(b[y]) * bc*math.cos(c) \
        #     - sb[x]*math.cos(a[x]) * sb[y]*math.sin(b[y]) * bc*math.sin(c) \
        #     - sb[x]*math.sin(a[x]) * sb[y]*math.sin(b[y]) * bc*math.cos(c) \
        #     - sb[x]*math.sin(a[x]) * sb[y]*math.cos(b[y]) * bc*math.sin(c)

        # sb[x]*math.cos(a[x]) --> JCmj[x]
        # sb[y]*math.cos(a[y]) --> JCmj[y]
        # sb[y]*math.sin(a[y]) --> JSmj[y]
        # sb[x]*math.sin(a[y]) --> JSmj[x]
        # bc *math.cos(c) --> JCmw
        # bc *math.sin(c) --> JSmw

        return  + JCmjx * JCmjy * JCwj - JCmjx * JSmjy * JSwj - JSmjx * JSmjy * JCwj - JSmjx * JCmjy * JSwj


    def _GCm(qu, p, Km0, Lm):
        LsK = Lm * cmath.sqrt(Km0)
        val = +qu.beta[p] ** 0.5 / LsK *cmath.sin(LsK) - qu.alpha[p] * _GSm(qu, p, Km0, Lm)
        return val.real


    def _GSm(qu, p, Km0, Lm):
        LsK = Lm * cmath.sqrt(Km0)
        val = 1 / (Lm * Km0 * qu.beta[p] ** 0.5) * (1 - cmath.cos(LsK))
        return val.real


    #steerers thick corrections
    TS = []
    TC = []
    for w,_ in enumerate(ind_cors):
        if thick_steerer:
            Lw = ring[ind_cors[w]].Length
            alpha = cor[w].alpha
            beta = cor[w].beta
            TS.append([Lw / (2 * beta[p] ** 0.5) for p in [x, y]])
            TC.append([beta[p] ** 0.5 - Lw * alpha[p] / (2 * beta[p] ** 0.5) for p in [x, y]])
        else:
            TS.append( [0, 0] )
            TC.append( [cor[w].beta[p] ** 0.5 for p in [x, y]] )

    one_over_sin_Qx_minus_Qy = 1 / (math.sin(math.pi * (Q[x] - Q[y])))
    one_over_sin_Qx_plus_Qy = 1 / (math.sin(math.pi * (Q[x] + Q[y])))
    sin_piQ = [math.sin(math.pi * Q[p]) for p in [x,y]]

    sqrt_beta_bpm = [b.beta**0.5 for b in bpm]

    for countm, m in enumerate(range(len(qua))):
        if cancel_callback is not None and cancel_callback():
            raise RuntimeError("LOCO run cancelled during analytical skew Jacobian calculation.")

        if thick_skew:
            is_a_quad = True
            Lm = ring[ind_skews[m]].Length
            Km0 = ring[ind_skews[m]].PolynomB[1]

            if Km0 == 0:
                is_a_quad = False
                if verbose:
                    print(f'{ring[ind_skews[m]].FamName} is not a quadrupole. '
                    f'PolynomA[1] = {ring[ind_skews[m]].PolynomA[1]:.3f}, '
                    f'PolynomB[1] = {ring[ind_skews[m]].PolynomB[1]:.3f}')

            if is_a_quad:
                GSm = [[_GSm(qua[m], p, _sign[p]*Km0, Lm) for p in [x, y]] for s in [h, v]]

                GCm = [[_GCm(qua[m], p, _sign[p]*Km0, Lm) for p in [x, y]] for s in [h, v]]

            else:
                # assume drift optics variation inside magnet selected to be a skew quadrupole
                beta = qua[m].beta
                alpha = qua[m].alpha
                GSm = [[Lm / 2 / beta[p] ** 0.5 for p in [x, y]] for s in [h, v]]
                GCm = [[beta[p] ** 0.5 - Lm * alpha[p] / 2 / beta[p] ** 0.5 for p in [x, y]] for s in [h, v]]

        else:
            GSm = [[0, 0] for s in [h, v]]
            GCm = [[(qua[m].beta[p]) ** 0.5 for p in [x, y]] for s in [h, v]]

        #print([(qua[m].beta[p]) ** 0.5 for p in [x, y]])
        #print(GCm[h])
        #print(GSm[h])
        #print(GCm[v])
        #print(GSm[v])

        for countj, j in enumerate(range(len(bpm))):

            sqrt_beta_j_x = sqrt_beta_bpm[j][x] # (bpm[j].beta[x]) ** 0.5
            sqrt_beta_j_y = sqrt_beta_bpm[j][y] # (bpm[j].beta[y]) ** 0.5

            tmj = [
                tau(
                    p,
                    qua[m],
                    bpm[j],
                    idx_a=ind_skews[m],
                    idx_b=ind_bpms[j],
                )
                for p in [x, y]
            ]

            for countw, w in enumerate(range(len(cor))):
                if verbose:
                    print(f'computing response of steerer {ring[ind_cors[w]].FamName}'
                          f' to BPM {ring[ind_bpms[j]].FamName}'
                          f' with a skew (thick={thick_skew}) quadrupole error in {ring[ind_skews[m]].FamName}'
                          f' with a (thick={thick_steerer}) steerers'
                          )

                twj = [
                    tau(
                        p,
                        cor[w],
                        bpm[j],
                        idx_a=ind_cors[w],
                        idx_b=ind_bpms[j],
                    )
                    for p in [x, y]
                ]
                tmw = [
                    tau(
                        p,
                        qua[m],
                        cor[w],
                        idx_a=ind_skews[m],
                        idx_b=ind_cors[w],
                    )
                    for p in [x, y]
                ]

                MV2H[j][w][m] = + 1 / 8 * \
                                sqrt_beta_j_x * \
                                (
                                one_over_sin_Qx_minus_Qy * (
                                + cosabc(tmj[x], -1, tmj[y], +1, twj[y], GCm[h], GSm[h], TC[w][y], TS[w][y]) / sin_piQ[y]
                                - cosabc(tmw[x], -1, tmw[y], +1, twj[x], GCm[h], GSm[h], TC[w][y], TS[w][y]) / sin_piQ[x]
                                )
                                +
                                one_over_sin_Qx_plus_Qy * (
                                 + cosabc(tmj[x], +1, tmj[y], -1, twj[y], GCm[h], GSm[h], TC[w][y], TS[w][y]) / sin_piQ[y]
                                 + cosabc(tmw[x], +1, tmw[y], +1, twj[x], GCm[h], GSm[h], TC[w][y], TS[w][y]) / sin_piQ[x]
                                 )
                                )

                #              -V- sign for AT
                MH2V[j][w][m] = - 1 / 8 * \
                                sqrt_beta_j_y * \
                                (
                                one_over_sin_Qx_minus_Qy * (
                                - cosabc(tmj[x], -1, tmj[y], -1, twj[x], GCm[v], GSm[v], TC[w][x], TS[w][x]) / sin_piQ[x] +
                                + cosabc(tmw[x], -1, tmw[y], -1, twj[y], GCm[v], GSm[v], TC[w][x], TS[w][x]) / sin_piQ[y]
                                )
                                +
                                one_over_sin_Qx_plus_Qy * (
                                + cosabc(tmj[x], +1, tmj[y], -1, twj[x], GCm[v], GSm[v], TC[w][x], TS[w][x]) / sin_piQ[x] +
                                + cosabc(tmw[x], +1, tmw[y], +1, twj[y], GCm[v], GSm[v], TC[w][x], TS[w][x]) / sin_piQ[y]
                                )
                                )

                """ THIN - THIN formulas for reference in paper Eq. 22
                if thick_skew or thick_steerer:
                else: # thin thin case, but the formula above should work also in this case
                    MV2H[j][w][m] = + 1/8 * \
                    ((bpm[j].beta[x] * cor[w].beta[y] * qua[m].beta[x] * qua[m].beta[y]) ** 0.5) * \
                       (
                       1 / (math.sin(math.pi * (Q[x] - Q[y]))) *
                       (math.cos(tmj[x] - tmj[y] + twj[y])/math.sin(math.pi * Q[y]) +
                        - math.cos(tmw[x] - tmw[y] + twj[x])/math.sin(math.pi * Q[x])
                        )
                       +
                       1 / (math.sin(math.pi * (Q[x] + Q[y]))) *
                       (math.cos(tmj[x] + tmj[y] - twj[y])/math.sin(math.pi * Q[y]) +
                        math.cos(tmw[x] + tmw[y] + twj[x])/math.sin(math.pi * Q[x])
                        )
                       )
                    #              -V- sign for AT
                    MH2V[j][w][m] = - 1/8 * \
                    ((bpm[j].beta[y] * cor[w].beta[x] * qua[m].beta[x] * qua[m].beta[y]) ** 0.5) * \
                       (
                       1 / (math.sin(math.pi * (Q[x] - Q[y]))) *
                       (- math.cos(tmj[x] - tmj[y] - twj[x])/math.sin(math.pi * Q[x]) +
                        + math.cos(tmw[x] - tmw[y] - twj[y])/math.sin(math.pi * Q[y])
                        )
                       +
                       1 / (math.sin(math.pi * (Q[x] + Q[y]))) *
                       (math.cos(tmj[x] + tmj[y] - twj[x])/math.sin(math.pi * Q[x]) +
                        math.cos(tmw[x] + tmw[y] + twj[y])/math.sin(math.pi * Q[y])
                        )
                       )
                """

        if progress_callback is not None:
            progress_callback(countm + 1, len(qua))

    return MH2V, MV2H


def _analytic_orm_variation_with_skew_quadrupole_vectorized(
        ring,
        ind_bpms=None,
        ind_cors=None,
        ind_skews=None,
        verbose=True,
        thick_skew=True,
        thick_steerer=True,
        opt_all_location=None,
        cancel_callback=None,
        progress_callback=None,
        timing_callback=None):
    """Vectorized evaluation of the existing analytical skew equations.

    The skew loop is retained so thick-element factors and cancellation remain
    easy to audit.  BPM and corrector dimensions are evaluated with NumPy
    arrays; no analytical term or sign convention differs from the legacy
    implementation above.
    """
    total_started = time.perf_counter()
    optics_started = time.perf_counter()
    if opt_all_location is None:
        _, _, opt_all_location = ring.linopt6(range(len(ring)))
    optics_seconds = time.perf_counter() - optics_started

    ind_bpms = np.asarray(ind_bpms, dtype=int)
    ind_cors = np.asarray(ind_cors, dtype=int)
    ind_skews = np.atleast_1d(np.asarray(ind_skews, dtype=int))
    bpm = opt_all_location[ind_bpms]
    cor = opt_all_location[ind_cors]
    qua = opt_all_location[ind_skews]
    tune = np.asarray(at.get_tune(ring, get_integer=True), dtype=float)

    bpm_mu = np.asarray([item.mu for item in bpm], dtype=float)
    cor_mu = np.asarray([item.mu for item in cor], dtype=float)
    skew_mu = np.asarray([item.mu for item in qua], dtype=float)

    def tau_matrix(mu_a, mu_b, idx_a, idx_b, plane):
        a = np.asarray(mu_a, dtype=float)
        b = np.asarray(mu_b, dtype=float)
        ia = np.asarray(idx_a)
        ib = np.asarray(idx_b)
        delta = b - a
        equal = np.isclose(b, a, rtol=0.0, atol=1e-12)
        wrap = np.where(equal, ib < ia, b < a)
        return delta + wrap * (2.0 * math.pi * tune[plane]) - math.pi * tune[plane]

    # Corrector-to-BPM phase is invariant for every skew parameter.
    twj = np.empty((2, len(bpm), len(cor)), dtype=float)
    for plane in (0, 1):
        twj[plane] = tau_matrix(
            cor_mu[:, plane][None, :], bpm_mu[:, plane][:, None],
            ind_cors[None, :], ind_bpms[:, None], plane,
        )

    ts = np.zeros((len(cor), 2), dtype=float)
    tc = np.empty((len(cor), 2), dtype=float)
    for w, ordinal in enumerate(ind_cors):
        beta = np.asarray(cor[w].beta, dtype=float)
        if thick_steerer:
            length = float(ring[int(ordinal)].Length)
            alpha = np.asarray(cor[w].alpha, dtype=float)
            ts[w] = length / (2.0 * np.sqrt(beta))
            tc[w] = np.sqrt(beta) - length * alpha / (2.0 * np.sqrt(beta))
        else:
            tc[w] = np.sqrt(beta)

    inv_minus = 1.0 / math.sin(math.pi * (tune[0] - tune[1]))
    inv_plus = 1.0 / math.sin(math.pi * (tune[0] + tune[1]))
    sin_tune = np.sin(math.pi * tune)
    sqrt_beta_bpm = np.sqrt(np.asarray([item.beta for item in bpm], dtype=float))
    mh2v = np.zeros((len(bpm), len(cor), len(qua)), dtype=float)
    mv2h = np.zeros_like(mh2v)

    def cosabc(ax, sign_by, by, sign_c, c, gc, gs, tcw, tsw):
        jsmjx = np.sin(ax) * gc[0] - np.cos(ax) * gs[0]
        jcmjx = np.cos(ax) * gc[0] + np.sin(ax) * gs[0]
        jsmjy = sign_by * (np.sin(by) * gc[1] - np.cos(by) * gs[1])
        jcmjy = np.cos(by) * gc[1] + np.sin(by) * gs[1]
        jswj = sign_c * (np.sin(c) * tcw - np.cos(c) * tsw)
        jcwj = np.cos(c) * tcw + np.sin(c) * tsw
        return (jcmjx * jcmjy * jcwj - jcmjx * jsmjy * jswj
                - jsmjx * jsmjy * jcwj - jsmjx * jcmjy * jswj)

    formula_started = time.perf_counter()
    for m, ordinal in enumerate(ind_skews):
        if cancel_callback is not None and cancel_callback():
            raise RuntimeError(
                "LOCO run cancelled during analytical skew Jacobian calculation."
            )
        length = float(ring[int(ordinal)].Length)
        beta = np.asarray(qua[m].beta, dtype=float)
        alpha = np.asarray(qua[m].alpha, dtype=float)
        if thick_skew:
            normal_k = float(ring[int(ordinal)].PolynomB[1])
            if normal_k != 0.0:
                gs = np.empty(2, dtype=float)
                gc = np.empty(2, dtype=float)
                for plane, signed_k in enumerate((normal_k, -normal_k)):
                    lsk = length * cmath.sqrt(signed_k)
                    gs[plane] = (
                        (1.0 - cmath.cos(lsk))
                        / (length * signed_k * math.sqrt(beta[plane]))
                    ).real
                    gc[plane] = (
                        math.sqrt(beta[plane]) / lsk * cmath.sin(lsk)
                        - alpha[plane] * gs[plane]
                    ).real
            else:
                if verbose:
                    print(
                        f"{ring[int(ordinal)].FamName} is not a quadrupole. "
                        f"PolynomA[1] = {ring[int(ordinal)].PolynomA[1]:.3f}, "
                        f"PolynomB[1] = {normal_k:.3f}"
                    )
                gs = length / (2.0 * np.sqrt(beta))
                gc = np.sqrt(beta) - length * alpha / (2.0 * np.sqrt(beta))
        else:
            gs = np.zeros(2, dtype=float)
            gc = np.sqrt(beta)

        tmj = np.empty((2, len(bpm)), dtype=float)
        tmw = np.empty((2, len(cor)), dtype=float)
        for plane in (0, 1):
            tmj[plane] = tau_matrix(
                skew_mu[m, plane], bpm_mu[:, plane],
                int(ordinal), ind_bpms, plane,
            )
            tmw[plane] = tau_matrix(
                skew_mu[m, plane], cor_mu[:, plane],
                int(ordinal), ind_cors, plane,
            )

        # BPM-dependent skew phases broadcast down columns; corrector-
        # dependent skew phases broadcast across rows.
        jx, jy = tmj[0][:, None], tmj[1][:, None]
        wx, wy = tmw[0][None, :], tmw[1][None, :]
        tx, ty = twj[0], twj[1]
        tcx, tcy = tc[:, 0][None, :], tc[:, 1][None, :]
        tsx, tsy = ts[:, 0][None, :], ts[:, 1][None, :]

        mv2h[:, :, m] = 0.125 * sqrt_beta_bpm[:, 0][:, None] * (
            inv_minus * (
                cosabc(jx, -1, jy, +1, ty, gc, gs, tcy, tsy) / sin_tune[1]
                - cosabc(wx, -1, wy, +1, tx, gc, gs, tcy, tsy) / sin_tune[0]
            )
            + inv_plus * (
                cosabc(jx, +1, jy, -1, ty, gc, gs, tcy, tsy) / sin_tune[1]
                + cosabc(wx, +1, wy, +1, tx, gc, gs, tcy, tsy) / sin_tune[0]
            )
        )
        mh2v[:, :, m] = -0.125 * sqrt_beta_bpm[:, 1][:, None] * (
            inv_minus * (
                -cosabc(jx, -1, jy, -1, tx, gc, gs, tcx, tsx) / sin_tune[0]
                + cosabc(wx, -1, wy, -1, ty, gc, gs, tcx, tsx) / sin_tune[1]
            )
            + inv_plus * (
                cosabc(jx, +1, jy, -1, tx, gc, gs, tcx, tsx) / sin_tune[0]
                + cosabc(wx, +1, wy, +1, ty, gc, gs, tcx, tsx) / sin_tune[1]
            )
        )
        if progress_callback is not None:
            stride = max(1, len(ind_skews) // 20)
            if (m + 1) % stride == 0 or m + 1 == len(ind_skews):
                progress_callback(m + 1, len(ind_skews))

    if timing_callback is not None:
        timing_callback({
            "skew_analytical_implementation": "vectorized",
            "optics_preparation_seconds": optics_seconds,
            "formula_seconds": time.perf_counter() - formula_started,
            "total_seconds": time.perf_counter() - total_started,
            "output_bytes": int(mh2v.nbytes + mv2h.nbytes),
            "thick_skew": bool(thick_skew),
            "thick_steerers": bool(thick_steerer),
        })
    return mh2v, mv2h


def _test_orm_skew_deriv(
        m=0,  # quadrupole index
        col=[2],  # column and row to plot for comparison
        row=[7],
        thick_skew=False,
        thick_steerer=False,
        use_quad_indexes=False,
        use_mp=False,
        ):

    import commissioningsimulations.config as config
    from commissioningsimulations.correction.ClosedOrbit import compute_orbit_response_matrix
    import matplotlib.pyplot as plt
    from os.path import exists
    import pickle
    import time

    # load lattice
    ring = at.load_mat(config.lattice_file_for_test, mat_key=config.lattice_variable_name)

    # get BPM indexes
    ind_bpms = list(at.get_refpts(ring, config.bpms_name_patter))

    # get correctors indexes
    # ind_cor = list(at.get_refpts(ring, config.correctors_fam_name_pattern))
    ind_cor = list(at.get_refpts(ring, config.correctors_for_optics_RM))

    # get quadruople indexes
    if use_quad_indexes:
        ind_qua = list(at.get_refpts(ring, config.normal_quadrupoles_fam_name_pattern))
    else:
        ind_qua = list(at.get_refpts(ring, config.skew_quadrupoles_fam_name_pattern))

    # get reference ORM
    orm0 = './Reference.pkl'
    if not exists(orm0):
        _, _, MH0, MV0, _, _, _, _, _, _, dh0, dv0, Q0 = \
            compute_orbit_response_matrix(ring,
                                          ind_bpms=ind_bpms,
                                          ind_cor=ind_cor,
                                          filename_cod_response=orm0)
    else:
        mat_cont = pickle.load(open(orm0, 'rb'))
        MH0 = mat_cont['MH2V']
        MV0 = mat_cont['MV2H']
        dh0 = mat_cont['hor_dispersion']
        dv0 = mat_cont['ver_dispersion']
        Q0 = mat_cont['tunes']

    # vary + one skew quadrupole
    dKL = 0.001
    K = ring[ind_qua[m]].PolynomA[1]
    ring[ind_qua[m]].PolynomA[1] = K + dKL / ring[ind_qua[m]].Length

    # get modified ORM
    ormq = f'./PlusSkew{ring[ind_qua[m]].FamName}.pkl'
    if not exists(ormq):
        _, _, MHqp, MVqp, _, _, _, _, _, _, dhqp, dvqp, Qqp = \
            compute_orbit_response_matrix(ring,
                                          ind_bpms=ind_bpms,
                                          ind_cor=ind_cor,
                                          filename_cod_response=ormq)
    else:
        mat_cont = pickle.load(open(ormq, 'rb'))
        MHqp = mat_cont['MH2V']
        MVqp = mat_cont['MV2H']
        dhqp = mat_cont['hor_dispersion']
        dvqp = mat_cont['ver_dispersion']
        Qqp = mat_cont['tunes']

    # vary - one skew quadrupole
    ring[ind_qua[m]].PolynomA[1] = K - dKL / ring[ind_qua[m]].Length

    # get modified ORM
    ormq = f'./MinusSkew{ring[ind_qua[m]].FamName}.pkl'
    if not exists(ormq):
        _, _, MHqm, MVqm, _, _, _, _, _, _, dhqm, dvqm, Qqm = \
            compute_orbit_response_matrix(ring,
                                          ind_bpms=ind_bpms,
                                          ind_cor=ind_cor,
                                          filename_cod_response=ormq)
    else:
        mat_cont = pickle.load(open(ormq, 'rb'))
        MHqm = mat_cont['MH2V']
        MVqm = mat_cont['MV2H']
        dhqm = mat_cont['hor_dispersion']
        dvqm = mat_cont['ver_dispersion']
        Qqm = mat_cont['tunes']

    # restore skew quadrupole
    ring[ind_qua[m]].PolynomA[1] = K

    # numeric derivative
    dMHn = (MHqp - MHqm) / 2 / dKL
    dMVn = (MVqp - MVqm) / 2 / dKL

    # compute analytic equivalent
    if thick_skew:
        thick_text = '_thickSkew'
    else:
        thick_text = '_thinSkew'

    if thick_steerer:
        thick_text = thick_text + '_thickSteerer'
    else:
        thick_text = thick_text + '_thinSteerer'

    print(f'start analytic {thick_text} orm derivative')
    start_time = time.time()
    dMHa, dMVa = analytic_orm_variation_with_skew_quadrupole(ring,
                                                             ind_bpms=ind_bpms,
                                                             ind_cors=ind_cor,
                                                             ind_skews=[ind_qua[m]],
                                                             thick_skew=thick_skew,
                                                             thick_steerer=thick_steerer,
                                                             verbose=False,
                                                             use_mp=use_mp)
    end_time = time.time()
    tottime = end_time - start_time
    print(f'time for {thick_text} analytic ORM derivative= {end_time-start_time} seconds')

    # modifications to make it equal to the numeric

    # plot std difference by column
    h_col_dif=[]
    for cc in range(len(ind_cor)):
        h_col_dif.append(np.std(dMHn[:, cc] - dMHa[:, cc, 0]))

    h_row_dif = []
    for rr in range(len(ind_bpms)):
        h_row_dif.append(np.std(dMHn[rr, :] - dMHa[rr, :, 0]))

    v_col_dif=[]
    for cc in range(len(ind_cor)):
        v_col_dif.append(np.std(dMVn[:, cc] - dMVa[:, cc, 0]))

    v_row_dif = []
    for rr in range(len(ind_bpms)):
        v_row_dif.append(np.std(dMVn[rr, :] - dMVa[rr, :, 0]))

    fig, ((axch, axcv), (axrh, axrv)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))

    fig.subplots_adjust(hspace=0.5)

    axch.bar(range(len(h_col_dif)), h_col_dif, label='hor columns')
    axch.set_xlabel('correctors')
    axch.set_ylabel('hor. std(num-ana)')

    axcv.bar(range(len(v_col_dif)), v_col_dif, label='ver columns')
    axcv.set_xlabel('correctors')
    axcv.set_ylabel('ver. std(num-ana)')

    axrh.bar(range(len(h_row_dif)), h_row_dif, label='hor rows')
    axrh.set_xlabel('BPMs')
    axrh.set_ylabel('hor. std(num-ana)')

    axrv.bar(range(len(v_row_dif)), v_row_dif, label='ver rows')
    axrv.set_xlabel('BPMs')
    axrv.set_ylabel('ver. std(num-ana)')


    # plot analytic vs numeric
    for cc, rr in zip(col, row):

        fig, ((axch, axcv), (axrh, axrv)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
        axch.plot(dMHn[:, cc], 'x', label=f'numeric, col={cc}')
        axch.plot(dMHa[:, cc, 0], label=f'analytic {thick_text}, col={cc}')
        axch.plot(dMHn[:, cc]-dMHa[:, cc, 0], ':', label=f'num - ana, col={cc}, cor {ring[ind_cor[cc]].FamName}')
        axch.legend()
        axch.set_ylabel('hor. [m/rad/k1]')
        axch.set_title(f'Skew {ring[ind_qua[m]].FamName} modified by {dKL}')
        axch.set_xlabel('BPMs')

        axcv.plot(dMVn[:, cc], 'x', label=f'numeric, col={cc}')
        axcv.plot(dMVa[:, cc, 0], label=f'analytic {thick_text}, col={cc}')
        axcv.plot(dMVn[:, cc]-dMVa[:, cc, 0], ':', label=f'num - ana, col={cc}, cor {ring[ind_cor[cc]].FamName}')
        axcv.legend()
        axcv.set_ylabel('ver. [m/rad/k1]')
        axcv.set_xlabel('BPMs')

        axrh.plot(dMHn[rr, :], 'x', label=f'numeric, row={rr}')
        axrh.plot(dMHa[rr, :, 0], label=f'analytic {thick_text}, row={rr}')
        axrh.plot(dMHn[rr, :] - dMHa[rr, :, 0], ':', label=f'num - ana, row={rr}, bpm {ring[ind_bpms[rr]].FamName}')
        axrh.legend()
        axrh.set_ylabel('hor. [m/rad/k1]')
        axrh.set_xlabel('correctors')

        axrv.plot(dMVn[rr, :], 'x', label=f'numeric, row={rr}')
        axrv.plot(dMVa[rr, :, 0], label=f'analytic {thick_text}, row={rr}')
        axrv.plot(dMVn[rr, :] - dMVa[rr, :, 0], ':', label=f'num - ana, row={rr}, bpm {ring[ind_bpms[rr]].FamName}')
        axrv.legend()
        axrv.set_ylabel('ver. [m/rad/k1]')
        axrv.set_xlabel('correctors')

    np.savetxt(f'Skew{ring[ind_qua[m]].FamName}_analytic{thick_text}_H.txt', dMHa[:, :, 0], fmt='%10.5f', delimiter=',')
    np.savetxt(f'Skew{ring[ind_qua[m]].FamName}_numeric_H.txt', dMHn, fmt='%10.5f', delimiter=',')
    np.savetxt(f'Skew{ring[ind_qua[m]].FamName}_analytic{thick_text}_V.txt', dMVa[:, :, 0], fmt='%10.5f', delimiter=',')
    np.savetxt(f'Skew{ring[ind_qua[m]].FamName}_numeric_V.txt', dMVn, fmt='%10.5f', delimiter=',')

    plt.show()
    np.savetxt(f'Skew{ring[ind_qua[m]].FamName}_analytic{thick_text}_H_timestdmax.txt',
               (tottime, np.max(h_col_dif), np.max(h_row_dif), np.max(v_col_dif), np.max(v_row_dif),
                np.std(h_col_dif), np.std(h_row_dif), np.std(v_col_dif), np.std(v_row_dif)), fmt='%10.5f',
               delimiter=',')

    return tottime, (np.max(h_col_dif), np.max(h_row_dif), np.max(v_col_dif), np.max(v_row_dif)), \
           (np.std(h_col_dif), np.std(h_row_dif), np.std(v_col_dif), np.std(v_row_dif))


if __name__ == '__main__':

    # _test_orm_skew_deriv(m=0, col=[0], row=[0], thick=False)  # ~ok
    # _test_orm_skew_deriv(m=0, col=[0], row=[0], thick=True)  # ~ok

    # _test_orm_skew_deriv(m=0, col=[10], row=[23], thick=False)  # not ok
    #_test_orm_skew_deriv(m=0, col=[10], row=[23], thick=True)  # not ok

    # varying magnet is a quadrupole
    #_test_orm_skew_deriv(m=32, col=[173], row=[3], thick_skew=False, thick_steerer=False, use_quad_indexes=True)  # ok
    #_test_orm_skew_deriv(m=32, col=[173], row=[3], thick_skew=True, thick_steerer=False, use_quad_indexes=True)  # not ok
    #_test_orm_skew_deriv(m=32, col=[173], row=[3], thick_skew=False, thick_steerer=True, use_quad_indexes=True)  # ok
    #_test_orm_skew_deriv(m=32, col=[173], row=[3], thick_skew=True, thick_steerer=True, use_quad_indexes=True)  # ok

    # varying magnet is not a quadrupole (drift)
    #_test_orm_skew_deriv(m=32, col=[173], row=[3], thick_skew=False, thick_steerer=False, use_quad_indexes=False)  # ok
    #_test_orm_skew_deriv(m=32, col=[173], row=[3], thick_skew=True, thick_steerer=False, use_quad_indexes=False)  # ok
    #_test_orm_skew_deriv(m=32, col=[173], row=[3], thick_skew=False, thick_steerer=True, use_quad_indexes=False)  # ok
    _test_orm_skew_deriv(m=32, col=[173], row=[3], thick_skew=True, thick_steerer=True, use_quad_indexes=False)  # ok

    # test multiprocessing
    # _test_orm_skew_deriv(m=32, col=[173], row=[3], thick_skew=True, thick_steerer=True, use_quad_indexes=False, use_mp=True)  # not ok

pass
