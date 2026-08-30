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

def analytic_orm_variation_with_normal_quadrupole(
        ring,
        ind_bpms=None,
        ind_cors=None,
        ind_quads=None,
        verbose=True,
        thick_quadrupole=True,
        thick_steerers=True,
        opt_all_location=None,
        use_mp=False,
        cancel_callback=None,
        implementation="vectorized",
        progress_callback=None,
        timing_callback=None):
    """
    Computes the derivative of the orbit response matrix compared to normal quadrupoles

    A.Franchi (ESRF), Z.Marti (CELLS),
    "Analytic formulas for the rapid evaluation of the orbit response matrix and chromatic functions from lattice
    parameters in circular accelerators" arXiv:1711:06589v2 17 Apr 2018

    :param ring: AT lattice
    :param ind_bpms: numpy array of np.uint32. indexes of BPMs
    :param ind_cors: numpy array of np.uint32. indexes of correctors
    :param ind_quads: numpy array of np.uint32. indexes of normal quadrupoles
    :param verbose: True/False
    :param thick_quadrupole:  True/False
    :param thick_steerers:  True/False
    :param opt_all_location: output of at.linopt6 at all ring elements
    :param use_mp: True/False use multiprocessing
    :param implementation: ``"vectorized"`` (production default) or
        ``"legacy"`` (original scientific/reference formula loops)
    :return: two 3D lists of floats.
    ORM_HorSteererToHorBPM_over_Kn[BPMS][CORRECTORS][QUAD], ORM_VerSteererToVerBPM_over_Kn[BPMS][CORRECTORS][QUAD],
    """

    implementation = str(implementation).strip().lower()
    if implementation not in {"legacy", "vectorized"}:
        raise ValueError(
            f"Unknown analytical implementation {implementation!r}; "
            "choose 'legacy' or 'vectorized'."
        )

    if len(ind_quads) == 0:
        shape = (len(ind_bpms), len(ind_cors), 0)
        return np.zeros(shape, dtype=float), np.zeros(shape, dtype=float)

    total_started = time.perf_counter()
    tagged_timing_callback = (
        (lambda data: timing_callback({
            "analytical_implementation": implementation, **data
        }))
        if timing_callback is not None else None
    )
    # loop quadrupoles
    if use_mp:

        n_cpu = multiprocessing.cpu_count()
        n_processes = min(n_cpu, len(ind_quads))
        quad_chunks = [
            chunk.tolist()
            for chunk in np.array_split(np.asarray(ind_quads, dtype=int), n_processes)
            if len(chunk)
        ]

        if verbose:
            print('RM derivative parallel computation using {} cores'.format(n_processes))

        worker = (
            _analytic_orm_variation_with_normal_quadrupole_legacy
            if implementation == "legacy"
            else _analytic_orm_variation_with_normal_quadrupole
        )
        with multiprocessing.Pool(processes=n_processes) as p:
            pending = p.starmap_async(worker,
                                 zip(repeat(ring),
                                     repeat(ind_bpms),
                                     repeat(ind_cors),
                                     quad_chunks,
                                     repeat(thick_quadrupole),
                                     repeat(thick_steerers),
                                     repeat(verbose),
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
                        raise RuntimeError("LOCO run cancelled during analytical normal Jacobian calculation.")

        MH = np.concatenate([result[0] for result in results], axis=2)
        MV = np.concatenate([result[1] for result in results], axis=2)
        if progress_callback is not None:
            progress_callback(len(ind_quads), len(ind_quads))
        if timing_callback is not None:
            timing_callback({
                "analytical_implementation": implementation,
                "multiprocessing_total_seconds": time.perf_counter() - total_started,
                "workers": n_processes,
                "chunks": len(quad_chunks),
            })

    else:  # sequential

        worker = (
            _analytic_orm_variation_with_normal_quadrupole_legacy
            if implementation == "legacy"
            else _analytic_orm_variation_with_normal_quadrupole
        )
        MH, MV = worker(
                        ring,
                        ind_bpms=ind_bpms,
                        ind_cors=ind_cors,
                        ind_quads=ind_quads,
                        verbose=verbose,
                        thick_quadrupole=thick_quadrupole,
                        thick_steerers=thick_steerers,
                        opt_all_location=opt_all_location,
                        cancel_callback=cancel_callback,
                        progress_callback=progress_callback,
                        timing_callback=tagged_timing_callback)
        if timing_callback is not None:
            timing_callback({
                "analytical_implementation": implementation,
                "serial_total_seconds": time.perf_counter() - total_started,
            })

    return MH, MV


def _analytic_orm_variation_with_normal_quadrupole(
        ring, ind_bpms=None, ind_cors=None, ind_quads=None,
        thick_quadrupole=False, thick_steerers=False, verbose=True,
        opt_all_location=None, cancel_callback=None, progress_callback=None,
        timing_callback=None):
    """Vectorized analytical normal-quadrupole ORM derivative."""
    total_started = time.perf_counter()
    optics_started = time.perf_counter()
    if opt_all_location is None:
        _, _, opt_all_location = ring.linopt6(range(len(ring)))
    optics_seconds = time.perf_counter() - optics_started
    if np.isscalar(ind_quads):
        ind_quads = [int(ind_quads)]

    ind_bpms = np.asarray(ind_bpms, dtype=int)
    ind_cors = np.asarray(ind_cors, dtype=int)
    ind_quads = np.asarray(ind_quads, dtype=int)
    bpm = opt_all_location[ind_bpms]
    cor = opt_all_location[ind_cors]
    qua = opt_all_location[ind_quads]
    tune = np.asarray(at.get_tune(ring, get_integer=True), dtype=float)[:2]

    bpm_mu = np.asarray([item.mu[:2] for item in bpm], dtype=float)
    cor_mu = np.asarray([item.mu[:2] for item in cor], dtype=float)
    bpm_beta = np.asarray([item.beta[:2] for item in bpm], dtype=float)
    cor_beta = np.asarray([item.beta[:2] for item in cor], dtype=float)
    cor_alpha = np.asarray([item.alpha[:2] for item in cor], dtype=float)
    bpm_s = np.asarray([item.s_pos for item in bpm], dtype=float)
    cor_s = np.asarray([item.s_pos for item in cor], dtype=float)

    def phase_advance(mu_w, mu_j, idx_w, idx_j):
        delta = mu_j - mu_w
        same = np.isclose(mu_j, mu_w, rtol=0.0, atol=1e-12)
        wrap = (~same & (mu_j < mu_w)) | (same & (idx_j < idx_w))
        return delta + wrap * (2.0 * np.pi * tune)

    def ordering(s_a, s_b, idx_a, idx_b):
        return (s_a < s_b) | ((s_a == s_b) & (idx_a < idx_b))

    response_preparation_started = time.perf_counter()
    twj = phase_advance(
        cor_mu[np.newaxis, :, :], bpm_mu[:, np.newaxis, :],
        ind_cors[np.newaxis, :, np.newaxis], ind_bpms[:, np.newaxis, np.newaxis],
    ) - np.pi * tune
    pwj = phase_advance(
        cor_mu[np.newaxis, :, :], bpm_mu[:, np.newaxis, :],
        ind_cors[np.newaxis, :, np.newaxis], ind_bpms[:, np.newaxis, np.newaxis],
    )

    cor_lengths = np.asarray([ring[index].Length for index in ind_cors], dtype=float)
    ts = cor_lengths[:, np.newaxis] / (2.0 * np.sqrt(cor_beta))
    tc = np.sqrt(cor_beta) - cor_lengths[:, np.newaxis] * cor_alpha / (2.0 * np.sqrt(cor_beta))
    if thick_steerers:
        jswj = np.sin(twj) * tc[np.newaxis, :, :] - np.cos(twj) * ts[np.newaxis, :, :]
        jcwj = np.cos(twj) * tc[np.newaxis, :, :] + np.sin(twj) * ts[np.newaxis, :, :]
        jcdwj = np.cos(pwj) * tc[np.newaxis, :, :] + np.sin(pwj) * ts[np.newaxis, :, :]
    else:
        sqrt_cor_beta = np.sqrt(cor_beta)[np.newaxis, :, :]
        jswj = sqrt_cor_beta * np.sin(twj)
        jcwj = sqrt_cor_beta * np.cos(twj)
        jcdwj = sqrt_cor_beta * np.cos(pwj)

    mh = np.zeros((len(bpm), len(cor), len(qua)), dtype=float)
    mv = np.zeros_like(mh)
    pi_b_c = ordering(
        bpm_s[:, np.newaxis], cor_s[np.newaxis, :],
        ind_bpms[:, np.newaxis], ind_cors[np.newaxis, :],
    ).astype(float)
    response_preparation_seconds = time.perf_counter() - response_preparation_started
    thick_coefficient_seconds = 0.0
    derivative_seconds = 0.0

    for m, qindex in enumerate(ind_quads):
        parameter_started = time.perf_counter()
        if cancel_callback is not None and cancel_callback():
            raise RuntimeError("LOCO run cancelled during analytical normal Jacobian calculation.")
        q = qua[m]
        q_mu = np.asarray(q.mu[:2], dtype=float)
        q_beta = np.asarray(q.beta[:2], dtype=float)
        q_alpha = np.asarray(q.alpha[:2], dtype=float)
        q_s = float(q.s_pos)
        tmj = phase_advance(
            q_mu, bpm_mu, np.full((len(bpm), 1), qindex), ind_bpms[:, np.newaxis],
        ) - np.pi * tune
        tmw = phase_advance(
            q_mu, cor_mu, np.full((len(cor), 1), qindex), ind_cors[:, np.newaxis],
        ) - np.pi * tune

        ibm = np.empty(2, dtype=float)
        ism_base = np.empty(2, dtype=float)
        icm_base = np.empty(2, dtype=float)
        length = float(ring[qindex].Length)
        is_quadrupole = isinstance(ring[qindex], at.Quadrupole)
        strength = float(ring[qindex].PolynomB[1]) if is_quadrupole else 0.0
        coefficient_started = time.perf_counter()
        for plane, sign in enumerate((1.0, -1.0)):
            if thick_quadrupole and is_quadrupole:
                km = sign * strength
                gamma = (1.0 + q_alpha[plane] ** 2) / q_beta[plane]
                skl = 2.0 * cmath.sqrt(km) * length
                ib = (0.5 * (q_beta[plane] + gamma / km)
                      + cmath.sin(skl) / (2.0 * skl) * (q_beta[plane] - gamma / km)
                      + q_alpha[plane] / (2.0 * km * length) * (cmath.cos(skl) - 1.0)).real
                sk = 2.0 * cmath.sqrt(km)
                is_value = (1.0 / (km * length) * (
                    0.5 * (1.0 - cmath.cos(sk * length))
                    + q_alpha[plane] / q_beta[plane] * (cmath.sin(sk * length) / sk - length)
                )).real
                ic_value = (ib - 1.0 / (km * q_beta[plane]) * (1.0 - cmath.sin(skl) / skl)).real
            elif thick_quadrupole:
                gamma = (1.0 + q_alpha[plane] ** 2) / q_beta[plane]
                ib = q_beta[plane] - q_alpha[plane] * length + gamma * length ** 2 / 3.0
                is_value = length - 2.0 * q_alpha[plane] * length ** 2 / (3.0 * q_beta[plane])
                ic_value = ib - 2.0 * length ** 2 / (3.0 * q_beta[plane])
            else:
                ib = q_beta[plane]
                is_value = q_beta[plane]
                ic_value = q_beta[plane]
            ibm[plane] = ib
            ism_base[plane] = is_value
            icm_base[plane] = ic_value
        coefficient_elapsed = time.perf_counter() - coefficient_started
        thick_coefficient_seconds += coefficient_elapsed

        pi_q_b = ordering(q_s, bpm_s, qindex, ind_bpms).astype(float)
        pi_q_c = ordering(q_s, cor_s, qindex, ind_cors).astype(float)
        order_term = pi_q_b[:, np.newaxis] - pi_q_c[np.newaxis, :] + pi_b_c

        for plane, output in ((0, mh), (1, mv)):
            sin_j = np.sin(2.0 * tmj[:, plane])
            cos_j = np.cos(2.0 * tmj[:, plane])
            sin_w = np.sin(2.0 * tmw[:, plane])
            cos_w = np.cos(2.0 * tmw[:, plane])
            if thick_quadrupole:
                ismj = sin_j * icm_base[plane] - cos_j * ism_base[plane]
                icmj = cos_j * icm_base[plane] + sin_j * ism_base[plane]
                ismw = sin_w * icm_base[plane] - cos_w * ism_base[plane]
                icmw = cos_w * icm_base[plane] + sin_w * ism_base[plane]
            else:
                ismj = q_beta[plane] * sin_j
                icmj = q_beta[plane] * cos_j
                ismw = q_beta[plane] * sin_w
                icmw = q_beta[plane] * cos_w
            if thick_quadrupole and thick_steerers:
                pcmwj = (icmw[np.newaxis, :] * (
                    np.cos(twj[:, :, plane]) * tc[np.newaxis, :, plane]
                    + np.sin(twj[:, :, plane]) * ts[np.newaxis, :, plane])
                    - 2.0 * ismw[np.newaxis, :] * ts[np.newaxis, :, plane] * np.cos(twj[:, :, plane]))
                psmwj = (ismw[np.newaxis, :] * (
                    np.sin(twj[:, :, plane]) * tc[np.newaxis, :, plane]
                    - np.cos(twj[:, :, plane]) * ts[np.newaxis, :, plane])
                    + 2.0 * icmw[np.newaxis, :] * ts[np.newaxis, :, plane] * np.sin(twj[:, :, plane]))
            else:
                pcmwj = jcwj[:, :, plane] * icmw[np.newaxis, :]
                psmwj = jswj[:, :, plane] * ismw[np.newaxis, :]
            numerator = (
                (jcwj[:, :, plane] * icmj[:, np.newaxis] + pcmwj
                 + jswj[:, :, plane] * ismj[:, np.newaxis] - psmwj)
                / (4.0 * np.sin(2.0 * np.pi * tune[plane]))
                + 0.5 * ibm[plane] * jswj[:, :, plane] * order_term
                + ibm[plane] * jcdwj[:, :, plane] / (4.0 * np.sin(np.pi * tune[plane]))
            )
            output[:, :, m] = (
                np.sqrt(bpm_beta[:, plane])[:, np.newaxis]
                / (2.0 * np.sin(np.pi * tune[plane])) * numerator
            )
        derivative_seconds += time.perf_counter() - parameter_started - coefficient_elapsed
        if verbose:
            print(f"Analytical normal Jacobian {m + 1} / {len(ind_quads)}")
        if progress_callback is not None:
            stride = max(1, len(ind_quads) // 20)
            if (m + 1) % stride == 0 or m + 1 == len(ind_quads):
                progress_callback(m + 1, len(ind_quads))

    if timing_callback is not None:
        timing_callback({
            "optics_preparation_seconds": optics_seconds,
            "bpm_corrector_preparation_seconds": response_preparation_seconds,
            "thick_coefficient_seconds": thick_coefficient_seconds,
            "per_parameter_derivative_seconds": max(0.0, derivative_seconds),
            "array_allocation_and_total_seconds": time.perf_counter() - total_started,
            "parameters": int(len(ind_quads)),
        })
    return mh, mv


def _analytic_orm_variation_with_normal_quadrupole_legacy(
        ring,
        ind_bpms=None,
        ind_cors=None,
        ind_quads=None,
        thick_quadrupole=False,
        thick_steerers=False,
        verbose=True,
        opt_all_location=None,
        cancel_callback=None,
        progress_callback=None,
        timing_callback=None):
    """
    analytic orbit response matrix derivative with integrated (KL) errors at normal quadrupoles

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

    # print(type(ind_quads))
    # Multiprocessing dispatches one lattice ordinal to each worker as a
    # plain Python ``int``.  Normalize every NumPy/Python scalar to the same
    # one-element sequence accepted by the sequential implementation.
    if np.isscalar(ind_quads):
        ind_quads = [int(ind_quads)]

    bpm = opt_all_location[ind_bpms]
    cor = opt_all_location[ind_cors]
    qua = opt_all_location[ind_quads]


    # full tunes
    Q = at.get_tune(ring, get_integer=True)

    # functions required by later formulas

    #def PI(a, b):
    #    val = 1
    #    if a.s_pos >= b.s_pos:
    #        val = 0
    #    return val

    def PI(a, b, idx_a=None, idx_b=None):
        """

        Ordering function.

        Returns 1 when a occurs before b in the forward lattice
        ordering and 0 otherwise.

        For identical s positions, lattice indices break the tie.

        Edited: E.M 20 Aug 2026
        """

        if a.s_pos < b.s_pos:
            return 1

        if a.s_pos > b.s_pos:
            return 0

        # Same longitudinal position
        if idx_a is not None and idx_b is not None:
            return int(idx_a < idx_b)

        # Preserve old behaviour if indices are unavailable
        return 0

    #def tau(pl, a, b):
    #    return dphi(pl, a, b) - math.pi*Q[pl]

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

    #def dphi(pl, w, j):

    #    d = j.mu[pl] - w.mu[pl]

    #    if j.mu[pl] < w.mu[pl]:
    #        d = d + 2*math.pi*Q[pl]

    #    return d


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

    MH = np.zeros(shape=(len(bpm), len(cor), len(qua)))
    MV = np.zeros(shape=(len(bpm), len(cor), len(qua)))

    x = 0
    y = 1
    _sign = [+1, -1]
    h = 0
    v = 1

    def Ib(qm, p, Km, Lm):
        # Km = abs(Km)
        gamma = (1 + qm.alpha[p] ** 2) / qm.beta[p]
        sKL = 2 * cmath.sqrt(Km) * Lm

        val = 1 / 2 * (qm.beta[p] + gamma / Km) + \
              cmath.sin(sKL) / (2 * sKL) * (qm.beta[p] - gamma / Km) + \
              qm.alpha[p] / (2 * Km * Lm) * (cmath.cos(sKL) - 1)

        return val.real

    def IS(qm, p, Km, Lm):
        #Km = abs(Km)
        sK = 2 * cmath.sqrt(Km)
        val = 1 / (Km * Lm) * (
                1 / 2 * (1 - cmath.cos( sK * Lm)) +
                qm.alpha[p] / qm.beta[p] * (cmath.sin(sK * Lm) / sK - Lm)
              )
        return val.real

    def IC(qm, p, Km, Lm):
        # Km = abs(Km)
        sKL = 2 * cmath.sqrt(Km) * Lm
        val = Ib(qm, p, Km, Lm) - 1 / (Km * qm.beta[p]) * (1 - cmath.sin(sKL) / sKL)
        return val.real


    def Ib_notquad(qm, p, Lm):
        # Km = abs(Km)
        gamma = (1 + qm.alpha[p] ** 2) / qm.beta[p]

        val = qm.beta[p] - qm.alpha[p]*Lm + gamma/3*Lm**2

        return val.real

    def IS_notquad(qm, p, Lm):
        #Km = abs(Km)

        val = Lm - 2/3 * qm.alpha[p] / qm.beta[p] * Lm**2

        return val.real

    def IC_notquad(qm, p, Lm):
        # Km = abs(Km)

        val = Ib_notquad(qm, p, Lm) - 2/3 / qm.beta[p] * Lm**2

        return val.real

    # thick steerers corrections
    TS = []
    TC = []
    for countw, w in enumerate(range(len(cor))):
        Lw = ring[ind_cors[w]].Length
        alpha = cor[w].alpha
        beta = cor[w].beta
        TS.append( [Lw / (2 * beta[p]**0.5) for p in [x, y]] )
        TC.append( [beta[p]**0.5 - Lw * alpha[p] / (2 * beta[p]**0.5) for p in [x, y]] )

    for countm, m in enumerate(range(len(qua))):
        if cancel_callback is not None and cancel_callback():
            raise RuntimeError("LOCO run cancelled during analytical normal Jacobian calculation.")
        if progress_callback is not None:
            stride = max(1, len(qua) // 20)
            if (countm + 1) % stride == 0 or countm + 1 == len(qua):
                progress_callback(countm + 1, len(qua))

        if thick_quadrupole:
            Lm = ring[ind_quads[m]].Length
            # if it is a quadrupole
            if isinstance(ring[ind_quads[m]], at.Quadrupole):
                Km = ring[ind_quads[m]].PolynomB[1]

                Ibm = [[Ib(qua[m], p, _sign[s]*Km, Lm) for p in [x, y]] for s in [h, v]]
                ISm = [[IS(qua[m], p, _sign[s]*Km, Lm) for p in [x, y]] for s in [h, v]]
                ICm = [[IC(qua[m], p, _sign[s]*Km, Lm) for p in [x, y]] for s in [h, v]]

            else:
                # # else, assume drift (only quads change optics)
                Ibm = [[Ib_notquad(qua[m], p, Lm) for p in [x, y]] for s in [h, v]]
                ISm = [[IS_notquad(qua[m], p, Lm) for p in [x, y]] for s in [h, v]]
                ICm = [[IC_notquad(qua[m], p, Lm) for p in [x, y]] for s in [h, v]]

        for countj, j in enumerate(range(len(bpm))):

            #tmj = [tau(p, qua[m], bpm[j]) for p in [x, y]]

            tmj = [
            tau(
                p,
                qua[m],
                bpm[j],
                idx_a=ind_quads[m],
                idx_b=ind_bpms[j],
            )
            for p in [x, y]
        ]

            for countw, w in enumerate(range(len(cor))):
                if verbose:
                    print(f'computing response of steerer {ring[ind_cors[w]].FamName}'
                          f' to BPM {ring[ind_bpms[j]].FamName}'
                          f' with a normal (thick={thick_quadrupole}) quadrupole error in {ring[ind_quads[m]].FamName}')

                #twj = [tau(p, cor[w], bpm[j]) for p in [x, y]]
                #tmw = [tau(p, qua[m], cor[w]) for p in [x, y]]
                #pwj = [dphi(p, cor[w], bpm[j]) for p in [x, y]]

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
                    idx_a=ind_quads[m],
                    idx_b=ind_cors[w],
                )
                for p in [x, y]
                ]

                pwj = [
                dphi(
                    p,
                    cor[w],
                    bpm[j],
                    idx_w=ind_cors[w],
                    idx_j=ind_bpms[j],
                )
                for p in [x, y]
                ]

                if thick_quadrupole:
                    # thick quadrupole
                    ISmj = [[math.sin(2 * tmj[p]) * ICm[p][s] - math.cos(2 * tmj[p]) * ISm[p][s] for p in [x, y]] for s in [h, v]]   # eq. C14
                    ICmj = [[math.cos(2 * tmj[p]) * ICm[p][s] + math.sin(2 * tmj[p]) * ISm[p][s] for p in [x, y]] for s in [h, v]]
                    ISmw = [[math.sin(2 * tmw[p]) * ICm[p][s] - math.cos(2 * tmw[p]) * ISm[p][s] for p in [x, y]] for s in [h, v]] # eq. C14
                    ICmw = [[math.cos(2 * tmw[p]) * ICm[p][s] + math.sin(2 * tmw[p]) * ISm[p][s] for p in [x, y]] for s in [h, v]]

                else:
                    # thin quadrupoles
                    Ibm = [[qua[m].beta[p] for p in [x, y]] for s in [h, v]]
                    ISmj = [[qua[m].beta[p] * math.sin(2 * tmj[p]) for p in [x, y]] for s in [h, v]]  # eq. C14
                    ICmj = [[qua[m].beta[p] * math.cos(2 * tmj[p]) for p in [x, y]] for s in [h, v]]
                    ISmw = [[qua[m].beta[p] * math.sin(2 * tmw[p]) for p in [x, y]] for s in [h, v]]  # eq. C14
                    ICmw = [[qua[m].beta[p] * math.cos(2 * tmw[p]) for p in [x, y]] for s in [h, v]]

                if thick_steerers:
                    # thick steerers
                    JSwj = [math.sin(twj[p]) * TC[w][p] - math.cos(twj[p]) * TS[w][p] for p in [x, y]]  # eq. C40
                    JCwj = [math.cos(twj[p]) * TC[w][p] + math.sin(twj[p]) * TS[w][p] for p in [x, y]]
                    JCdwj = [math.cos(pwj[p]) * TC[w][p] + math.sin(pwj[p]) * TS[w][p] for p in [x, y]]
                else:
                    # thin steerers
                    JSwj = [cor[w].beta[p] ** 0.5 * math.sin(twj[p]) for p in [x, y]]
                    JCwj = [cor[w].beta[p] ** 0.5 * math.cos(twj[p]) for p in [x, y]]
                    JCdwj =[cor[w].beta[p] ** 0.5 * math.cos(pwj[p]) for p in [x, y]]

                if thick_quadrupole and thick_steerers:
                    # thick quadrupoles and steerers
                    PCmwj = [[ICmw[p][s] * (math.cos(twj[p])*TC[w][p] + math.sin(twj[p])*TS[w][p] )
                             - 2 * ISmw[p][s]*TS[w][p] * math.cos(twj[p]) for p in [x, y]] for s in [h, v]]

                    PSmwj = [[ISmw[p][s] * (math.sin(twj[p]) * TC[w][p] - math.cos(twj[p]) * TS[w][p])
                             + 2 * ICmw[p][s] * TS[w][p] * math.sin(twj[p]) for p in [x, y]] for s in [h, v]]

                else:
                    #elif (thick_quadrupole and not(thick_steerers)) or (not(thick_quadrupole) and thick_steerers):
                    # thin quads thick steerers or thin quad thick steerers
                    PCmwj = [[JCwj[p] * ICmw[p][s] for p in [x, y]] for s in [h, v]]
                    PSmwj = [[JSwj[p] * ISmw[p][s] for p in [x, y]] for s in [h, v]]

                #else:
                #    # thin quad and thin steerers
                #    PCmwj = [cor[w].beta[p] ** 0.5 * math.cos(twj[p]) * qua[m].beta[p] * math.cos(2*tmw[p]) for p in [x, y]]
                #
                #    PSmwj = [cor[w].beta[p] ** 0.5 * math.sin(twj[p]) * qua[m].beta[p] * math.sin(2*tmw[p]) for p in [x, y]]

                #  V   sign swap intentional to recover numeric ORM. AT sign conventions
                MH[j][w][m] = + ((bpm[j].beta[x]) ** 0.5) \
                              / (2 * math.sin(math.pi * Q[x])) * \
                              (
                              +
                              1 / (4 * math.sin(2 * math.pi * Q[x])) *
                              (JCwj[x]*ICmj[x][h] + PCmwj[x][h] + JSwj[x]*ISmj[x][h] - PSmwj[x][h])
                              +
                              0.5 * Ibm[x][h] * JSwj[x] *
                              (
                                PI(
                                    qua[m],
                                    bpm[j],
                                    idx_a=ind_quads[m],
                                    idx_b=ind_bpms[j],
                                )
                                -
                                PI(
                                    qua[m],
                                    cor[w],
                                    idx_a=ind_quads[m],
                                    idx_b=ind_cors[w],
                                )
                                +
                                PI(
                                    bpm[j],
                                    cor[w],
                                    idx_a=ind_bpms[j],
                                    idx_b=ind_cors[w],
                                )
                            )
                              +
                              Ibm[x][h] * JCdwj[x] /
                              (4 * math.sin(math.pi * Q[x]))
                              )



                MV[j][w][m] = + ((bpm[j].beta[y]) ** 0.5) \
                              / (2 * math.sin(math.pi * Q[y])) * \
                              (
                              +
                              1 / (4 * math.sin(2 * math.pi * Q[y])) *
                              (JCwj[y]*ICmj[y][v] + PCmwj[y][v] + JSwj[y]*ISmj[y][v] - PSmwj[y][v])
                              +
                              0.5 * Ibm[y][v] * JSwj[y] * # * cor[w].beta[y]**0.5 * math.sin(twj[y]) *
                              (
                                PI(
                                    qua[m],
                                    bpm[j],
                                    idx_a=ind_quads[m],
                                    idx_b=ind_bpms[j],
                                )
                                -
                                PI(
                                    qua[m],
                                    cor[w],
                                    idx_a=ind_quads[m],
                                    idx_b=ind_cors[w],
                                )
                                +
                                PI(
                                    bpm[j],
                                    cor[w],
                                    idx_a=ind_bpms[j],
                                    idx_b=ind_cors[w],
                                )
                            )
                              +
                              Ibm[y][v] * JCdwj[y] /  # * cor[w].beta[y]**0.5 * math.cos(dphi(y, cor[w], bpm[j])) /
                              (4 * math.sin(math.pi * Q[y]))
                              )
                """
                # if anything thick (actually works also for thin-thin and it is faster!)
                if thick_quadrupole or thick_steerers:
                else: # all thin
                               #  V   sign swap intentional to recover numeric ORM. AT sign conventions
                    MH[j][w][m] = + ((bpm[j].beta[x] * cor[w].beta[x]) ** 0.5) * qua[m].beta[x] \
                                   / (2 * math.sin(math.pi * Q[x])) * \
                                   (
                                   +
                                   math.cos(twj[x]) /
                                   (4 * math.sin(2 * math.pi * Q[x])) * (math.cos(2 * tmj[x]) + math.cos(2 * tmw[x]))
                                   +
                                   math.sin(twj[x]) /
                                   (4 * math.sin(2 * math.pi * Q[x])) * (math.sin(2 * tmj[x]) - math.sin(2 * tmw[x]))
                                   +
                                   0.5 * math.sin(twj[x]) *
                                   (PI(qua[m], bpm[j]) - PI(qua[m], cor[w]) + PI(bpm[j], cor[w]))
                                   +
                                   math.cos(dphi(x, cor[w], bpm[j])) / (4 * math.sin(math.pi * Q[x]))
                                   )

                    MV[j][w][m] = + ((bpm[j].beta[y] * cor[w].beta[y]) ** 0.5) * qua[m].beta[y] \
                                   / (2 * math.sin(math.pi * Q[y])) * \
                                   (
                                   +
                                   math.cos(twj[y]) /
                                   (4 * math.sin(2 * math.pi * Q[y])) * (math.cos(2 * tmj[y]) + math.cos(2 * tmw[y]))
                                   +
                                   math.sin(twj[y]) /
                                   (4 * math.sin(2 * math.pi * Q[y])) * (math.sin(2 * tmj[y]) - math.sin(2 * tmw[y]))
                                   +
                                   0.5 * math.sin(twj[y]) *
                                   (PI(qua[m], bpm[j]) - PI(qua[m], cor[w]) + PI(bpm[j], cor[w]))
                                   +
                                   math.cos(dphi(y, cor[w], bpm[j])) / (4 * math.sin(math.pi * Q[y]))
                                   )
                """

                """
                if np.isnan(MH[j,w,m]):
                    print(f'MH[{j},{w},{m}] is NaN for '
                          f'Quad {m} {ring[ind_quads[m]].FamName} '
                          f'Cor {w} {ring[ind_cors[w]].FamName} ')
                """

    return MH, MV
