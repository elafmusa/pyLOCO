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
        use_mp=False):
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
    :return: two 3D lists of floats.
    ORM_HorSteererToHorBPM_over_Kn[BPMS][CORRECTORS][QUAD], ORM_VerSteererToVerBPM_over_Kn[BPMS][CORRECTORS][QUAD],
    """

    MH = np.zeros(shape=(len(ind_bpms), len(ind_cors), len(ind_quads)))
    MV = np.zeros(shape=(len(ind_bpms), len(ind_cors), len(ind_quads)))

    # loop quadrupoles
    if use_mp:

        n_cpu = multiprocessing.cpu_count()
        n_processes = n_cpu

        if verbose:
            print('RM derivative parallel computation using {} cores'.format(n_processes))

        with multiprocessing.Pool() as p:
            results = p.starmap( _analytic_orm_variation_with_normal_quadrupole,
                                 zip(repeat(ring),
                                     repeat(ind_bpms),
                                     repeat(ind_cors),
                                     ind_quads,    # loop index
                                     repeat(thick_quadrupole),
                                     repeat(thick_steerers),
                                     repeat(verbose),
                                     repeat(opt_all_location)
                                     )
                                 )

        for m, _ in enumerate(ind_quads):
            _MH = results[m][0]
            _MV = results[m][1]
            MH[:, :, m] = _MH[:,:,0]
            MV[:, :, m] = _MV[:,:,0]

    else:  # sequential

        MH, MV = _analytic_orm_variation_with_normal_quadrupole(
                        ring,
                        ind_bpms=ind_bpms,
                        ind_cors=ind_cors,
                        ind_quads=ind_quads,
                        verbose=verbose,
                        thick_quadrupole=thick_quadrupole,
                        thick_steerers=thick_steerers,
                        opt_all_location=opt_all_location)

    return MH, MV
def _analytic_orm_variation_with_normal_quadrupole(
        ring,
        ind_bpms=None,
        ind_cors=None,
        ind_quads=None,
        thick_quadrupole=False,
        thick_steerers=False,
        verbose=True,
        opt_all_location=None):
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
    if isinstance(ind_quads, np.uint32):
        ind_quads = [ind_quads]

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
