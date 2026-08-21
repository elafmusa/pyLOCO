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

def analytic_orm_variation_with_skew_quadrupole(
        ring,
        ind_bpms=None,
        ind_cors=None,
        ind_skews=None,
        verbose=True,
        thick_skew=True,
        thick_steerer=True,
        opt_all_location=None,
        use_mp=False):
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

    MH2V = np.zeros(shape=(len(ind_bpms), len(ind_cors), len(ind_skews)))
    MV2H = np.zeros(shape=(len(ind_bpms), len(ind_cors), len(ind_skews)))

    # loop quadrupoles
    if use_mp:

        n_cpu = multiprocessing.cpu_count()
        n_processes = n_cpu
        if verbose:
            print('parallel computation using {} cores'.format(n_processes))

        with multiprocessing.Pool() as p:
            results = p.starmap(_analytic_orm_variation_with_skew_quadrupole,
                                 zip(repeat(ring),
                                     repeat(ind_bpms),
                                     repeat(ind_cors),
                                     ind_skews,    # loop index
                                     repeat(verbose),
                                     repeat(thick_skew),
                                     repeat(thick_steerer),
                                     repeat(opt_all_location)
                                     )
                                 )

        for m, _ in enumerate(ind_skews):
            _MH2V=results[m][0]
            _MV2H = results[m][1]
            MH2V[:, :, m] = _MH2V[:,:,0]
            MV2H[:, :, m] = _MV2H[:,:,0]

    else:  # sequential

        MH2V, MV2H = _analytic_orm_variation_with_skew_quadrupole(
                        ring,
                        ind_bpms=ind_bpms,
                        ind_cors=ind_cors,
                        ind_skews=ind_skews,
                        verbose=verbose,
                        thick_skew=thick_skew,
                        thick_steerer=thick_steerer,
                        opt_all_location=opt_all_location)

    return MH2V, MV2H


def _analytic_orm_variation_with_skew_quadrupole(
        ring,
        ind_bpms=None,
        ind_cors=None,
        ind_skews=None,
        verbose=True,
        thick_skew=True,
        thick_steerer=True,
        opt_all_location=None):
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

    if isinstance(ind_skews, np.uint32):
        ind_skews = [ind_skews]

    bpm = opt_all_location[ind_bpms]
    cor = opt_all_location[ind_cors]
    qua = opt_all_location[ind_skews]

    # full tunes
    Q = at.get_tune(ring, get_integer=True)

    def tau(pl, a, b, idx_a=None, idx_b=None):
        return dphi(pl, a, b, idx_a=idx_a, idx_b=idx_b) - math.pi*Q[pl]

    def dphi(pl, a, b, idx_a=None, idx_b=None):
        """Forward phase advance, using lattice order to break phase ties."""

        d = b.mu[pl] - a.mu[pl]

        if np.isclose(b.mu[pl], a.mu[pl], rtol=0.0, atol=1e-12):
            if idx_a is not None and idx_b is not None and idx_b < idx_a:
                d = d + 2*math.pi*Q[pl]
        elif b.mu[pl] < a.mu[pl]:
            d = d + 2*math.pi*Q[pl]

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
                tau(p, qua[m], bpm[j], ind_skews[m], ind_bpms[j])
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
                    tau(p, cor[w], bpm[j], ind_cors[w], ind_bpms[j])
                    for p in [x, y]
                ]
                tmw = [
                    tau(p, qua[m], cor[w], ind_skews[m], ind_cors[w])
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

    return MH2V, MV2H


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
