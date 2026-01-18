from RDT import get_rdts
import at
import numpy as np
def ORM_rdts_new(dkick, ring, quads_ind, used_bpm):
    cd = []
    ca = []
    etay = []

    [elemdata0, beamdata, elemdata] = at.get_optics(ring, used_bpm)
    Eta_yy0 = elemdata.dispersion[:,2]

    for quad_index in quads_ind:
        a = ring[quad_index].PolynomA[1]

        ring[quad_index].PolynomA[1] = dkick + a

        [elemdata0, beamdata, elemdata] = at.get_optics(ring, used_bpm)
        Eta_yy1 =  elemdata.dispersion[:,2]

        F1001C, F1010C, f2000, f0020 = get_rdts(ring, used_bpm)

        F1001R_c = np.real(F1001C)
        F1001I_c = np.imag(F1001C)
        F1010R_c = np.real(F1010C)
        F1010I_c = np.imag(F1010C)
        difference = np.array([F1001R_c, F1001I_c])
        addit = np.array([F1010R_c, F1010I_c])

        ring[quad_index].PolynomA[1] = a

        cd.append(difference.flatten())
        ca.append(addit.flatten())
        etay.append(Eta_yy1)

    CD = np.squeeze(cd) / dkick
    CA = np.squeeze(ca) / dkick
    Etay = np.squeeze(etay) / dkick

    return CD, CA, Etay


def ORM_mu(dkick, ring, quads_ind, used_bpm):
    cxx = []
    cyy = []
    etax = []
    qxx = []
    qyy = []
    bx =[]
    by =[]

    lindata0, tune, chrom, lindata = ring.linopt(get_chrom=True, refpts=used_bpm)
    s_pos = lindata['s_pos']
    mux0 = lindata['mu'][:, 0] / (2 * np.pi)
    muy0 = lindata['mu'][:, 1] / (2 * np.pi)

    mux0_diff = np.append(np.diff(mux0), mux0[-1] - mux0[0])
    muy0_diff = np.append(np.diff(muy0), muy0[-1] - muy0[0])

    Eta_xx0 = lindata['dispersion'][:, 0]

    qx0 = tune[0]
    qy0 = tune[1]

    beta_x0 = lindata['beta'][:, 0]
    beta_y0 = lindata['beta'][:, 1]


    for quad_index in quads_ind:

        a = ring[quad_index].PolynomB[1]

        ring[quad_index].PolynomB[1] = dkick + a


        lindata0, tune, chrom, lindata = ring.linopt(get_chrom=True, refpts=used_bpm)
        s_pos = lindata['s_pos']
        mux = lindata['mu'][:, 0] / (2 * np.pi)
        muy = lindata['mu'][:, 1] / (2 * np.pi)

        Eta_xx = lindata['dispersion'][:, 0]

        qx = tune[0]
        qy = tune[1]

        beta_x = lindata['beta'][:, 0]
        beta_y = lindata['beta'][:, 1]



        mux_diff = np.append(np.diff(mux), mux[-1] - mux[0])
        muy_diff = np.append(np.diff(muy), muy[-1] - muy[0])

        mux1 = mux_diff - mux0_diff
        muy1 = muy_diff - muy0_diff

        Eta_xx1 = Eta_xx - Eta_xx0

        qx1 = qx - qx0
        qy1 = qy - qy0

        beta_x1 = beta_x - beta_x0
        beta_y1 = beta_y - beta_y0


        cxx.append(mux1)
        cyy.append(muy1)

        etax.append(Eta_xx1)

        qxx.append(qx1)
        qyy.append(qy1)

        bx.append(beta_x1)
        by.append(beta_y1)

        ring[quad_index].PolynomB[1] = a

    Cxx = np.squeeze(cxx)/dkick
    Cyy = np.squeeze(cyy)/dkick
    Etax = np.squeeze(etax)/dkick

    Qx = np.squeeze(qxx)/dkick
    Qy = np.squeeze(qyy)/dkick

    Bx = np.squeeze(bx)/dkick
    By = np.squeeze(by)/dkick



    return Cxx, Cyy, Etax, Qx, Qy, Bx, By
