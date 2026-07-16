import numpy as np
import matplotlib.pyplot as plt
import at
from matplotlib.cm import get_cmap
from pySC.tuning.response_measurements import measure_RFFrequencyOrbitResponse
import logging
LOGGER = logging.getLogger(__name__)
#from pySC.core.beam import bpm_reading
#from dispersion import modelDispersion, measureDispersion

def collect_ring_metrics(SC, twiss, elements_indices, RF_indices, rfStep, useIdealRing, trackMode):
    """
    Returns metrics in the SAME UNITS used by `analyze_ring` prints:
      - rms_orbit_*_um           [µm]
      - rms_beta_beating_*_pct   [%]
      - rms_dispersion_err_*_mm  [mm]
      - emittance_*_pm           [pm]
    """
    ring = SC.lattice.design if useIdealRing else SC.lattice.ring

    # raw SI from your helpers
    rmsx, rmsy = rms_orbits(SC, useIdealRing=useIdealRing)               # [m]
    bx_rms_err, by_rms_err = get_beta_beat(SC, twiss, elements_indices)  # [fraction]
    dx_rms_err, dy_rms_err = get_dispersion_err(SC, twiss, useIdealRing)  # [m]
    emit0, _, _ = ring.ohmi_envelope()                                                        # [m·rad]
    emittance_h = emit0['emitXY'][0]
    emittance_v = emit0['emitXY'][1]

    # convert to printed units
    rmsx_um = float(rmsx * 1e6)
    rmsy_um = float(rmsy * 1e6)
    bx_pct  = float(bx_rms_err * 100.0)
    by_pct  = float(by_rms_err * 100.0)
    dx_mm   = float(dx_rms_err * 1e3)
    dy_mm   = float(dy_rms_err * 1e3)
    emit_h_pm = float(emittance_h * 1e12)
    emit_v_pm = float(emittance_v * 1e12)

    # tune & chrom may be numpy types/tuples; make them JSON/npz-friendly
    tune_int_frac = at.get_tune(ring, get_integer=True)
    chrom = at.get_chrom(ring)
    try:
        tune_int_frac = tuple(map(float, tune_int_frac))  # safe cast
    except Exception:
        pass
    try:
        chrom = tuple(map(float, chrom))
    except Exception:
        pass

    return {
        "rms_orbit_x_um": rmsx_um,
        "rms_orbit_y_um": rmsy_um,
        "rms_beta_beating_x_pct": bx_pct,
        "rms_beta_beating_y_pct": by_pct,
        "rms_dispersion_err_x_mm": dx_mm,
        "rms_dispersion_err_y_mm": dy_mm,
        "emittance_h_pm": emit_h_pm,
        "emittance_v_pm": emit_v_pm,
        "tune_integer_fractional": tune_int_frac,
        "chromaticity": chrom,
        "useIdealRing": bool(useIdealRing),
        "trackMode": str(trackMode),
        "rfStep": int(rfStep),
    }


def analyze_ring(SC, twiss,  elements_indices=None, RF_indices=None, rfStep=40,
                 useIdealRing=True, trackMode='ORB', makeplot=False):
    """
    Analyze ring optics: orbit, beta beating, dispersion errors, and optionally plot them.
    """
    if elements_indices is None:
        elements_indices = SC.bpm_system.indices
    if RF_indices is None:
        RF_indices = SC.rf_settings.main.indices

    _, _, twiss0 = at.get_optics(SC.lattice.design, elements_indices)
    _, _, twiss_err = at.get_optics(SC.lattice.ring, elements_indices)


    ring = SC.lattice.design if useIdealRing else SC.lattice.ring

    rmsx, rmsy = rms_orbits(SC, useIdealRing=useIdealRing)
    bx_rms_err, by_rms_err = get_beta_beat(SC, twiss0, twiss_err)
    dx_rms_err, dy_rms_err = get_dispersion_err(SC, twiss0, twiss_err)
    emit0, bbb, eee = SC.lattice.design.ohmi_envelope()
    emittance_h0 = emit0['emitXY'][0]
    emittance_v0 = emit0['emitXY'][1]

    emit, bbb, eee = SC.lattice.ring.ohmi_envelope()
    emittance_h = emit['emitXY'][0]
    emittance_v = emit['emitXY'][1]
    print('----------------------- Analyzing the Lattice ------------------------')
    print(f"RMS horizontal orbit         : {rmsx * 1e6:8.4f} µm")
    print(f"RMS vertical orbit           : {rmsy * 1e6:8.4f} µm")
    print(f"RMS horizontal beta beating  : {bx_rms_err * 100:8.4f} %")
    print(f"RMS vertical beta beating    : {by_rms_err * 100:8.4f} %")
    print(f"RMS horizontal dispersion err: {dx_rms_err * 1e3:8.4f} mm")
    print(f"RMS vertical dispersion err  : {dy_rms_err * 1e3:8.4f} mm")
    print(f"Ideal Tune                         : {at.get_tune(SC.lattice.design, get_integer=True)}")
    print(f"Ideal Chromaticity                 : {at.get_chrom(SC.lattice.design)}")
    print(f"Tune                         : {at.get_tune(SC.lattice.ring, get_integer=True)}")
    print(f"Chromaticity                 : {at.get_chrom(SC.lattice.ring)}")
    print('Ideal emittance_h', emittance_h0 * 1e12, 'emittance_v', emittance_v0 * 1e12)
    print('With errors emittance_h', emittance_h * 1e12, 'emittance_v', emittance_v * 1e12)
    print('----------------------------------------------------------------------')

    if makeplot:
        plot_orbits(SC,twiss,useIdealRing)
        plot_beta_beat(SC, twiss, elements_indices)
        plot_dispersion_err(SC, twiss, RF_indices, rfStep)

def calculate_rms(data):
    return np.sqrt(np.mean(np.square(data)))

def rms_orbits(SC, useIdealRing=False):

    orbit_x, orbit_y = SC.bpm_system.capture_orbit(use_design=useIdealRing)
    return calculate_rms(orbit_x), calculate_rms(orbit_y)


def get_beta_beat(SC, twiss_ref,twiss_err):


    if (twiss_err.beta.shape[0] != twiss_ref.beta.shape[0] or
        twiss_err.beta.shape[1] != twiss_ref.beta.shape[1]):
        raise ValueError(
            f"Twiss beta shape mismatch: "
            f"twiss_ref {twiss_ref.beta.shape}, twiss_err {twiss_err.beta.shape}"
        )

    bx = (twiss_err.beta[:, 0] - twiss_ref.beta[:, 0]) / twiss_ref.beta[:, 0]
    by = (twiss_err.beta[:, 1] - twiss_ref.beta[:, 1]) / twiss_ref.beta[:, 1]

    return calculate_rms(bx), calculate_rms(by)



def get_dispersion_err(SC, twiss_ref,twiss_err):



    dx = twiss_err.dispersion[:, 0] - twiss_ref.dispersion[:, 0]
    dy = twiss_err.dispersion[:, 2] - twiss_ref.dispersion[:, 2]
    return calculate_rms(dx), calculate_rms(dy)


def plot_data(s_pos, data, xlabel, ylabel, title):
    plt.figure(figsize=(7, 3))
    plt.plot(s_pos, data, color='navy')  # Deep blue color
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    # plt.grid(True, which='both', linestyle=':', color='gray')
    plt.tight_layout()
    plt.show()

def plot_orbits(SC, twiss,useIdealRing):
    ring = SC.lattice.design if useIdealRing else SC.lattice.ring
    _, _, twiss = at.get_optics(ring, SC.bpm_system.indices)


    if useIdealRing:
        _, _, elemdata0 = at.get_optics(ring, SC.bpm_system.indices)
        orbit_x = elemdata0.closed_orbit[:, 0]
        orbit_y = elemdata0.closed_orbit[:, 2]
    else:


        orbit_x, orbit_y = SC.bpm_system.capture_orbit()

    s_pos = twiss.s_pos
    plot_data(s_pos, orbit_x * 1e6, "s [m]", "Horizontal orbit $[\mu\mathrm{m}]$", "Horizontal closed orbit")
    plot_data(s_pos, orbit_y * 1e6, "s [m]", "Vertical orbit $[\mu\mathrm{m}]$", "Vertical closed orbit")


def plot_beta_beat(SC, twiss_ref, elements_indices, useIdealRing=False):

    ring = SC.lattice.design if useIdealRing else SC.lattice.ring
    _, _, twiss_err = at.get_optics(ring, elements_indices)
    s_pos = twiss_err.s_pos
    bx = (twiss_err.beta[:, 0] - twiss_ref.beta[:, 0]) / twiss_ref.beta[:, 0]
    by = (twiss_err.beta[:, 1] - twiss_ref.beta[:, 1]) / twiss_ref.beta[:, 1]
    plot_data(s_pos, bx * 100, "s [m]", r"$\Delta\beta_x / \beta_x$ [%]", "Horizontal beta beating")
    plot_data(s_pos, by * 100, "s [m]", r"$\Delta\beta_y / \beta_y$ [%]", "Vertical beta beating")


def plot_dispersion_err(SC, twiss, elements_indices, rfStep , useIdealRing=True):
    if elements_indices is None:
        elements_indices = SC.rf_settings.main.indices
    ring = SC.lattice.design if useIdealRing else SC.lattice.ring

    _, _, twiss = at.get_optics(ring, SC.bpm_system.indices)

    eta = measure_RFFrequencyOrbitResponse(SC, delta_frf=50, normalize=True)
    dx_meas = eta[:len(eta) // 2]
    dy_meas = eta[len(eta) // 2:]


    s_pos = twiss.s_pos

    if len(dx_meas) != len(twiss.dispersion[:, 0]) or len(dy_meas) != len(twiss.dispersion[:, 2]):
        raise ValueError("Dispersion length mismatch between measured and model Twiss data")

    dx = dx_meas - twiss.dispersion[:, 0]
    dy = dy_meas - twiss.dispersion[:, 2]
    plot_data(s_pos, dx * 1000, "s [m]", r"$\Delta\eta_x$ [mm]", "Horizontal dispersion error")
    plot_data(s_pos, dy * 1000, "s [m]", r"$\Delta\eta_y$ [mm]", "Vertical dispersion error")



def twiss_beat(twiss_ref, twiss_ref_bpms, twiss_all,etax_all, etay_all, plot=False, labels=None, colors=None):
    """
    Compute and optionally plot beta and dispersion beating over multiple iterations.

    Parameters
    ----------
    twiss_ref : optics object
        Reference Twiss (usually from the ideal lattice).
    twiss_all : list of optics objects
        List of optics objects to compare against the reference.
    plot : bool
        Whether to plot the beta and dispersion beating.
    labels : list of str, optional
        Custom labels for the legend.
    colors : list of str, optional
        Custom colors for each line.

    Returns
    -------
    bx_rms_list, by_rms_list, dx_rms_list, dy_rms_list : lists of float
        RMS beta beating and dispersion errors for each input Twiss.
    """
    s_pos = twiss_ref.s_pos
    s_pos_bpms = twiss_ref_bpms.s_pos
    bx_rms_list, by_rms_list, dx_rms_list, dy_rms_list = [], [], [], []

    # Default labels
    if labels is None:
        labels = ["Before correction"] + [f"Iteration {i}" for i in range(1, len(twiss_all))]

    if colors is None:
        cmap = get_cmap('tab20')
        colors = [cmap(i % 20) for i in range(len(twiss_all))]

    line_styles = ['-', '-', '-', '-'] * ((len(twiss_all) + 3) // 4)

    if plot:
        init_font = plt.rcParams["font.size"]
        plt.rcParams.update({'font.size': 14})
        fig, ax = plt.subplots(2, 2, figsize=(14, 6), sharex="all")

    for i, twiss_err in enumerate(twiss_all):
        # Beta beating
        bx = (twiss_err.beta[:, 0] - twiss_ref.beta[:, 0]) / twiss_ref.beta[:, 0]
        by = (twiss_err.beta[:, 1] - twiss_ref.beta[:, 1]) / twiss_ref.beta[:, 1]
        bx_rms = calculate_rms(bx) * 100
        by_rms = calculate_rms(by) * 100

        # Dispersion error

        dx = etax_all[i] - twiss_ref_bpms.dispersion[:, 0]
        dy = etay_all[i] - twiss_ref_bpms.dispersion[:, 2]
        #dx = twiss_err.dispersion[:, 0] - twiss_ref.dispersion[:, 0]
        #dy = twiss_err.dispersion[:, 2]  - twiss_ref.dispersion[:, 2]

        dx_rms = calculate_rms(dx)
        dy_rms = calculate_rms(dy)

        # Store RMS values
        bx_rms_list.append(bx_rms)
        by_rms_list.append(by_rms)
        dx_rms_list.append(dx_rms * 1000)
        dy_rms_list.append(dy_rms * 1000)

        if plot:
            color = colors[i]
            style = line_styles[i]

            ax[0, 0].plot(s_pos, bx * 100, label=labels[i], color=color, linestyle=style)
            ax[1, 0].plot(s_pos, by * 100, label=labels[i], color=color, linestyle=style)
            ax[0, 1].plot(s_pos_bpms, dx * 1000, label=labels[i], color=color, linestyle=style)
            ax[1, 1].plot(s_pos_bpms, dy * 1000, label=labels[i], color=color, linestyle=style)

    if plot:
        ylabels = [r'$\Delta\beta_x/\beta_x$ [%]', r'$\Delta\beta_y/\beta_y$ [%]',
                   r'$\Delta D_x$ [mm]', r'$\Delta D_y$ [mm]']

        ax[0, 0].set_ylabel(ylabels[0])
        ax[1, 0].set_ylabel(ylabels[1])
        ax[0, 1].set_ylabel(ylabels[2])
        ax[1, 1].set_ylabel(ylabels[3])
        ax[1, 0].set_xlabel("s [m]")
        ax[1, 1].set_xlabel("s [m]")

        for row in ax:
            for col in row:
                col.grid(True, linestyle=':', alpha=0.6)
                col.legend(loc='upper right', fontsize=9, frameon=True,
                           framealpha=0.9, borderpad=0.4, handletextpad=0.6,
                           borderaxespad=0.4, labelspacing=0.3)

        fig.tight_layout()
        plt.show()
        plt.rcParams.update({'font.size': init_font})

    return bx_rms_list, by_rms_list, dx_rms_list, dy_rms_list



def get_orbit(ring, SC, bpm_ords, trackMode='ORB', nTurns=1, use_model=False):
    """
    Get orbit from either the model or measurement.

    Parameters
    ----------
    SC : SimulatedCommissioning
    bpm_ords : list
        BPM indices
    useIdealRing : bool
    trackMode : 'ORB' or 'TBT'
    nTurns : int
    use_model : bool

    Returns
    -------
    orbit_x, orbit_y : 1D arrays
    """

    orbit_x, orbit_y = SC.bpm_system.capture_orbit(use_design=use_model)

    return orbit_x, orbit_y



def extract_orbit_xy(orbit, use_model):
    """
    Extracts orbit_x and orbit_y depending on the orbit format.

    Parameters
    ----------
    orbit : array
        Orbit data, shape depends on use_model.
    use_model : bool
        If True, orbit is model-based and needs reshaping.

    Returns
    -------
    orbit_x, orbit_y : 1D arrays
    """
    if use_model:
        orbit_x = np.ravel(np.transpose(orbit[0, :, :, :], axes=(2, 1, 0)))
        orbit_y = np.ravel(np.transpose(orbit[2, :, :, :], axes=(2, 1, 0)))
    else:
        orbit_x = orbit[0]
        orbit_y = orbit[1]
    return orbit_x, orbit_y



def modelDispersion(SC, refpts=None):
    """
    Calculates the model dispersion (orbit-based) based on current setpoints
    Returns:
        Dx : The horizontal dispersion given in [m].
        Dy : The vertical dispersion given in [m].
    """

    if refpts is None:
        refpts = SC.bpm_system.indices
        LOGGER.info('Calculating model dispersion at all bpms.')

    _, _, twiss =  SC.lattice.design.get_optics(refpts=refpts)
    Dx = twiss.dispersion[:, 0]  # horizontal dispersion
    Dy = twiss.dispersion[:, 2]  # vertical dispersion
    return Dx, Dy
