from pyLOCO.helpers import load_config
from pyLOCO.pyloco import pyloco, remove_bad_bpms, save_fit_dict
from pyLOCO.analysis import plot_beta, plot_eta, plot_matrices
import numpy as np
import at
import os
def run_pyloco_from_model(measured_orm, sigma_w,  measured_eta_x_,  measured_eta_y_, config_path: str | None = None,
                        config_module: str | None = None):

    import at.lattice.elements
    import multiprocessing
    import os
    multiprocessing.set_start_method("fork", force=True)

    cfgmod = load_config(config_path=config_path, config_module=config_module)

    ring = at.load_lattice('betamodel.mat', use='betamodel')
    ring.disable_6d()
    cor_indices =  at.get_refpts(ring, 'S[HFDIJ]*')
    used_bpm = at.get_refpts(ring, at.elements.Monitor)
    quad_indices = at.get_refpts(ring, at.elements.Quadrupole)
    QD3 = at.get_refpts(ring, 'QD3[AE]*')
    QF4 = at.get_refpts(ring, 'QF4[ABDE]*')
    QD5 = at.get_refpts(ring, 'QD5[BD]*')
    combined = np.concatenate((QD3, QF4, QD5))
    quad_indices = np.sort(combined)
    CAVords =  at.get_refpts(ring, at.elements.RFCavity)
    skew_ord = at.get_refpts(ring, 'S[HFDIJ]*')
    Corords = [cor_indices , cor_indices]
    CMstep = [[100e-6] * len(Corords[0]),
              [100e-6] * len(Corords[1])]

    # ============================================================================== #
    #               Remove bad BPMs from measurment data
    # ============================================================================== #

    used_bpms_ords = used_bpm

    bad_bpm_positions = np.array([27, 58, 157, 286])
    Noise_BPMx_cleaned = np.delete(sigma_w[:len(sigma_w)//2], bad_bpm_positions)
    Noise_BPMy_cleaned = np.delete(sigma_w[len(sigma_w)//2:], bad_bpm_positions)
    sigma_w = np.concatenate((Noise_BPMx_cleaned, Noise_BPMy_cleaned))[:,
              np.newaxis]  # Note: weight matrix shape for pyloco
    used_bpms_ords = np.delete(used_bpm, bad_bpm_positions)

    # sigma_w = np.concatenate((Noise_BPMx, Noise_BPMy))[:, np.newaxis]
    # used_bpms_ords = used_bpm

    measured_eta_x_ = np.delete(measured_eta_x_, bad_bpm_positions)
    measured_eta_y_ = np.delete(measured_eta_y_, bad_bpm_positions)

    measured_eta_x = measured_eta_x_
    measured_eta_y = measured_eta_y_

    measured_orm, removed = remove_bad_bpms(measured_orm,
                                            bad_bpm_positions,
                                            total_bpms=len(used_bpm),
                                            axis=0,
                                            input_type="positions")
    # If include dispersion is true

    # eta = np.concatenate([measured_eta_x, measured_eta_y])
    # measured_orm = np.hstack((measured_orm, eta.reshape(-1, 1)))

    CMstep = [[100e-6] * len(Corords[0]),
              [100e-6] * len(Corords[1])]

    config_path = os.path.abspath("pyloco_config.py")
    load_config(config_path=config_path)
    from pyloco_config import FitInitConfig, fixed_parameters, loco_options
    fit_cfg = FitInitConfig()

    # --- define arguments ---

    nHorCOR = len(Corords[0])
    nVerCOR = len(Corords[1])
    nHBPM = nVBPM = len(used_bpms_ords)
    nIter = 2
    fit_list = [
        "quads",
        'skew_quads',
        'quads_tilt',
        "hbpm_gain",
        "vbpm_gain",
        "hcor_cal",
        "vcor_cal",
        "HCMEnergyShift",
        "VCMEnergyShift",
        'hbpm_coupling',
        'vbpm_coupling',
        'hcor_coupling',
        'vcor_coupling'
    ]
    fit_list = [
        "quads"
    ]
    remove_coupling_ = True
    includeDispersion = False
    fixedpathlength = False
    hor_dispersion_weight = 1
    ver_dispersion_weight = 1

    fit_results, fit_dict, ring_pyloco,_,_,_,_,_ = pyloco(
        ring,

        # --- general control ---
        algorithm=loco_options.algorithm,
        nIter=nIter,

        # --- indices & number of elements ---
        used_bpms_ords=used_bpms_ords,
        used_cor_ords=Corords,
        quads_ords=quad_indices,
        skew_ords=skew_ord,
        CAVords=CAVords,
        nHBPM=nHBPM,
        nVBPM=nVBPM,
        nHorCOR=nHorCOR,
        nVerCOR=nVerCOR,
        quads_tilt_ind=quad_indices,
        inetial_fit_parameters=None,

        # --- measurement data ---
        orm_measured=measured_orm,
        weights=sigma_w,
        includeDispersion=includeDispersion,
        measured_eta_x=measured_eta_x,
        measured_eta_y=measured_eta_y,
        hor_dispersion_weight= hor_dispersion_weight,
        ver_dispersion_weight= ver_dispersion_weight,

        # --- correctors kicks & RF ---
        CMstep=CMstep,
        rfStep=fixed_parameters.rfstep,
        Frequency=fixed_parameters.Frequency,

        # --- features ---
        fit_list = fit_list,
        individuals=loco_options.individuals,
        remove_coupling_=remove_coupling_,

        # --- outliers & normalization ---
        outlier_rejection=loco_options.outlier_rejection,
        sigma_outlier=loco_options.sigma_outlier,
        apply_normalization=loco_options.apply_normalization,
        normalization_mode=loco_options.normalization_mode,

        # --- SVD selection ---
        svd_selection_method=loco_options.svd_selection_method,
        svd_threshold=loco_options.svd_threshold,
        cut_=loco_options.cut_,
        show_svd_plot=loco_options.show_svd_plot,

        # --- LM options ---
        nLMIter=loco_options.nLMIter,
        Starting_Lambda=loco_options.Starting_Lambda,
        max_lm_lambda=loco_options.max_lm_lambda,
        scaled=loco_options.scaled,

        # --- more options ---
        plot_fit_parameters=loco_options.plot_fit_parameters,
        auto_correct_delta=loco_options.auto_correct_delta,
        fixedpathlength= fixedpathlength,
        fit_cfg=fit_cfg,
    )

    # --- save results ---
    from pathlib import Path
    #save_fit_dict(fit_dict, Path("./output/fit_results_all_parameters"))
    ring_pyloco.save('./output/ring_pyloco.mat', mat_key='ring')

    # --- save results ---
    return fit_results, fit_dict, ring





