from pyLOCO.pyloco import pyloco, remove_bad_bpms, plot_data
from pyLOCO.helpers import load_config
import numpy as np
import at

def run_pyloco_from_model(measured_orm, sigma_w,  measured_eta_x,  measured_eta_y, config_path: str | None = None,
                        config_module: str | None = None):

    import at.lattice.elements
    import multiprocessing
    multiprocessing.set_start_method("fork", force=True)

    cfgmod = load_config(config_path=config_path, config_module=config_module)

    ring = at.load_lattice('betamodel.mat', use='betamodel')
    cor_indices =  at.get_refpts(ring, 'S[HFDIJ]*')
    used_bpms_ords = at.get_refpts(ring, at.elements.Monitor)
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
    ring.disable_6d()
    import os
    config_path = os.path.abspath("pyloco_config.py")
    load_config(config_path=config_path)
    from pyloco_config import FitInitConfig, fixed_parameters, loco_options
    fit_cfg = FitInitConfig()

    # --- define arguments ---

    nHorCOR = len(Corords[0])
    nVerCOR = len(Corords[1])
    nHBPM = nVBPM = len(used_bpms_ords)
    nIter = 5
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
        "quads",
        "hbpm_gain",
        "vbpm_gain",
        "hcor_cal",
        "vcor_cal",
        "HCMEnergyShift"
    ]
    remove_coupling_ = True
    includeDispersion = False
    fixedpathlength = False
    hor_dispersion_weight = 10
    ver_dispersion_weight = 10

    fit_results, fit_dict, ring = pyloco(
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
    np.save("./output/loco_lm_fit_results_iterations.npy", fit_results)

    # --- save results ---
    return fit_results, fit_dict, ring





