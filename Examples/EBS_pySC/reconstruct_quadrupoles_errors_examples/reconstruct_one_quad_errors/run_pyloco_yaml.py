"""
run_pyloco_yaml.py
reads all options from pyloco_config.yaml.
"""
import os
import multiprocessing
from pathlib import Path

import numpy as np
import yaml
import at
import at.lattice.elements

from pyLOCO.pyloco import pyloco, remove_bad_bpms, save_fit_dict
from pyloco_config2 import FitInitConfig  # dataclass used by pyloco internals


# ── Config loader ─────────────────────────────────────────────────────────

def load_yaml_config(path: str | Path = "pyloco_config2.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Main runner ───────────────────────────────────────────────────────────

def run_pyloco_yaml(
    measured_orm,
    sigma_w,
    measured_eta_x_,
    measured_eta_y_,
    config_path: str | Path = "pyloco_config2.yaml",
):
    multiprocessing.set_start_method("fork", force=True)

    cfg = load_yaml_config(config_path)
    lat_cfg  = cfg["lattice"]
    el_cfg   = cfg["elements"]
    loco_cfg = cfg["loco"]
    fi_cfg   = cfg["fit_init"]
    fp_cfg   = cfg["fixed_parameters"]
    out_cfg  = cfg["output"]

    # ── Load lattice ──────────────────────────────────────────────────────
    ring = at.load_lattice(lat_cfg["file"], use=lat_cfg["use"])
    ring.disable_6d()

    # ── Element indices ───────────────────────────────────────────────────
    cor_indices  = at.get_refpts(ring, el_cfg["corrector_pattern"])
    used_bpm     = at.get_refpts(ring, getattr(at.elements, el_cfg["bpm_type"]))
    CAVords      = at.get_refpts(ring, getattr(at.elements, el_cfg["cavity_type"]))
    skew_ord     = at.get_refpts(ring, el_cfg["skew_pattern"])
    Corords      = [cor_indices, cor_indices]

    quad_indices = np.sort(np.concatenate([
        at.get_refpts(ring, pat) for pat in el_cfg["quad_groups"]
    ]))

    # ── Remove bad BPMs ───────────────────────────────────────────────────
    bad = np.asarray(cfg["bad_bpms"])
    n   = len(used_bpm)

    sigma_w = np.concatenate([
        np.delete(sigma_w[:n], bad),
        np.delete(sigma_w[n:], bad),
    ])[:, np.newaxis]

    used_bpms_ords   = np.delete(used_bpm, bad)
    measured_eta_x   = np.delete(measured_eta_x_, bad)
    measured_eta_y   = np.delete(measured_eta_y_, bad)
    measured_orm, _  = remove_bad_bpms(
        measured_orm, bad, total_bpms=n, axis=0, input_type="positions"
    )

    # ── CM step ───────────────────────────────────────────────────────────
    step    = cfg["cm_step"]
    CMstep  = [[step] * len(Corords[0]), [step] * len(Corords[1])]

    nHBPM = nVBPM = len(used_bpms_ords)
    nHorCOR = len(Corords[0])
    nVerCOR = len(Corords[1])

    # ── FitInitConfig from YAML ───────────────────────────────────────────
    fit_cfg = FitInitConfig(
        fit_list        = loco_cfg["fit_list"],
        block_order     = fi_cfg["block_order"],
        init_policy     = fi_cfg["init_policy"],
        CMstep          = CMstep,
        rfStep          = fp_cfg["rfstep"],
        individuals     = loco_cfg["individuals"],
        quads_attr      = fi_cfg["quads_attr"],
        quads_attr_index= fi_cfg["quads_attr_index"],
        skew_attr       = fi_cfg["skew_attr"],
        skew_attr_index = fi_cfg["skew_attr_index"],
        quads_tilt_attr_R1  = fi_cfg["quads_tilt_attr_R1"],
        quads_tilt_attr_R2  = fi_cfg["quads_tilt_attr_R2"],
        quads_tilt_method   = fi_cfg["quads_tilt_method"],
    )

    # ── Run pyloco ────────────────────────────────────────────────────────
    fit_results, fit_dict, ring_pyloco, *_ = pyloco(
        ring,

        # general
        algorithm   = loco_cfg["algorithm"],
        nIter       = loco_cfg["nIter"],

        # indices
        used_bpms_ords  = used_bpms_ords,
        used_cor_ords   = Corords,
        quads_ords      = quad_indices,
        skew_ords       = skew_ord,
        CAVords         = CAVords,
        nHBPM           = nHBPM,
        nVBPM           = nVBPM,
        nHorCOR         = nHorCOR,
        nVerCOR         = nVerCOR,
        quads_tilt_ind  = quad_indices,
        inetial_fit_parameters = None,

        # measurement data
        orm_measured            = measured_orm,
        weights                 = sigma_w,
        includeDispersion       = loco_cfg["includeDispersion"],
        measured_eta_x          = measured_eta_x,
        measured_eta_y          = measured_eta_y,
        hor_dispersion_weight   = loco_cfg["hor_dispersion_weight"],
        ver_dispersion_weight   = loco_cfg["ver_dispersion_weight"],

        # correctors & RF
        CMstep      = CMstep,
        rfStep      = fp_cfg["rfstep"],
        Frequency   = fp_cfg["Frequency"],

        # features
        fit_list        = loco_cfg["fit_list"],
        individuals     = loco_cfg["individuals"],
        remove_coupling_= loco_cfg["remove_coupling_"],

        # outliers & normalisation
        outlier_rejection   = loco_cfg["outlier_rejection"],
        sigma_outlier       = loco_cfg["sigma_outlier"],
        apply_normalization = loco_cfg["apply_normalization"],
        normalization_mode  = loco_cfg["normalization_mode"],

        # SVD
        svd_selection_method = loco_cfg["svd_selection_method"],
        svd_threshold        = loco_cfg["svd_threshold"],
        cut_                 = loco_cfg["cut_"],
        show_svd_plot        = loco_cfg["show_svd_plot"],

        # LM
        nLMIter         = loco_cfg["nLMIter"],
        Starting_Lambda = loco_cfg["Starting_Lambda"],
        max_lm_lambda   = loco_cfg["max_lm_lambda"],
        scaled          = loco_cfg["scaled"],

        # misc
        plot_fit_parameters = loco_cfg["plot_fit_parameters"],
        auto_correct_delta  = loco_cfg["auto_correct_delta"],
        fixedpathlength     = loco_cfg["fixedpathlength"],
        fit_cfg             = fit_cfg,
    )

    # ── Save results ──────────────────────────────────────────────────────
    ring_pyloco.save(out_cfg["ring_file"], mat_key=out_cfg["mat_key"])

    return fit_results, fit_dict, ring
