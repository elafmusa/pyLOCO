#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, time, traceback
from pathlib import Path
from typing import Any
import numpy as np
import at
from at import get_refpts
from pySC import SimulatedCommissioning
from pySC.tuning.response_measurements import measure_OrbitResponseMatrix, measure_RFFrequencyOrbitResponse
from pySC.tuning.averaging import get_average_orbit

import improve_pyLOCO_latest as drv
from pyLOCO.pyloco import get_fit_param_block
from set_correction import set_correction
from analyze_ring import analyze_ring

CASES = {
    "linear_numerical": ("Linear", "Numerical"),
    "linear_analytical": ("Linear", "Analytical"),
    "analytical_numerical": ("Analytical", "Numerical"),
    "analytical_analytical": ("Analytical", "Analytical"),
    "tracking_numerical": ("Tracking", "Numerical"),
    "tracking_analytical": ("Tracking", "Analytical"),
}

def jdefault(x: Any):
    if isinstance(x, Path): return str(x)
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.floating): return float(x)
    if isinstance(x, np.bool_): return bool(x)
    raise TypeError(type(x).__name__)

def save_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=jdefault))

def update_status(case_dir: Path, **kwargs):
    p = case_dir / "status.json"
    cur = {}
    if p.exists():
        try: cur = json.loads(p.read_text())
        except Exception: pass
    cur.update(kwargs)
    cur["updated_unix"] = time.time()
    save_json(p, cur)

def prepare_sc(seed_file: Path):
    sc = SimulatedCommissioning.from_json(seed_file, lattice_file=str(drv.LATTICE_FILE))
    sc.lattice.ring.enable_6d()
    sc.lattice.design.enable_6d()
    x0, y0 = sc.bpm_system.capture_orbit()
    sc.bpm_system.reference_x = x0
    sc.bpm_system.reference_y = y0
    sc.tuning.chromaticity.controls_1 = [c for c in sc.control_arrays["sd"] if c.endswith("B3")]
    sc.tuning.chromaticity.controls_2 = [c for c in sc.control_arrays["sf"] if c.endswith("B3")]
    return sc

def selection(sc):
    data = np.load(drv.CORRECTOR_FILE)
    hc = data["hcor_inds"].astype(int).tolist()
    vc = data["vcor_inds"].astype(int).tolist()
    bpms = np.asarray(sc.bpm_system.indices, dtype=int)
    q = np.asarray(get_refpts(sc.lattice.ring, at.elements.Quadrupole), dtype=int)
    names = sc.magnet_arrays["cysf"] + sc.magnet_arrays["cxysf"] + sc.magnet_arrays["cxsf"] + sc.magnet_arrays["cxysf2"]
    sq = np.asarray(sorted(sc.magnet_settings.magnets[n].sim_index for n in names), dtype=int)
    cav = np.asarray(get_refpts(sc.lattice.ring, at.elements.RFCavity), dtype=int)
    ids = np.sort(np.concatenate([get_refpts(sc.lattice.ring, "BPM_09_DW*"),
                                  get_refpts(sc.lattice.ring, "BPM_01_DW*")])).astype(int)
    cm = [np.full(len(hc), drv.CORRECTOR_KICK_RAD), np.full(len(vc), drv.CORRECTOR_KICK_RAD)]
    return bpms, [hc, vc], q, sq, cav, ids, cm

def measure(sc, cors, bpms, cm, cycle_dir):
    t0 = time.perf_counter()
    idxmap = sc.magnet_settings.index_mapping
    hcorr = [idxmap[c] + "/B1L" for c in cors[0]]
    vcorr = [idxmap[c] + "/A1L" for c in cors[1]]

    t = time.perf_counter()
    orm = np.asarray(measure_OrbitResponseMatrix(sc, hcorr, vcorr,
        dkick=drv.CORRECTOR_KICK_RAD, normalize=False, bipolar=True), float)
    orm_time = time.perf_counter() - t
    orm[:, :min(drv.FLIP_FIRST_N_ORM_COLUMNS, orm.shape[1])] *= -1.0

    t = time.perf_counter()
    eta = np.asarray(measure_RFFrequencyOrbitResponse(sc, delta_frf=drv.RF_STEP_HZ,
        normalize=False, bipolar=True), float).ravel()
    rf_time = time.perf_counter() - t
    ex, ey = np.split(eta, 2)

    t = time.perf_counter()
    _, _, xs, ys = get_average_orbit(sc, drv.BPM_AVERAGES)
    noise_time = time.perf_counter() - t
    sigma = np.concatenate([np.asarray(xs).ravel(), np.asarray(ys).ravel()])

    np.savez_compressed(cycle_dir / "measured_inputs.npz",
        measured_orm=orm, measured_eta_x=ex, measured_eta_y=ey, sigma_w=sigma,
        hcor_inds=np.asarray(cors[0]), vcor_inds=np.asarray(cors[1]),
        used_bpms_ords=bpms, CMstep_h=cm[0], CMstep_v=cm[1],
        rf_step_hz=drv.RF_STEP_HZ, rf_frequency_hz=drv.PETRA_IV_RF_FREQUENCY_HZ)
    timing = {"orm_measurement_s": orm_time, "rf_measurement_s": rf_time,
              "bpm_noise_s": noise_time, "measurement_total_s": time.perf_counter() - t0}
    save_json(cycle_dir / "measurement_timing.json", timing)
    return orm, ex, ey, sigma, timing

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=sorted(CASES))
    ap.add_argument("--seed-file", required=True, type=Path)
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--stage1-iter", type=int, default=4)
    ap.add_argument("--stage2-iter", type=int, default=4)
    a = ap.parse_args()

    orm_calc, jac_calc = CASES[a.case]
    stage1_orm, stage2_orm = drv.benchmark_stage_response_matrix_calculators(orm_calc)
    case_dir = a.output_root / a.case
    case_dir.mkdir(parents=True, exist_ok=True)
    update_status(case_dir, state="STARTING", completed=False, case=a.case,
                  orm_calculator=orm_calc,
                  stage1_orm_calculator=stage1_orm,
                  stage2_orm_calculator=stage2_orm,
                  jacobian_calculator=jac_calc)

    drv.STAGE1_RESPONSE_MATRIX_CALCULATOR = stage1_orm
    drv.STAGE2_RESPONSE_MATRIX_CALCULATOR = stage2_orm
    drv.QUAD_JACOBIAN_CALCULATOR = jac_calc
    drv.SKEW_JACOBIAN_CALCULATOR = jac_calc
    drv.STAGE1_NITER = a.stage1_iter
    drv.STAGE2_NITER = a.stage2_iter
    drv.FORCE_RECOMPUTE_JACOBIANS = True
    drv.SAVE_JACOBIANS = True

    try:
        sc = prepare_sc(a.seed_file)
        bpms, cors, q, sq, cav, ids, cm = selection(sc)
        save_json(case_dir / "optics_before.json",
                  analyze_ring(sc, elements_indices=bpms, special_elements=ids,
                               useIdealRing=False, makeplot=False, return_dict=True))
        sc.to_json(case_dir / "machine_before.json")
        campaign_t0 = time.perf_counter()
        timing_summary = {}

        for cyc in range(1, a.cycles + 1):
            cdir = case_dir / f"cycle_{cyc:02d}"
            cdir.mkdir(parents=True, exist_ok=True)
            update_status(case_dir, state=f"CYCLE_{cyc}_MEASURING", cycle=cyc)

            cyc_t0 = time.perf_counter()
            save_json(cdir / "optics_before_cycle.json",
                      analyze_ring(sc, elements_indices=bpms, special_elements=ids,
                                   useIdealRing=False, makeplot=False, return_dict=True))
            sc.to_json(cdir / "machine_before_cycle.json")

            orm, ex, ey, sigma, mt = measure(sc, cors, bpms, cm, cdir)

            update_status(case_dir, state=f"CYCLE_{cyc}_LOCO_RUNNING", cycle=cyc)
            fit_t0 = time.perf_counter()
            result = drv.run_latest_pyloco_two_stage(
                model_ring=sc.lattice.design, CMstep=cm, CAVords=cav,
                quad_indices=q, skew_quad_indices=sq, used_cor_ords=cors,
                used_bpms_ords=bpms, measured_orm=orm, sigma_w=sigma,
                measured_eta_x=ex, measured_eta_y=ey,
                output_dir=cdir / "pyloco",
                stage1_response_matrix_calculator=stage1_orm,
                stage2_response_matrix_calculator=stage2_orm)
            fit_wall = time.perf_counter() - fit_t0
            _, fit_dict, _, _, _, _ = result

            fq = np.asarray(get_fit_param_block(fit_dict, "quads"), float).ravel()
            dq = fq - np.asarray([sc.lattice.design[i].K for i in q], float)
            fs = np.asarray(get_fit_param_block(fit_dict, "skew_quads"), float).ravel()
            ds = fs - np.asarray([drv.current_skew_strength(sc.lattice.design[i]) for i in sq], float)
            np.savez_compressed(cdir / "corrections.npz", delta_q=dq, delta_skew=ds,
                                fitted_quads=fq, fitted_skews=fs,
                                quad_indices=q, skew_indices=sq)

            update_status(case_dir, state=f"CYCLE_{cyc}_APPLYING", cycle=cyc)
            t = time.perf_counter()
            set_correction(sc, -dq, q, individuals=True, skewness=False)
            sc.lattice.ring.enable_6d(); sc.lattice.design.enable_6d()
            save_json(cdir / "optics_after_quad.json",
                      analyze_ring(sc, elements_indices=bpms, special_elements=ids,
                                   useIdealRing=False, makeplot=False, return_dict=True))
            set_correction(sc, -ds, sq, individuals=True, skewness=True)
            sc.lattice.ring.enable_6d(); sc.lattice.design.enable_6d()
            save_json(cdir / "optics_after_skew.json",
                      analyze_ring(sc, elements_indices=bpms, special_elements=ids,
                                   useIdealRing=False, makeplot=False, return_dict=True))
            apply_s = time.perf_counter() - t

            update_status(case_dir, state=f"CYCLE_{cyc}_COMMISSIONING", cycle=cyc)
            t = time.perf_counter()
            sc.tuning.correct_orbit(parameter=20)
            sc.tuning.tune.correct(measurement_method="cheat")
            sc.tuning.chromaticity.correct(gain=0.8)
            comm_s = time.perf_counter() - t

            final_optics = analyze_ring(sc, elements_indices=bpms, special_elements=ids,
                                        useIdealRing=False, makeplot=False, return_dict=True)
            save_json(cdir / "optics_after_commissioning.json", final_optics)
            sc.to_json(cdir / "machine_after_cycle.json")

            cyc_timing = dict(mt)
            cyc_timing.update({"loco_fit_wall_s": fit_wall, "correction_apply_s": apply_s,
                               "commissioning_s": comm_s,
                               "cycle_total_s": time.perf_counter() - cyc_t0})
            save_json(cdir / "timing.json", cyc_timing)
            timing_summary[f"cycle_{cyc:02d}"] = cyc_timing
            update_status(case_dir, state=f"CYCLE_{cyc}_COMPLETE",
                          last_completed_cycle=cyc, cycle=cyc)

        total = time.perf_counter() - campaign_t0
        timing_summary["campaign_total_s"] = total
        save_json(case_dir / "timing_summary.json", timing_summary)
        save_json(case_dir / "optics_final.json", final_optics)
        sc.to_json(case_dir / "machine_final.json")
        update_status(case_dir, state="COMPLETE", completed=True,
                      last_completed_cycle=a.cycles, campaign_total_s=total)

    except BaseException as exc:
        update_status(case_dir, state="FAILED", completed=False, error=repr(exc))
        (case_dir / "fatal_error.traceback.log").write_text(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
