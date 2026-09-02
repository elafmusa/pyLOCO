#!/usr/bin/env python3
"""Generate an evidence report for the uncorrected fixed-seed PETRA III profile."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import at
import numpy as np

REPOSITORY = Path(__file__).resolve().parents[4]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from Examples.Demo.start_pysc_server import (build_profile_machine, catalog_for,
                                              realized_error_statistics)
from pyLOCO.response_matrix import response_matrix


def stats(values):
    values = np.asarray(values, dtype=float)
    return {"rms": float(np.sqrt(np.mean(values**2))),
            "maximum_absolute": float(np.max(np.abs(values))),
            "minimum": float(np.min(values)), "maximum": float(np.max(values))}


def main() -> int:
    profile, sc = build_profile_machine("petra3_realistic")
    _, nominal_sc = build_profile_machine("petra3")
    bpms = np.asarray(sc.bpm_system.indices, dtype=int)
    design = sc.lattice.design.disable_6d(copy=True)
    ring = sc.lattice.ring.disable_6d(copy=True)
    true_orbit = np.asarray(sc.lattice.get_orbit(indices=bpms))
    rotated = np.einsum("ijk,jk->ik", sc.bpm_system._rot_matrices, true_orbit)
    indicated = np.vstack(((rotated[0] - sc.bpm_system.offsets_x) * (1 + sc.bpm_system.calibration_errors_x),
                           (rotated[1] - sc.bpm_system.offsets_y) * (1 + sc.bpm_system.calibration_errors_y)))

    _, design_data, design_at_bpms = at.get_optics(design, refpts=bpms, get_chrom=True, get_w=True)
    _, error_data, error_at_bpms = at.get_optics(ring, refpts=bpms, get_chrom=True, get_w=True)
    beta_beat = 100.0 * (error_at_bpms.beta - design_at_bpms.beta) / design_at_bpms.beta

    hidx = [sc.magnet_settings.magnets[name.rsplit("/", 1)[0]].sim_index for name in sc.tuning.HCORR]
    vidx = [sc.magnet_settings.magnets[name.rsplit("/", 1)[0]].sim_index for name in sc.tuning.VCORR]
    orm = response_matrix(ring, bpm_ords=bpms, cm_ords=(hidx, vidx),
                          dkick=(np.full(len(hidx), 1.0e-5), np.full(len(vidx), 1.0e-5)),
                          calculator="Linear", coupling_orm=True,
                          includeDispersion=False, fixedpathlength=False,
                          Frequency=sc.rf_settings.main.frequency, HarmNumber=3840)
    nominal_orm = response_matrix(design, bpm_ords=bpms, cm_ords=(hidx, vidx),
                                  dkick=(np.full(len(hidx), 1.0e-5), np.full(len(vidx), 1.0e-5)),
                                  calculator="Linear", coupling_orm=True,
                                  includeDispersion=False, fixedpathlength=False,
                                  Frequency=sc.rf_settings.main.frequency, HarmNumber=3840)
    nh, nb = len(hidx), len(bpms)
    hh, hv = orm[:nb, :nh], orm[nb:, :nh]
    vh, vv = orm[:nb, nh:], orm[nb:, nh:]
    nominal_hh, nominal_hv = nominal_orm[:nb, :nh], nominal_orm[nb:, :nh]
    nominal_vh, nominal_vv = nominal_orm[:nb, nh:], nominal_orm[nb:, nh:]
    norm = lambda a: float(np.linalg.norm(a))

    samples = np.asarray([sc.bpm_system.capture_orbit() for _ in range(100)])
    noise_x, noise_y = np.std(samples[:, 0, :], axis=0), np.std(samples[:, 1, :], axis=0)
    nominal_true = np.asarray(nominal_sc.lattice.get_orbit(indices=nominal_sc.bpm_system.indices))
    nominal_samples = np.asarray([nominal_sc.bpm_system.capture_orbit() for _ in range(100)])
    nominal_noise_x = np.std(nominal_samples[:, 0, :], axis=0)
    nominal_noise_y = np.std(nominal_samples[:, 1, :], axis=0)
    f0 = float(sc.rf_settings.main.frequency)
    sc.rf_settings.main.set_frequency(f0 + 1500.0)
    sc.rf_settings.main.set_frequency(f0 - 1500.0)
    sc.rf_settings.main.set_frequency(f0)

    report = {
        "profile": profile.key, "random_seed": profile.configuration["random_seed"],
        "requested_error_distribution": profile.configuration["error_distribution"],
        "realized_errors": realized_error_statistics(sc),
        "mapping_integrity": {"bpms": len(sc.bpm_system.names),
                              "horizontal_correctors": len(sc.tuning.HCORR),
                              "vertical_correctors": len(sc.tuning.VCORR),
                              "all_unique": len(set(sc.bpm_system.names)) == 246 and
                              len(set(sc.tuning.HCORR)) == 219 and len(set(sc.tuning.VCORR)) == 194},
        "official_nominal": {
            "mapping_integrity": {"bpms": len(nominal_sc.bpm_system.names),
                                  "horizontal_correctors": len(nominal_sc.tuning.HCORR),
                                  "vertical_correctors": len(nominal_sc.tuning.VCORR)},
            "true_orbit_x_m": stats(nominal_true[0]), "true_orbit_y_m": stats(nominal_true[1]),
            "measured_bpm_noise_x_m": stats(nominal_noise_x),
            "measured_bpm_noise_y_m": stats(nominal_noise_y),
            "rf_hz": float(nominal_sc.rf_settings.main.frequency),
            "horizontal_dispersion_m": stats(design_at_bpms.dispersion[:, 0]),
            "vertical_dispersion_m": stats(design_at_bpms.dispersion[:, 2]),
            "tunes": np.asarray(design_data.tune).tolist(),
            "chromaticities": np.asarray(design_data.chromaticity).tolist(),
            "h_to_v_relative_orm_norm": norm(nominal_hv) / norm(nominal_hh),
            "v_to_h_relative_orm_norm": norm(nominal_vh) / norm(nominal_vv),
        },
        "orbit_at_bpms": {"true_x_m": stats(true_orbit[0]), "true_y_m": stats(true_orbit[1]),
                           "bpm_indicated_noiseless_x_m": stats(indicated[0]),
                           "bpm_indicated_noiseless_y_m": stats(indicated[1])},
        "optics": {"beta_x_beating_percent": stats(beta_beat[:, 0]),
                   "beta_y_beating_percent": stats(beta_beat[:, 1]),
                   "horizontal_dispersion_m": stats(error_at_bpms.dispersion[:, 0]),
                   "vertical_dispersion_m": stats(error_at_bpms.dispersion[:, 2]),
                   "design_tunes": np.asarray(design_data.tune).tolist(),
                   "realistic_tunes": np.asarray(error_data.tune).tolist(),
                   "design_chromaticities": np.asarray(design_data.chromaticity).tolist(),
                   "realistic_chromaticities": np.asarray(error_data.chromaticity).tolist()},
        "coupling": {"hh_frobenius_norm_m_per_rad": norm(hh),
                     "hv_frobenius_norm_m_per_rad": norm(hv),
                     "vh_frobenius_norm_m_per_rad": norm(vh),
                     "vv_frobenius_norm_m_per_rad": norm(vv),
                     "h_to_v_relative_norm": norm(hv) / norm(hh),
                     "v_to_h_relative_norm": norm(vh) / norm(vv),
                     "normal_mode_tune_separation": float(abs(error_data.tune[0] - error_data.tune[1]))},
        "instrumentation": {"measured_bpm_noise_x_m": stats(noise_x),
                            "measured_bpm_noise_y_m": stats(noise_y)},
        "rf": {"original_hz": f0, "restored_hz": float(sc.rf_settings.main.frequency),
               "restoration_difference_hz": float(sc.rf_settings.main.frequency - f0)},
    }
    output = profile.directory / "validation.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    catalog = catalog_for(sc, profile, host="127.0.0.1", port=13131)
    profile.catalog_path.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
