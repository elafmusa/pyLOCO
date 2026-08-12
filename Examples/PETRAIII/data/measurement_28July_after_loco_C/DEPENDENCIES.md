# PETRA III 28 July after-LOCO case C input manifest

The original archive remains read-only at
`/Users/musa/Desktop/measurment_28July_after_loco/Run_LOCO/`.

| Category | Original input used by `C.ipynb` | Purpose | Migrated/reused location |
|---|---|---|---|
| Measurement | `measurements/Optics_before/ORM_before_LOCO_merged.h5` | Full measured ORM used by the archived run (492×377 before BPM removal) | `ORM_before_LOCO_merged.h5` |
| Measurement | `measurements/Optics_before/Dispersion_RF_minus1500Hz_2026-07-28_11-10-21.h5` and `Dispersion_RF_plus1500Hz_2026-07-28_11-09-23.h5` | Notebook subtracts minus − plus | Replaced by the equivalent prepared `Dispersion_RF_difference.h5`; source plus/minus files intentionally not copied |
| Measurement | `measurements/Optics_before/Dispersion_RF_difference.h5` | Prepared minus − plus orbit difference (`mean_orbit_x/y`) | `Dispersion_RF_difference.h5` |
| Measurement | `measurements/Optics_before/BPM_noise_2026-07-28_11-11-15.h5` | BPM standard deviations (`std_orbit_x/y`) | `BPM_noise_2026-07-28_11-11-15.h5` |
| Reference | `dk_k.npy` | Optional 193-family reference ΔK/K series used in the longitudinal comparison plot | `reference_family_dk_over_k.npy` |
| Model | Original pre-August-6 `p3_v24.mat`, preserved byte-identically as `Run_LOCO/p3_v24_from_matlab.mat` and elsewhere in the archive | 3,693-element PETRA III v24 lattice used by C | `../p3_v24_C_original.mat` |
| Shared selection | `quad_ind_2024.npy` | 398 normal quadrupoles/tilts | `../quad_ind_2024.npy` (byte-identical existing file) |
| Shared selection | `quad_family_groups_new.npy` | Ordered 193-family grouping | `../quad_family_groups_new.npy` (byte-identical existing file) |
| Shared selection | `skew_ind_2026.npy` | 16 skew indices | `../skew_ind_2026.npy` (byte-identical existing file) |
| Shared names | `HCM_names_control.txt` | 184 horizontal correctors | `../HCM_names_control.txt` (byte-identical existing file) |
| Shared names | `VCM_names_control.txt` | 193 vertical correctors | `../VCM_names_control.txt` (byte-identical existing file) |
| Shared names | `BPM_names.txt` | Ordered 246 BPM selection before removal | `../BPM_names.txt` (byte-identical existing file) |
| Configuration | `pyloco_config.py` plus values overridden in `C.ipynb` | Solver/RF/initialization defaults | `../../configs/measurement_28July_after_loco_C.yaml` |

Corrector steps are constants constructed by the notebook (100 µrad for every
H/V corrector), so there is no corrector-step input file to copy.

Old output, plots, Jacobian caches, pickles, fitted lattices, correction JSON,
and notebook files were intentionally not copied because they are generated
results, not runtime inputs.

## Stale notebook-path resolution

The saved C result has a 476×378 fitted matrix: 238 retained BPMs per plane,
377 correctors, and one dispersion column. That proves the executed analysis
used the full merged ORM. The notebook source now resolves its wildcard to the
later-added 492×20 short ORM, which cannot produce the archived result.

Likewise, the `p3_v24.mat` currently beside C was replaced on 2026-08-06 with
a 7,673-element lattice after C ran. C's saved fitted lattice has 3,693
elements, and its maximum selected lattice index is 3,683. The migrated model
is the byte-identical 3,693-element version preserved elsewhere in the archive
(SHA-256 `e1dbab6e5a293adf73c8987c9bdc96f860adb6fb8becbad2d8e3b6f2a7952512`).
