# EBS multi-stage FIT

This maintained example documents the continuation workflow in the historical
`example_skew_coupling_C3.ipynb` from the read-only
`pyLOCO_old_backup/Examples_backup/EBS/EBS_MDT_26Jan_2026` tree.

## Historical C3 sequence

All four calls used the same measured ORM, BPM weights and dispersion. Each
stage used two outer iterations, disabled 6D tracking, included dispersion and
set `force_recompute=True`. The fitted ring, fit dictionary and fit-result
history were passed explicitly to the next call.

1. **Calibration:** horizontal/vertical BPM gains and H/V corrector calibration;
   coupling removed; dispersion weights 5/5; user SVD cut 100.
2. **BPM coupling:** horizontal/vertical BPM coupling; coupling retained;
   dispersion weights 10/10; cut 0.
3. **Normal optics:** normal quadrupoles; coupling removed; dispersion weights
   5/5; cut 110.
4. **Skew/coupling:** skew quadrupoles and horizontal/vertical BPM coupling;
   coupling retained; dispersion weights 5/5; cut 50.

The old `continue_from_previous` therefore meant more than loading a fitted
lattice: `previous_ring`, `previous_fit_dict`, and `previous_fit_results` were
all supplied. The Jacobian was recomputed for every stage. Measurements and BPM
noise weights stayed fixed; the enabled parameters, coupling treatment,
dispersion weights, and SVD cut changed.

## Maintained equivalent

`recipe.json` describes the same four-stage strategy using the current backend
mapping. It intentionally contains no measurement path. Bind it in FIT to a
current EBS lattice and measurement session, run compatibility preflight, then
execute the stages in order.

Every completed stage writes a continuation checkpoint containing:

- `ring_pyloco.mat` — fitted lattice;
- `fit_dict.pkl` — fitted model and calibration/coupling parameter state;
- `fit_results.npy` — fit history;
- `blocks.pkl` — parameter ordering;
- `summary.json` — chi-squared history and configuration provenance.

The FIT run/session manifest binds the recipe, current measurements, reference
lattice checksum and stage checkpoint directories. Reloading the session uses
exactly these persisted artifacts; it does not depend on hidden Python state.

The recipe is a strategy template. Before using it with another EBS measurement,
populate the authoritative element selections/groups and identity requirements
for that lattice, then pass compatibility preflight. Ordinals must never be
transferred to a different lattice without stable-name verification.
