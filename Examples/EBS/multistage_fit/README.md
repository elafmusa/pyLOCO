# ESRF-EBS Case C with the current FIT interface

This directory reproduces the continuation workflow in the historical
`example_skew_coupling_C3.ipynb` using the maintained FIT backend and GUI.
The preserved measurement files and lattice are copied here because Case C is
not the same dataset or lattice as the newer pySC commissioning example.

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

## Open in the GUI

Launch `pyloco-gui`, choose **Open**, and select
`EBS_case_C_multistage.pyloco.json`. The project contains the full four-stage
recipe. Use **Preview full workflow** to inspect it before running.

Every completed stage writes a continuation checkpoint containing:

- `ring_pyloco.mat` — fitted lattice;
- `fit_dict.pkl` — fitted model and calibration/coupling parameter state;
- `fit_results.npy` — fit history;
- `blocks.pkl` — parameter ordering;
- `summary.json` — chi-squared history and configuration provenance.

The FIT run/session manifest binds the recipe, current measurements, reference
lattice checksum and stage checkpoint directories. Reloading the session uses
exactly these persisted artifacts; it does not depend on hidden Python state.

The project retains 317 BPMs after removing selected-list positions 27, 231 and
286, 32 horizontal plus 32 vertical correctors, 252 individual normal
quadrupoles, and 288 individual `PolynomA[1]` skew components hosted by EBS
sextupoles. These are the exact selections in the historical notebook.

To reuse the strategy with another measurement, replace its ORM, dispersion,
and BPM-noise files. Keep the lattice and device ordering unchanged unless the
new files carry an independently verified ordering.

`recipe.json` is the reusable strategy-only file. It deliberately contains no
measurement paths; the GUI project binds it to the preserved Case C inputs.

## Standalone run

From the repository root:

```bash
python Examples/EBS/multistage_fit/run_case_c_multistage.py --preflight-only
python Examples/EBS/multistage_fit/run_case_c_multistage.py
```

Replacement measurements can be supplied without editing the recipe:

```bash
python Examples/EBS/multistage_fit/run_case_c_multistage.py \
  --orm /path/to/orm.h5 \
  --dispersion /path/to/dispersion.h5 \
  --bpm-noise /path/to/bpm_noise.h5 \
  --output /path/to/results \
  --session /path/to/fit-session.json
```

The replacement files must preserve the Case C shapes, units, normalization,
row/column ordering and RF convention. The loader validates dimensions, while
scientific provenance and ordering must still be verified for new data.
