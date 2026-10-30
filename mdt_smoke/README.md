# pyLOCO modern EBS MDT smoke test

This bundle is the reproducible **Measure → FIT → Correct** smoke test for the
modern 4434-element ESRF-EBS pyAML/pySC simulator. It uses the measurements
acquired on 18 October 2026. It is not the historical 4432-element Case C
example.

## Safety boundary

- The prepared FIT is deliberately restricted to `QD3J`, `QF4J`, and `QD5B`.
- Opening, validating, and comparing are read-only. `Run LOCO` performs an
  offline calculation; it does not write to the accelerator.
- Do not use Correct's Apply action during this smoke test. Perform mapping,
  read-only strength acquisition, and preview only.
- The copied pyAML configuration and measurement metadata describe a live,
  write-capable adapter. Their presence does **not** authorize machine writes.
- A restricted two-iteration simulator FIT completed successfully during MDT
  verification. Its generated results remain local and are not publication
  fixtures or evidence that the configuration is suitable for the real machine.

## Modern package inputs

- Complete FIT project: `configs/EBS_pyaml_MDT_smoke.pyloco.json`
- Four-stage proposal: `configs/EBS_pyaml_MDT_four_stage_REVIEW_REQUIRED.pyloco.json`
- Modern selection audit: `configs/EBS_modern_selection_provenance.json`
- Measurement session: `measurements/measurement-session.pyloco-session.json`
- Exact modern lattice: `lattice/betamodel.mat`, variable `betamodel`
- Validated measurement mapping: `mappings/EBS_measurement_devices.validated.local.json`
- Correct mapping: `mappings/EBS_three_quadrupoles.correct.validated.json`
- Exact hashes and preflight: `VALIDATION.json`

The FIT is restricted to three individual normal quadrupoles:
QD3J (ordinal 27), QF4J (36), and QD5B (51). With only five horizontal and five
vertical correctors, this is a workflow/compatibility smoke test. It does not
identify a full-machine error distribution, BPM/corrector calibration,
coupling, skew errors, or RF calibration.

The older `output/`, `run_fit_smoke.py`, and `ebs_small_orm_fit.yaml` are
retained evidence from the 13 October smoke exercise. They are not inputs to
this package.

## Two deliberately separate FIT projects

### 1. Restricted software smoke test

`EBS_pyaml_MDT_smoke.pyloco.json` remains the validated end-to-end project. It
fits only QD3J/27, QF4J/36, and QD5B/51 and is the project to use for today's
software workflow verification.

### 2. Realistic four-stage proposal — review required

`EBS_pyaml_MDT_four_stage_REVIEW_REQUIRED.pyloco.json` reproduces the Case C
selection strategy by resolving patterns on the modern lattice itself:

| Group | AT pattern | Matches | Physical types |
|---|---|---:|---|
| QD3 | `QD3[AE]*` | 62 | Quadrupole |
| QF4 | `QF4[ABDE]*` | 126 | Quadrupole |
| QD5 | `QD5[BD]*` | 64 | Quadrupole |
| Normal total | three patterns above | 252 | Quadrupole |
| Skew candidates | `S[HFDIJ]*` | 288 | 192 Sextupole, 96 Multipole |

All 288 skew candidates expose `PolynomA[1]`. There are no duplicate or
unmatched selections. Exact names, zero-based ordinals, types, patterns, and
lattice checksum are recorded in `EBS_modern_selection_provenance.json`.

The stages are Calibration → BPM coupling → Normal optics → Skew and coupling.
Stages 2–4 inherit the preceding fitted state. Stage-specific Case C iteration,
dispersion-weight, coupling-mask, and SVD rank choices are preserved where they
remain meaningful, but no historical magnet ordinal is copied.

This project is intentionally marked **REVIEW REQUIRED**. Today's 5H/5V ORM is
retained unchanged and is not claimed to constrain 252 normal plus 288 skew
candidates. Historical BPM exclusion positions `[27, 231, 286]` are not
applied because their physical identities have not been revalidated against
the modern session.

### Moving to a fuller ORM inventory

1. Acquire a new Measure session with the intended BPM and H/V correctors.
2. Open the four-stage proposal while choosing **Keep current measurements**.
3. Run **Find machine mappings** against the same checksum-bound 4434-element
   lattice. All measured identities must map exactly.
4. Use **Use mapped devices in FIT…**, review the ordered proposal, and confirm.
   This replaces only BPM/H/V FIT selections; it must not sort measurement
   order or alter arrays.
5. Review/reset exclusions by physical device identity. Do not transfer old
   positional exclusions automatically.
6. Review every stage, especially rank selection versus available response
   columns, then Validate. Save As a new project; preserve this proposal.
7. Use the pre-FIT initial-model comparison before considering Run LOCO.

## Launch

From the pyLOCO checkout, with the MDT environment active:

```bash
cd <PYLOCO_CHECKOUT>
PYTHONPATH=. python -m pyLOCO.measure.app
PYTHONPATH=. python -m pyLOCO.gui.app
# After a FIT result exists:
PYTHONPATH=. python -m pyLOCO.correct.app --results mdt_smoke/results/<run-directory>
```

Open `mdt_smoke/configs/EBS_pyaml_MDT_smoke.pyloco.json`, verify the selections,
click Validate, then use **Compare imported measurements with initial model…**.
Only run the short offline fit if the MDT operator explicitly decides to do so.
Selecting a stage edits it; Run LOCO executes the entire validated recipe.
Do not use the four-stage review project for today's smoke run merely because
its structural preflight passes; its measurement sufficiency still requires
scientific review and a more complete corrector inventory.

The reproducible package-build command is:

```bash
PYTHONPATH=. python mdt_smoke/build_modern_ebs_package.py \
  --session <MEASUREMENT_SESSION_MANIFEST> \
  --lattice <MODERN_EBS_BETAMODEL_MAT> \
  --pyaml-config <ESRF_APPROVED_PYAML_YAML> \
  --correct-mapping <ESRF_APPROVED_CORRECT_MAPPING> \
  --output mdt_smoke
```

## Measure → FIT

Before using Measure's optional reference-model ORM or RF-dispersion comparison
with pyAML, load the approved lattice and YAML, select the measured inventories,
and click **Verify device alignment**. The full interactive Measure inventory was
verified at **896/896** exact mappings. The restricted 18 October FIT session
contains **330** measured devices (320 BPMs, 5 H and 5 V correctors). These are
different scopes and must not be conflated. Alignment is bound to lattice/YAML
checksums and ordered inventory; changing any of them requires revalidation.
Built-in pySC profiles retain their profile-bound alignment.

The 18 October session manifest associates one validated schema-1.0 file for each role:
BPM noise, dispersion, and ORM. FIT must retain the measured ordering: 320 BPMs
per plane, X rows then Y rows, and five H-corrector columns then five
V-corrector columns. The saved ORM is an unscaled orbit difference in metres;
the 100 µrad bipolar kick separations are separate metadata. Dispersion inputs
are the saved `measured_eta_x` and `measured_eta_y` vectors in metres.

The dispersion acquisition separation is +200 Hz, while its canonical vector
is `orbit_negative − orbit_positive`; the corresponding signed FIT/model step
is therefore −200 Hz. This is intentional and preserved.

The project pins all measurement SHA-256 hashes and contains 330/330 exact
device-to-lattice mappings in HDF5 order. No identity is inferred from array
position, naming similarity, or dimensions.

## FIT → Correct

After an explicitly authorized restricted offline run, open its result in
Correct. The three fitted quadrupoles resolve through the packaged schema-v2
mapping to B2L controls in m⁻¹. FIT ΔK is m⁻²; Correct calculates
Δ(KL)=ΔK·L in m⁻¹. Do not apply corrections during this smoke test.

Before any live correction:

1. Select the intended EBS pyAML profile and verify its configuration hash.
2. Load and verify the explicit mapping against that same lattice/profile.
3. Connect with the required write authorization.
4. Read current strengths and readbacks; resolve any unavailable calibration or
   current-limit diagnostics.
5. Generate a fresh preview. Apply requires a separate explicit confirmation.
6. Keep the durable transaction journal; restoration/rollback must remain
   available until every applied readback is verified.

Measure requires RF or corrector writes only for automatic dispersion/ORM;
BPM noise is read-only and manual RF dispersion issues no RF writes. FIT is
offline. Correct can load, map, and review offline, but a truthful target
preview needs machine readbacks; Apply requires enabled writes and explicit
confirmation.

## Likely failures

- **Incomplete session:** FIT reports missing ORM, dispersion, or noise. Use the
  packaged manifest and its exact files.
- **MAT variable error:** the EBS file contains multiple variables; `use:
  betamodel` is required.
- **Shape/order mismatch:** expected ORM shape is 640×10 and dispersion/noise
  length is 320 per plane. Never transpose or rescale merely to pass loading.
- **Mapping blocked:** a lattice name/ordinal is missing, ambiguous, duplicated,
  or targets the wrong magnet family. Do not map by array position.
- **Preview unavailable:** no connection/readback, stale mapping/configuration,
  missing magnetic length, or an unsupported correction type. Do not substitute
  model nominal values for live readbacks.
- **Apply blocked:** writes are disabled, confirmation is absent, a readback is
  stale, or a previous transaction still requires restoration.

## Deferred scientific issues

- The restricted two-iteration simulator fit is a software smoke result only;
  no scientific performance claim is made for the three-parameter fit.
- Live pyAML quadrupole readbacks, calibration, current limits, and rollback
  behavior were not exercised in this offline task.
- Real-machine B2L/A2L interpretation, units, sign conventions, calibration and
  limits remain explicit ESRF verification items before any correction write.
- Interactive stage-by-stage pause/continue is deferred until after MDT. A safe
  implementation requires session/recipe fingerprints, contiguous checkpoint
  validation, invalidation of downstream checkpoints, and reopen/resume tests.
