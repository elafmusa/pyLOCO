# ESRF MDT smoke-test checklist

## Measure handoff

- [ ] Session contains BPM noise, dispersion, and ORM.
- [ ] Noise and dispersion contain 320 values per plane in metres.
- [ ] ORM is 640×10, unscaled orbit difference in metres.
- [ ] Inventory is 320 BPM, 5 H correctors, and 5 V correctors.
- [ ] ORM kick is bipolar with total 100 µrad separation.
- [ ] RF states are ±100 Hz; canonical vector is negative−positive.

## FIT

- [ ] Lattice is 4434 elements and MAT variable is `betamodel`.
- [ ] Measurement mapping is 330/330 valid in HDF5 order.
- [ ] Normal quadrupoles are exactly QD3J/27, QF4J/36, QD5B/51.
- [ ] Only `quads` is enabled; no skew, tilt, calibration, coupling, or energy shift.
- [ ] One LM stage and two outer iterations.
- [ ] Validate passes before Run is considered.
- [ ] Pre-FIT initial-model ORM and RF comparison opens read-only.
- [ ] Optional short fit is run only by explicit MDT decision.

## Four-stage proposal — separate scientific review

- [ ] Project title says `REVIEW REQUIRED`.
- [ ] Modern pattern audit reports 252 normal and 288 skew candidates.
- [ ] Stages are Calibration, BPM coupling, Normal optics, Skew and coupling.
- [ ] Stages 2–4 start from the previous stage.
- [ ] Historical positional BPM exclusions were not silently carried over.
- [ ] Current 5H/5V ORM limitation is acknowledged; extended fit is not run.
- [ ] A future full ORM is remapped in exact HDF5 order and saved as a new project.

## Correct — read-only

- [ ] Exact FIT result is selected.
- [ ] Three-entry schema-v2 mapping revalidates as B2L/m⁻¹.
- [ ] Current strengths are read without writes.
- [ ] ΔK, length, and Δ(KL) conversion is reviewed.
- [ ] Preview only; no correction is applied.

## Outcome

- [ ] GUI startup passed.
- [ ] Project validation passed.
- [ ] Pre-FIT comparison passed or limitation recorded.
- [ ] Optional restricted fit passed or was intentionally not run.
- [ ] Correct preview passed or limitation recorded.
- [ ] No machine writes occurred.
