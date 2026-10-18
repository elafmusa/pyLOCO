# ESRF pyLOCO MDT shift checklist

Print this page and mark each item. Detailed recovery steps are in `ESRF_MDT_RUNBOOK.md`.

## Environment and target

- [ ] Published branch and expected commit recorded and matched.
- [ ] Correct virtual environment active; import paths and package versions saved.
- [ ] `pip check` result reviewed; any known exception explicitly accepted by the team.
- [ ] Target clearly identified: modern EBS simulator or real ESRF machine.
- [ ] Trusted lattice/YAML selected; automatic MAT detection shows 4434 elements and variable `betamodel` (or the intended variable was selected explicitly).
- [ ] Real-machine units, signs, permissions, limits, and interlocks confirmed with Kostas.

## Measure

- [ ] Connection and readback inventory pass with writes disabled.
- [ ] Ordered BPM/H/V-corrector identities reviewed.
- [ ] For full pyAML Measure comparison, **Verify device alignment** reports 896/896; for the restricted FIT inventory it reports 330/330. Details show exact AT Device identities and B1L/A1L components.
- [ ] Changing the lattice, YAML, or device selection is understood to invalidate alignment; comparison stays disabled until reverified.
- [ ] BPM noise saved and metadata valid.
- [ ] RF dispersion saved; signed RF step recorded; RF restored and read back.
- [ ] ORM saved; kick direction/value recorded; all correctors restored and read back.
- [ ] Review & Save shows BPM noise, dispersion, and ORM present.
- [ ] Session manifest path recorded and **Validate for pyLOCO** passes.
- [ ] Remember: Save Project saves Measure configuration, not the measurement session.

## FIT

- [ ] `EBS_pyaml_MDT_smoke.pyloco.json` opens with lattice, all three measurements, mappings, selections, and one stage.
- [ ] 320 BPM + 5 H + 5 V identities map 330/330 to the active lattice.
- [ ] Smoke selections include only approved quadrupoles 27, 36, and 51.
- [ ] Pre-FIT measured versus initial-model ORM and X/Y RF-response comparisons reviewed.
- [ ] RMS, units, sign conventions, exclusions, kicks, RF step, calculator, and provenance recorded.
- [ ] Full FIT **Validate** passes before Run.
- [ ] Selecting a stage is understood to edit it; **Run LOCO runs the entire validated recipe**.
- [ ] Restricted smoke result, run request, checkpoints, logs, and project snapshot preserved.
- [ ] Saved restricted smoke evidence shows both configured iterations completed; no new fit is required for publication testing.
- [ ] Four-stage review project is not run with the restricted 5H/5V ORM as though fully constrained.

## Correct — read-only MDT scope

- [ ] Exact FIT result and lattice provenance confirmed.
- [ ] pyAML connection established with writes disabled.
- [ ] Magnet mappings, component, ordinal, checksum, length, and readbacks validate.
- [ ] Correction preview reviewed in explicit physical/control units.
- [ ] Offline-preview warning recorded: live strengths, readbacks, calibration and limits were not verified; B2L/A2L unit/sign calibration remains an ESRF decision.
- [ ] No Apply action performed; zero writes confirmed.
- [ ] Preview/journal evidence preserved.

## Shutdown and handover

- [ ] No acquisition or restoration remains active.
- [ ] RF and corrector readbacks equal recorded references within tolerance.
- [ ] Write authorizations disabled; GUIs disconnected.
- [ ] Simulator stopped; port 13131 no longer listening.
- [ ] HDF5 files, session manifest, projects, mappings, results, checkpoints, journals, screenshots, package list, and logs archived.
- [ ] Outstanding warnings and ESRF-specific assumptions entered in the shift log.

**NO-GO:** unknown target, missing identity mapping, non-finite data, unit/sign ambiguity, failed restoration, stale checksum, incomplete provenance, or unapproved write permission.
