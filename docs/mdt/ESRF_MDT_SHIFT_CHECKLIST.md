# ESRF pyLOCO MDT — one-page shift checklist

**Target:** ☐ Local pySC simulator  ☐ Real ESRF machine (**expert approval required**)

**Date / shift:** __________

**Tested application revision present:** ☐ `6081c0aee5c0efd0c9ccdd4469571bfe5c8b1bc6`

**STOP / NO-GO:** wrong machine, missing mapping, changed file/checksum, unknown unit/sign, nonfinite data, failed restoration, unresolved safety warning, or unapproved writes. **No Apply is authorized by this checklist.**

## A. Before connecting

- [ ] Branch `integration/pyaml-gui-mdt` checked out; documentation HEAD recorded; `6081c0a` is an ancestor; working tree clean.
- [ ] Virtual environment active; Python/pyAT/pyAML/pySC versions and import paths recorded. `pip check` reviewed (known version conflict not silently ignored).
- [ ] Correct modern EBS `betamodel.mat` selected: **4,434** elements, MAT key `betamodel`.
- [ ] Target YAML verified: simulator file for local test or ESRF-approved machine file for hardware.
- [ ] Start with **writes disabled**; confirm target, units, permissions, interlocks with ESRF expert.

## B. Measure → save

- [ ] **Measure → Machine:** Test connection and confirm BPM/H/V corrector and RF capabilities. For an approved simulator ORM, authorize writes normally and connect once; record whether restoration evidence is **independent readback** or **adapter value matched — independent readback unavailable**.
- [ ] **Devices:** Review exact device order.
- [ ] Reference lattice shows MAT variable `betamodel`, **4,434** elements. Click **Verify device alignment**; full simulator inventory expects **896/896** exact mappings. Any lattice/YAML/inventory change requires revalidation.
- [ ] **Measurement:** BPM noise saved to `.h5`.
- [ ] RF dispersion saved to `.h5`; RF returned to nominal by **readback**.
- [ ] ORM saved to `.h5`; corrector kicks/signs recorded and all correctors restored. Evidence level recorded: ☐ independent readback ☐ verified simulator adapter value. Any mismatch/failed read is **NO-GO**.
- [ ] **Review & Save:** All three measurement types shown, **Validate for pyLOCO** passes.
- [ ] Record **measurement session** `*.pyloco-session.json` path: __________________________

**Reminder:** *Save Project* stores Measure settings, not HDF5 data; the session manifest references the HDF5 files.

## C. FIT → review → run

- [ ] Obtain the external ESRF MDT data package; open `configs/EBS_pyaml_MDT_smoke.pyloco.json` with **Use measurements saved in project**. This project and its HDF5/lattice/YAML inputs are not in GitHub.
- [ ] Modern lattice + three HDF5 inputs restored; expect **320 BPM + 5 H + 5 V**, 330/330 mappings for the documented smoke session.
- [ ] Stage fits **QD3J, QF4J, QD5B only**, not the four-stage REVIEW REQUIRED proposal.
- [ ] **Pre-FIT measurement checks:** ORM + X/Y RF dispersion Measured/Model/Residual and RMS reviewed; units and signs clear.
- [ ] **Validate** passes; save working copy; **Run LOCO** only for the approved restricted test.
- [ ] Results χ², 3 fitted parameter identities, ORM and output artifacts reviewed; model corrector ORM is 640×10 and RF response is shown separately.
- [ ] Save run directory / project path: ___________________________________________

## D. Correct — READ ONLY

- [ ] **Open in pyLOCO Correct**; trusted pyAML connected with writes disabled.
- [ ] Correct mapping **3/3**, correct B2L component/identity, current readbacks and lengths.
- [ ] **Review & Validate → Preview machine changes** succeeds; record proposed targets and units.
- [ ] **Read and resolve every warning**. Simulator smoke previously had 1 warning.
- [ ] Confirm **NO Apply**, **NO machine writes**, and no pending transaction.
- [ ] Preview evidence path: ___________________________________________________

## E. Finish and hand over

- [ ] Check actual RF/corrector restoration readbacks; disable writes and disconnect.
- [ ] Stop simulator with `Ctrl-C` (if used). Verify port 13131 is no longer listening.
- [ ] Archive `.h5`, session manifest, project copies, mappings, results/checkpoints, logs, screenshots, warning text and `pip freeze` **outside public Git**.
- [ ] Write down unresolved issues and confirm handover to ESRF expert.

**Full instructions:** `docs/mdt/ESRF_MDT_RUNBOOK.md`

**Versions:** `docs/mdt/INSTALLATION_AND_DEPENDENCIES.md`

**Files:** `mdt_smoke/PACKAGE_MANIFEST.md` and `mdt_smoke/README.md`
