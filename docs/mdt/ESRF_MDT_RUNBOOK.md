# pyLOCO at ESRF — step-by-step MDT runbook

**Scope:** modern EBS (4,434 AT elements), from **Measure → FIT → Correct**. The documented smoke test was exercised with the **pyAML/pySC simulator**. **Real ESRF machine operation has not been verified by these tests. Do not apply corrections under this runbook.**

Use the [one-page shift checklist](ESRF_MDT_SHIFT_CHECKLIST.md) during the shift and [installation notes](INSTALLATION_AND_DEPENDENCIES.md) for versions. The package also contains `mdt_smoke/README.md`, `CHECKLIST.md`, and `PACKAGE_MANIFEST.md`.

## 0. Find the files you need

Set these two locations first. Paths starting with `<...>` are placeholders, **not literal Terminal commands**.

| Name | Where to find it | Used for |
|---|---|---|
| pyLOCO checkout | `<PYLOCO_CHECKOUT>` | Source, GUIs and publishable MDT instructions |
| MDT package guide | `<PYLOCO_CHECKOUT>/mdt_smoke/` | Builder, checklist, manifest and portable selection audit |
| **Smoke FIT project** | External ESRF package: `configs/EBS_pyaml_MDT_smoke.pyloco.json` | Tested *three-quadrupole* software smoke setup; not stored in GitHub |
| **Four-stage REVIEW project** | External ESRF package: `configs/EBS_pyaml_MDT_four_stage_REVIEW_REQUIRED.pyloco.json` | Review only; **do not run with the limited 5H/5V ORM** |
| Selection audit | `mdt_smoke/configs/EBS_modern_selection_provenance.json` | Names, AT ordinals, selection patterns |
| Simulator/MAT lattice | `<EBS_PYSC_BUNDLE>/betamodel.mat` | External modern EBS model, MAT variable `betamodel`; not stored in GitHub |
| Simulator pyAML YAML | `<EBS_PYSC_BUNDLE>/betamodel_conf_pyaml_test.yaml` | External simulator configuration; not stored in GitHub |
| Simulator launcher | `<EBS_PYSC_BUNDLE>/run_pysc_realistic.py` | pySC simulator on port 13131 |
| Saved session manifest | **Open its actual path from Measure → Review & Save**; local session example: `measurements/measurement-session.pyloco-session.json` | References BPM-noise, RF-dispersion and ORM HDF5 files |
| October 18 sample ORM | External ESRF package: `measurements/pyloco-orm-20261018-170011.h5` | ORM only, **not the complete session and not stored in GitHub** |
| FIT outputs | Results → Files / selected run's results directory | `run_request.json`, summary, fitted model, checkpoints |
| Correct mapping | External ESRF package if approved; otherwise Correct → Machine / Mapping | Verified control identities for preview; not stored in GitHub |

**Before relying on a packaged path**, read `mdt_smoke/PACKAGE_MANIFEST.md` and confirm that the file exists after checkout. Some simulator/lattice files are external and must be supplied separately. Do not publish machine-specific YAML, access information, operational journals, or nonredistributable lattice data.

## 1. Install and identify the correct version

The tested application-code revision is `6081c0aee5c0efd0c9ccdd4469571bfe5c8b1bc6` on `integration/pyaml-gui-mdt`. A later documentation-only commit may become branch HEAD; that changes the documentation revision, not the tested application-code revision. Record both hashes.

```bash
git clone https://github.com/elafmusa/pyLOCO.git pyloco-mdt
cd pyloco-mdt
git fetch origin
git switch --track -c mdt-local origin/integration/pyaml-gui-mdt
git log --oneline -3
# Confirm the history contains the tested application revision:
git merge-base --is-ancestor 6081c0aee5c0efd0c9ccdd4469571bfe5c8b1bc6 HEAD
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[gui]'
python -m pip check
```

**Pass:** branch HEAD and tested application revision are recorded, the checkout is clean, and GUI dependencies install. **If not:** stop and save the error log. The demonstrated pyAML/pySC environment has an unresolved `pip check` version conflict; do not claim a clean compatible integration install until resolved with Kostas.

To see versions and paths:

```bash
which python
python -c 'import pyLOCO,at; print(pyLOCO.__file__); print(at.__file__)'
python -m pip list | grep -E 'accelerator|pySC|PySide6|numpy|h5py|pyLOCO'
```

## 2. Simulator preparation (skip for real-machine studies)

**Simulator-only.** The external bundle must contain `betamodel.mat`, `betamodel_conf_pyaml_test.yaml`, and `run_pysc_realistic.py`; `validate_realistic_ebs.py` and `test_realistic_ebs.py` are the documented preflight tests.

```bash
cd "<EBS_PYSC_BUNDLE>"
<SIMULATOR_PYTHON> validate_realistic_ebs.py
<SIMULATOR_PYTHON> -m pytest -q test_realistic_ebs.py
<SIMULATOR_PYTHON> run_pysc_realistic.py --serve --port 13131
```

Keep its Terminal running. **Pass:** simulator starts on loopback port 13131. **If not:** check the YAML, imports, terminal output and whether the port is already occupied. Do not terminate an unidentified process.

## 3. Start the GUIs

Open separate Terminal windows using the correct environment:

```bash
source "<PYLOCO_CHECKOUT>/.venv/bin/activate"
pyloco-measure
# separate terminal: pyloco-gui
# separate terminal: pyloco-correct
```

**Pass:** all windows open without tracebacks. **If not:** verify `which python`, the selected checkout, and the installation output.

## 4. Connect Measure and verify the target

1. **Measure → Machine:** choose **pyAML** and the trusted YAML (simulator YAML for the local test; ESRF-approved YAML for hardware).
2. Start with **writes disabled**. Press **Test connection** and review the target identity, inventory and read capabilities.
3. For an approved simulator ORM test, enable corrector writes, accept the normal write warning and press **Test connection once**. A verified simulator must show its runtime identity/configuration fingerprint and either **independent readback** or **Adapter value matched — independent readback unavailable**. No separate simulator checkbox or second connection is required. Unknown targets remain blocked when independent restoration evidence is unavailable.
4. Check target/mode, BPM/H/V corrector inventories, RF readback, backend permissions and output directory.
4. **Only for approved acquisition**, enable the minimum necessary corrector or RF write capability after confirming the target.

For Measure's model comparison, select the reference lattice. MAT files are inspected for valid AT lattices: the modern EBS file selects `betamodel` and must show **4,434 elements**. If more than one valid lattice exists, choose it explicitly. Then select the BPM/H/V inventory and click **Verify device alignment**. The shared FIT/Measure mapper requires exact AT `Device` identities, unique ordinals and the correct `B1L`/`A1L` steering components; it preserves measurement order and fails closed. The full simulator inventory was verified at **896/896**. Changing the lattice, MAT variable, YAML or device inventory invalidates alignment.

**Pass:** correct identity and expected readbacks. **If not:** do not enable writes; check connection/YAML and consult the ESRF expert. Simulator success does not authorize real-machine writes.

## 5. Measure and save

Navigate **Machine → Devices → Measurement → Review & Save**. **Keep the BPM and corrector order**; it determines ORM row/column identity.

| Measurement | What to do | Evidence of success |
|---|---|---|
| **BPM noise** | Select the BPM inventory; choose BPM noise; configure readings and delay; start read-only acquisition | Samples finite; HDF5 saved with BPM names and units |
| **RF dispersion** | Review signed RF separation and orbit readings; authorize RF writes only for approved target; start automatic dispersion | RF restored to original reference and verified by readback; HDF5 records sign and RF state |
| **ORM** | Review ordered H/V correctors, kick size and bipolar/one-sided mode; authorize corrector writes only as approved; start ORM | All correctors restored; evidence explicitly states either independent readback or verified simulator adapter-value matching; ORM HDF5 saved with names and kick metadata |

**Stop on failed restoration.** Disable writes; retain logs and the session. Restarting the GUI is not proof of restoration.

For the local pySC2pyaml simulator, a fresh public `magnet.strength.get()` is accepted only after the runtime and configuration SHA-256 are verified. It is simulator adapter evidence, not an independent hardware readback. Alternative simulators, including Simone's ESRF simulator, require an authoritative public runtime attestation containing a stable instance identity, configuration fingerprint and confirmation that `strength.get()` is a fresh simulator-state read. Editable YAML labels, hostnames and ports are not identity proof. Real-machine ORM requires the ESRF-approved readback and operating procedure.

### Three different saved artifacts

- **Individual measurements:** separate `.h5` files with numerical data and acquisition metadata.
- **Measurement session:** `*.pyloco-session.json` manifest that points to the measurement files. The manifest **does not embed their arrays**.
- **Measure project:** `*.pyloco-measure.json` with application/backend setup. **Save Project** is *not* Save Measurement Session.

In **Review & Save**, check that BPM noise, dispersion and ORM are listed; click **Validate for pyLOCO** and record the exact manifest path. Use **Open session in pyLOCO Fit** to hand off, or load in FIT through **Measurements → Open Measurement Session…**. If a role is missing, the session is partial; never substitute invented arrays.

## 6. Open the complete FIT smoke project

1. Start **FIT** (`pyloco-gui`), click **Open**.
2. Choose `<ESRF_MDT_PACKAGE>/configs/EBS_pyaml_MDT_smoke.pyloco.json`. This local-only project is not included in GitHub; obtain the package through the approved ESRF transfer route.
3. If FIT is empty, choose **Use measurements saved in this project**. **Keep current measurements** is only appropriate when a deliberately imported session should replace saved bindings.
4. In **Machine Components**, verify modern `betamodel` lattice, **4,434 elements**.
5. In **Measurements**, verify BPM noise, dispersion and ORM from the intended session.
6. In **Machine Components → Machine Lattice**, click **Find machine mappings**. For the documented smoke inventory expect **320 BPM + 5H + 5V = 330/330** mapped. If adapting new data, use **Use mapped devices in FIT…**, inspect the ordered proposal, and confirm.
7. In **FIT**, ensure the restricted single stage fits only normal quadrupoles `QD3J`, `QF4J`, `QD5B` (modern zero-based AT ordinals **27, 36, 51**, identity-confirmed). Two LM iterations were used for the documented smoke.
8. Use **Save As** for your working copy. Do not overwrite packaged or historical baseline files.

**Pass:** project title changes, lattice/measured files/selections appear, mappings validate. **If empty:** capture the open-dialog message and console traceback; never silently build an unrelated configuration. The previous failure involved string-valued `HarmNumber` and project restoration; use the fixed build.

## 7. Compare measurements with the initial model

In **FIT → Pre-FIT measurement checks**, choose **Compare imported measurements with initial model…**.

- This is **read-only**, before optimizer execution.
- Inspect **ORM comparison**: Measured, Model and Residual (**Measured − Model**), including block RMS.
- Inspect **RF dispersion**: horizontal and vertical independently, measured/model/residual RMS, **RF orbit response [m]** versus optional **physical dispersion D [m]**.
- Inspect **Inputs and provenance**: active lattice, MAT variable, measurement files, calculator, device order, kick/RF conventions and exclusions.

**Pass:** dimensions/identities compatible, plots and RMS render, no control writes. **If comparison complains about missing fitted quadrupoles:** that is a version/validation problem; comparison readiness should not require selected fit magnets.

## 8. Validate and run the restricted FIT

1. In **FIT**, review parameter groups, optimizer/weights, corrector counts, response calculator and exclusions.
2. Click **Validate**. Resolve errors before proceeding.
3. Click **Run LOCO**; this executes **every configured stage automatically**, not just the highlighted table row.
4. In **Results**, inspect χ² history, **Parameters**, ORM and Optics, run request and output paths. Preserve the saved results.

**Documented simulator smoke observation (18 Oct 2026):** two iterations completed in about 19 seconds; reduced χ² decreased approximately `1.653e9 → 1.628e9` (~1.5%). This verifies the restricted software execution path, **not satisfactory EBS optics recovery**. At the tested application revision, Results separates the metadata-verified appended RF-response column before comparing the measured and model corrector ORMs; RF response remains a separate dispersion quantity.

## 9. Correct: mapping and **preview only**

1. From FIT **Results**, click **Open in pyLOCO Correct…**, or load the exact result in Correct.
2. In **Correct → Machine / Mapping**, connect to the trusted pyAML target **with writes disabled**, then find and validate mappings. For the smoke test verify **3/3** exact QD3J/QF4J/QD5B mappings and their `B2L` components. Operational device endpoints remain in the ESRF-controlled mapping, not this public guide.
3. In **Correction Plan**, check included devices, correction fraction, current/fitted values, magnetic length and ΔK/Δ(KL) representation.
4. In **Review & Validate**, click **Preview machine changes**. Inspect current strength, requested change, proposed target, readback, units and warnings.
5. **Stop here. Do not click Apply.** Export/store preview evidence if needed.

**Documented simulator observation:** 3/3 mapped; read-only preview verified with no writes; 0 current-limit violations, **1 warning requiring inspection**; the screenshot showed **100% correction fraction**, which is a preview example, *not an approved hardware application value*. **Real B2L/A2L units/signs remain ESRF verification required.**

## 10. Review the full four-stage proposal — do not run with smoke data

Open `<ESRF_MDT_PACKAGE>/configs/EBS_pyaml_MDT_four_stage_REVIEW_REQUIRED.pyloco.json` **for review only**. It is an external local package input, not a GitHub file. It proposes:

1. Calibration: BPM gains and corrector calibration H/V.
2. BPM coupling: BPM coupling H/V.
3. Normal optics: **252** modern-lattice normal quads selected by historical family patterns.
4. Skew/coupling: **288** modern skew candidates and BPM coupling.

Stages 2–4 inherit prior checkpoints; **Run LOCO runs the complete validated recipe automatically.** A safe pause/review/continue mode is **deferred until after MDT**. Today's 5H/5V ORM is **not** established as sufficient for the full fit. Acquire an adequate new ORM, verify new device mappings/exclusions and operator-review conditions first.

## 11. Keep evidence and shut down safely

Create an operational evidence directory **outside the source checkout** and retain: measurement HDF5, session manifest, project copies, lattice/mapping provenance, run request, results/checkpoints, correction preview, warnings, screenshots, console logs and package versions. **Do not commit operational data or transaction journals to a public branch.**

At the end: verify RF/corrector readbacks restored; confirm no pending Correct transaction; disable writes; disconnect GUIs; `Ctrl-C` to stop the simulator; verify `lsof -nP -iTCP:13131 -sTCP:LISTEN` has no listener.

## Quick troubleshooting: where to look first

| Problem | Check this | Safe response |
|---|---|---|
| FIT opens as **Untitled** | Project open dialog/log, external `configs/EBS_pyaml_MDT_smoke.pyloco.json`, MAT/measurement paths | Use the tested build; capture traceback; don't hand-rebuild silently |
| Missing measurement files | Measure → Review & Save manifest, FIT → Measurements, package manifest | Rebind exact files, never same-shaped substitutes |
| 42/330 mapped | Historical **4432** lattice used with modern measurement | Select modern **4434** lattice and repeat mapping |
| Pre-FIT blocked by quad selection | Comparison readiness vs FIT execution readiness | Use fixed build; comparison should not require fitted quads |
| ORM 640×10 vs 640×11 | Wrong build or incomplete result provenance | Use the tested revision; verify saved metadata and exact stage/iteration; never blindly truncate |
| Wrong dispersion | RF sign/step, physical D conversion, X/Y separation | Compare like quantities with stated units and masks |
| Correct unmapped | pyAML connection, exact B2L controls, lattice provenance | Keep writes disabled and resolve mappings |
| Warning/unknown unit/restoration | Correct warning details, readbacks, journal | **NO-GO** for machine writes; consult ESRF expert |

## Publication and hardware boundary

Record the current documentation HEAD and confirm that it contains tested application revision `6081c0a`. pyAML/pySC version conflicts and the real-machine YAML, control-unit conventions, limits, interlocks and authorization must be agreed with Kostas. **This runbook never authorizes live Apply.**
