# ESRF pyLOCO MDT runbook

Purpose: a reproducible Measure -> FIT -> Correct shift procedure for the modern 4434-element EBS model. This runbook does not authorize machine writes. Read it together with [INSTALLATION_AND_DEPENDENCIES.md](INSTALLATION_AND_DEPENDENCIES.md) and the package-local `mdt_smoke/README.md`.

Status key:

- **Simulator-tested**: exercised with the modern EBS pyAML/pySC simulator.
- **Code-verified**: implementation and UI path inspected, but not exercised against ESRF hardware.
- **ESRF verification required**: confirm with the machine expert before use.

## 1. Obtain the exact MDT revision

The final public MDT branch has not yet been approved or published. Replace `<PUBLISHED_MDT_BRANCH>` and `<EXPECTED_COMMIT>` only with the identifiers announced for the shift.

```bash
git clone https://github.com/elafmusa/pyLOCO.git pyloco-mdt
cd pyloco-mdt
git fetch origin
git switch --track -c mdt-local origin/<PUBLISHED_MDT_BRANCH>
test "$(git rev-parse HEAD)" = "<EXPECTED_COMMIT>"
git status --short
```

Expected: the commit matches and the status is empty. If the branch is missing, the hash differs, or files are modified, stop and obtain the published identifiers. Do not substitute the development worktree.

For the current unpublished integration snapshot only, the audited commit is `2b6623246eec8bb87e8702b04c9fa0eb08ceda2e` on local branch `integration/pyaml-gui-review`; the worktree contains uncommitted development changes and is not a publication identifier.

## 2. Create the environment

### 2.1 Core Measure/FIT/Correct GUI

```bash
cd <PYLOCO_CHECKOUT>
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[gui]'
python -m pip check
```

Expected: installation completes and `pip check` reports no broken requirements. If it fails, save the full output and do not mix packages from another environment.

### 2.2 Simulator/control-system integrations

The modern EBS simulator currently depends on external pyAML, pySC2pyaml, and EBS bundle inputs. Follow the version decision recorded in `INSTALLATION_AND_DEPENDENCIES.md`; do not combine `.[pysc-demo]` with the demonstrated pySC 1.5.6 environment and assume compatibility.

Verify imports and record versions:

```bash
python -c "import at, h5py, numpy, PySide6, pyLOCO; print(pyLOCO.__file__)"
python -m pip list | grep -E 'accelerator|pySC|PySide6|numpy|h5py|matplotlib|PyYAML|pyLOCO'
```

If the source path is not inside `<PYLOCO_CHECKOUT>`, reactivate the intended environment or reinstall editable pyLOCO.

## 3. Launch applications

Use separate terminals with the same activated environment:

```bash
source <PYLOCO_CHECKOUT>/.venv/bin/activate
pyloco-measure
```

```bash
source <PYLOCO_CHECKOUT>/.venv/bin/activate
pyloco-gui
```

```bash
source <PYLOCO_CHECKOUT>/.venv/bin/activate
pyloco-correct
```

Expected: each main window opens without a traceback. If a console command is missing, run `python -m pip install -e '.[gui]'` from the checkout and recheck the active interpreter with `which python`.

## 4. Start and stop the modern EBS simulator

### 4.1 Preflight

The modern package expects an external bundle containing at least:

- `<EBS_PYSC_BUNDLE>/betamodel.mat`
- `<EBS_PYSC_BUNDLE>/betamodel_conf_pyaml_test.yaml`
- `<EBS_PYSC_BUNDLE>/run_pysc_realistic.py`
- `<EBS_PYSC_BUNDLE>/validate_realistic_ebs.py`
- `<EBS_PYSC_BUNDLE>/test_realistic_ebs.py`

Run offline validation before starting a server:

```bash
cd <EBS_PYSC_BUNDLE>
<SIMULATOR_PYTHON> validate_realistic_ebs.py
<SIMULATOR_PYTHON> -m pytest -q test_realistic_ebs.py
```

Expected: validation and tests pass. If they fail, do not start the server or edit the lattice to suppress the failure; preserve the logs.

### 4.2 Start

```bash
cd <EBS_PYSC_BUNDLE>
<SIMULATOR_PYTHON> run_pysc_realistic.py --serve --port 13131
```

Expected: the script validates the configuration, starts on loopback port 13131, and remains in the foreground. Keep this terminal visible. Only one simulator may own the port.

The repository also provides `python Examples/Demo/start_pysc_server.py --profile ebs`; that is a generic repository profile and must not be confused with the external modern realistic EBS bundle used by the MDT package.

### 4.3 Stop

Use `Ctrl-C` in the simulator terminal, then verify the listener is gone:

```bash
lsof -nP -iTCP:13131 -sTCP:LISTEN
```

Expected: no output. Do not kill an unknown process solely because it uses the expected port; identify it first.

## 5. Connect Measure to the simulator

Status: **Simulator-tested**.

1. Open pyLOCO Measure and select the **Machine** tab.
2. Select the pyAML/live backend and choose `<EBS_PYSC_BUNDLE>/betamodel_conf_pyaml_test.yaml`, or the package-local YAML when following its relocation instructions.
3. Confirm the displayed configuration path, machine/profile, available capabilities, and output directory.
4. For the first connection, disable corrector and RF writes. Select **Test connection**.
5. Confirm BPM, horizontal-corrector, vertical-corrector, RF, and readback inventories. Resolve missing devices before continuing.
6. Enable only the capability required by the next acquisition and only after the simulator identity is confirmed.

Expected: status is connected and inventories match the intended modern EBS profile. If connection fails, confirm the server terminal, port 13131, selected YAML, active Python environment, and pyAML imports. Never switch on writes to diagnose a read failure.

## 6. Switch between simulator and real ESRF configurations

Status: simulator path **Simulator-tested**; real machine **ESRF verification required**.

1. Finish or cancel any acquisition and verify restoration.
2. Save the current Measure project if its application settings are needed.
3. Start a new Measure project or disconnect before selecting another YAML.
4. Select the trusted YAML for the intended target. Do not edit a simulator YAML in place to point at production.
5. Keep all writes disabled and perform a connection/readback inventory first.
6. Confirm the target name, endpoints, selected devices, units, signs, permissions, and output directory with the shift expert.
7. Enable a write capability only for its explicitly approved measurement.

Failure rule: an unknown endpoint, unexpected device count, missing readback, unit ambiguity, or permission ambiguity blocks acquisition. Save diagnostics and consult Kostas.

## 7. Acquire the three measurements

Use the tabs **Machine -> Devices -> Measurement -> Review & Save**. Device ordering is scientific metadata: do not reorder lists merely to make dimensions match.

### 7.0 Verify Measure reference-model alignment (required for pyAML comparisons)

This check is read-only and is separate from acquisition. After connecting pyAML, selecting the authoritative `.mat` reference lattice, and choosing the BPM/H/V-corrector inventories, confirm the displayed MAT variable and element count. Measure detects valid AT lattice variables structurally; the modern EBS file must show `betamodel` and 4434 elements. If several valid lattices exist, select one explicitly. The selected variable is saved in the Measure project and restored on reopen.

1. In **Machine -> Optional reference model**, select **Verify device alignment**.
2. Require `mapped/total` to be complete. The interactive full pyAML inventory check mapped **896/896** devices (320 BPM, 288 H correctors, 288 V correctors). The restricted 18 October FIT session maps **330/330** devices (320 BPM, 5 H, 5 V).
3. Use **Details...** to review measured identity, role, AT element, zero-based ordinal, steering component, and status.
4. Only then enable **Compare measurements with reference model (diagnostic only)**.

Measure reuses FIT's exact AT `Device` mapping and validates the lattice checksum, pyAML configuration checksum, ordered inventory fingerprint, unique ordinals, and `B1L`/`A1L` steering capability. It does not infer ordinals from numeric name fragments. Changing the lattice, YAML, or selected inventory invalidates alignment and disables comparison until it is verified again. Acquisition and automatic HDF5/session saving remain available if alignment is incomplete; only the model comparison is blocked.

Expected: the model ORM and RF-dispersion response use the same ordered BPM/corrector inventory as the measurement. Missing, duplicate, ambiguous, wrong-plane, wrong-component, or stale-provenance entries fail closed. Built-in verified pySC profiles retain their profile-bound device resolver.

### 7.1 BPM noise

1. On **Devices**, select the ordered BPM inventory and review exclusions.
2. On **Measurement**, choose BPM noise, set readings and delay, and inspect the plan.
3. Confirm the status says read-only.
4. Select **Start BPM-noise measurement**.

Expected: orbit samples complete, an HDF5 file is written, and BPM identities/dataset names/units appear in Review & Save. A non-finite sample or identity mismatch is a failure; do not discard entries silently.

### 7.2 RF dispersion

1. Choose dispersion and review the signed RF step, number of readings, and live orbit mode.
2. Authorize RF writes only for the confirmed simulator or approved real-machine procedure.
3. Select **Start automatic dispersion measurement**.
4. Observe the reference, positive/negative RF states as configured, and final restoration.

Expected: the original RF value is restored and independently read back; an HDF5 file records ordered BPM identities, RF step/sign convention, orbit datasets, and units. If restoration is incomplete, stop all further writes and preserve the evidence.

### 7.3 ORM

1. Select the ordered H and V correctors on **Devices**. Inspect the effective kick table and exclusions.
2. Choose ORM, then confirm one-sided or bipolar direction, common/per-device kick values, and units.
3. Authorize corrector writes only for the intended target.
4. Select **Start ORM measurement**.
5. Monitor each corrector state and the final restored-reference display.

Expected: every corrector returns to its recorded reference and an HDF5 file is written with ordered BPM/H/V identities, kick metadata, matrices, and units. If a reference or intermediate array contains non-finite values, treat it as an acquisition failure; do not save or fit a fabricated replacement.

### 7.4 Restoration check

For RF and correctors, require all of the following:

- the acquisition reports restoration complete;
- final readbacks equal the recorded references within the configured tolerance;
- no pending transaction/restoration warning remains;
- the simulator or control-system log contains no failed set/readback;
- the HDF5 metadata describes the actual direction and kick/RF separation.

On any failure: stop acquisition, disable writes, keep logs and HDF5/session files, and escalate. Restarting the GUI is not evidence of restoration.

## 8. Understand what is saved

Measure distinguishes three artifacts:

1. **Individual HDF5 measurement files**: numerical BPM noise, dispersion, or ORM data plus acquisition metadata.
2. **Measurement-session manifest**: links the available HDF5 files and session provenance for FIT.
3. **Measure Project**: application/configuration settings. **Save Project** and **Save Project As** do not replace the HDF5 files or session manifest.

In **Review & Save**, confirm BPM noise, dispersion, and ORM each show present, note any missing role, and copy the displayed session-manifest path. Use **Validate for pyLOCO**. A session is complete for the intended FIT only when all required roles and compatible identity/units metadata pass validation.

FIT reopens a session using **Measurements -> Open Measurement Session...**. Individual HDF5 files may also be imported by role, but the manifest is the preferred portable handoff.

## 9. Open the prepared modern EBS FIT project

Status: **Simulator-tested GUI path** after the project-loading fix.

Project:

```text
<PYLOCO_CHECKOUT>/mdt_smoke/configs/EBS_pyaml_MDT_smoke.pyloco.json
```

1. Keep the complete `mdt_smoke` directory structure together; the project uses portable relative references.
2. Start FIT with `pyloco-gui` and select **Open**.
3. Select the project above.
4. When FIT is empty, choose **Use measurements saved in this project**. Choose **Keep current measurements** only when a deliberately imported, validated session should replace the saved bindings.
5. Confirm the title/project name changes and the GUI restores:
   - `betamodel` from the 4434-element MAT lattice;
   - BPM noise, dispersion, and ORM from 18 October;
   - 320 BPMs, 5 H correctors, and 5 V correctors;
   - 330 validated device mappings;
   - normal quadrupoles at zero-based AT ordinals 27, 36, and 51;
   - one restricted FIT stage.
6. If files are missing, use the explicit locate/relocation workflow; never select similarly shaped unrelated files.
7. Use **Save As** for an adapted copy. Do not overwrite the packaged baseline.

Expected: lattice, measurements, mappings, selections, and FIT recipe are visible. An empty-looking project is a failure, not a successful open. Capture the dialog technical details and console traceback.

The original bug was a string-valued harmonic number during GUI activation plus fragile nested relative measurement references. Current project loading normalizes numeric configuration values, resolves package-relative paths, and rolls back activation transactionally on failure.

## 10. Inspect mappings and FIT selections

1. Open **Machine Components**.
2. Confirm the reference lattice path, MAT variable `betamodel`, 4434 elements, and checksum.
3. In measurement-device mapping, select **Find machine mappings**.
4. Require 330/330 validated mappings. Review measured device, role, AT element, zero-based AT ordinal, and `B1L`/`A1L` component.
5. Remember: finding mappings verifies identities; it does not silently change FIT selections.
6. If adapting another project, use **Use mapped devices in FIT...**, inspect ordered current/proposed selections and ORM shape, then explicitly confirm.
7. Revalidate exclusions whenever the ordered inventory changes.

Expected for the smoke package: model/measured ORM inventory is 640 rows by 10 corrector columns before exclusions. Missing, ambiguous, duplicate, wrong-plane, wrong-component, or stale-checksum mappings block use.

## 11. Compare imported measurements with the initial model

1. In FIT, open **Pre-FIT measurement checks**.
2. Select **Compare imported measurements with initial model...**.
3. This requires comparison readiness—lattice, measurements, mappings, exclusions, kick/RF conventions, and response calculator—but not fitted quadrupole selections or optimizer readiness.
4. Inspect measured, model, and residual ORM views.
5. Inspect separate horizontal and vertical RF-response views and RMS values.
6. If physical dispersion is selected, confirm the displayed signed RF/slip-factor conversion and units. Do not treat RF orbit response as physical dispersion.
7. Expand technical provenance to record lattice/MAT/checksum, input files, calculator, kicks, RF step, devices, and exclusions.

Expected: comparison completes without running a Jacobian or optimizer and without changing imported data. A mapping or unit mismatch must fail closed. The post-FIT Results views remain scientifically distinct.

## 12. Validate and run the restricted smoke FIT

1. Review the single stage, its three normal quadrupoles, weights, solver settings, exclusions, and iteration count.
2. Select **Validate**. Resolve every run-readiness error.
3. Record the validation summary and save an adapted copy with **Save As** if settings changed.
4. Only after approval, select **Run LOCO**. Selecting a table row edits that stage; it does **not** limit execution. **Run LOCO executes the entire validated recipe.**
5. Do not run the four-stage review project with the 5H/5V measurements as though it were sufficiently constrained.

Verified simulator evidence: the restricted three-quadrupole smoke FIT completed two configured iterations and wrote results/checkpoint/provenance artifacts. This is software-path verification, not a physics-quality EBS correction and not authorization for a real-machine correction.

Stage-by-stage pause/review/continue remains deliberately deferred until after MDT. Selecting a stage edits its settings; the published workflow runs the complete validated recipe automatically. Safe future implementation requires recipe/session fingerprints, contiguous checkpoint validation, stale-downstream invalidation, and reopen/continue tests.

## 13. Inspect FIT results and preserve evidence

1. In **Results**, confirm selected stage and iteration labels.
2. Review convergence, ORM residuals, optics, and fitted parameters.
3. Use Results ORM for post-FIT measured/model comparison; do not confuse it with the pre-FIT initial-model comparison.
4. Preserve the run request, backend options, project snapshot, model ORM/RF artifacts, checkpoints, logs, and exported summaries.
5. Never edit historical result files in place.

If a restored model artifact cannot be matched to the current project and measurements by provenance, fail closed rather than selecting a recent result by filename.

## 14. Review the proposed four-stage configuration

Review-only project:

```text
<PYLOCO_CHECKOUT>/mdt_smoke/configs/EBS_pyaml_MDT_four_stage_REVIEW_REQUIRED.pyloco.json
```

It contains:

1. Calibration: BPM gains and H/V corrector calibration.
2. BPM coupling: H/V BPM coupling.
3. Normal optics: modern-lattice QD3/QF4/QD5 selection.
4. Skew and coupling: validated modern-lattice skew candidates plus H/V BPM coupling.

Stages 2-4 inherit the previous fitted state. Review `mdt_smoke/configs/EBS_modern_selection_provenance.json` for patterns, names, types, and zero-based ordinals. Historical 4432-element ordinals were not copied.

The current 5H/5V ORM is not claimed to constrain the extended fit. Acquire a scientifically adequate corrector inventory, rebuild/validate identity mappings in exact measurement order, apply mapped BPM/corrector selections explicitly, review exclusions/weights, and save a new project before considering execution.

## 15. FIT -> Correct read-only preview

Status: the three-quadrupole FIT-to-Correct mapping and read-only preview completed against saved simulator evidence. ESRF live mapping/units remain **ESRF verification required**.

1. Launch `pyloco-correct`.
2. Load the exact FIT result/project bundle selected for review; confirm lattice and run provenance.
3. Select the intended pyAML machine YAML with writes disabled.
4. Use **Find machine mappings** and require validated normal/skew magnet mappings, component, ordinal, checksum, length, and readback evidence.
5. Read current strengths only.
6. Generate **Preview machine changes** and inspect current/target/delta values, units, limits, identities, and journal destination.
7. Stop before **Apply**. The MDT smoke workflow is read-only.

Expected: preview performs GET/readback operations only and reports zero writes. The saved offline preview validates the three explicit B2L mappings, but its remaining operational warning is that live pyAML strengths, readbacks, calibration and limits were not acquired. B2L/A2L real-machine integrated-strength units, signs and calibration remain open questions for Kostas. Unmapped magnets, missing lengths/readbacks, stale mappings, unit uncertainty, or changed machine state block application.

## 16. Evidence preservation

Create a shift evidence directory outside the source tree:

```bash
mkdir -p <MDT_DATA_DIR>/logs <MDT_DATA_DIR>/projects <MDT_DATA_DIR>/results <MDT_DATA_DIR>/journals
git -C <PYLOCO_CHECKOUT> rev-parse HEAD > <MDT_DATA_DIR>/logs/pyloco-commit.txt
python -m pip freeze > <MDT_DATA_DIR>/logs/python-packages.txt
```

Preserve, without rewriting:

- HDF5 measurements and session manifest;
- Measure and FIT project copies;
- mapping bundles and checksum evidence;
- validation output and screenshots;
- run request/backend options/checkpoints/results;
- Correct preview and transaction journal, including failed/restored records;
- simulator and GUI console logs.

Do not commit operational data, personal paths, live endpoints, or journals to a public branch.

## 17. Safe shutdown

1. Do not close Measure while an acquisition or restoration is active.
2. Verify RF and every perturbed corrector are restored by readback.
3. In Correct, leave no pending transaction; for this runbook, no Apply is permitted.
4. Save project copies and record manifest/result paths.
5. Disable all write authorizations and disconnect the GUI.
6. Stop the simulator with `Ctrl-C` and verify port 13131 is closed.
7. Archive logs and evidence; record all unresolved warnings in the shift log.

## 18. Troubleshooting map

| Symptom | First check | Safe action | Classification |
|---|---|---|---|
| pyAML connection fails | YAML path, active environment, simulator terminal/port, adapter traceback | Disable writes; retest connection/read inventory | control-system/configuration |
| Wrong lattice or MAT variable | displayed path, `betamodel`, element count 4434, checksum | reopen the exact packaged lattice; do not accept another candidate by name alone | configuration |
| FIT project appears empty | project-open dialog details and console traceback | cancel/rollback; verify package-relative files and current code revision | GUI/packaging |
| BPM/corrector mappings unresolved | HDF5 identities, mapping checksum, AT ordinal/component | generate/review mapping; never infer by position or dimensions | identity/data |
| ORM shape or units mismatch | ordered devices, exclusions, kick convention, HDF5 metadata | correct configuration or reacquire; never truncate/reorder arrays | data/science |
| Dispersion mismatch | RF step/sign, RF response versus physical dispersion, slip-factor conversion | compare in RF-response units first; record provenance | data/science |
| Pre-FIT comparison requests quadrupoles | comparison-readiness versus run-readiness path | update to the fixed build; comparison must not require fit groups | GUI/validation |
| Run disabled | full validation details, stages, fitted element groups, optimizer settings | repair configuration and revalidate | configuration |
| Stage row selection seems ignored | selected row edits configuration only | validate and remember Run executes the full recipe | usability |
| Correct preview blocked | mapping, component, checksum, length/readback, units | remain read-only; resolve authoritative evidence | safety |
| Restoration incomplete | acquisition/transaction log and independent readback | disable writes, preserve journal, escalate immediately | safety/control system |

## 19. Go/no-go criteria

Proceed only when the short checklist passes. Real-machine writes additionally require explicit ESRF authorization, verified unit/sign conventions, known limits/interlocks, and a rehearsed restoration procedure. Simulator success alone is not authorization for hardware use.
