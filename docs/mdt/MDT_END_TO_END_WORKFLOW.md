# ESRF MDT end-to-end workflow

This guide applies to the publication candidate based on published
`integration/pyaml-gui-mdt` revision
`86e412bb0531060189f83c483edb999b49309b93`. The candidate's tested
application-code revision is `462b2d11f077464ac602c23d3016dbbd8b80606a`;
the later documentation-only commit changes repository HEAD without changing
that tested application code. It describes the verified Measure → FIT →
Correct workflow for a simulator study. It does not authorize real-machine
writes.

## Before the shift

Obtain the ESRF-approved local inputs listed in
[`MDT_REQUIRED_FILES.md`](MDT_REQUIRED_FILES.md). The GitHub branch contains the
package builder and public selection provenance, but it intentionally does not
contain the modern lattice, pyAML YAML, October 18 HDF5 files, validated local
mappings, prepared FIT project, or generated FIT results.

Build a local package from the checkout root:

```bash
python mdt_smoke/build_modern_ebs_package.py \
  --session <SESSION_MANIFEST> \
  --lattice <MODERN_4434_ELEMENT_BETAMODEL_MAT> \
  --pyaml-config <APPROVED_SIMULATOR_PYAML> \
  --correct-mapping <APPROVED_THREE_QUAD_CORRECT_MAPPING> \
  --output <LOCAL_MDT_PACKAGE>
```

Expected result: `VALIDATION.json` reports no project/preflight errors,
`330/330` mapped devices, ORM shape `640 × 10`, MAT variable `betamodel`, and
normal-quadrupole ordinals `27, 36, 51` (`QD3J`, `QF4J`, `QD5B`).

The generated project is the portable starter. Do not substitute a blank
published project: exact BPM/corrector order, mapping evidence and compatibility
can only be established after the external scientific inputs are supplied.

## 1. Measure

Launch:

```bash
pyloco-measure
```

1. On **Machine**, select **pyAML**, load Simone's approved simulator YAML,
   choose the intended runtime mode, and enable only the writes required for
   the planned measurement.
2. Click **Test Connection** once. Confirm the displayed runtime identity and
   capabilities. A positively identified simulator may report
   **Adapter value matched — independent readback unavailable**; this is not an
   independent hardware readback.
3. Load the 4434-element `.mat` reference model and select `betamodel`. Click
   **Verify device alignment**. Do not compare with the model unless the exact
   ordered inventory is validated.
4. On **Devices**, either create the BPM/H/V selections or use **Device
   selection configuration → Load selection…**. Review the identities and
   counts after loading. This file is a Measure selection file, not a FIT
   mapping.
5. Acquire BPM noise, RF dispersion, then ORM. For ORM, confirm the kick
   direction, kick separation and the separate corrector settling delay.
6. After every write-capable acquisition, review restoration status. Stop if
   restoration is mismatched, unreadable or unresolved. For a verified
   simulator, `strength.get()` evidence is reported as adapter-value evidence,
   never as independent readback.
7. On **Review & Save**, confirm all three roles are present and that the
   session is complete for FIT. Record the three HDF5 paths and the session
   manifest path. Individual measurements and the manifest are saved by the
   acquisition workflow. **Save Project** saves Measure configuration, not the
   measurement session.
8. Click **Validate for pyLOCO**. Preserve the HDF5 files and manifest together.

The **Open session in pyLOCO Fit** action imports measurements only. It does not
load the FIT lattice, machine selections, mapping, quadrupoles, exclusions,
weights, solver options, or fitting recipe.

## 2. FIT — loading order

The project-first order below is recommended because it makes the retained FIT
configuration most obvious to the operator:

1. Launch FIT:

   ```bash
   pyloco-gui
   ```

2. Click **Open** and select
   `<LOCAL_MDT_PACKAGE>/configs/EBS_pyaml_MDT_smoke.pyloco.json`.
3. Confirm the 4434-element lattice, `betamodel`, 320 BPMs, 5 H correctors,
   5 V correctors, three normal quadrupoles, and one restricted fitting stage.
4. On **Measurements**, click **Open Measurement Session…** and select the
   freshly saved manifest. This replaces only the measurement bindings and
   measurement-derived kick/RF metadata. It preserves the project lattice,
   machine ordinals, fitting groups, exclusions, weights, solver options and
   recipe. It deliberately marks compatibility **validation required**.
5. On **Machine Components**, click **Find machine mappings**. This verifies
   identity only and does not change FIT selections. Require every BPM and
   corrector to be uniquely mapped against the selected lattice.
6. If the fresh inventory differs from the starter project, click **Use mapped
   devices in FIT…**, review the exact HDF5-order proposal and exclusions, and
   confirm. Never sort ordinals or substitute positional matches.
7. Click **Validate**. Run must remain disabled until compatibility validation
   succeeds.
8. Under **Pre-FIT measurement checks**, click **Compare imported measurements
   with initial model…**. Review measured/model/residual ORM and separate X/Y
   RF response. This does not run the optimizer.
9. Save As a new project beside the local package. Do not overwrite the public
   template or the original measurement session.
10. Click **Run LOCO** only for the restricted smoke stage. Selecting a stage
    edits it; **Run LOCO executes the complete validated recipe**.

The restricted example fits only `QD3J`, `QF4J` and `QD5B` for two iterations.
It is a software demonstration with a 5H/5V ORM, not a complete optics study.

### Session-first order

The alternative order is also supported: import the measurement session into
an empty FIT window, open the starter project, and choose **Keep current
measurements**. FIT retains the active arrays and ordered HDF5 identities while
loading the project's lattice, machine selections, stages, weights and solver
configuration. The project's mapping is kept only after exact revalidation
against the retained inventory, lattice checksum and available machine-
configuration fingerprint. A changed measurement binding requires **Validate**
before Run. Missing, reordered, stale or incompatible mapping evidence is
cleared; use **Find machine mappings** and validate again. FIT never substitutes
the empty active-window mapping or maps devices by position.

## 3. Save and review FIT results

1. In **Results**, select the intended stage and iteration.
2. Check **ORM**: measured, initial model, fitted model, residual before and
   residual after must be available. RF response is handled separately; the
   combined model artifact may contain 11 columns, but ORM comparison must use
   the ten metadata-identified corrector columns.
3. Check **Optics/Dispersion** and the run summary.
4. Keep the whole results directory. Do not extract only `fit_results.npy`;
   Correct and Results use the request, lattice provenance, parameter metadata
   and other saved artifacts.

## 4. Correct — read-only demonstration

See [`CORRECT_DEMONSTRATION.md`](CORRECT_DEMONSTRATION.md) for the complete
procedure. The short path is:

1. Launch with the results directory:

   ```bash
   pyloco-correct --results <FIT_RESULTS_DIRECTORY> --iteration 2
   ```

2. Select the pyAML backend and connect to the approved simulator with writes
   disabled for this demonstration.
3. On **Machine / Mapping**, load the explicit, checksum-bound mapping using
   **Load completed mapping…**. Require all three entries to validate.
4. Click **Read current integrated strengths**. This is read-only and does not
   prove physical calibration.
5. Click **Preview machine changes**. Confirm the identity, current KL, target
   KL, magnetic length and `Δ(KL)=ΔK·L` values.
6. Do not click **Apply…** during this smoke test.

## Stop conditions

Stop and preserve diagnostics if any of these occurs:

- a missing HDF5 role or changed SHA-256;
- a lattice other than 4434 elements or MAT variable other than `betamodel`;
- an unresolved, ambiguous, duplicate, wrong-plane or wrong-component mapping;
- measurement ORM shape other than `640 × 10` for the restricted example;
- Run enabled before post-import validation;
- a Correct result/mapping lattice checksum mismatch;
- an unavailable or incompatible integrated-strength unit;
- any unresolved restoration or transaction journal.
