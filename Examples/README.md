# pyLOCO examples

Each main example has three files:

- a `.py` script for a complete reproducible run;
- a `.ipynb` notebook that explains the same workflow step by step;
- a `pyloco_config.yaml` file containing the settings a user is most likely to change.

## Which example should I start with?

For a guided introduction, start with the **Summer Project FODO notebooks**.
They begin with a familiar Accelerator Toolbox lattice and introduce pyLOCO and
Jacobian calculations gradually.

For a complete application script, start with the **EBS single-quadrupole
example**. It shows the basic LOCO workflow without requiring pySC or measured
machine data.

## Available examples

### 1. Summer Project: FODO tutorials

Location: `Summer_project/`

These educational notebooks use the simple FODO ring from the Accelerator
Toolbox Primer:

- `from_fodo_to_pyloco.ipynb` introduces the complete basic workflow: create an
  ORM, assign one quadrupole error, fit it with pyLOCO, apply the correction,
  and verify the ORM and optics.


Run these notebooks in the listed order. The numbered Jacobian notebooks are
student exercises: cells marked `STUDENT TASK — YOUR CODE HERE` are
intentionally incomplete. The analytical support module must remain in the
`FODO_Jacobian/` directory beside them.

### 2. EBS: one quadrupole error

Location: `reconstruct_quadrupoles_errors_examples/reconstruct_one_quad_errors/`

This example:

1. starts with an ideal EBS lattice;
2. assigns a known error to one quadrupole;
3. simulates an orbit-response matrix (ORM);
4. uses pyLOCO to reconstruct the error;
5. applies the fitted correction back to the machine model;
6. checks that the corrected ORM and optics are close to the ideal lattice.

Use this example to learn the basic pyLOCO workflow.

### 3. EBS: multiple quadrupole errors

Location: `reconstruct_quadrupoles_errors_examples/reconstruct_multiple_quad_errors/`

This example repeats the first workflow with several quadrupole errors. It
compares every assigned error with its reconstructed value and shows the ORM,
beta beating, fit convergence, and corrected machine.

Use this example after the single-error tutorial.

### 4. EBS: simulated commissioning with pySC

Location: `EBS_pySC/com_simu_loco_example/`

This example demonstrates how pySC and pyLOCO work together:

1. pySC provides an EBS machine containing realistic magnet, alignment, BPM,
   and corrector errors;
2. pySC measures the ORM, RF-frequency response, and BPM uncertainty;
3. pyLOCO reconstructs the machine errors from those measurements;
4. the opposite fitted quadrupole errors are applied through pySC;
5. pySC measures the corrected machine again;
6. ORM residuals, beta beating, and dispersion are compared before and after
   correction.

This example requires pySC. Set `measurement.source: cached` in its YAML file
to use the preserved pySC measurement instead of simulating a new one.

### 5. PETRA III: measured ORM

Files: `PETRAIII/example_measured_orm.py` and
`PETRAIII/example_measured_orm.ipynb`

This example uses a measured PETRA III ORM. It explains BPM and corrector
selection, removal of bad BPMs, measurement uncertainty, fitted parameters,
and validation of the fitted response.

Use this example to learn how pyLOCO is applied to real machine data.

All PETRA III scripts accept an alternate measurement configuration, for
example `python example_measured_orm.py --config configs/before_correction.yaml`.
Paths in a YAML file are resolved relative to that file, and `output.directory`
lets each machine study keep its results separate without copying the Python
workflow.

To continue a PETRA III fit from a previous stage, enable the `resume` section
in the YAML and point `resume.directory` at the earlier run directory (the one
containing `results/`) or directly at its `results/` directory. The workflow
starts from `ring_pyloco.mat`, restores the final values in `fit_dict.pkl`, and
uses the current YAML fit list and solver settings. Parameters newly added to
the second-stage fit list use their normal initial values; overlapping blocks
use their values from the previous fit. Run the same example command with the
new configuration, and choose a different `output.directory` or `run_name` so
the earlier result is not overwritten.

### 6. PETRA III: coupling fit

Files: `PETRAIII/example_measured_coupling.py` and
`PETRAIII/example_measured_coupling.ipynb`

This example keeps the cross-plane ORM blocks and fits coupling-related
parameters such as skew quadrupoles, quadrupole tilts, and BPM/corrector
coupling. It is a more advanced measured-data workflow.

### 7. PETRA III: constrained fit

Files: `PETRAIII/example_measured_constrained.py` and
`PETRAIII/configs/constrained.yaml`

This thin example uses the same measured-data workflow with dispersion and
`ConstraintConfig`. The YAML controls the fit list, constraint sigmas and
weights, and output directory. The bundled constrained configuration uses the
PETRA III machine-study family groups; setting `data.quadrupole_mode:
individual` preserves the individual-quadrupole path used by the standard fit.
The run saves the constraint settings and quadrupole corrections alongside the
standard residual, convergence, optics, and coupling plots.

Each measured-data run creates `plots/`, `correction/`, and `results/` below
its configured output directory. The correction folder separates the direct
family solution (`delta_q_families.npy` and
`quadrupole_family_corrections.csv`) from its mapping to physical magnets
(`delta_q_expanded.npy` and `quadrupole_corrections_expanded.csv`). The latter
is an expansion of the family fit, not an individual-quadrupole refit.

To add another machine measurement, copy the concise constrained YAML (or the
default `pyloco_config.yaml`) and change `lattice.file`, the paths under
`data`, `bad_bpm_positions`, `rf`, and `output`. Fit parameter lists and solver
settings live under `loco`; family mode is selected with
`data.quadrupole_mode: family` plus `data.quadrupole_family_groups`; priors live
under `constraints`. No Python example needs to be copied.

### 8. PETRA III: comparison with MATLAB LOCO

Files: `PETRAIII/example_matlab_comparison.py` and
`PETRAIII/example_matlab_comparison.ipynb`

This example repeats the standard one-iteration PETRA III fit and compares all
1,500 fitted parameters with the preserved MATLAB LOCO result. The plots show
the full parameter vectors, parameter-by-parameter agreement, and differences
for each physical parameter family.

Use this example to validate consistency between pyLOCO and MATLAB LOCO.

## Dependencies

The AT-only EBS and PETRA III examples require:

- NumPy
- Matplotlib
- PyYAML
- h5py
- Accelerator Toolbox
- pyLOCO

The EBS simulated-commissioning example additionally requires pySC:

```bash
pip install accelerator-commissioning
```

The files under `PETRAIII/data/` are required measurement, lattice, and
selection inputs. They should remain beside the PETRA III examples.

## Running an example

From the repository root, for example:

```bash
python Examples/reconstruct_quadrupoles_errors_examples/reconstruct_one_quad_errors/example_one_quad_error.py
```

For the educational version, open the matching notebook and run its cells from
top to bottom.
