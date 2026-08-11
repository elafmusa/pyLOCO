# pyLOCO examples

Each main example has three files:

- a `.py` script for a complete reproducible run;
- a `.ipynb` notebook that explains the same workflow step by step;
- a `pyloco_config.yaml` file containing the settings a user is most likely to change.

## Which example should I start with?

Start with the **EBS single-quadrupole example**. It is the smallest example and
shows the basic idea of LOCO without requiring pySC or measured machine data.

## Available examples

### 1. EBS: one quadrupole error

Location: `reconstruct_quadrupoles_errors_examples/reconstruct_one_quad_errors/`

This example:

1. starts with an ideal EBS lattice;
2. assigns a known error to one quadrupole;
3. simulates an orbit-response matrix (ORM);
4. uses pyLOCO to reconstruct the error;
5. applies the fitted correction back to the machine model;
6. checks that the corrected ORM and optics are close to the ideal lattice.

Use this example to learn the basic pyLOCO workflow.

### 2. EBS: multiple quadrupole errors

Location: `reconstruct_quadrupoles_errors_examples/reconstruct_multiple_quad_errors/`

This example repeats the first workflow with several quadrupole errors. It
compares every assigned error with its reconstructed value and shows the ORM,
beta beating, fit convergence, and corrected machine.

Use this example after the single-error tutorial.

### 3. EBS: simulated commissioning with pySC

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

### 4. PETRA III: measured ORM

Files: `PETRAIII/example_measured_orm.py` and
`PETRAIII/example_measured_orm.ipynb`

This example uses a measured PETRA III ORM. It explains BPM and corrector
selection, removal of bad BPMs, measurement uncertainty, fitted parameters,
and validation of the fitted response.

Use this example to learn how pyLOCO is applied to real machine data.

### 5. PETRA III: coupling fit

Files: `PETRAIII/example_measured_coupling.py` and
`PETRAIII/example_measured_coupling.ipynb`

This example keeps the cross-plane ORM blocks and fits coupling-related
parameters such as skew quadrupoles, quadrupole tilts, and BPM/corrector
coupling. It is a more advanced measured-data workflow.

### 6. PETRA III: comparison with MATLAB LOCO

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
