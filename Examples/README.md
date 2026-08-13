# pyLOCO examples

This directory contains learning examples for students and application examples
for simulated and measured accelerator machines.

## Recommended starting point

If you are new to pyLOCO, start with the single-quadrupole example. It introduces
the complete LOCO workflow with one known error:

```bash
python3 Examples/reconstruct_quadrupoles_errors_examples/reconstruct_one_quad_errors/example_one_quad_error.py
```

The example:

1. loads an ideal EBS lattice;
2. adds one quadrupole error;
3. simulates an orbit-response matrix;
4. reconstructs the error with pyLOCO;
5. applies the correction and checks the result.

Its settings are in the adjacent `pyloco_config.yaml` file.

## Real PETRA III measurement

For a real measured-machine application, use the PETRA III coupling example:

```bash
python3 Examples/PETRAIII/example_measured_coupling.py
```

The user settings are in:

```text
Examples/PETRAIII/pyloco_config_coupling.yaml
```

This advanced example fits the normal and cross-plane response using normal and
skew quadrupoles, quadrupole tilts, BPM calibration/coupling, and corrector
calibration/coupling.

## Other maintained examples

### Multiple quadrupole errors

After the single-error example, run:

```bash
python3 Examples/reconstruct_quadrupoles_errors_examples/reconstruct_multiple_quad_errors/example_multiple_quad_errors.py
```

This reconstructs several simultaneous EBS quadrupole errors.

### Simulated commissioning with pySC

`Examples/EBS_pySC/com_simu_loco_example/` combines pySC measurements with a
pyLOCO reconstruction. It requires the optional `accelerator-commissioning`
package.

### Reusable measured-machine runner

`Examples/measured_machine/` provides one runner for different accelerators.
Machine selection, data paths, fitted parameters, constraints, and solver
settings are controlled from YAML. See its [README](measured_machine/README.md).

### PETRA III MATLAB comparison

`Examples/PETRAIII/example_matlab_comparison.py` compares a pyLOCO fit with a
preserved MATLAB LOCO result.

## Student summer project

The educational notebooks are in `Examples/Summer_project/`. They use a simple
FODO lattice so students can study the calculations before working with a large
accelerator model.

Start with:

```text
Examples/Summer_project/from_fodo_to_pyloco.ipynb
```

It introduces the response matrix, a quadrupole error, the LOCO fit, correction,
and verification of the corrected optics.

Then continue with the Jacobian exercises in this order:

1. `FODO_Jacobian/01_fodo_jacobian_methods.ipynb`
2. `FODO_Jacobian/02_fodo_jacobian_with_errors.ipynb`
3. `FODO_Jacobian/03_fodo_iterative_loco_comparison.ipynb`

Cells marked `Open analysis — implementation required` are intentionally
incomplete so alternative methods can be evaluated. Keep the two
`analytic_orm_*.py` support files in the `FODO_Jacobian` directory.

## Installation

Install pyLOCO from the repository root:

```bash
python3 -m pip install -e .
```

For the pySC example, also install:

```bash
python3 -m pip install accelerator-commissioning
```

Run Python scripts from the repository root. Open educational notebooks in
Jupyter and execute their cells from top to bottom.
