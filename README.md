# LOCO Reference

- Safranek, J. (1997). Experimental determination of storage ring optics using orbit response measurements. *Nuclear Instruments and Methods in Physics Research Section A*, **388**, 27–36. https://doi.org/10.1016/S0168-9002(97)00309-4

# Installation

Optional (creat vertual environment)
```
python -m venv pyloco_venv
source pyloco_venv/bin/activate
```

```
git clone https://github.com/elafmusa/pyLOCO
cd pyLOCO
pip install -e .    # or: pip install .
```

# GUI

The GUI is a front end to the same `pyLOCO.pyloco` calculation used by the
Python examples; it does not contain a separate fitting implementation.

Install the optional GUI dependencies:

```
pip install -e ".[gui]"
```

Launch the GUI:
```
python -m pyLOCO.gui.app
```
or directly
```
pyloco-gui
```

## Basic workflow

1. Create a project and select **Lattice** to load an Accelerator Toolbox
   lattice (`.mat`, `.m`, or `.json`).
2. On **Machine Elements**, select the BPMs, horizontal/vertical correctors,
   quadrupoles, and (when needed) skew quadrupoles and cavities. The selection
   order defines the ORM and fitted-parameter order.
3. On **Measurements**, import the measured ORM and assign the `orm` role.
   HDF5, MATLAB, NPY, and NPZ files are supported. Dispersion is required only
   when dispersion fitting is enabled; BPM-noise data is optional and unit
   weights are used when it is omitted.
4. On **LOCO Configuration**, choose the analytic **Linear** or numerical
   **Tracking** ORM calculator, corrector/RF steps, fitted parameter blocks,
   solver iterations, and SVD settings. Switch to **Advanced** mode for
   coupling, normalization, constraints, RF, momentum compaction, and
   parameter-initialization controls.
5. Select **Run LOCO**. Progress and errors appear on **Results**. A completed
   run saves the initial model ORM, fitted ORM and histories, fit dictionary,
   summary, and fitted lattice in the displayed results directory. Use
   **Compare ORMs** to inspect measured, model, and difference plots.

Use **Export configuration** to save the current controls as JSON or YAML and
**Import configuration** to restore them. Import also understands the
user-facing `pyloco_config.yaml` files used by the maintained EBS, pySC, and
PETRA III examples, resolves their relative lattice/data/output paths, and
populates the corresponding GUI controls. Invalid paths, missing inputs,
inconsistent ORM dimensions, and invalid corrector-step arrays are reported
before or at run startup with a specific message.

For a small starting point, see the EBS single-quadrupole workflow in
`Examples/reconstruct_quadrupoles_errors_examples/reconstruct_one_quad_errors/`.
For advanced fitted-parameter and measured-data settings, see
`Examples/PETRAIII/pyloco_config.yaml`.

The pySC example can load and fit its cached HDF5 measurements in the GUI.
Generating new measurements and applying corrections to a live pySC
`SimulatedCommissioning` object remain Python-script operations because that
object is an in-memory commissioning session, not a lattice/measurement file.


# Examples

The repository contains examples for both simulated and measured machines:

- **EBS single quadrupole error:** the simplest introduction to reconstructing
  and correcting a known optics error.
- **EBS multiple quadrupole errors:** reconstruction and correction of several
  simultaneous errors.
- **EBS with pySC:** pySC performs simulated measurements, pyLOCO reconstructs
  the errors, and the fitted correction is applied back to the pySC machine.
- **PETRA III measured ORM:** fitting real machine-response data, including bad
  BPM handling and measurement uncertainty.
- **PETRA III coupling:** an advanced fit of the cross-plane response.
- **PETRA III constrained fit:** YAML-configured quadrupole-family constraints,
  dispersion weighting, and per-run output directories.
- **PETRA III MATLAB comparison:** comparison of all fitted pyLOCO parameters
  with a preserved MATLAB LOCO result.

See [Examples/README.md](Examples/README.md) for a simple description of every
workflow and guidance on which example to run first.

The pySC example requires the optional pySC package:

```bash
pip install accelerator-commissioning
```

# Questions or suggestions

Please contact: <elaf.musa@desy.de>
