# Generic measured-machine workflow

The same runner supports PETRA III, EBS, and normally formatted new machines:

```bash
python Examples/measured_machine/example_measured_machine.py --config Examples/measured_machine/configs/petra_iii.yaml
python Examples/measured_machine/example_measured_machine.py --config Examples/measured_machine/configs/ebs.yaml
```

Use `--prepare-only` to load and validate the lattice, measurements, element
selections, dimensions, family mappings, and constraints without running LOCO.

## How do I run pyLOCO on a new machine?

1. Prepare an Accelerator Toolbox lattice.
2. Provide an HDF5 ORM and its dataset name.
3. Identify horizontal/vertical BPMs and correctors in YAML.
4. Optionally provide dispersion and BPM-noise datasets.
5. Select physical quadrupoles and optional family groups/skew quadrupoles.
6. Copy `configs/template.yaml` and change paths, selectors, units, and counts.
7. Choose `fit_parameters` and independently enable or disable `constraints`.
8. Run this same Python script with the new YAML.

Selectors support explicit indices, NPY index files, ordered name files,
Accelerator Toolbox element types, exact family names, shell-style patterns,
and regular expressions. Relative paths are resolved from the YAML file.

Python changes are still needed for a genuinely new file format (not HDF5/NPY
or an AT-readable lattice), a measurement convention that cannot be expressed
as transpose/order/scale settings, or machine-specific correction application
to a live control system. The LOCO preparation and fit itself needs no machine
class or machine-specific workflow.
