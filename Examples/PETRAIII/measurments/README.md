# PETRA III measured-machine example

This folder contains one runner for the PETRA III measurement taken on
12 August, before the quadrupole change.

## Files to use

```text
example_measured_machine.py
configs/measurement_12Aug_before_change_quad.yaml
```

Normally, change only the YAML file. It selects the measurement data, machine
elements, fit parameters, constraints, solver settings, and output directory.

## Run

From the repository root:

```bash
python3 Examples/PETRAIII/measurments/example_measured_machine.py
```

To check the inputs without starting the fit:

```bash
python3 Examples/PETRAIII/measurments/example_measured_machine.py --prepare-only
```

The runner automatically uses:

```text
Examples/PETRAIII/measurments/configs/measurement_12Aug_before_change_quad.yaml
```

To use another YAML file:

```bash
python3 Examples/PETRAIII/measurments/example_measured_machine.py \
  --config path/to/config.yaml
```

For the complete machine-independent YAML format, see
[the measured-machine guide](../../measured_machine/README.md).
