# Testing pyLOCO

The tests check configuration parsing, independent parameter/constraint
selection, response-matrix behavior, PETRA III preparation, and the generic
measured-machine workflow. Normal LOCO users do not need to modify these files.

## Installation

Install pyLOCO and pytest from the repository directory:

```bash
python3 -m pip install -e .
python3 -m pip install pytest
```

## Run the tests

Run the complete suite:

```bash
python3 -m pytest -v
```

Run only the configuration and generic measured-machine tests:

```bash
python3 -m pytest -v \
  tests/test_user_config.py \
  tests/test_measured_machine_workflow.py
```

Run the PETRA III workflow tests:

```bash
python3 -m pytest -v tests/test_petra_workflow.py
```

## Lightweight machine preparation

Validate the lattice, measurements, element selections, dimensions, and
configuration without calculating or fitting an ORM:

```bash
python3 Examples/measured_machine/example_measured_machine.py \
  --config Examples/measured_machine/configs/petra_iii.yaml \
  --prepare-only

python3 Examples/measured_machine/example_measured_machine.py \
  --config Examples/measured_machine/configs/ebs.yaml \
  --prepare-only
```
