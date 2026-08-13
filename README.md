# pyLOCO

pyLOCO is a Python implementation of Linear Optics from Closed Orbits (LOCO)
for fitting accelerator models to measured or simulated orbit-response data.

## Install

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/elafmusa/pyLOCO.git
cd pyLOCO
python3 -m pip install -e .
```

For the graphical interface:

```bash
python3 -m pip install -e ".[gui]"
pyloco-gui
```

## Run an example

Start with the simple single-quadrupole example:

```bash
python3 Examples/reconstruct_quadrupoles_errors_examples/reconstruct_one_quad_errors/example_one_quad_error.py
```

For the PETRA III measurement from 12 August, before the quadrupole change:

```bash
python3 Examples/PETRAIII/measurments/example_measured_machine.py
```

Its settings are in:

```text
Examples/PETRAIII/measurments/configs/measurement_12Aug_before_change_quad.yaml
```

## Tests

Install pytest and run:

```bash
python3 -m pip install pytest
python3 -m pytest
```

## More information

- [Examples and which one to choose](Examples/README.md)
- [Measured-machine configuration](Examples/measured_machine/README.md)
- [PETRA III measurement example](Examples/PETRAIII/measurments/README.md)
- [PETRA III GUI files](Examples/PETRAIII/GUI/README.md)
- [Testing guide](tests/README.md)

LOCO reference: J. Safranek, “Experimental determination of storage ring
optics using orbit response measurements,” *NIM A* 388 (1997), 27–36,
[doi:10.1016/S0168-9002(97)00309-4](https://doi.org/10.1016/S0168-9002(97)00309-4).

Questions: <elaf.musa@desy.de>
