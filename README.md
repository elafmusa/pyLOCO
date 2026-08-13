# pyLOCO

pyLOCO is a Python implementation of the Linear Optics from Closed Orbits (LOCO)
for fitting accelerator models to measured or error-simulated data.

LOCO reference: J. Safranek, “Experimental determination of storage ring
optics using orbit response measurements,” *NIM A* 388 (1997), 27–36,
[doi:10.1016/S0168-9002(97)00309-4](https://doi.org/10.1016/S0168-9002(97)00309-4).

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

For a real PETRA III measurement including transverse coupling:

```bash
python3 Examples/PETRAIII/example_measured_coupling.py
```

Its settings are in:

```text
Examples/PETRAIII/pyloco_config_coupling.yaml
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


Questions ot suggestions: <elaf.musa@desy.de>
