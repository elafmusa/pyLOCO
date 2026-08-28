# pyLOCO

pyLOCO is a Python implementation of the Linear Optics from Closed Orbits (LOCO)
for fitting accelerator models to measured or error-simulated data.

LOCO reference: J. Safranek, “Experimental determination of storage ring
optics using orbit response measurements,” *NIM A* 388 (1997), 27–36,
[doi:10.1016/S0168-9002(97)00309-4](https://doi.org/10.1016/S0168-9002(97)00309-4).

## Citing pyLOCO

If pyLOCO contributes to your work, please cite:

> E. Musa, I. Agapov, K. Paraschou, J. Keil, and S. Liuzzo, “PyLOCO: A Python Framework for Linear Optics Correction in Storage Rings,” presented at the 17th International Particle Accelerator Conference (IPAC’26), Deauville, France, May 2026, paper WEP5011. [JACoW contribution](https://indico.jacow.org/event/95/contributions/13338/)

```bibtex
@inproceedings{musa_pyloco_ipac26,
  author = {Musa, Elaf and Agapov, Ilya and Paraschou, Konstantinos and Keil, Joachim and Liuzzo, Simone},
  title = {PyLOCO: A Python Framework for Linear Optics Correction in Storage Rings},
  booktitle = {Proceedings of the 17th International Particle Accelerator Conference (IPAC'26)},
  address = {Deauville, France},
  year = {2026},
  month = {May},
  note = {Paper WEP5011},
  url = {https://indico.jacow.org/event/95/contributions/13338/}
}
```

## Install and launch

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/elafmusa/pyLOCO.git
cd pyLOCO
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[gui]"
```

## GUI

Launch the graphical interface from the pyLOCO directory:

```bash
pyloco-gui
```

Alternatively, launch it with Python:

```bash
python -m pyLOCO.gui.app
```

On Windows, activate the environment with `.venv\Scripts\activate` instead of
`source .venv/bin/activate`.

## Recommended examples

If you are new to pyLOCO, start with the simple single-quadrupole example:

```bash
python3 Examples/reconstruct_quadrupoles_errors_examples/reconstruct_one_quad_errors/example_one_quad_error.py
```

For a real measured-machine application, use the PETRA III coupling example:

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


Questions or suggestions: <elaf.musa@desy.de>

## Contributors

- Elaf Musa

With thanks to Ilya Agapov, Joachim Keil, Konstantinos Paraschou, Simone
Liuzzo, and Ahmed El Deeb for their support and contributions.
