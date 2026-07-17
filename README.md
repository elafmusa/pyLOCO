# LOCO Reference

- Safranek, J. (1997). Experimental determination of storage ring optics using orbit response measurements. *Nuclear Instruments and Methods in Physics Research Section A*, **388**, 27–36. https://doi.org/10.1016/S0168-9002(97)00309-4

# Questions or suggestions

Please contact: <elaf.musa@desy.de>

# Installation

Optional (creat vertual environment)
```
python -m venv venv
source venv/bin/activate
```

```
git clone https://github.com/elafmusa/pyLOCO
cd pyLOCO
pip install -e .    # or: pip install .
```

# GUI

The developmet of the GUI is currently ongoing.
Install the optional GUI dependencies:

```
pip install -e ".[gui]"
```

Launch the GUI:
```
python -m pyLOCO.gui.app
```
**pyLOCO** is primarily designed for LOCO analysis using experimental measurements.

The repository also includes examples that demonstrate how to use **pyLOCO** with simulated commissioning, for example with **pySC**. To install pySC:

```bash
pip install accelerator-commissioning
```
