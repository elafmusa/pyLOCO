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

The developmet of the GUI is currently ongoing.
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

