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

**pyLOCO** is primarily designed for LOCO analysis using experimental measurements.

The repository also includes examples that demonstrate how to use **pyLOCO** with simulated commissioning, for example with **pySC**. To install pySC:

```bash
pip install accelerator-commissioning
```

# Questions or suggestions

Please contact: <elaf.musa@desy.de>


## GUI Bad BPM list

The GUI measurement importer supports an optional **Bad BPM list** role. Import a
`.npy`, `.npz`, `.h5`, `.hdf5`, or `.mat` file containing a one-dimensional array
named `bad_bpm_positions` (preferred) or `bad_bpms`; if neither name is present,
the first array/dataset in the file is used. Values are **0-based BPM positions**
in the BPM list, not ORM row indices. For example:

```python
bad_bpm_positions = np.array([24, 104, 108, 111, 123, 138, 144, 153, 161, 162, 243])
```

Before the GUI calls `pyloco()`, the backend validates that the Bad BPM list is
one-dimensional, integer-valued, unique, and within the valid BPM position range.
When present, the same positions are removed from all BPM-indexed inputs so the
LOCO fit dimensions stay consistent:

- horizontal and vertical BPM noise arrays (`Noise_BPMx`, `Noise_BPMy`), followed
  by reconstruction of the concatenated BPM weight vector;
- `used_bpms_ords`, with `nHBPM` and `nVBPM` updated to the remaining BPM count;
- horizontal and vertical measured dispersion (`measured_eta_x`,
  `measured_eta_y`);
- both horizontal and vertical BPM row blocks of the measured ORM via
  `remove_bad_bpms(..., axis=0, input_type="positions")`.

If no Bad BPM list is imported, the GUI backend preserves the existing workflow
and passes the measurement arrays to `pyloco()` unchanged.
