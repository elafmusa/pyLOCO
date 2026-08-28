# PETRA III GUI project

`petra_iii.pyloco.json` is the canonical, verified PETRA III measured-ORM
project.
It restores the lattice, measured ORM/noise files, machine-element selections,
fit parameters, constraints, and LOCO controls.

This project is configured as a **non-coupling fit**. It fits normal
quadrupoles, BPM gains, corrector calibration, and the horizontal corrector
energy shift.
Skew quadrupoles, quadrupole tilt, BPM coupling, and corrector coupling are
disabled. The original PETRA III measurement files are preserved unchanged.

From the repository root, launch the GUI:

```bash
pyloco-gui
```

Then choose **File → Open** and select:

```text
Examples/PETRAIII/GUI/petra_iii.pyloco.json
```

The project should report complete validation. Select **Run LOCO**, then use
the **Results** workspace to inspect:

- **Overview** for chi-square convergence and summary values.
- **ORM** for the measured, initial-model, fitted-model, and residual plots.
- **Optics** for available lattice diagnostics.
- **Parameters** for fitted changes from their initialization values.

Opening the project does not start a fit. The real PETRA III calculation is
larger and slower than the single-quadrupole demonstration.

## Coupling fit project

`petra_iii_coupling.pyloco.json` is the portable PETRA III coupling project
reconstructed from the validated measured-machine run. It uses the tracked
PETRA III lattice and measurement resources and enables the four coupling fit
blocks (`hbpm_coupling`, `vbpm_coupling`, `hcor_coupling`, and
`vcor_coupling`) together with the validated family-quadrupole, skew,
quadrupole-tilt, gain, and corrector-calibration configuration. Open it from
the same **File → Open** action; no machine-specific absolute path is stored
in the project file.

## Verified reference result

This exact saved project was run successfully with the current GUI backend on
14 August 2026. The run completed in about 51 seconds and produced:

- Initial chi-square: `521130`
- Final chi-square: `10314.47`
- Chi-square reduction: `98.02%`
- ORM residual RMS before fitting: `160.76 micrometres`
- ORM residual RMS after fitting: `72.03 micrometres`

The project uses the measured per-corrector steps from `data/CMstep.npz`. Do
not replace these with a single uniform corrector step if you want to reproduce
the reference result.

For the machine-independent command-line workflow, use
`Examples/measured_machine/configs/petra_iii.yaml` instead.
