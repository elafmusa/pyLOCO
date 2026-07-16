# pyLOCO GUI Design Document

## Purpose and User Perspective

This document describes a future **machine-independent PySide6 GUI** for pyLOCO from the point of view of an accelerator physicist running LOCO for the first time. The GUI should make the standard LOCO workflow visible, guided, and reproducible without hiding pyLOCO's advanced capabilities.

The design remains **design-only**: no GUI source code is proposed for implementation in this change.

## 1. User-Centered LOCO Workflow

The GUI should be organized around the actions a physicist performs during a LOCO study, not around the internal repository layout. The primary navigation should therefore follow this workflow:

```text
Project → Machine Model → Measurements → Fit Setup → Run → Inspect Results → Export/Apply
```

### 1.1 Project

The user starts by creating or opening a **LOCO project**. A project is a reproducible container for:

- machine model and lattice files;
- measurement files and dataset mappings;
- BPM, corrector, RF cavity, quadrupole, skew-quadrupole, and tilt-element selections;
- LOCO options, fit blocks, and numerical settings;
- bad BPM/corrector masks and unit conversions;
- output artifacts, logs, plots, and exported scripts.

The first screen should answer practical questions:

- “What machine/model am I analyzing?”
- “Where are my measurement files?”
- “Where will results be saved?”
- “Can this project be rerun later or shared with a colleague?”

### 1.2 Machine Model

The model page should guide the user through loading an Accelerator Toolbox-compatible lattice and selecting machine elements. It should not assume PETRA III-specific names or directory conventions. The same GUI must support PETRA III, PETRA IV, EBS, FCC-ee, or future machines by using configurable selectors and plugins.

The user should be able to:

- load a lattice/model file;
- inspect elements in a table and longitudinal layout;
- select BPMs, horizontal correctors, vertical correctors, RF cavities, quadrupoles, skew quadrupoles, and tilt elements;
- define selections by element type, family name, common name, wildcard pattern, index list, imported text file, or machine plugin;
- verify counts and expected matrix dimensions before importing data.

### 1.3 Measurements

The measurement page should help the user import and validate measured data:

- orbit response matrix (ORM);
- BPM noise/standard deviations;
- dispersion or RF-frequency response;
- corrector kick steps;
- optional bad BPM/corrector masks;
- machine metadata and acquisition notes.

The GUI should show immediate previews:

- matrix shape and expected shape;
- plane ordering and column ordering;
- units and scale factors;
- NaN or outlier warnings;
- direct and coupled ORM blocks;
- whether dispersion is appended as an extra column.

### 1.4 Fit Setup

The fit setup page should expose a **Basic mode** by default and an **Advanced mode** on demand.

In Basic mode, users choose from physical presets:

- optics-only quadrupole fit;
- BPM and corrector calibration fit;
- coupling fit;
- dispersion-enabled fit;
- multi-stage “start simple, then add coupling/calibration” workflow.

In Advanced mode, users can edit the full `fit_list`, SVD settings, LM/GN options, normalization settings, constraints, Jacobian cache paths, attribute targets, and custom `pyloco_config.py` values.

### 1.5 Run

The run page should provide a live run monitor. LOCO fits can be long and numerically complex, so the GUI must show progress and intermediate diagnostic plots rather than appearing frozen.

The monitor should show:

- current phase: model ORM, Jacobian, weighting, outlier filtering, SVD, LM/GN solve, trial evaluation, final chi-square;
- current outer iteration and LM inner iteration;
- elapsed time and estimated remaining time when possible;
- current chi-square and previous chi-square;
- accepted/rejected LM steps and lambda changes;
- number of selected SVD modes;
- number of outliers removed;
- latest residual plots and singular-value plots;
- log messages and warnings.

### 1.6 Inspect Results

The result page should help the physicist decide whether the fit is trustworthy:

- before/after ORM residuals;
- chi-square history;
- fit parameters grouped by physical block;
- BPM gain/coupling tables;
- corrector calibration/coupling tables;
- quadrupole/skew/tilt correction tables;
- dispersion residuals;
- beta beating and optics comparisons when reference optics are available;
- delta-chi-square contributions by block when enabled;
- warnings about suspicious corrections, rank deficiency, or inconsistent units.

### 1.7 Export or Apply

The GUI should always support offline export before any live-machine integration:

- reproducible Python script;
- project file snapshot;
- HDF5/NumPy/CSV result tables;
- final model ORM and residuals;
- fit dictionary and parameter vector history;
- correction tables for machine applications;
- human-readable run report.

Live machine application, if added later, must be an expert plugin with explicit confirmation, dry-run preview, setpoint restore checks, and machine-specific safety rules.

## 2. Basic and Advanced Modes

### 2.1 Basic Mode

Basic mode is the default. It is intended for accelerator physicists who understand LOCO concepts but do not want to edit Python scripts.

Basic mode should provide:

- workflow pages with “Next” guidance;
- machine-independent templates;
- physical terminology instead of Python variable names where possible;
- sensible defaults from the selected template or plugin;
- dimension checks before a run can start;
- simple fit presets;
- compact result summaries;
- plain-language warnings and suggestions;
- one-click script export for reproducibility.

Basic mode should hide or collapse advanced details such as:

- full block-order editing;
- raw `pyloco()` keyword lists;
- low-level lattice attribute names;
- individual Jacobian file paths unless a cache is detected;
- detailed normalization internals;
- multiprocessing options.

### 2.2 Advanced Mode

Advanced mode should preserve the full flexibility currently available from scripts and notebooks.

Advanced mode should expose:

- complete `fit_list` editing and block ordering;
- initial policy and initial-value overrides;
- element attribute targets such as `PolynomB[1]`, `PolynomA[1]`, or machine-specific attributes;
- individual/family fitting;
- SVD threshold/cut strategies;
- LM/GN parameters;
- outlier rejection details;
- component/global normalization;
- fixed-path-length and fixed-momentum options;
- coupling-row removal;
- dispersion weighting;
- Jacobian cache paths and recomputation policy;
- constraint definitions;
- custom `pyloco_config.py` import/export;
- raw backend call preview.

The user should be able to switch between modes without losing settings. When an advanced setting differs from the Basic preset, Basic mode should show a small “advanced settings modified” indicator.

## 3. Project Explorer

The main window should include a persistent **Project Explorer** on the left, similar to analysis environments familiar to scientists.

```text
LOCO Project
├── Machine Model
│   ├── Lattice
│   ├── BPMs
│   ├── Correctors
│   ├── RF Cavities
│   ├── Quadrupoles
│   └── Skew/Tilt Elements
├── Measurements
│   ├── ORM
│   ├── BPM Noise
│   ├── Dispersion
│   └── Masks / Bad BPMs
├── Fit Setup
│   ├── Preset
│   ├── Fit Blocks
│   ├── Solver
│   ├── SVD
│   └── Jacobians
├── Runs
│   ├── Run 001
│   ├── Run 002
│   └── Comparison
├── Results
│   ├── Residuals
│   ├── Parameters
│   ├── Optics
│   └── Exports
└── Plugins
    ├── Machine Plugin
    ├── Importers
    └── Exporters
```

The Project Explorer should provide:

- status badges: valid, warning, missing, running, complete, failed;
- quick navigation to relevant pages;
- right-click actions such as reload, validate, preview, export, duplicate run, or compare runs;
- support for multiple runs in one project;
- a clear distinction between input data, configuration, computed outputs, and exported products.

## 4. Machine Independence

The GUI must not be designed as a PETRA-only tool. It should treat machine-specific knowledge as data, configuration, or plugin behavior.

### 4.1 Machine-neutral core concepts

The core GUI should only assume generic LOCO concepts:

- lattice/model;
- BPMs;
- horizontal and vertical correctors;
- RF cavities;
- quadrupoles;
- skew quadrupoles or coupling knobs;
- optional tilt elements;
- ORM, BPM noise, dispersion;
- fit blocks, constraints, solver options;
- correction tables and result exports.

### 4.2 Machine profile

Each project may select a **machine profile**. A profile describes defaults but does not hard-code behavior into the GUI.

A profile can provide:

- display name, institution, and machine version;
- default unit conventions;
- default element selection patterns;
- default fit presets;
- default HDF5/NumPy dataset names;
- recommended solver/SVD settings;
- safe limits for correction magnitudes;
- optional live-control adapter name;
- documentation links.

The GUI should ship with generic profiles and example profiles, but users must be able to create custom profiles for PETRA IV, EBS, FCC-ee, or future machines without editing GUI source code.

### 4.3 Machine-independent data mapping

Measurement import should use explicit mapping rather than directory assumptions. For example, the user or profile maps:

- HDF5 dataset `/measured_orm` → ORM matrix;
- `/Noise_BPMx` and `/Noise_BPMy` → BPM weights;
- `/measured_eta_x` and `/measured_eta_y` → dispersion vectors;
- text files → BPM/corrector names;
- metadata attributes → acquisition settings.

The GUI should save these mappings inside the project so the analysis is reproducible.

## 5. Repository Structure Summary

Although the GUI should be organized around the LOCO workflow, developers still need to understand the current repository. pyLOCO is a compact numerical package with examples and measurement scripts outside the importable package.

```text
pyLOCO/
├── pyLOCO/                         # Importable backend package
│   ├── pyloco.py                   # Main LOCO engine, fitting loop, Jacobians, SVD/LM/GN, utilities
│   ├── response_matrix.py          # Model ORM/dispersion response calculation from an AT lattice
│   ├── initial_fit.py              # Builds initial parameter vectors and named block slices
│   ├── set_parameters.py           # Applies fit parameters to lattice attributes/R1/R2 tilt matrices
│   ├── analysis.py                 # Plotting and optics/dispersion helper functions
│   ├── helpers.py                  # Dynamic pyloco_config loader
│   └── __init__.py
├── Examples/
│   ├── PETRAIII/                   # PETRA III AT lattice example, measured HDF5/NumPy data, notebooks
│   └── EBS_pySC/                   # ESRF-EBS/pySC simulation and reconstruction examples
├── Measurments_scripts_pydoocs/    # PETRA/DOOCS measurement scripts for ORM, BPM noise, dispersion
├── README.md                       # Installation notes and LOCO reference
└── pyproject.toml                  # Minimal packaging metadata
```

### 5.1 Core package

- `pyLOCO/pyloco.py` is the central engine. It contains weighting, coupling removal, outlier handling, chi-square calculations, Jacobian assembly, normalization, SVD selection, Gauss-Newton/Levenberg-Marquardt solvers, and the public `pyloco()` function.
- `pyLOCO/response_matrix.py` computes model orbit response matrices with Accelerator Toolbox (`at`), optional dispersion columns, RF shifts, corrector coupling, fixed-path-length corrections, and linear/tracking-style calculation branches.
- `pyLOCO/initial_fit.py` converts a selected `fit_list` into an ordered numerical parameter vector and an `OrderedDict` of block slices.
- `pyLOCO/set_parameters.py` writes fitted values back into lattice elements, supporting scalar/array element attributes, family/individual fitting, skew blocks, and quadrupole tilt matrices.
- `pyLOCO/helpers.py` loads a user `pyloco_config.py` by path or module name and registers it as the canonical `pyloco_config` module.
- `pyLOCO/analysis.py` contains Matplotlib plotting helpers for beta beating, dispersion, and ORM visualization; it is useful for a GUI but should not be part of the compute core.

### 5.2 Examples and data

- `Examples/PETRAIII/` demonstrates a workflow based on an AT `.mat` lattice, measured ORM/noise/dispersion HDF5 files, corrector/BPM name mapping text files, precomputed reference arrays, notebooks, and scripts for one-iteration, multi-iteration, and coupling fits.
- `Examples/EBS_pySC/com_simu_loco_example/` demonstrates pySC-based simulated commissioning: generating or loading a simulated machine state, measuring ORM/noise/dispersion, running pyLOCO through `run_pyloco_from_model.py`, and applying corrections to a simulated machine.
- `Examples/EBS_pySC/reconstruct_quadrupoles_errors_examples/` contains smaller one- and two-quadrupole-error reconstruction examples.

### 5.3 Measurement scripts

- `Measurments_scripts_pydoocs/ORM_measurment/` measures orbit response matrices using DOOCS channels, magnet setpoints, bidirectional kicks, and HDF5 output.
- `Measurments_scripts_pydoocs/BPM_noise/` measures BPM noise by repeated orbit acquisition and saves standard deviations.
- `Measurments_scripts_pydoocs/dispersion_measurment/` measures dispersion-like orbit changes and saves HDF5 output.

## 6. Main Execution Workflow of pyLOCO

The current backend execution model is script-driven. The GUI should reproduce this workflow visually and export an equivalent script.

1. **Load a configuration module**
   - Example scripts call `load_config(config_path=...)` so existing backend imports can resolve `pyloco_config`.
   - The config module defines `LOCOOptions`, `RMConfig`, `FitInitConfig`, `fixed_parameters`, `get_mcf()`, and `_cfg_get()`.

2. **Load or construct the accelerator model**
   - PETRA III examples load an Accelerator Toolbox lattice with `at.load_lattice()`.
   - EBS examples either load a `betamodel.mat` lattice or use pySC to generate/load a simulated commissioning state.

3. **Resolve machine element selections**
   - Scripts determine BPM ordinals, horizontal/vertical corrector ordinals, RF cavity ordinals, quadrupole ordinals, skew ordinals, and optional quadrupole-tilt ordinals.
   - The GUI should generalize this through selectors and machine profiles rather than hard-coded example paths.

4. **Load measured data**
   - ORM data is read from HDF5 and arranged as a matrix with rows for BPM planes and columns for correctors; dispersion can be appended as an additional column when enabled.
   - BPM noise is loaded and concatenated into a weight vector.
   - Measured horizontal and vertical dispersion arrays are loaded separately when used.
   - Bad BPMs may be removed by `remove_bad_bpms()` before fitting.

5. **Define LOCO run options**
   - The script selects the `fit_list`, algorithm (`lm` or `gn`), iteration counts, SVD options, normalization mode, outlier rejection, dispersion weights, coupling-removal mode, and Jacobian caching behavior.
   - The GUI should present this as Basic presets plus Advanced raw controls.

6. **Call `pyloco()`**
   - `pyloco()` creates initial parameter vectors, then for each outer iteration:
     1. Computes the model ORM with `response_matrix()`.
     2. Applies BPM gain/coupling matrix corrections.
     3. Builds the full Jacobian from enabled blocks.
     4. Applies fixed-momentum/fixed-path-length energy-shift terms when relevant.
     5. Flattens measured/model matrices using Fortran ordering.
     6. Builds weights with `weight_matrix()`.
     7. Optionally removes coupling rows from the fit system.
     8. Optionally removes statistical outliers.
     9. Applies RF and/or Jacobian normalization.
     10. Computes initial chi-square.
     11. Solves either an LM inner loop with trial-ring acceptance or a GN step.
     12. Applies accepted fit parameters to lattice/state.
     13. Recomputes model ORM and chi-square after correction.
     14. Stores iteration parameter vectors, structured fit dictionaries, chi-square history, optional delta-chi-square history, and block metadata.

7. **Analyze and apply results**
   - Scripts compare fitted parameters with reference data, plot optics/ORM residuals, save arrays, and optionally apply corrections to pySC or lattice objects.

The public `pyloco()` return value is currently a tuple containing fit parameter history, structured fit dictionaries, the final ring, final model ORM, final BPM correction matrix, chi-square history, delta-chi-square history, and parameter block slices.

## 7. Modules That Should Remain Independent from the GUI

The GUI should be a client of the pyLOCO backend, not a replacement for it. These modules should remain importable and runnable without Qt/PySide dependencies:

- **Numerical engine:** `pyLOCO.pyloco`
  - Keep fitting, Jacobian generation, normalization, SVD selection, outlier logic, and solver code GUI-free.
  - Refactor later only by introducing backend-neutral dataclasses/callbacks, never by importing GUI classes.

- **Response calculation:** `pyLOCO.response_matrix`
  - Keep Accelerator Toolbox computation independent from display concerns.

- **Initial parameter construction:** `pyLOCO.initial_fit`
  - The GUI can present block selections visually, but vector/block construction should stay here or in a backend service layer.

- **Lattice parameter application:** `pyLOCO.set_parameters`
  - GUI should call service methods that eventually use these functions; widgets should not directly mutate AT lattice internals.

- **Configuration loading:** `pyLOCO.helpers`
  - The GUI can offer file pickers and editors, but canonical config loading should stay backend-neutral.

- **Measurement scripts:** `Measurments_scripts_pydoocs/*`
  - Machine control must remain isolated behind explicit adapters and safety dialogs. The first GUI version should import existing measured files rather than drive live DOOCS changes.

- **Examples:** `Examples/*`
  - Examples should remain reproducible scripts/notebooks. The GUI can provide “Open example project” templates that reference these workflows.

`pyLOCO.analysis` is partly presentation-oriented. For the GUI, prefer extracting reusable numerical summaries from plotting functions, while implementing actual plots with Qt-compatible canvases.

## 8. Proposed Modular PySide6 Architecture

Use a layered architecture that separates user interface, application orchestration, backend adapters, plugins, and persistent project state.

```text
PySide6 UI layer
    ↓ signals/slots, actions, view models
Application/service layer
    ↓ typed requests/results, worker jobs, progress events
Plugin layer
    ↓ machine profiles, importers, exporters, plotting panels, live-control adapters
Backend adapter layer
    ↓ calls pyLOCO, at, h5py, numpy, config loader
pyLOCO numerical core
```

### 8.1 Architectural goals

- Support accelerator physicists who prefer forms, presets, validation messages, and plots over Python scripts.
- Preserve full pyLOCO flexibility through Advanced mode exposing raw fit blocks, config values, file paths, and custom `pyloco_config.py` loading.
- Keep compute jobs responsive by running LOCO fits, response-matrix calculations, and file imports in worker threads/processes.
- Represent every GUI session as a reproducible project file that can export an equivalent Python script.
- Keep machine-specific behavior in profiles/plugins rather than in the core GUI.

### 8.2 Key components

#### UI layer (`pyLOCO/gui/`)

- PySide6 windows, dialogs, widgets, model/view table models, plot canvases, and icons/resources.
- Workflow-oriented pages: Project, Machine Model, Measurements, Fit Setup, Run Monitor, Results, Export.
- No direct long-running numerical work in widgets.
- Widgets emit high-level intents such as `run_requested`, `import_orm_requested`, or `fit_block_enabled`.

#### View-model layer

- Qt-friendly state wrappers for project status, selected files, fit block table, element selections, run options, and results.
- Responsible for validation messages shown in the GUI.
- Converts backend dataclasses to table rows and plot-ready arrays.
- Tracks whether settings are Basic-compatible or Advanced-modified.

#### Service layer

- `ProjectService`: load/save GUI project files, manage recent projects, track dirty state, manage multiple runs.
- `ProfileService`: load machine profiles and expose defaults without coupling the GUI to a machine.
- `ConfigService`: load `pyloco_config.py`, inspect `LOCOOptions`, `RMConfig`, `FitInitConfig`, and `fixed_parameters`.
- `LatticeService`: load AT lattices, summarize element families, resolve BPM/corrector/quadrupole selections.
- `MeasurementService`: import HDF5/NumPy/text files; validate dimensions; preview ORM/noise/dispersion.
- `LocoRunService`: build a typed run request and execute `pyloco()` in a worker.
- `ResultService`: normalize and serialize results, build residuals, chi-square summaries, and export correction tables/scripts.
- `PluginService`: discover, validate, load, and sandbox plugin contributions.

#### Backend adapters

- Thin wrappers around existing functions: `load_config()`, `response_matrix()`, `pyloco()`, `remove_bad_bpms()`, `at.load_lattice()`, `h5py.File`, etc.
- Adapters should translate backend tuples into named result dataclasses to reduce GUI fragility.

#### Worker layer

- Use `QThread`/`QRunnable` or a dedicated `concurrent.futures` executor for IO and light computation.
- For full LOCO runs, use a worker process when possible because current Jacobian generation already uses multiprocessing and the GUI must remain responsive.
- Worker emits structured events: started, phase changed, iteration summary, plot data updated, warning, result, failed, cancelled.

#### Project model

A GUI project should store:

- paths to lattice/config/data files;
- selected machine profile and plugin versions;
- machine element selections and counts;
- measurement dataset mappings and unit conversions;
- bad BPM/corrector lists and data-cleaning choices;
- fit block selections and block-specific advanced settings;
- algorithm/SVD/LM/outlier/normalization/dispersion options;
- Jacobian cache paths and recomputation flags;
- run history, result metadata, and links to saved result arrays.

Prefer a human-readable YAML or JSON project file, plus an output directory for large NumPy/HDF5 result artifacts.

## 9. Recommended GUI Folder Structure

Recommended initial structure, without moving existing backend modules:

```text
pyLOCO/
├── gui/
│   ├── __init__.py
│   ├── app.py                         # QApplication setup and main entry point
│   ├── main_window.py                 # MainWindow shell, menus, docks, project explorer
│   ├── resources/                     # Icons, Qt resources, style sheets
│   ├── models/                        # Qt item/table models and typed GUI state
│   │   ├── project.py
│   │   ├── machine_profile.py
│   │   ├── fit_blocks_model.py
│   │   ├── element_table_model.py
│   │   ├── run_monitor_model.py
│   │   └── results_model.py
│   ├── services/                      # GUI-facing orchestration services
│   │   ├── project_service.py
│   │   ├── profile_service.py
│   │   ├── config_service.py
│   │   ├── lattice_service.py
│   │   ├── measurement_service.py
│   │   ├── loco_run_service.py
│   │   ├── plugin_service.py
│   │   └── result_service.py
│   ├── backend/                       # Thin adapters around current pyLOCO/AT/HDF5 APIs
│   │   ├── pyloco_adapter.py
│   │   ├── at_adapter.py
│   │   ├── hdf5_adapter.py
│   │   └── config_adapter.py
│   ├── workers/
│   │   ├── base_worker.py
│   │   ├── import_worker.py
│   │   ├── response_worker.py
│   │   └── loco_worker.py
│   ├── plugins/                       # Built-in plugin contracts and optional built-ins
│   │   ├── api.py
│   │   ├── machine_profile.py
│   │   ├── importer.py
│   │   ├── exporter.py
│   │   ├── plot_panel.py
│   │   └── live_control.py
│   ├── widgets/
│   │   ├── project_explorer.py
│   │   ├── project_setup_page.py
│   │   ├── lattice_browser.py
│   │   ├── measurement_import_page.py
│   │   ├── fit_configuration_page.py
│   │   ├── run_monitor_page.py
│   │   ├── run_control_panel.py
│   │   ├── log_console.py
│   │   ├── results_summary.py
│   │   └── plotting/
│   │       ├── matrix_plot.py
│   │       ├── residual_plot.py
│   │       ├── optics_plot.py
│   │       ├── svd_plot.py
│   │       ├── chi_square_plot.py
│   │       └── correction_plot.py
│   ├── dialogs/
│   │   ├── preferences_dialog.py
│   │   ├── mode_switch_dialog.py
│   │   ├── config_editor_dialog.py
│   │   ├── machine_profile_dialog.py
│   │   ├── data_mapping_dialog.py
│   │   ├── bad_bpm_dialog.py
│   │   ├── jacobian_cache_dialog.py
│   │   ├── plugin_manager_dialog.py
│   │   ├── export_dialog.py
│   │   └── about_dialog.py
│   └── utils/
│       ├── validation.py
│       ├── units.py
│       ├── exceptions.py
│       └── script_export.py
└── ... existing backend files unchanged
```

For packaging, add a console entry point later, for example `pyloco-gui = pyLOCO.gui.app:main`, after the design is approved and dependencies are declared.

## 10. Main Windows, Dialogs, Widgets, and Plotting Panels

### 10.1 Main window

Use a workflow-centered central area with:

- Project Explorer on the left;
- active workflow page in the center;
- validation/messages panel at the bottom;
- optional plot/log docks on the right or bottom;
- Basic/Advanced mode toggle visible in the toolbar.

Primary workflow pages:

1. **Project**
2. **Machine Model**
3. **Measurements**
4. **Fit Setup**
5. **Run Monitor**
6. **Results**
7. **Export / Apply**

### 10.2 Menus and global actions

- **File:** New Project, Open Project, Save Project, Save As, Open Example, Export Script, Export Results, Quit.
- **Project:** Validate Project, Duplicate Run, Compare Runs, Project Report.
- **Data:** Import ORM, Import BPM Noise, Import Dispersion, Edit Bad BPMs, Validate Dimensions.
- **Model:** Load Lattice, Reload Config, Select Machine Profile, Compute Model ORM Preview, Inspect Elements.
- **Run:** Start LOCO, Stop/Cancel, Continue From Previous, Recompute Jacobians, Clear Output.
- **Plugins:** Plugin Manager, Reload Plugins, Open Plugin Folder.
- **View:** Toggle panels, reset layout, theme/unit preferences, Basic/Advanced mode.
- **Help:** Documentation links, about dialog, backend version details.

### 10.3 Core widgets/pages

#### Project page

- Project name and output directory.
- Machine profile selector: generic AT, PETRA III example, EBS pySC example, user-defined profile, future PETRA IV/FCC-ee profiles.
- Project Explorer overview.
- Recent runs and latest status.
- Reproducibility summary and script export preview.

#### Machine Model page

- Lattice file picker and loader status.
- Machine-independent element table with index, family name, common name, type, position, and key attributes.
- Selection helpers for BPMs, H/V correctors, quadrupoles, skew quadrupoles, RF cavities, and tilt elements.
- Pattern-based, file-based, and plugin-provided selectors.
- Longitudinal lattice overview plot.
- Counts and dimension checks visible at all times.

#### Measurements page

- ORM importer with HDF5/NumPy dataset selector and matrix preview.
- BPM-noise importer with horizontal/vertical arrays and weight-vector preview.
- Dispersion importer with horizontal/vertical arrays and optional append-to-ORM behavior.
- Dataset mapping editor for machine-independent input formats.
- Bad BPM manager accepting positions or lattice indices, reusing the current `remove_bad_bpms()` conventions.
- Unit selectors and scaling preview.

#### Fit Setup page

- Basic preset selector with short explanations and recommended defaults.
- Fit block table with enabled state, physical meaning, length, initial policy, attribute target, units, and advanced options.
- Basic-mode presets:
  - Optics-only quadrupole fit
  - BPM/corrector calibration fit
  - Coupling fit
  - Dispersion-enabled fit
  - Multi-stage commissioning fit
- Advanced panels:
  - algorithm: LM/GN selection, outer iterations, LM inner iterations, lambda settings;
  - SVD: threshold/cut selection and singular-value preview when available;
  - data conditioning: coupling-row removal, outlier rejection, normalization, dispersion weights;
  - physics options: fixed path length, fixed momentum, energy-shift blocks;
  - Jacobian cache: normal/skew/tilt paths, recomputation policy, metadata preview;
  - raw backend call preview.

#### Live Run Monitor page

The live run monitor is a first-class workflow page, not just a log window.

It should include:

- run phase timeline;
- progress bar per phase and per iteration;
- live chi-square plot;
- live residual RMS plot;
- latest ORM residual heatmap;
- singular-value plot after SVD is available;
- LM lambda and accepted/rejected trial indicators;
- outlier count and outlier residual plot;
- worker status, elapsed time, output directory, and latest saved artifacts;
- structured log console with filters for info, warning, error, and backend stdout.

The monitor should remain useful even before backend callbacks exist by capturing stdout/logging. Later backend-neutral progress callbacks should provide richer structured events.

#### Results page

- Fit result table grouped by block.
- Before/after ORM residual metrics.
- Chi-square history and optional delta-chi-square group contributions.
- Correction export table for quadrupoles, skew quadrupoles, BPM gains/couplings, corrector calibrations/couplings, RF, and energy-shift terms.
- Result comparison between multiple runs.
- Final ring/model save actions.
- Export reproducible Python script.

### 10.4 Dialogs

- **Mode switch dialog:** explains which advanced settings are hidden or modified when returning to Basic mode.
- **Machine profile dialog:** create/edit machine-independent defaults, element selectors, units, and dataset mappings.
- **Config editor dialog:** read-only/basic mode plus advanced editable mode for dataclass fields; changes should save to project-specific config or export a new `pyloco_config.py`.
- **Data mapping dialog:** map HDF5 datasets to ORM/noise/dispersion roles.
- **Bad BPM dialog:** import from text, paste lists, toggle by table, convert positions/indices, preview resulting matrix dimensions.
- **Jacobian cache dialog:** inspect cache shape, iteration, included dispersion, corrector kicks, and metadata.
- **Plugin manager dialog:** list installed plugins, enabled state, compatibility, trusted source, and capabilities.
- **Export dialog:** choose NumPy/HDF5/CSV/JSON/Python script outputs.
- **Preferences dialog:** units, plotting style, recent projects, compute worker defaults.

### 10.5 Plotting panels

Use Matplotlib with QtAgg initially because current analysis code already uses Matplotlib. Consider pyqtgraph later for high-speed matrix interaction.

- **ORM matrix plot:** measured, model, after-fit, residual, difference, direct/coupled blocks.
- **Response standard deviation bars:** direct/coupled/model bars inspired by existing analysis helper.
- **Residual histograms and outlier plot:** before/after residual distributions and rejected points.
- **SVD plot:** singular values with selected modes highlighted.
- **Chi-square plot:** chi-square versus iteration, LM trial history when available.
- **Fit parameter plot:** grouped block corrections, sortable/filterable.
- **Optics plot:** beta beating and dispersion plots when reference/final optics are available.
- **BPM/corrector map:** optional longitudinal `s` position view of BPMs, correctors, quads, and flagged bad BPMs.
- **Run comparison plot:** compare chi-square, residuals, and correction distributions between multiple runs.

## 11. GUI Communication with the pyLOCO Backend

### 11.1 Recommended backend API boundary

The first GUI implementation should avoid changing numerical behavior. Add a thin adapter that builds explicit request/result objects around existing functions.

Suggested request object:

```python
@dataclass
class LocoRunRequest:
    ring: Any
    project_id: str
    machine_profile: str | None
    config_module_path: Path | None
    used_bpms_ords: np.ndarray
    used_cor_ords: tuple[np.ndarray, np.ndarray]
    quads_ords: np.ndarray
    skew_ords: np.ndarray | None
    cav_ords: np.ndarray | None
    measured_orm: np.ndarray
    weights: np.ndarray
    measured_eta_x: np.ndarray | None
    measured_eta_y: np.ndarray | None
    cm_step: tuple[np.ndarray, np.ndarray]
    fit_list: list[str]
    options: dict[str, Any]
    jacobian_files: dict[str, Path | None]
```

Suggested result object:

```python
@dataclass
class LocoRunResult:
    run_id: str
    fit_results_all: list[np.ndarray]
    fit_dict_all: dict[int, dict[str, np.ndarray]]
    final_ring: Any
    orm_model_after: np.ndarray
    bpm_correction_matrix: np.ndarray
    chi2_history: list[float]
    delta_chi2_history: list[np.ndarray]
    blocks: dict[str, slice]
    logs: list[str]
    output_files: dict[str, Path]
```

### 11.2 Runtime communication pattern

1. UI validates the project and constructs a `LocoRunRequest`.
2. `LocoWorker` runs outside the GUI thread, preferably in a worker process for full LOCO runs.
3. Worker calls `load_config()` if needed, then delegates to `pyloco()`.
4. Worker captures stdout/logging into progress events.
5. Worker converts the tuple returned by `pyloco()` into `LocoRunResult`.
6. Main thread updates the Project Explorer, run monitor, result models, and plot panels.

### 11.3 Progress and cancellation

The current `pyloco()` function mainly communicates through `print()` and returns only after completion. For a better GUI, introduce optional backend-neutral hooks in a later milestone:

- `progress_callback(event: LocoEvent) -> None`
- `cancel_token.is_cancelled() -> bool`
- optional `result_callback(iteration_result)`

These hooks should be plain Python callables/objects, not Qt signals. The GUI adapter can translate them to Qt signals.

Suggested event types:

- `RunStarted`
- `PhaseStarted`
- `PhaseProgress`
- `IterationStarted`
- `JacobianStarted`
- `JacobianFinished`
- `SvdComputed`
- `LmTrialEvaluated`
- `OutliersUpdated`
- `ChiSquareUpdated`
- `PlotDataUpdated`
- `RunFinished`
- `RunFailed`
- `RunCancelled`

### 11.4 Error handling

- Convert backend exceptions into structured GUI messages with traceback details hidden under “Details”.
- Validate dimensions before running:
  - `orm_measured.shape == (nHBPM + nVBPM, nHorCOR + nVerCOR [+ dispersion column])`
  - `weights.shape == (nHBPM + nVBPM,)` or compatible column vector
  - dispersion arrays match BPM counts
  - fit block lengths match selected element counts
- Warn before live-machine or DOOCS-related actions; keep those disabled in the first release.

### 11.5 Reproducibility

Every GUI run should be exportable as a Python script that reproduces the backend call. The script should include:

- `load_config()` call;
- lattice/data loading;
- machine profile and plugin metadata;
- BPM/corrector/quadrupole selections;
- bad BPM removal;
- exact `pyloco()` keyword arguments;
- result save commands.

This preserves pyLOCO flexibility for expert users while allowing non-Python users to operate safely through the GUI.

## 12. Plugin Architecture for Future Extensions

The GUI should include a plugin architecture from the start, even if the first implementation only ships built-in plugins. Plugins allow future machines and workflows to be added without modifying the core GUI.

### 12.1 Plugin types

#### Machine profile plugins

Provide machine defaults for PETRA III, PETRA IV, EBS, FCC-ee, or local user machines:

- element selection rules;
- default units;
- default dataset mappings;
- recommended LOCO presets;
- correction safety limits;
- display labels and documentation links.

#### Measurement importer plugins

Support additional data sources:

- HDF5 variants;
- MATLAB files;
- NumPy archives;
- CSV tables;
- accelerator control-system exports;
- simulated commissioning outputs.

#### Exporter plugins

Support site-specific outputs:

- correction tables for operations;
- machine-study reports;
- archiver-compatible files;
- notebook templates;
- script templates.

#### Plot panel plugins

Add specialized diagnostics:

- coupling diagnostics;
- optics-beating summaries;
- lattice-section views;
- beta/dispersion comparison panels;
- machine-specific quality metrics.

#### Live-control plugins

Optional expert-only plugins for online measurement or correction application:

- DOOCS;
- EPICS;
- TANGO;
- simulation backends;
- future control-system adapters.

Live-control plugins must be disabled by default and require explicit user confirmation.

### 12.2 Plugin manifest

Each plugin should declare:

- plugin name and version;
- plugin API compatibility;
- plugin type/capabilities;
- supported machine profiles;
- required optional dependencies;
- whether it can access live controls;
- trust/source metadata;
- entry points for discovery.

### 12.3 Plugin safety boundaries

- Core numerical pyLOCO modules must not depend on plugins.
- Plugins should interact through stable service interfaces and typed project data.
- Live-control plugins should run behind safety confirmation and dry-run layers.
- Plugin errors should not crash the GUI; they should disable the plugin and report diagnostics.
- Project files should record plugin names and versions used for reproducibility.

## 13. Development Roadmap

### Milestone 0 — User review and design approval

- Review this design with accelerator physicists who have not previously used pyLOCO.
- Validate the workflow wording, Basic/Advanced split, and Project Explorer concept.
- Decide the first supported offline workflow, ideally one PETRA-style file-based example plus one generic AT-lattice workflow.
- Confirm required PySide6, Matplotlib QtAgg, PyYAML/JSON, and packaging dependencies.

### Milestone 1 — Project shell, Project Explorer, and Basic mode skeleton

- Add GUI package skeleton without modifying numerical behavior.
- Implement main window with Project Explorer and Basic/Advanced mode toggle.
- Implement project file load/save.
- Implement generic machine profile model.
- Implement config loader preview.
- No LOCO execution yet.

### Milestone 2 — Machine model and measurement import

- Implement lattice loading and element browser.
- Implement machine-independent selectors and dimension counters.
- Implement measurement file import previews.
- Add dataset mapping editor.
- Add bad BPM manager.

### Milestone 3 — Validation and script export

- Add dimension validation for ORM, weights, dispersion, and element selections.
- Add fit-block selection model and Basic presets.
- Add Advanced raw backend-call preview.
- Export a reproducible Python script from GUI state.
- Compare exported script with existing examples.

### Milestone 4 — First runnable LOCO GUI workflow

- Implement `LocoRunRequest`, `LocoRunResult`, and backend adapter.
- Run `pyloco()` in a worker with captured logs.
- Display live run monitor with phase status, logs, and final chi-square history.
- Support cancellation at process boundary if fine-grained cancellation is not yet available.

### Milestone 5 — Live plots and result analysis

- Add live chi-square and residual plots.
- Add measured/model/residual ORM plots.
- Add SVD plot.
- Add fit-parameter grouped plots and tables.
- Add optics plots when reference/final optics are available.
- Add HDF5/NumPy/CSV exports for fit results and corrections.

### Milestone 6 — Advanced configuration and Jacobian cache management

- Add advanced editor for `LOCOOptions`, `RMConfig`, `FitInitConfig`, and `fixed_parameters`.
- Add Jacobian cache inspection and cache reuse controls.
- Add multi-stage fit continuation using existing `continue_from_previous` arguments.
- Add delta-chi-square visualization.

### Milestone 7 — Plugin infrastructure

- Define plugin API and manifest format.
- Add Plugin Manager dialog.
- Add built-in generic machine profile plugin.
- Add example PETRA III and EBS profiles as plugins or profile files.
- Add importer/exporter plugin examples.

### Milestone 8 — Backend event hooks and robust cancellation

- Add optional callback/cancel-token support to backend functions without importing Qt.
- Emit structured iteration events rather than relying on captured `print()` output.
- Add richer LM inner-loop progress and SVD selection events.
- Add unit tests for adapter request/result conversion.

### Milestone 9 — Optional live measurement integration

- Wrap DOOCS or other control-system measurement scripts behind explicit plugin interfaces.
- Add machine-protection confirmations, dry-run mode, and setpoint restore checks.
- Keep live measurement features disabled by default and separated from offline analysis.

### Milestone 10 — Packaging, documentation, and user testing

- Add optional GUI dependency group, e.g. `pip install pyLOCO[gui]`.
- Add `pyloco-gui` entry point.
- Create quick-start tutorials for generic AT workflows, PETRA-style measured data, and EBS pySC simulation.
- Run usability tests with accelerator physicists who do not routinely edit Python scripts.

## 14. Design Principles for the Target Users

- **Workflow first:** organize the interface around LOCO tasks, not repository files or Python modules.
- **Basic by default, Advanced when needed:** make the first fit approachable while preserving expert control.
- **Machine-independent:** support PETRA III, PETRA IV, EBS, FCC-ee, and future machines through profiles and plugins.
- **Transparent dimensions:** always show BPM/corrector counts, matrix shapes, units, and removed rows.
- **Live feedback:** show progress, plots, warnings, and logs during long numerical runs.
- **Safe by default:** first release should be offline/file-based; live machine actions require explicit expert plugins.
- **Reproducible:** every GUI run should map to a clear `pyloco()` call and save a project snapshot.
- **Backend independence:** pyLOCO must remain usable from scripts, notebooks, and automated pipelines without GUI dependencies.
