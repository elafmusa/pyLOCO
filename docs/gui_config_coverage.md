# GUI configuration coverage

This document maps the current public configuration and backend dataclasses to the desktop GUI. The source of truth is `pyLOCO/config.py`, `pyLOCO/user_config.py`, and `pyLOCO/measured_machine/workflow.py`. “Preserved” means a YAML value survives load/save but has no dedicated editor.

| Config area | Backend/config field | GUI support | GUI location | Mode | Notes |
|---|---|---|---|---|---|
| Machine | `machine.name` | Preserved | Project name is separately editable | Basic | Descriptive YAML metadata |
| Machine | `lattice.file` | Yes | Machine → lattice | Basic | MAT/Python lattice selection |
| Machine | `lattice.disable_6d` | Preserved | — | — | Measured-machine preparation option |
| Elements | BPM/H/V corrector/quadrupole/skew/cavity selectors | Yes | Machine → Machine Elements | Basic | GUI stores resolved lattice ordinals; YAML selector specification is preserved |
| Elements | selector `indices`, `indices_file`, `names_file`, `element_type`, `family_name`, `pattern`, `regex` | Partial | Machine → Edit/Select | Advanced | Equivalent selectors available; original YAML is preserved |
| Elements | `name_attribute`, `allow_repeated_names`, `order`, `optional` | Preserved | — | — | Applied by measured-machine workflow during YAML preparation |
| Elements | quadrupole `mode` | Partial | Fit → individual fitting | Advanced | Individual flag supported; family-group file remains preserved |
| Elements | `family_groups_file` | Preserved/pass-through | YAML only | Advanced | Intentionally YAML-only for now because the measured-machine preparation workflow owns family mapping and validation; candidate for future Advanced-mode GUI support |
| Measurement | `data.orm.file` / legacy scalar path | Yes | Measurements → file import | Basic | Relative paths resolve against YAML |
| Measurement | ORM `dataset` | Yes | Imported YAML metadata | Advanced | Used by GUI runner |
| Measurement | ORM `transpose`, `scale` | Yes | Imported YAML metadata | Advanced | Used without reducing precision |
| Measurement | ORM `row_order`, `column_order` | Preserved | — | — | No reordering algorithm is invented by GUI |
| Measurement | ORM `corrector_steps` | Yes | Fit → Response Matrix | Basic | Horizontal and vertical values |
| Measurement | ORM `remove_correctors` | Preserved | — | — | Measured-machine workflow owns name/position removal |
| Measurement | dispersion file/enable | Yes | Measurements + Fit | Basic | Required validation when enabled |
| Measurement | dispersion dataset names/scales | Yes | Imported YAML metadata | Advanced | Used by GUI runner |
| Measurement | BPM-noise file/default | Yes | Measurements | Basic | Defaults to unit sigma when absent |
| Measurement | BPM-noise dataset names/scales | Yes | Imported YAML metadata | Advanced | Used by GUI runner |
| Measurement | `bad_bpm_positions` / bad-BPM file | Yes | Measurements / config | Basic | Positions within selected BPM list |
| Measurement | `expected_counts` | Preserved | — | — | Safety checks specific to measured-machine preparation |
| RF | `frequency_hz`, `harmonic_number`, `step_hz` | Yes | Fit → Fixed Parameters / Response Matrix | Advanced | Exact numeric values retained |
| Response matrix | `calculator` | Yes | Fit → Response Matrix | Basic | Linear, Analytical (uncoupled optics), or Tracking (`Numerical` backend value) |
| Response matrix | `bidirectional` | Yes | Fit → Response Matrix | Basic | Central difference toggle |
| Response matrix | `includeDispersion` | Yes | Fit → Response Matrix | Basic | Synchronized with LOCO option |
| Response matrix | `delta_coupling`, `coupling_orm` | Yes | Fit → Response Matrix | Advanced | Numerical coupling controls |
| Response matrix | `NewVectorizedMethod` | Yes | Fit → Response Matrix | Advanced | Backend vectorized path |
| Response matrix | `fixedpathlength`, `log_info` | Yes | Fit → Response Matrix | Advanced | Lower-level controls |
| Response matrix | `HCMCoupling`, `VCMCoupling`, `Frequency`, `HarmNumber`, `RFAttr` | Yes | Fit → Fixed/advanced fields | Advanced | Literal values supported |
| Fit parameters | all 14 backend blocks in `BLOCK_ORDER` | Yes | Fit → Fit Parameters | Basic/Advanced | Group YAML switches map to canonical backend blocks |
| Fit parameters | `individuals` | Yes | Fit → Fit Parameters | Advanced | Family-vs-individual behavior |
| Initialization | `init_policy`, overrides, `init` | Yes | Fit → Parameter Initialization | Advanced | Python literals validated |
| Initialization | `CMstep`, per-corrector NPZ | Yes | Fit → Parameter Initialization | Advanced | NPZ `hor`/`ver` arrays validated |
| Initialization | quad/skew attributes and indices | Yes | Fit → Parameter Initialization | Advanced | Defaults follow backend dataclass |
| Initialization | quadrupole-tilt R1/R2 attributes and method | Yes | Fit → Parameter Initialization | Advanced | No new methods invented |
| Solver | `algorithm` | Yes | Fit → Solver | Basic | LM or GN |
| Solver | `nIter` | Yes | Fit → Solver | Basic | New iterations for resumed runs |
| Solver | `nLMIter`, `Starting_Lambda`, `max_lm_lambda`, `scaled` | Yes | Fit → Solver | Advanced | LM-only scaled control is disabled for GN |
| SVD | `svd_selection_method` | Yes | Fit → Regularization / SVD | Basic | threshold, rank, user input, interactive |
| SVD | `svd_threshold`, `cut_`, `show_svd_plot` | Yes | Fit → Regularization / SVD | Basic/Advanced | Interactive plotting retained |
| Rejection | `outlier_rejection`, `sigma_outlier` | Yes | Fit → Advanced fitting | Basic | Pre-run value validation |
| Normalization | `apply_normalization`, `normalization_mode` | Yes | Fit → Advanced fitting | Advanced | Current backend modes retained |
| Dispersion | horizontal/vertical weights | Yes | Fit → Advanced fitting | Advanced | Independent weights |
| LOCO | `plot_fit_parameters`, `auto_correct_delta`, `fixedpathlength`, `remove_coupling_` | Yes | Fit → Advanced fitting | Advanced | Passed through unchanged |
| LOCO | `skew_individuals` | Preserved/pass-through | No dedicated widget | Advanced | Candidate for future Advanced-mode GUI support |
| LOCO | `tilt_individuals` | Preserved/pass-through | No dedicated widget | Advanced | Candidate for future Advanced-mode GUI support |
| LOCO | `calculate_delta_chi2` | Preserved/pass-through | No dedicated widget | Advanced | Candidate for future Advanced-mode GUI support |
| Constraints | `enable` | Yes | Fit → Constraints | Basic | Independent of fit-parameter switches |
| Constraints | quad absolute/relative sigma, minimum sigma | Yes | Fit → Constraints | Basic/Advanced | Mutually exclusive sigma mode |
| Constraints | quad/skew default and selected weight/families | Yes | Fit → Constraints | Basic/Advanced | Common-weight workflow supports changing 1 to 5 |
| Constraints | quad/skew `weighted_families` | Yes | Fit → Constraints tables | Advanced | Add/remove/edit compact table |
| Constraints | explicit `weights`, `mask` vectors | Yes | Fit → Constraints | Advanced | Legacy/current compatibility |
| Resume | `enabled`, `directory` | Yes | Fit → Initialization / Resume | Basic | Select run or results directory |
| Resume | ring/fit-dict/fit-results filenames | Yes | Fit → Initialization / Resume | Advanced | Defaults from `FitResumeConfig` |
| Resume | previous metadata | Yes | Fit → Initialization / Resume | Basic | Iterations, final χ², blocks and timestamp when persisted |
| Output | `directory` | Yes | Project/config import/export | Basic | Used as GUI results root |
| Output | `root`, `run_name` | Preserved | — | — | Measured-machine naming convention |
| Diagnostics | measured-machine plotting/output settings | Preserved | Results workspace | — | GUI has its own non-destructive results views |
| Momentum compaction | automatic/user value | Yes | Fit → Fixed Parameters | Advanced | Valid finite scalar required |

## Round-trip policy

Known values are translated through typed GUI state. The complete source YAML is also retained and used as the base for export, so unknown future fields survive. The GUI reports paths it cannot edit in `LocoConfiguration.uneditable_fields`. Imported relative data, lattice, output, and resume paths are resolved for execution; their original source mappings remain available for lossless preservation where the GUI has not edited them.

## Resume semantics

Resume consumes the previous fitted lattice (`ring_pyloco.mat`) and fit dictionary (`fit_dict.pkl`), plus optional fit history (`fit_results.npy`). It initializes the new run from those fitted values; it does not resume an interrupted solver stack or LM inner-loop counter. `nIter` is the number of new outer corrections. Shared parameter blocks are restored, while newly selected blocks use their normal initialization policy. GUI-produced runs now save these resume-compatible artifacts.

## Deliberately non-editable fields

Machine-specific expected counts, arbitrary selector details, corrector-removal recipes, and measured-machine diagnostics are preserved but not duplicated as generic widgets. Their semantics require the YAML-driven measured-machine preparation workflow; exposing them as generic controls without that execution path would be misleading.
