# EBS MDT standalone four-stage LOCO

This example runs the 24 October 2026 EBS MDT measurement entirely from
Python. It does not create a Qt application, connect to pyAML, authorize
writes, or communicate with a machine.

The saved FIT project is the source of the lattice, exact measurement-device
order, kick metadata, fitting selections, weights, optimizer configuration and
complete Machine Mapping. The separate recipe changes only the fitted
parameter blocks and ORM block mask at each stage:

1. BPM gains and horizontal/vertical corrector calibration (diagonal ORM)
2. BPM coupling (off-diagonal ORM)
3. selected normal quadrupoles (diagonal ORM)
4. mapped skew quadrupoles plus BPM/corrector coupling (off-diagonal ORM)

Every stage after the first resumes from the preceding checkpoint through
`pyLOCO.gui.fit_workflow.execute_workflow`. The original project, measurements
and reference lattice are never modified.

## Required local scientific files

The project uses repository-relative paths to these external MDT artifacts:

| Role | Expected path from the repository root |
| --- | --- |
| BPM noise | `measurements/pyloco-bpm-noise-20261024-000753.h5` |
| RF dispersion | `measurements/pyloco-dispersion-20261024-000810.h5` |
| ORM | `measurements/pyloco-orm-20261024-001135.h5` |
| Measure session | `measurements/measurement-session-20261024-000753.pyloco-session.json` |
| AT lattice | `measurements/lattice/1b6744878147-betamodel.mat` (`betamodel`) |

These scientific inputs are intentionally not copied into the example. The
preflight checks their actual bytes, identities, ordering and metadata. It
requires 320 BPMs, 5 horizontal correctors, 5 vertical correctors, a 640×10
ORM, the 4,434-element lattice, RF frequency 352372169.3993786 Hz and harmonic
number 992. Historical 32+32 selections and 4,432-element ordinals are not
accepted.

The dispersion file uses the older negative-minus-positive on-disk convention.
The shared loader converts it in memory to the canonical positive total RF
separation. The source HDF5 file is not changed.

## Mac Terminal commands

From the MDT candidate checkout:

```bash
cd "$HOME/Desktop/pyLOCO/mdt-urgent-candidate"
source "$HOME/Desktop/pyLOCO-pyaml-meeting/.venv-pyaml/bin/activate"
export PYTHONPATH="$PWD"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/pyloco-matplotlib"
```

Validate everything without starting LOCO:

```bash
python Examples/EBS/mdt_multistage/run_ebs_mdt_multistage.py \
  --preflight-only
```

Run a short four-stage integration check (one outer and one LM iteration per
stage, with the same measurements, selections and parameter blocks):

```bash
python Examples/EBS/mdt_multistage/run_ebs_mdt_multistage.py \
  --smoke-test \
  --output "$PWD/Examples/EBS/mdt_multistage/results-smoke" \
  --session "$PWD/Examples/EBS/mdt_multistage/smoke-session.json"
```

Run the configured four-stage workflow:

```bash
python Examples/EBS/mdt_multistage/run_ebs_mdt_multistage.py \
  --output "$PWD/Examples/EBS/mdt_multistage/results" \
  --session "$PWD/Examples/EBS/mdt_multistage/fit-run-session.json"
```

Resume after an interruption. Completed stage directories are reused and are
not overwritten:

```bash
python Examples/EBS/mdt_multistage/run_ebs_mdt_multistage.py \
  --output "$PWD/Examples/EBS/mdt_multistage/results" \
  --session "$PWD/Examples/EBS/mdt_multistage/fit-run-session.json" \
  --resume
```

Use `--project` and `--recipe` to run reviewed replacements. A session path
that already exists is rejected unless `--resume` is supplied.

## Outputs

The workflow session records one immutable checkpoint per completed stage.
The existing FIT backend saves each stage into its own timestamped directory,
including:

- `ring_pyloco.mat` and `final_lattice.mat`
- `fit_dict.pkl`, `fit_dict.json`, `fit_results.npy` and `blocks.pkl`
- initial/fitted ORM artifacts and `loco_results.npz`
- `optics_results.npz`, `summary.json`, logs and run request
- `iterations/iteration_NNN/` fitted lattice, ORM, parameters, optics and
  iteration metadata
- numerical SVD/Jacobian artifacts when enabled and available

At completion, the script writes a compact comparison JSON beside the workflow
session and prints stage-by-stage χ², ORM residual RMS, βx/βy beating RMS and
the fitted-parameter change norm. Missing optional SVD data is reported as
unavailable; it is never invented or recomputed.

The runner calls the same `ProjectMetadata`, `LocoRunRequest`,
`run_loco_request` and `execute_workflow` interfaces used by FIT. Consequently
the command-line and GUI paths consume equivalent backend inputs and produce
the same result layout.
