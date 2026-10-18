# pyLOCO MDT installation and dependency audit

Audit date: 18 October 2026. This document records observed installations; it is not a promise that every dependency is available from public PyPI.

## Status labels

- **Simulator-tested:** exercised with the modern EBS pyAML/pySC simulator artifacts.
- **Code-verified:** source and entry points were inspected, but the integration was not exercised against ESRF hardware.
- **ESRF verification required:** machine configuration, units, permissions, or readbacks must be confirmed on site.

## Supported installation layers

### Core GUI installation

From a pyLOCO checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[gui]'
python -m pip check
```

The repository metadata declares Python 3.10 or newer and these runtime dependencies: `accelerator-toolbox`, `h5py`, `matplotlib`, `numpy`, `PyYAML`, and `tomli` on Python older than 3.11. The `gui` extra adds `PySide6` and `matplotlib`.

### pySC demo installation

`pyproject.toml` currently defines:

```bash
python -m pip install -e '.[gui,pysc-demo]'
```

The `pysc-demo` extra pins `accelerator-commissioning==1.5.2`. This supports the repository demo profile, but it is **not** the same dependency set as the demonstrated modern EBS pyAML simulator environment below.

### Modern EBS pyAML/pySC integration

The demonstrated local integration used private/local artifacts in addition to pyLOCO. They must not be assumed to exist on a new workstation:

- pySC2pyaml source checkout, commit `cdce2131681b840dddecd1b365723c628872d83d`
- pyAML source checkout, commit `c7cf416deb2b5f4856ca857f644b521f3c83b88d`
- modern EBS simulator bundle containing `betamodel.mat`, `betamodel_conf_pyaml_test.yaml`, and `run_pysc_realistic.py`

The observed working environment contained:

| Component | Observed version | Evidence/status | Publication guidance |
|---|---:|---|---|
| Python | 3.14.0 | Simulator-tested locally | Public metadata requires Python >=3.10; do not narrow it based only on this workstation. |
| pyLOCO | 0.4.0, source commit `2b6623246eec8bb87e8702b04c9fa0eb08ceda2e` | Simulator-tested integration worktree | Publish the final branch name and commit separately after approval. |
| Accelerator Toolbox (`accelerator-toolbox`) | 0.8.0 | Simulator-tested | Installed through pyLOCO dependency metadata. |
| pySC (`accelerator-commissioning`) | 1.5.6 | Simulator-tested with the external realistic launcher | Conflicts with pyLOCO's `pysc-demo` 1.5.2 pin and pyAML 0.3.1 metadata's 1.5.3 pin; resolve before declaring a clean public extra. |
| pyAML (`accelerator-middle-layer`) | 0.3.1 | Simulator-tested | Current source revision includes the diagnostic-device API used by pyLOCO; package metadata requests pySC 1.5.3. |
| pySC2pyaml | 0.1.0, editable local source | Simulator-tested | Treat as an external source checkout until a redistributable release/source URL is agreed. |
| PySide6 | 6.12.0 | Simulator-tested | Installed by `gui` extra. |
| NumPy | 2.5.3 | Simulator-tested | No tighter pyLOCO constraint was verified. |
| h5py | 3.16.0 | Simulator-tested | No tighter pyLOCO constraint was verified. |
| Matplotlib | 3.11.2 | Simulator-tested | No tighter pyLOCO constraint was verified. |
| PyYAML | 6.0.3 | Simulator-tested | Required for project and machine YAML. |
| Tango client | Not verified in this audit | ESRF verification required | Install through the ESRF-supported environment; confirm the exact package and version with Kostas. |

The local integration environment fails `python -m pip check` because `accelerator-middle-layer 0.3.1` declares `accelerator-commissioning==1.5.3` while 1.5.6 is installed. It was retained only as evidence of the demonstrated simulator setup. Do not publish this mismatch as a recommended clean installation.

For source-based pyAML evaluation, the inspected revision can be installed with:

```bash
python -m pip install \
  'accelerator-middle-layer[tango-pyaml] @ git+https://github.com/python-accelerator-middle-layer/pyaml.git@c7cf416deb2b5f4856ca857f644b521f3c83b88d'
```

This command has **not** been validated as a complete clean-room MDT install in this audit. It may require ESRF-provided Tango dependencies and still declares pySC 1.5.3.

## Version and import verification

Run after activating the intended environment:

```bash
python - <<'PY'
from importlib.metadata import version
from pathlib import Path
import at, pyLOCO

for package in (
    'pyLOCO', 'accelerator-toolbox', 'accelerator-middle-layer',
    'accelerator-commissioning', 'pySC2pyaml', 'PySide6',
    'numpy', 'h5py', 'matplotlib', 'PyYAML',
):
    try:
        print(f'{package}: {version(package)}')
    except Exception as exc:
        print(f'{package}: unavailable ({exc})')
print('pyLOCO import:', Path(pyLOCO.__file__).resolve())
print('pyAT import:', Path(at.__file__).resolve())
PY
python -m pip check
```

Record the output in the shift log. An unexpected import path is a release-blocking environment issue.

## Proposed public dependency work

No dependency metadata change is made by this documentation task. Before publishing a combined pyAML/pySC extra:

1. Agree one compatible pySC version across pyLOCO, pyAML, and pySC2pyaml.
2. Confirm a redistributable pySC2pyaml source or package.
3. Test a clean virtual environment with `pip check` passing.
4. Test the EBS simulator, one read-only Measure connection, FIT project relocation/open, and Correct read-only preview.
5. Add an optional extra only after those commands are reproducible.

## Public versus internal material

Suitable for the public README or installation guide:

- core `.[gui]` installation and the three console commands;
- generic simulator instructions using repository-owned profiles;
- portable `mdt_smoke` layout and relative-path project behavior;
- safety principles: explicit authorization, preview, journaling, and restoration checks;
- machine-neutral guidance for supplying a trusted YAML configuration.

Suitable for a public MDT example only after licensing and redistribution review:

- sanitized FIT projects and selection provenance;
- small, non-sensitive measurement examples;
- mapping schemas without private endpoints or personal paths;
- documented external-input placeholders for lattice and machine YAML files.

Keep internal to ESRF/local operations:

- Tango hosts, credentials, access-control details, and operator identities;
- production device endpoints and write permissions;
- nonredistributable lattice/simulator data;
- personal absolute paths;
- live correction limits and calibration decisions not yet approved;
- transaction journals and measurement/run artifacts containing operational provenance.

## Items requiring confirmation from Kostas

1. Supported Python, Tango client, pyAML, pySC, and pySC2pyaml versions for the ESRF control environment.
2. The authoritative real-machine YAML and whether it may be redistributed.
3. Corrector kick, RF frequency, BPM orbit, and quadrupole integrated-strength units and signs at the pyAML boundary.
4. Whether `B2L`/`A2L` readbacks and setpoints use the exact signed integrated-strength convention expected by Correct.
5. Production write authorization, limits, interlocks, ownership, and restoration procedure.
6. The approved public MDT branch name and immutable commit.
