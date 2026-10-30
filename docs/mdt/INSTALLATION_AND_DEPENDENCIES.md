# Installing pyLOCO for the ESRF MDT

**Status: environment audit dated 18 October 2026.** These are **observed local versions**, not a validated clean-room installation or an approved ESRF hardware environment. For the exact GUI procedure and filenames, see [ESRF_MDT_RUNBOOK.md](ESRF_MDT_RUNBOOK.md).

## 1. Identify the checkout

Use branch `integration/pyaml-gui-mdt`. The tested application-code revision is `6081c0aee5c0efd0c9ccdd4469571bfe5c8b1bc6`. A later documentation-only commit may be branch HEAD; record that HEAD as the documentation revision and confirm that `6081c0a` is its ancestor.

## 2. Standard pyLOCO GUI installation

From the correct checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[gui]'
python -m pip check
```

Launch applications separately with `pyloco-measure`, `pyloco-gui`, and `pyloco-correct`.

**If a command fails:** save the console output, check `which python` and `python -c 'import pyLOCO; print(pyLOCO.__file__)'`. Do not use an unrelated active virtual environment.

## 3. Local simulator integration: observed versions

| Package | Observed on local simulator | What this means |
|---|---|---|
| Python | **3.14.0** | pyLOCO metadata says Python >= 3.10; only this integration environment was tested here |
| pyLOCO | **0.4.0**, tested application commit `6081c0aee5c0efd0c9ccdd4469571bfe5c8b1bc6` | Published MDT application-code revision; later documentation-only HEAD does not change it |
| Accelerator Toolbox (`accelerator-toolbox`) | **0.8.0** | Tested locally |
| pyAML (`accelerator-middle-layer`) | **0.3.1**, source revision `c7cf416deb2b5f4856ca857f644b521f3c83b88d` | Source-based integration; package metadata pins pySC 1.5.3 |
| pySC (`accelerator-commissioning`) | **1.5.6** | Used in modern realistic EBS demo |
| pySC2pyaml | **0.1.0**, source revision `cdce2131681b840dddecd1b365723c628872d83d` | External editable checkout, not included automatically |
| PySide6 | **6.12.0** | GUI environment |
| NumPy / h5py / Matplotlib | **2.5.3 / 3.16.0 / 3.11.2** | Observed versions |
| PyYAML | **6.0.3** | Observed version |
| Tango client | **Not verified** | ESRF expert must confirm supported environment |

**Known blocker for a clean recommended simulator install:** pyAML 0.3.1 declares `accelerator-commissioning==1.5.3`, while the demonstrated environment installed pySC 1.5.6. Also the repository `.[pysc-demo]` extra pins pySC 1.5.2. Thus the working local simulator environment **fails `python -m pip check`**. Do not present this combination as officially compatible or hide the conflict. Resolve it with Kostas and test from a new virtual environment before publishing a unified installation extra.

The audited source revisions:
- pyAML: `c7cf416deb2b5f4856ca857f644b521f3c83b88d`
- pySC2pyaml: `cdce2131681b840dddecd1b365723c628872d83d`

Verify availability and licensing of those source repositories and the EBS simulator bundle. A local editable checkout is not proof of a publicly reproducible install.

## 4. Check the installed environment

```bash
which python
python - <<'PY'
from importlib.metadata import version
import pyLOCO, at
for name in ['pyLOCO','accelerator-toolbox','accelerator-middle-layer',
             'accelerator-commissioning','pySC2pyaml','PySide6','numpy','h5py','matplotlib','PyYAML']:
    try: print(f'{name}: {version(name)}')
    except Exception as exc: print(f'{name}: NOT AVAILABLE ({exc})')
print('pyLOCO location:', pyLOCO.__file__)
print('pyAT location:', at.__file__)
PY
python -m pip check
```

Save the output and `python -m pip freeze` with shift evidence. If dependencies conflict, ask the integration owners for a supported combination before using ESRF hardware.

## 5. Where the extra files are

| Item | Location or acquisition |
|---|---|
| pyLOCO example projects and guide | `<PYLOCO_CHECKOUT>/mdt_smoke/` and `docs/mdt/` |
| Modern realistic EBS simulator | **External** `<EBS_PYSC_BUNDLE>`; not stored in GitHub |
| Modern lattice | `<EBS_PYSC_BUNDLE>/betamodel.mat` |
| pyAML simulator config | `<EBS_PYSC_BUNDLE>/betamodel_conf_pyaml_test.yaml` |
| Simulator runner | `<EBS_PYSC_BUNDLE>/run_pysc_realistic.py` |
| Trusted live ESRF YAML / Tango | **ESRF-supplied**; not bundled or verified here |
| Package content list | `<PYLOCO_CHECKOUT>/mdt_smoke/PACKAGE_MANIFEST.md` |

Do not assume that external simulator/lattice files come with `git clone`. Verify paths before launching.

## 6. What belongs in the online repository?

**Public documentation:** core GUI installation; three app launch commands; optionally supported pyAML integration described as external; sanitized example projects/mapping schemas; portable paths; read-only safeguards.

**Only after compatibility/licensing review:** pinned optional pyAML/pySC dependencies, redistributable pySC2pyaml checkout, simulator and EBS data.

**Internal/ESRF-controlled:** production Tango hosts and credentials, machine YAML and permissions, live setpoint limits, operational journals, personal absolute paths, unapproved control-unit calibrations.

**Before recommending a new public optional extra:** agree compatible pyAML/pySC/pySC2pyaml versions, prove `pip check` passes in a clean venv, verify simulator connection and read-only Correct preview, and obtain approval for redistribution.

## 7. Questions for Kostas / ESRF

1. Which Python, pyAML, pySC, pySC2pyaml and Tango versions are approved for the machine environment?
2. Where is the official real-machine YAML, and who may distribute it?
3. What are the definitive BPM, kick, signed RF and B2L/A2L units and calibration/readback conventions?
4. What are the approved hardware-write procedures, limits, interlocks and restoration rules?
5. Which later documentation HEAD has been approved for the meeting, while retaining application revision `6081c0a`?

**Until these are answered, simulator verification does not authorize live machine correction.**

## Corrector restoration evidence

After the ordinary corrector-write authorization and one successful pyAML connection, Measure may use a fresh public `magnet.strength.get()` as **adapter-value evidence** for the approved simulator procedure. This is not an independent hardware readback, and it does not require a particular pySC runtime class or `simulator_identity()` API. Unknown and real-machine targets still require the ESRF-approved readback and operating procedure; a YAML label, hostname, port or `mode=live` is not evidence of independent physical restoration.
