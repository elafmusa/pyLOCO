# pyAML device access

pyLOCO Measure can load an existing python-accelerator-middle-layer configuration
through the **pyAML** backend. Corrector and RF frequency writes default to enabled,
supporting ORM and automatic dispersion acquisition. The default RF plant is `RF`.
Correct also supports normal and skew integrated-strength corrections (see below).

Install pyAML and the control-system plugin referenced by your configuration into
the same Python environment as pyLOCO. They are imported only when connecting.

### Tested EBS pySC/pyAML environment

The EBS demonstration has been exercised with:

- Python 3.14.0
- `accelerator-middle-layer==0.3.1`
- `accelerator-commissioning==1.5.6`
- `pySC2pyaml==0.1.0`, installed from the local `../pySC2pyaml` checkout
- `accelerator-toolbox==0.8.0`
- `numpy==2.5.3`
- `PySide6==6.12.0`

The pyAML BPM API changed in upstream PR #464. Current pyAML exposes arrays as
`machine.diagnostic.bpms`, while the tested 0.3.1 release exposes
`machine.bpms`. pyLOCO detects those attributes explicitly, prefers the current
hierarchy when both are present, and retains the 0.3.1 fallback. No array is
reordered: device names and the X/Y columns are consumed in the order returned
by the selected BPM array.

Create a dedicated environment; do not install pyLOCO's current `pysc-demo`
extra into it because that extra pins `accelerator-commissioning==1.5.2`:

```bash
python3.14 -m venv .venv-pyaml
.venv-pyaml/bin/python -m pip install --upgrade pip
.venv-pyaml/bin/python -m pip install -e '.[gui]'
.venv-pyaml/bin/python -m pip install accelerator-middle-layer==0.3.1
.venv-pyaml/bin/python -m pip install accelerator-commissioning==1.5.6
.venv-pyaml/bin/python -m pip install ../pySC2pyaml
```

There is a known metadata conflict in this exact tested combination:
`accelerator-middle-layer==0.3.1` declares
`accelerator-commissioning==1.5.3`, while the demonstrated working environment
uses 1.5.6. Consequently `pip check` reports that mismatch even though the tested
pyAML connection and BPM-noise workflow operate correctly. `pySC2pyaml==0.1.0`
does not narrow the commissioning version. Do not hide the mismatch with a
pyLOCO dependency declaration: either use the resolver-clean 1.5.3 combination,
or deliberately install 1.5.6 as the final command above and record the exception.
The durable fix is for a compatible `accelerator-middle-layer` release to widen
its commissioning requirement; that should be verified before making 1.5.6 a
normal pyLOCO optional dependency.

For testing the post-#464 API before a numbered release, install a pinned pyAML
revision together with its Tango extra. Revision `c7cf416deb2b` was inspected
for this integration (it contains merged PR #464):

```bash
python3.14 -m venv .venv-pyaml-current
.venv-pyaml-current/bin/python -m pip install --upgrade pip
.venv-pyaml-current/bin/python -m pip install -e '.[gui]'
.venv-pyaml-current/bin/python -m pip install \
  'accelerator-middle-layer[tango-pyaml] @ git+https://github.com/python-accelerator-middle-layer/pyaml.git@c7cf416deb2b'
```

That revision still declares `accelerator-commissioning==1.5.3`. Keep the
realistic EBS pySC demonstration, which is validated with 1.5.6 and
`pySC2pyaml==0.1.0`, in its separate environment until the upstream constraint
is resolved.

### Contribution boundaries

The pyAML adapter, shared backend registration, Measure connection workflow,
Correct connection/mapping/transaction workflow, and their pyAML tests originate
from Kostas's PR #35. The persistent Measure status header, compact measurement
statistics table, 1.0-second new-measurement default, and corresponding GUI tests
are the subsequent pyLOCO GUI additions. The realistic EBS machine-state example
is intentionally maintained separately from both sets of application changes.

1. Open `pyloco-measure`, then select **pyAML** in Machine setup.
2. Click **Browse…** beside **Configuration file** to select your YAML or JSON
   configuration in the file browser. Then use **Test connection** to load it.
3. Mode defaults to `live` for a configured control-system connection (including
   virtual accelerators). Change it to `design` for a configured model if needed.
4. Set array names (defaults: `BPM`, `HCorr`, `VCorr`).
   The RF plant name defaults to `RF`. Both **corrector magnet writes** and
   **RF frequency writes** are checked by default; uncheck either to disable it.
   Opening a project resets these options to enabled, but does not connect or write.
   Live-mode write access still requires confirmation on connection. For a machine
   without RF, disable RF writes and clear the RF plant name.
5. Click **Test connection**. This loads the configuration and reads all configured
   BPM positions and corrector strengths without calling any setters.
6. Select BPMs and acquire BPM noise. Corrector tables expose strength reads in
   the KICK.RBV column when refreshed; these are pyAML individual `strength.get()` values,
   not an independently verified physical readback or a separate setpoint.

BPM-noise acquisition is read-only even when the connected adapter has separately
authorized magnet or RF writes. Raw X/Y orbit arrays are stored in metres with
dimensions `(reading, BPM)` and the configured BPM-array order is preserved.
`Noise_BPMx` and `Noise_BPMy` are the per-BPM standard deviations across readings.
The default deliberately matches the historical accelerator measurement script:
`numpy.std(..., axis=0, ddof=0)` for pyLOCO's `(reading, BPM)` arrays, equivalent
to `axis=1` for the former `(BPM, reading)` layout. A sample-standard-deviation
option (`ddof=1`) is also available in the Measure form. Each HDF5 file records
its actual convention and `ddof`; validation recomputes the stored vectors using
that value. The GUI's Mean and RMS are the arithmetic mean and quadratic mean
across the resulting per-BPM standard deviations. At least two readings are
required by the acquisition workflow.

The completed BPM-noise plots use the original, zero-based BPM-array index on
the horizontal axis. Hovering or clicking either X or Y point identifies the
device from that acquired array and reports both noise values in nm. A click is
only a view selection: it highlights the same BPM in both planes and does not
change measurement data or fitting inputs. The searchable/sortable review table
stores the immutable original index separately from its current row order.
The single compact review table supports multiple persistent visual highlights;
its Highlight checkboxes correspond to emphasized markers and subtle dashed
index guides. The selected BPM's optional note is edited in the details panel.
Highlights, free-text review notes, manual flags, and threshold flags are
distinct annotations. Selecting another point changes the inspected BPM but
does not remove highlights.
The selected-BPM review panel provides explicit `Unreviewed`, `Good`, `Suspect`,
and `Bad` states, a persistent-highlight control, and notes. Table rows update
immediately and show a compact note preview with the full note in a tooltip.
Review edits are not written implicitly: **Save Review** writes the optional
HDF5 `review` group and the independent Saved/Unsaved indicator reports its
state. Advanced X/Y thresholds start Off and remain collapsed until requested.
The optional CSV export contains identities, original indices, noise values,
highlights, statuses, threshold flags, and notes.
The table's review-status filter can show all BPMs, Bad and Suspect together,
or an individual review state. It combines with name/index search and sorting,
reports the visible match count, and never changes stored annotations or BPM
identity order.
Optional threshold flags and manually checked bad-BPM entries are review
annotations in the HDF5 `review` group; they do not remove or reorder
`Noise_BPMx`, `Noise_BPMy`, `names/bpms`, or the raw readings. Automatic use of
these annotations as LOCO fitting exclusions is not implemented. Longitudinal
position plotting and AT lattice position mapping are intentionally out of
scope; BPM index is the only plot coordinate.

Magnet writes use `machine.magnet.get(name).strength.set(value)` and readbacks use
that individual magnet's `strength.get()`. Magnet arrays are used only to discover
the configured names, never for magnet reads or writes (including restoration).
RF writes use the selected plant's `frequency.set()`. All pyAML unit metadata is
ignored: BPM, corrector and RF values pass through unchanged in both directions.
pyLOCO still uses metres/radians/Hz internally and in its measurement conventions,
so the configuration must provide compatible numeric values; the adapter does not
check or convert them. Nonfinite values and nonpositive RF
frequencies are rejected. Machine-specific limits/interlocks remain the
responsibility of the pyAML configuration and control system; verify these before
enabling writes. Acquisitions attempt restoration on completion, failure and
cancellation and report restoration failures. This is not a hardware safety guarantee.

For `../p3_example/petra3_conf.pyaml.yaml`, use mode `live` and the default array
names. That file configures a pySC server at localhost:13131; start the matching
server separately. The adapter does not start a server or modify your YAML.
Resolve any configuration dependencies (including referenced files) before use.
Only load trusted configurations: pyAML configurations select Python classes.

The adapter preserves configured device order, batches BPM reads by array, rejects
invalid shapes/nonfinite data, and never queries accessor unit metadata.
Unknown or missing units do not block access. Configuration
settings are saved in Measure projects; reopening requires **Test connection**
again. Editing settings immediately invalidates the old inventory and connection.

Implementation: `pyLOCO/control_system/pyaml_adapter.py`, registered through
`InterfaceRegistry` and selected by `pyLOCO/measure/main_window.py`.

## ESRF Tango configuration (read-only connection check)

The upstream ESRF example inspected at revision `c7cf416deb2b` is
`examples/use_cases/config/esrf/esrf.yaml`. It defines the EBS lattice
`sr/lattices/ebs.mat`, Tango host `ebs-simu-3:10000`, BPM array `BPM`, horizontal
and vertical corrector arrays `HCorr` and `VCorr`, RF plant `RF`, skew array
`Skews`, and tune-correction quadrupole array `QForTune`. Therefore the Measure
defaults identify the configured BPM, corrector and RF devices exactly.

For a connection-only check, select pyAML/live, choose that file, keep the four
Measure names above, **uncheck corrector magnet writes and RF frequency writes**,
and then click **Test connection**. This path calls getters only. Its Tango BPM
catalog maps horizontal and vertical position attributes in metres. It does not
start a server, issue a correction, or call a setter.

The file still loads its live Tango control-system class even when a design
machine is selected, so install the `tango-pyaml` extra before loading it. In the
inspection environment the checkout itself was not installed (and therefore
lacked the build-generated `pyaml/_version.py`), and after bypassing only that
packaging artifact the load stopped at the missing `tango` module. Consequently
no Tango connection was attempted and no live ESRF device count is claimed here.

The realistic pySC demonstration uses the older configuration schema with
top-level `eclass:` and nested `class:` keys. Current upstream examples use
top-level and nested `type:` keys. Do not mechanically rewrite that saved
realistic configuration: its working `accelerator-middle-layer==0.3.1` /
`pySC2pyaml==0.1.0` loader owns the legacy schema. Conversely, use the upstream
`type:`-based ESRF file with current pyAML. Mixing the two schemas was the cause
of the earlier `eclass`-related loader incompatibility; it is separate from the
BPM-device hierarchy change.

## pyLOCO Correct

1. Load the FIT Results directory in Correct using **Results…**. Normal family
   fits are expanded to physical-magnet correction rows by the existing loader.
2. Select **pyAML** in the backend selector. **Machine / Mapping** adapts to
   pyAML configuration and connection. There is no separate correction tab.
3. Select the configuration file and mode (default `live`). Set **Normal
   quadrupole array** (default `Quads`) and **Skew quadrupole array** (default
   `SkewQuads`). Both arrays must exist; either can be empty if that correction
   family is unused. No BPM or RF configuration is needed. These settings are
   remembered locally after a successful connection.

For the inspected ESRF example, use `QForTune` and `Skews`, not the generic
defaults. `QForTune` contains only the configured `QD2*` and `QF1*` families;
it is not full EBS quadrupole coverage. Do not apply a general LOCO correction
against it unless the fitted-device mapping and intended physical coverage have
been reviewed explicitly.
4. Click **Connect / discover quadrupoles** and confirm live write access if
   enabled. Connection reads individual strengths but does not write them.
5. Review matching in **Correction Plan**. FIT lattice names match exact, unique
   `get_lattice_names()` identities within the corresponding array. pyAML's
   default identity is the magnet name. Array order never determines matching.
   Wildcards, indexed selectors, multi-element identities and ambiguous matches
   require an explicit mapping. Missing included mappings block preview.
6. Select inclusion and scaling in the existing Correction Plan. Use a global
   fraction greater than 0% and at most 100% (10% remains the default).
   In **Review & Validate**, click **Preview machine changes**,
   inspect targets, then **Apply…** and confirm. Repeat preview/apply to advance
   cumulative correction by the selected fraction per pass, up to 100%.
   A pass that would exceed 100% is rejected, not silently reduced. Restore
   original strengths before changing the fraction of an applied sequence.

Normal and skew pyAML strengths are assumed to be integrated K×L. Each step uses
the existing signed/scaled `final_delta` times the positive source-lattice magnet
length, with cumulative targets anchored to the original strength. No unit
metadata is read, and no hardware-current conversion is done by Correct.
Only individual `machine.magnet.get(name).strength.get()/set()` calls perform
magnet I/O; arrays provide device discovery only.

### Explicit mapping

Use **Create mapping template…** in **Machine / Mapping** and fill in only each
`control_name`, or provide a JSON/YAML file containing just the overrides needed:

```yaml
mappings:
  - lattice_ordinal: 42
    lattice_name: QF1
    control_name: QF_POWER_SUPPLY_1
  - lattice_ordinal: 57
    lattice_name: SQ1
    control_name: SKEW_POWER_SUPPLY_1
```

Ordinals are zero-based source-lattice indices, not correction-table row numbers.
Component, unit and validation status are derived and checked when the file is
loaded. Existing full-format files containing those fields remain supported.
This is the same JSON/YAML mapping schema used by pySC. Control names must
identify devices on the selected backend; names need not be identical between
backends. pyLOCO validates supplied pyAML component/unit metadata and uses
integrated B2L/A2L strengths.
Legacy pyAML files using `correction_type` and `magnet_name` remain readable.
Targets must belong to the matching configured array. Overrides take precedence
over automatic identity matching; duplicate targets, wrong-family mappings and
unknown source identities are rejected. Many lattice elements mapped to one
magnet are unsupported. Excluded unresolved rows do not prevent preview.

### Reusable validated mapping bundles

**Find machine mappings** first uses exact backend lattice identities, then a
remembered or profile-associated authoritative mapping. A reusable version-2
bundle records the reference-lattice checksum and element count, pyAML
configuration checksum, array names, inventory fingerprint, exact source and
target identities, component, unit and provenance. It is revalidated on every
load; stored status text is never evidence of validity. A changed lattice,
configuration or inventory invalidates the bundle and blocks unresolved
included corrections. Legacy JSON/YAML mappings remain loadable but must be
revalidated against the current lattice and inventory before use.

### Restoration and recovery

Writes require a fresh preview and separate confirmation. Before application,
Correct rechecks configuration and source hashes, inventory, mapping and current
strengths. It journals each pending write before issuing it and verifies readback
with relative tolerance 1e-10 and absolute tolerance 1e-12.

**Cancel / restore** stops between magnet operations and attempts restoration;
it cannot interrupt a blocking control-system call. Failures also trigger
restoration, including the magnet whose setter raised. **Restore original
strengths** returns all changed magnets to the original pre-correction snapshot,
not merely to the previous cumulative pass. Failed restoration is shown explicitly
and blocks further correction or connection changes. Closing Correct keeps pyAML
strengths unchanged; restoration is optional. The journal remains available for
later recovery and restoration after reconnecting to the same configuration.
Closing is blocked only while a pyAML operation is actively running.

Durable journals are written under `correction-transactions/pyaml` relative to
the working directory. After a crash, reconnect explicitly to the same unchanged
configuration and use **Recover journal…**, then **Restore original strengths**.
Recovery does not write until Restore is confirmed. Configuration identity includes
the root file hash, absolute path, mode, array names and discovered membership;
keep any externally referenced configuration/calibration files unchanged as well.
Journals are trusted local recovery artifacts, not signed machine authorization.

Machine-specific limits, interlocks and permissions remain the responsibility of
the configured control system. A restoration attempt is not a hardware safety
guarantee. Existing pySC workflows remain separate and unchanged.
