# pyAML device access

pyLOCO Measure can load an existing python-accelerator-middle-layer configuration
through the **pyAML** backend. Corrector and RF frequency writes default to enabled,
supporting ORM and automatic dispersion acquisition. The default RF plant is `RF`.
Correct also supports normal and skew integrated-strength corrections (see below).

Install pyAML and the control-system plugin referenced by your configuration into
the same Python environment as pyLOCO. They are imported only when connecting.
For this workspace, the local packages are `../pyaml` and, for the supplied PETRA
simulation configuration, `../pySC2pyaml` (install with `pip install -e PATH`).

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

Use **Generate mapping template…** in **Machine / Mapping** and fill in each `control_name`, or
provide a JSON/YAML file containing just the overrides needed:

```yaml
mappings:
  - lattice_ordinal: 42
    lattice_name: QF1
    control_name: QF_POWER_SUPPLY_1
    component: B2L
    unit: m^-1
  - lattice_ordinal: 57
    lattice_name: SQ1
    control_name: SKEW_POWER_SUPPLY_1
    component: A2L
    unit: m^-1
```

Ordinals are zero-based source-lattice indices, not correction-table row numbers.
This is the same JSON/YAML mapping schema used by pySC. Control names must
identify devices on the selected backend; names need not be identical between
backends. pyAML ignores unit metadata and always uses integrated strengths.
Legacy pyAML files using `correction_type` and `magnet_name` remain readable.
Targets must belong to the matching configured array. Overrides take precedence
over automatic identity matching; duplicate targets, wrong-family mappings and
unknown source identities are rejected. Many lattice elements mapped to one
magnet are unsupported. Excluded unresolved rows do not prevent preview.

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
