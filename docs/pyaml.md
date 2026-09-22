# pyAML device access

pyLOCO Measure can load an existing python-accelerator-middle-layer configuration
through the **pyAML** backend. Corrector and RF frequency writes default to enabled,
supporting ORM and automatic dispersion acquisition. The default RF plant is `RF`.
Applying fitted quadrupole corrections is not implemented.

Install pyAML and the control-system plugin referenced by your configuration into
the same Python environment as pyLOCO. They are imported only when connecting.
For this workspace, the local packages are `../pyaml` and, for the supplied PETRA
simulation configuration, `../pySC2pyaml` (install with `pip install -e PATH`).

1. Open `pyloco-measure`, then select **pyAML** in Machine setup.
2. Choose your YAML configuration using **Choose YAML…**.
3. Set the mode explicitly: `design` for a configured model, or `live` for a
   configured control-system connection. A virtual accelerator also uses `live`.
4. Set array names (defaults: `BPM`, `HCorr`, `VCorr`).
   The RF plant name defaults to `RF`. Both **corrector magnet writes** and
   **RF frequency writes** are checked by default; uncheck either to disable it.
   Opening a project resets these options to enabled, but does not connect or write.
   Live-mode write access still requires confirmation on connection. For a machine
   without RF, disable RF writes and clear the RF plant name.
5. Click **Test connection**. This loads the configuration and reads all configured
   BPM positions and corrector strengths without calling any setters.
6. Select BPMs and acquire BPM noise. Corrector tables expose strength reads in
   the KICK.RBV column when refreshed; these are pyAML `strengths.get()` values,
   not an independently verified physical readback or a separate setpoint.

Writes address only the selected corrector's `strength.set()` (not the entire
array), or the selected RF plant's `frequency.set()`. Values are converted from
radians/Hz to the accessor's declared units. Nonfinite values and nonpositive RF
frequencies are rejected. Machine-specific limits/interlocks remain the
responsibility of the pyAML configuration and control system; verify these before
enabling writes. Acquisitions attempt restoration on completion, failure and
cancellation and report restoration failures. This is not a hardware safety guarantee.

For `../p3_example/petra3_conf.pyaml.yaml`, use mode `live` and the default array
names. That file configures a pySC server at localhost:13131; start the matching
server separately. The adapter does not start a server or modify your YAML.
Resolve any configuration dependencies (including referenced files) before use.
Only load trusted configurations: pyAML configurations select Python classes.

The adapter preserves configured device order, batches reads by array, rejects
invalid shapes/nonfinite data, and converts declared position/strength units to
metres/radians. Unknown units fail rather than being guessed. Configuration
settings are saved in Measure projects; reopening requires **Test connection**
again. Editing settings immediately invalidates the old inventory and connection.

Implementation: `pyLOCO/control_system/pyaml_adapter.py`, registered through
`InterfaceRegistry` and selected by `pyLOCO/measure/main_window.py`.
The shared registry exposes this adapter, but the Correct GUI's machine-write
workflow is not implemented for pyAML.
