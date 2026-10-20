# pyLOCO Measure backend diagnostics

This guide describes the current source paths used during ESRF MDT troubleshooting. It intentionally omits credentials, complete machine configurations, and production endpoint values.

## Call-flow map

```text
MeasureMainWindow (GUI)
  ├─ Test connection → ConnectionTestWorker
  │    → InterfaceRegistry.create()
  │    → PyAMLAdapter.__init__() → public pyAML Accelerator/live accessors
  │    → adapter.test_connection()
  ├─ BPM noise → BpmNoiseAcquirer.acquire() → adapter.read_many()
  ├─ RF dispersion → AutomaticDispersionAcquirer.acquire()
  │    → adapter.get_rf_frequency()/set_rf_frequency()
  │    → cancellation-safe RF restoration and comparison with final getter value
  ├─ ORM → ORMAcquirer.acquire()
  │    → adapter.write(setpoint) → settle → readback evidence → BPM reads
  │    → non-cancellable restoration → evidence/tolerance classification
  ├─ reference comparison → exact Device mapping → model_orm()/model_dispersion()
  └─ save → write_bpm_noise()/write_dispersion()/write_orm()
       → validate_measurement_file() → save_session()
```

## Important files and functions

| Operation | First source/function | Downstream boundary and tests |
|---|---|---|
| Test Connection | `pyLOCO/measure/main_window.py`: `MeasureMainWindow._test_connection`, `ConnectionTestWorker.run`, `_connection_test_completed` | `pyLOCO/control_system/backends.py`: `InterfaceRegistry.create`; `pyLOCO/control_system/pyaml_adapter.py`: `PyAMLAdapter.__init__`, `test_connection`; tests: `test_measure_pyaml.py`, `test_pyaml_adapter.py` |
| BPM noise | `pyLOCO/measure/main_window.py`: `start_acquisition` | `pyLOCO/measure/acquisition.py`: `BpmNoiseAcquirer.acquire` → adapter `read_many`; tests: `test_measure_milestone2.py` |
| Automatic RF dispersion | `pyLOCO/measure/main_window.py`: `_start_automatic_dispersion` | `pyLOCO/measure/automatic.py`: `AutomaticDispersionAcquirer.acquire` → public adapter `get_rf_frequency`, `set_rf_frequency`; tests: `test_measure_milestone3_dispersion.py`, `test_measure_selection_safety_focus.py` |
| RF cancellation/restoration | `pyLOCO/measure/automatic.py`: `AutomaticDispersionAcquirer.acquire` cleanup | GUI reporting: `_on_cancelled`, `_on_failed`, `_restoration_diagnostic_text`; tests above |
| ORM perturbation | `pyLOCO/measure/main_window.py`: `_start_orm`, `ORMAcquisitionWorker.run` | `pyLOCO/measure/acquisition.py`: `ORMAcquirer.acquire` → adapter `write`, corrector evidence/read method, BPM reads; tests: `test_measure_milestone4_orm.py` |
| ORM cancellation/restoration | `pyLOCO/measure/acquisition.py`: nested `restore()` in `ORMAcquirer.acquire` | Evidence is classified as independent, verified simulator adapter value, unavailable, or failed; GUI `_on_cancelled` preserves the write block on unresolved restoration; tests: `test_measure_orm_cancellation_safety.py` |
| Device selection persistence | `pyLOCO/measure/main_window.py`: `_save_selection_configuration`, `_load_selection_configuration`, `_selection_snapshot` | `pyLOCO/measure/project.py`; tests: `test_measure_selection_safety_focus.py` |
| Device-to-AT mapping | `pyLOCO/measure/main_window.py`: `_verify_device_alignment`, `_validated_alignment_ordinals` | `pyLOCO/gui/measurement_mapping.py`: `find_exact_device_mappings`, `validate_mapping`, `ordered_ordinals`; tests: `test_measure_validated_device_alignment.py` |
| Reference model comparison | `pyLOCO/measure/reference_model.py`: `model_orm`, `model_dispersion` | Ordered validated AT ordinals from the mapping path; display in `MeasureMainWindow._show_orm_result`, `_show_dispersion_result` |
| HDF5 and session saving | `pyLOCO/measure/main_window.py`: `_save_result` | `pyLOCO/data_schema/measurement.py`: `write_bpm_noise`, `write_dispersion`, `write_orm`; `pyLOCO/data_schema/session.py`: `save_session`; schema tests in `test_milestone1_schemas.py` |

Failures are shown in the Measure status bar, acquisition log, warning/critical dialogs, and restoration diagnostic text. Saved HDF5 metadata and the session manifest provide run evidence; full machine configuration contents are not copied into diagnostic messages.

## Corrector evidence layers

```text
pyLOCO Measure
  → pyLOCO control-system adapter
    → public pyAML magnet.strength / magnet.hardware accessor
      → underlying DeviceAccess / simulator or control system
```

- **Set command issued** means the adapter accepted `write()`/`strength.set()`; it does not prove the hardware reached the target.
- **Adapter get-only verification** means the final public `strength.get()` matched the original value. It is not an independent hardware readback.
- **Independent readback verified** is used only when an authoritative public readback accessor or a distinct validated readback channel supplied the final value.
- **Unavailable/failed** blocks ORM writes before acquisition or preserves the post-failure safety block.

The installed pyAML scalar wrapper exposes `strength.get()` but not public `strength.readback()`. Real-machine ORM therefore fails closed. For the pySC2pyaml backend, source inspection verifies that the underlying `pySCDeviceAccess.get()` performs a fresh `pySC_client.read(address)` and `readback()` aliases that same server read. This is useful simulator restoration evidence but is not an independent channel. For an authoritatively verified simulator, a fresh `strength.get()` is accepted automatically after the normal corrector-write authorization and one successful connection. It is labelled **Adapter value matched — independent readback unavailable**. The local fallback recognizes an installed `pySC2pyaml.controlsystem.pySCControlSystem` runtime and binds its capability to the configuration SHA-256. Other simulator runtimes may use a public `simulator_identity()` attestation containing `verified`, `backend`, `instance_id`, `configuration_sha256`, `strength_get_is_fresh`, and evidence fields. Configuration labels, hostnames, ports, `mode: live`, or user confirmation are never sufficient. Kostas/pyAML should still expose a distinct `readback()` when an authoritative hardware readback exists, preserving units and quality/timestamp metadata.

## Simone ESRF simulator check

1. Record pyAML and control-adapter source revisions and the simulator configuration SHA-256.
2. In Python, print `type(accelerator.controls()[mode])`, `accelerator.controls().keys()`, and the public magnet `strength` methods. Do not include credentials or confidential endpoints.
3. Confirm whether the runtime is `pySC2pyaml.controlsystem.pySCControlSystem`, or whether it supplies an authoritative public simulator identity/capability attestation.
4. Read one corrector through `magnet.strength.get()` before and after a deliberately small simulator-only write, and confirm the simulator itself records the same change.
5. In Measure, authorize corrector writes through the normal safety prompt and click Test connection once. Confirm the verified simulator identity/fingerprint and the label **Adapter value matched — independent readback unavailable**.
6. Run a minimal ORM and a cancellation test. Preserve the Measure log and corrector restoration diagnostics. A successful message must say simulator adapter `get()` matched and must not say independent hardware readback.
7. If restoration is unresolved, stop writes, preserve diagnostics, and reconnect only after independently checking the simulator state.

For a non-pySC2pyaml ESRF simulator, obtain from Simone: the exact active control-system class/module, package commit, public set/get/readback semantics, configuration hash, simulator-side write/read log with timestamps, and evidence that the returned value is recomputed by the simulator rather than cached client state.

## Troubleshooting entry points

| Problem | First place to investigate |
|---|---|
| Connection timeout | `ConnectionTestWorker`, `_test_connection`, `InterfaceRegistry.create`, `PyAMLAdapter.test_connection` |
| RF value does not change | `AutomaticDispersionAcquirer.acquire`, `PyAMLAdapter.set_rf_frequency/get_rf_frequency`, configured pyAML RF plant, then external simulator logs |
| RF restoration mismatch | `pyLOCO/measure/automatic.py` cleanup and `_restoration_diagnostic_text` |
| Corrector restoration failure | `ORMAcquirer.acquire` nested `restore`, `PyAMLAdapter.corrector_restoration_evidence/read_corrector_readback` |
| Incorrect device selection | `_resolve_correctors`, `_selection_snapshot`, `pyLOCO/measure/project.py` |
| Reference-model mismatch | `_verify_device_alignment`, `pyLOCO/gui/measurement_mapping.py`, `pyLOCO/measure/reference_model.py` |
| Missing measurement file | `_save_result`, `pyLOCO/data_schema/measurement.py`, `pyLOCO/data_schema/session.py` |

For external ESRF simulator problems, record the requested setpoint, the value returned by `get()`, a genuinely independent readback if exposed, timestamps, and BPM orbit timestamps. Do not infer external simulator behavior from the local pySC compatibility server.

