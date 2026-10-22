from types import SimpleNamespace
import sys
import os
from pathlib import Path

import numpy as np
import pytest

from pyLOCO.control_system import AdapterCapability, InterfaceRegistry
from pyLOCO.control_system.pyaml_adapter import PyAMLAdapter


class Array:
    def __init__(self, names, values, units):
        self._names = names
        self.values = values
        self.units = units
        self.calls = 0
        self.positions = self.strengths = self

    def names(self):
        return self._names

    def get(self):
        self.calls += 1
        return self.values

    def unit(self):
        return self.units

    def __getitem__(self, index):
        def set_value(value):
            self.values[index] = value
        return SimpleNamespace(strength=SimpleNamespace(unit=lambda: self.units[index], get=lambda: self.values[index], set=set_value))


def make_adapter(*, bpm_hierarchy="legacy", include_legacy_with_new=False):
    bpms = Array(["B2", "B1"], [[1000, 2000], [3000, 4000]], [["nm", "nm"]] * 2)
    horizontal = Array(["H1"], [2], ["mrad"])
    vertical = Array(["V1"], [3], ["rad"])
    machine = SimpleNamespace(magnets={"HCorr": horizontal, "VCorr": vertical},
                              magnet={"H1": horizontal[0], "V1": vertical[0]})
    if bpm_hierarchy == "new":
        machine.diagnostic = SimpleNamespace(bpms={"BPM": bpms})
        if include_legacy_with_new:
            machine.bpms = {"BPM": Array(["WRONG"], [[9, 9]], [["m", "m"]])}
    elif bpm_hierarchy == "legacy":
        machine.bpms = {"BPM": bpms}
    else:
        raise ValueError(bpm_hierarchy)
    return PyAMLAdapter("unused.yaml", accelerator=SimpleNamespace(live=machine), rf_name="", allow_writes=False, allow_rf_writes=False), bpms


@pytest.mark.parametrize(
    ("hierarchy", "expected_path"),
    [("new", "diagnostic.bpms"), ("legacy", "bpms")],
)
def test_bpm_hierarchy_compatibility_preserves_physical_identity(hierarchy, expected_path):
    adapter, bpms = make_adapter(bpm_hierarchy=hierarchy)
    assert adapter.bpm_api_path == expected_path
    assert adapter.groups["bpm"] is bpms
    assert [device["name"] for device in adapter.list_devices("bpm")] == ["B2", "B1"]
    assert [sample.value for sample in adapter.read_many(adapter.list_channels()[:4]).values()] == [1000, 2000, 3000, 4000]


def test_current_bpm_hierarchy_is_preferred_when_both_are_exposed():
    adapter, bpms = make_adapter(bpm_hierarchy="new", include_legacy_with_new=True)
    assert adapter.groups["bpm"] is bpms
    assert adapter.bpm_api_path == "diagnostic.bpms"


def test_missing_bpm_hierarchy_has_clear_error():
    horizontal = Array(["H1"], [2], ["mrad"])
    vertical = Array(["V1"], [3], ["rad"])
    machine = SimpleNamespace(
        magnets={"HCorr": horizontal, "VCorr": vertical},
        magnet={"H1": horizontal[0], "V1": vertical[0]},
        rf={},
    )
    with pytest.raises(ValueError, match="diagnostic\\.bpms.*legacy bpms"):
        PyAMLAdapter(
            "unused.yaml",
            accelerator=SimpleNamespace(live=machine),
            rf_name="",
            allow_writes=False,
            allow_rf_writes=False,
        )


def test_batch_reads_preserve_order_and_ignore_units():
    adapter, bpms = make_adapter()
    devices = adapter.list_devices("bpm")
    assert [device["name"] for device in devices] == ["B2", "B1"]
    channels = [devices[1]["y_channel"], devices[0]["x_channel"]]
    readings = adapter.read_many(channels)
    assert list(readings) == channels
    assert [sample.value for sample in readings.values()] == pytest.approx([4000, 1000])
    assert bpms.calls == 1
    # Read-only sessions preserve backend-native values because no write
    # calibration has been authorized or validated.
    assert adapter.read(adapter.list_devices("hcor")[0]["readback_channel"]).value == 2


def test_read_only_and_invalid_channels():
    adapter, bpms = make_adapter()
    assert AdapterCapability.WRITE not in adapter.capabilities
    with pytest.raises(PermissionError):
        adapter.write(adapter.list_devices("hcor")[0]["setpoint_channel"], .001)
    with pytest.raises(PermissionError):
        adapter.require(AdapterCapability.WRITE)
    with pytest.raises(KeyError):
        adapter.read_many([adapter.list_channels()[0], "unknown"])
    assert bpms.calls == 0


@pytest.mark.parametrize("values", [[[1, 2]], [[np.nan, 2], [3, 4]]])
def test_rejects_invalid_orbits(values):
    adapter, bpms = make_adapter()
    bpms.values = values
    with pytest.raises(ValueError, match="Invalid pyAML"):
        adapter.test_connection()


def test_accepts_unknown_units():
    adapter, bpms = make_adapter()
    bpms.units = "counts"
    assert adapter.test_connection()["bpms"] == 2


def test_non_corrector_unit_accessors_are_never_called(monkeypatch):
    adapter, frequency = writable_adapter(allow_writes=True, allow_rf_writes=True)
    def fail_unit():
        raise AssertionError("Unit metadata must not be queried")
    for group in adapter.groups.values():
        monkeypatch.setattr(group, "unit", fail_unit)
    monkeypatch.setattr(frequency, "unit", fail_unit)
    assert adapter.test_connection()["bpms"] == 2
    channel = adapter.list_devices("hcor")[0]["setpoint_channel"]
    assert adapter.write(channel, .123).value == pytest.approx(.123)
    adapter.set_rf_frequency(500e6 + 10)
    assert adapter.get_rf_frequency() == 500e6 + 10


def test_registry_requires_explicit_configuration():
    with pytest.raises(ValueError, match="configuration file"):
        InterfaceRegistry().create("pyaml")


def test_invalid_mode_rejected_before_loading():
    with pytest.raises(ValueError, match="mode"):
        PyAMLAdapter("missing.yaml", mode="typo")


def test_missing_design_machine_has_clear_error():
    with pytest.raises(ValueError, match="no design machine"):
        PyAMLAdapter("unused.yaml", mode="design", accelerator=SimpleNamespace(design=None))


def test_registry_loads_selected_file_and_mode(monkeypatch):
    import sys

    adapter, _ = make_adapter()
    calls = []
    accelerator = SimpleNamespace(live=adapter.machine)
    def load(path):
        calls.append(path)
        return accelerator
    monkeypatch.setitem(sys.modules, "pyaml.accelerator", SimpleNamespace(Accelerator=SimpleNamespace(load=load)))
    session = InterfaceRegistry(pyaml_options={"config_path": "selected.yaml", "mode": "live", "rf_name": "", "allow_writes": False, "allow_rf_writes": False}).create("pyaml")
    assert calls == ["selected.yaml"]
    assert session.adapter.machine is accelerator.live
    assert session.adapter.test_connection()["bpms"] == 2


class Frequency:
    def __init__(self):
        self.value = 500e6
        self.writes = []

    def get(self):
        return self.value

    def set(self, value):
        self.writes.append(value)
        self.value = value

    def unit(self):
        return "MHz"


def writable_adapter(**options):
    base, bpms = make_adapter()
    frequency = Frequency()
    base.machine.rf = {"RF": SimpleNamespace(frequency=frequency)}
    simulator_attested=options.pop("simulator_attested",bool(options.get("allow_get_only_restoration",False)))
    if simulator_attested:
        base.accelerator.simulator_identity=lambda:{"verified":True,"backend":"pySC","instance_id":"pytest-local-pysc","configuration_sha256":"1"*64,"evidence":"test runtime attestation"}
    options = {"allow_writes": False, "allow_rf_writes": False, **options}
    adapter = PyAMLAdapter("unused.yaml", accelerator=base.accelerator, rf_name="RF", **options)
    return adapter, frequency


class DiagnosticAccessor:
    def __init__(self, setpoint, readback, unit):
        self.setpoint=setpoint; self.readback_value=readback; self.unit_name=unit
    def get(self):return self.setpoint
    def readback(self):return self.readback_value
    def unit(self):return self.unit_name


def test_corrector_diagnostics_report_only_exposed_quantities_with_metadata_units():
    adapter, _=writable_adapter(allow_writes=True,corrector_values_in_radians=True)
    magnet=adapter.machine.magnet["H1"]
    magnet.strength=DiagnosticAccessor(2.0,1.9,"mrad")
    magnet.hardware=DiagnosticAccessor(12.0,11.8,"A")
    values=adapter.read_corrector_diagnostics("H1")
    assert values["KICK.SP"]=={"status":"available","value":.002,"unit":"rad","error":""}
    assert values["KICK.RBV"]=={"status":"available","value":.0019,"unit":"rad","error":""}
    assert values["CURRENT.SP"]=={"status":"available","value":12.0,"unit":"A","error":""}
    assert values["CURRENT.RBV"]=={"status":"available","value":11.8,"unit":"A","error":""}


def test_corrector_diagnostics_do_not_duplicate_setpoint_as_missing_readback_or_current():
    adapter,_=writable_adapter(allow_writes=True,corrector_values_in_radians=True)
    values=adapter.read_corrector_diagnostics("H1")
    assert values["KICK.SP"]["status"]=="available" and values["KICK.SP"]["unit"]=="rad"
    assert values["KICK.RBV"]["status"]=="unsupported"
    assert values["CURRENT.SP"]["status"]=="unsupported"
    assert values["CURRENT.RBV"]["status"]=="unsupported"


def test_pyaml_get_is_adapter_value_without_runtime_attestation():
    adapter,_=writable_adapter(allow_writes=True,corrector_values_in_radians=True)
    evidence=adapter.corrector_restoration_evidence("H1")
    value,read_evidence=adapter.read_corrector_readback("H1")
    assert evidence["level"]=="adapter_value"
    assert "independent readback unavailable" in evidence["source"]
    assert read_evidence==evidence and value==pytest.approx(.002)


def test_get_evidence_does_not_require_caller_flag_or_runtime_attestation():
    adapter,_=writable_adapter(allow_writes=True,corrector_values_in_radians=True,allow_get_only_restoration=True,simulator_attested=False)
    assert adapter.corrector_restoration_evidence("H1")["level"]=="adapter_value"


def test_alternative_runtime_attestation_requires_fresh_get_capability():
    base,_=make_adapter()
    base.accelerator.simulator_identity=lambda:{"verified":True,"backend":"esrf-simulator","instance_id":"sim-1","configuration_sha256":"2"*64,"strength_get_is_fresh":True,"evidence":"public runtime capability"}
    adapter=PyAMLAdapter("unused.yaml",accelerator=base.accelerator,rf_name="",allow_writes=True,allow_rf_writes=False,corrector_values_in_radians=True)
    assert adapter.simulator_identity["backend"]=="esrf-simulator"
    assert adapter.corrector_restoration_evidence("H1")["level"]=="adapter_value"
    other,_=make_adapter(); other.accelerator.simulator_identity=lambda:{"verified":True,"backend":"esrf-simulator","instance_id":"sim-2","configuration_sha256":"3"*64}
    blocked=PyAMLAdapter("unused.yaml",accelerator=other.accelerator,rf_name="",allow_writes=True,allow_rf_writes=False,corrector_values_in_radians=True)
    assert blocked.corrector_restoration_evidence("H1")["level"]=="adapter_value"


@pytest.mark.parametrize("bad_value",[np.nan,np.inf])
def test_pyaml_get_verification_rejects_nonfinite_values(bad_value):
    adapter,_=writable_adapter(allow_writes=True,corrector_values_in_radians=True)
    adapter.machine.magnet["H1"].strength=SimpleNamespace(
        get=lambda:bad_value,set=lambda value:None,unit=lambda:"rad")
    with pytest.raises(RuntimeError,match="non-finite"):
        adapter.read_corrector_readback("H1")


def test_pyaml_get_verification_reports_failed_read():
    adapter,_=writable_adapter(allow_writes=True,corrector_values_in_radians=True)
    def fail():raise OSError("simulator unavailable")
    adapter.machine.magnet["H1"].strength=SimpleNamespace(
        get=fail,set=lambda value:None,unit=lambda:"rad")
    with pytest.raises(RuntimeError,match="verification read failed.*simulator unavailable"):
        adapter.read_corrector_readback("H1")


def test_pysc_runtime_identity_is_bound_to_configuration_hash(tmp_path,monkeypatch):
    class RuntimePySC:pass
    monkeypatch.setitem(sys.modules,"pySC2pyaml",SimpleNamespace())
    monkeypatch.setitem(sys.modules,"pySC2pyaml.controlsystem",SimpleNamespace(pySCControlSystem=RuntimePySC))
    config=tmp_path/"simulator.yaml"; config.write_text("test fixture\n")
    base,_=make_adapter(); runtime=RuntimePySC(); base.accelerator.controls=lambda:{"live":runtime}
    adapter=PyAMLAdapter(config,accelerator=base.accelerator,rf_name="",allow_writes=True,allow_rf_writes=False,corrector_values_in_radians=True,allow_get_only_restoration=True)
    identity=adapter.simulator_identity
    assert identity["verified"] is True and identity["backend"]=="pysc2pyaml"
    assert len(identity["configuration_sha256"])==64 and identity["instance_id"]
    other=tmp_path/"other.yaml"; other.write_text("different fixture\n")
    base2,_=make_adapter(); base2.accelerator.controls=lambda:{"live":RuntimePySC()}
    adapter2=PyAMLAdapter(other,accelerator=base2.accelerator,rf_name="",allow_writes=True,allow_rf_writes=False,corrector_values_in_radians=True,allow_get_only_restoration=True)
    assert adapter2.simulator_identity["instance_id"]!=identity["instance_id"]


def test_pyaml_uses_public_independent_strength_readback_when_available():
    adapter,_=writable_adapter(allow_writes=True,corrector_values_in_radians=True)
    adapter.machine.magnet["H1"].strength=DiagnosticAccessor(2.0,1.9,"mrad")
    evidence=adapter.corrector_restoration_evidence("H1")
    value,read_evidence=adapter.read_corrector_readback("H1")
    assert evidence["level"]=="independent"
    assert read_evidence==evidence
    assert value==pytest.approx(.0019)


def test_single_magnet_write_converts_si_radians_to_declared_unit():
    adapter, frequency = writable_adapter(allow_writes=True, corrector_values_in_radians=True)
    channel = adapter.list_devices("hcor")[0]["setpoint_channel"]
    assert adapter.write(channel, .004).value == .004
    assert adapter.groups["hcor"].values == [4.0]
    assert adapter.groups["vcor"].values == [3]
    with pytest.raises(ValueError):
        adapter.write(channel, np.nan)
    with pytest.raises(PermissionError):
        adapter.write(adapter.list_channels()[0], 1)
    with pytest.raises(PermissionError):
        adapter.set_rf_frequency(500e6)
    assert frequency.writes == []


def test_corrector_writes_reject_non_angular_strength_units():
    base, _ = make_adapter()
    base.groups["hcor"].units[0] = "1/m"
    with pytest.raises(ValueError, match="not a verified angular unit"):
        PyAMLAdapter("unused.yaml", accelerator=base.accelerator, rf_name="", allow_writes=True, allow_rf_writes=False, corrector_values_in_radians=True)


def test_magnet_write_and_readback_never_access_arrays(monkeypatch):
    adapter, _ = writable_adapter(allow_writes=True)
    def fail_array(*args):
        raise AssertionError("Magnet operations must not access arrays")
    monkeypatch.setattr(Array, "__getitem__", fail_array)
    for kind in ("hcor", "vcor"):
        monkeypatch.setattr(adapter.groups[kind], "get", fail_array)
        monkeypatch.setattr(adapter.groups[kind], "set", fail_array, raising=False)
    adapter.groups["vcor"].values[0] = np.nan
    channel = adapter.list_devices("hcor")[0]["setpoint_channel"]
    assert adapter.write(channel, .004).value == .004
    assert adapter.read(channel).value == .004
    assert np.isnan(adapter.groups["vcor"].values[0])
    adapter.groups["hcor"].values[0] = np.nan
    with pytest.raises(ValueError, match="magnet reading for H1"):
        adapter.read(channel)


def test_rf_read_write_ignores_units_and_requires_opt_in():
    adapter, frequency = writable_adapter(allow_rf_writes=True)
    assert adapter.get_rf_frequency() == 500e6
    adapter.set_rf_frequency(500e6 + 1000)
    assert frequency.value == 500e6 + 1000
    for value in (0, -1, np.inf, np.nan):
        with pytest.raises(ValueError):
            adapter.set_rf_frequency(value)
    assert len(frequency.writes) == 1
    assert AdapterCapability.WRITE not in adapter.capabilities


def test_orm_restores_pyaml_correctors():
    from pyLOCO.measure.acquisition import BpmDevice, CorrectorDevice, ORMAcquirer
    adapter, _ = writable_adapter(
        allow_writes=True,
        corrector_values_in_radians=True,
        simulator_attested=False,
    )
    assert adapter.simulator_identity is None
    assert adapter.corrector_restoration_evidence("H1")["level"] == "adapter_value"
    acquirer = ORMAcquirer(adapter,
        [BpmDevice(**item) for item in adapter.list_devices("bpm")],
        [CorrectorDevice(**item) for item in adapter.list_devices("hcor")], [])
    result = acquirer.acquire([1e-6], [], readings=2, delay_seconds=0)
    assert result.restoration_status == ("restored",)
    assert result.intermediate_restoration_status == ("restored",)
    original = result.original_setpoints_rad[0]
    assert result.actual_state_a_rad == pytest.approx((original + 0.5e-6,))
    assert result.actual_state_b_rad == pytest.approx((original - 0.5e-6,))
    assert adapter.groups["hcor"].values == pytest.approx([2])


def test_installed_pysc2pyaml_runtime_authorizes_get_evidence_and_restores():
    """Exercise the production identity check without contacting or writing a server."""
    runtime_module=pytest.importorskip("pySC2pyaml.controlsystem")
    config=Path(os.environ.get("PYLOCO_PYSC_PYAML_CONFIG",""))
    if not config.is_file():pytest.skip("verified local pySC2pyaml configuration is not installed")
    runtime=object.__new__(runtime_module.pySCControlSystem)
    bpms=Array(["B1"],[[0.0,0.0]],[["m","m"]])
    horizontal=Array(["H1"],[2e-6],["rad"]); vertical=Array(["V1"],[3e-6],["rad"])
    machine=SimpleNamespace(
        diagnostic=SimpleNamespace(bpms={"BPM":bpms}),
        magnets={"HCorr":horizontal,"VCorr":vertical},
        magnet={"H1":horizontal[0],"V1":vertical[0]}, rf={},
    )
    accelerator=SimpleNamespace(live=machine,controls=lambda:{"live":runtime})
    adapter=PyAMLAdapter(config,accelerator=accelerator,rf_name="",allow_writes=True,
        allow_rf_writes=False,corrector_values_in_radians=True,allow_get_only_restoration=True)
    assert adapter.simulator_identity["backend"]=="pysc2pyaml"
    assert adapter.corrector_restoration_evidence("H1")["level"]=="adapter_value"
    from pyLOCO.measure.acquisition import BpmDevice,CorrectorDevice,ORMAcquirer
    result=ORMAcquirer(adapter,[BpmDevice(**adapter.list_devices("bpm")[0])],
        [CorrectorDevice(**adapter.list_devices("hcor")[0])],[]).acquire(
            [1e-6],[],direction="positive",readings=2,delay_seconds=0,settling_delay_seconds=0)
    assert result.restoration_status==("restored",)
    assert adapter.corrector_restoration_evidence("H1")["level"]=="adapter_value"
    assert horizontal.values==pytest.approx([2e-6])


@pytest.mark.parametrize("cancelled", [False, True])
def test_rf_measurement_restores_original(cancelled):
    from threading import Event
    from pyLOCO.measure.acquisition import AcquisitionCancelled, BpmDevice
    from pyLOCO.measure.automatic import AutomaticDispersionAcquirer
    adapter, frequency = writable_adapter(allow_rf_writes=True)
    acquirer = AutomaticDispersionAcquirer(adapter, [BpmDevice(**item) for item in adapter.list_devices("bpm")])
    cancel = Event()
    def status(stage, details):
        if cancelled and details.get("label") == "positive":
            cancel.set()
    if cancelled:
        with pytest.raises(AcquisitionCancelled):
            acquirer.acquire(1000, 2, 0, cancel_event=cancel, status=status)
    else:
        result = acquirer.acquire(1000, 2, 0)
        assert result.restoration_status == "restored"
    assert frequency.value == 500e6
    assert frequency.writes[-1] == 500e6


def test_rf_failed_restore_is_reported_even_on_cancellation():
    from threading import Event
    from pyLOCO.measure.acquisition import AcquisitionCancelled, BpmDevice
    from pyLOCO.measure.automatic import AutomaticDispersionAcquirer
    adapter, frequency = writable_adapter(allow_rf_writes=True)
    cancel = Event()
    def status(stage, details):
        if details.get("label") == "positive":
            cancel.set()
        if stage == "restoring":
            def fail(value):
                raise RuntimeError("RF unavailable")
            frequency.set = fail
    acquirer = AutomaticDispersionAcquirer(adapter, [BpmDevice(**item) for item in adapter.list_devices("bpm")])
    with pytest.raises(AcquisitionCancelled) as error:
        acquirer.acquire(1000, 2, 0, cancel_event=cancel, status=status)
    assert error.value.restoration_status == "restore_failed"
    assert error.value.restoration_diagnostics["status"] == "restore_failed"
    assert error.value.restoration_diagnostics["final_readback_hz"] is None


def test_rf_writes_require_named_plant():
    base, _ = make_adapter()
    with pytest.raises(ValueError, match="RF plant"):
        PyAMLAdapter("unused.yaml", accelerator=base.accelerator, rf_name="", allow_rf_writes=True)


def test_default_write_capabilities_and_rf_name():
    base, frequency = writable_adapter()
    adapter = PyAMLAdapter("unused.yaml", accelerator=base.accelerator)
    assert AdapterCapability.WRITE in adapter.capabilities
    assert AdapterCapability.RF_WRITE in adapter.capabilities
    assert adapter.rf is frequency
    assert adapter.machine is base.accelerator.live
    assert frequency.writes == []


def test_orm_cancellation_restores_changed_corrector():
    from threading import Event
    from pyLOCO.measure.acquisition import BpmDevice, CorrectorDevice, ORMAcquirer, AcquisitionCancelled
    adapter, _ = writable_adapter(
        allow_writes=True,
        corrector_values_in_radians=True,
        simulator_attested=False,
    )
    cancel = Event()
    original_write = adapter.write
    def write(channel, value):
        result = original_write(channel, value)
        cancel.set()
        return result
    adapter.write = write
    acquirer = ORMAcquirer(adapter,
        [BpmDevice(**item) for item in adapter.list_devices("bpm")],
        [CorrectorDevice(**item) for item in adapter.list_devices("hcor")], [])
    with pytest.raises(AcquisitionCancelled) as error:
        acquirer.acquire([1e-6], [], readings=2, delay_seconds=0, cancel_event=cancel)
    assert error.value.restoration_status == "restored"
    assert adapter.groups["hcor"].values == pytest.approx([2])
