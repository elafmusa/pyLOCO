from types import SimpleNamespace

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
    options = {"allow_writes": False, "allow_rf_writes": False, **options}
    adapter = PyAMLAdapter("unused.yaml", accelerator=base.accelerator, rf_name="RF", **options)
    return adapter, frequency


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
    adapter, _ = writable_adapter(allow_writes=True, corrector_values_in_radians=True)
    acquirer = ORMAcquirer(adapter,
        [BpmDevice(**item) for item in adapter.list_devices("bpm")],
        [CorrectorDevice(**item) for item in adapter.list_devices("hcor")], [])
    result = acquirer.acquire([1e-6], [], readings=2, delay_seconds=0)
    assert result.restoration_status == ("restored",)
    assert adapter.groups["hcor"].values == pytest.approx([2])


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
    from pyLOCO.measure.acquisition import BpmDevice
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
    with pytest.raises(RuntimeError, match="RF restoration failed") as error:
        acquirer.acquire(1000, 2, 0, cancel_event=cancel, status=status)
    assert error.value.restoration_status == "restore_failed"


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
    adapter, _ = writable_adapter(allow_writes=True, corrector_values_in_radians=True)
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
