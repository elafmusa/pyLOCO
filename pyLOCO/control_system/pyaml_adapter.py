"""Capability-gated bridge from configured pyAML arrays to the measurement GUI."""
from __future__ import annotations

from time import time

import numpy as np

from .adapters import AdapterCapability, ChannelSample, WritableAdapter


class PyAMLAdapter(WritableAdapter):
    def __init__(self, config_path, *, mode="design", bpm_group="BPM",
                 hcor_group="HCorr", vcor_group="VCorr", accelerator=None,
                 rf_name="RF", allow_writes=True, allow_rf_writes=True):
        if mode not in ("design", "live"):
            raise ValueError("pyAML mode must be design or live")
        if accelerator is None:
            from pyaml.accelerator import Accelerator
            accelerator = Accelerator.load(str(config_path))
        self.accelerator = accelerator
        self.machine = getattr(accelerator, mode)
        if self.machine is None:
            raise ValueError(f"The pyAML configuration has no {mode} machine")
        self.allow_writes = bool(allow_writes)
        self.allow_rf_writes = bool(allow_rf_writes)
        self.rf = self.machine.rf.get(rf_name).frequency if rf_name else None
        if self.allow_rf_writes and self.rf is None:
            raise ValueError("Select an RF plant before enabling RF writes")
        self.groups = {
            "bpm": self.machine.bpms.get(bpm_group),
            "hcor": self.machine.magnets.get(hcor_group),
            "vcor": self.machine.magnets.get(vcor_group),
        }
        for kind, group in self.groups.items():
            if group is None:
                raise ValueError(f"The pyAML configuration has no selected {kind} array")
        self._catalog = {}
        self._channels = {}
        for kind, group in self.groups.items():
            names = tuple(group.names())
            if len(set(names)) != len(names):
                raise ValueError(f"Duplicate names in pyAML {kind} group")
            devices = []
            for index, name in enumerate(names):
                if kind == "bpm":
                    channels = [f"PYAML:BPM:{name}:{plane}" for plane in ("X", "Y")]
                    for plane, channel in enumerate(channels):
                        self._channels[channel] = (kind, index, plane)
                    devices.append(dict(name=name, x_channel=channels[0], y_channel=channels[1]))
                else:
                    channel = f"PYAML:{kind}:{name}:STRENGTH"
                    self._channels[channel] = (kind, index, None)
                    devices.append(dict(name=name, setpoint_channel=channel,
                                        readback_channel=channel,
                                        plane="Horizontal" if kind == "hcor" else "Vertical"))
            self._catalog[kind] = tuple(devices)

    @property
    def capabilities(self):
        result = {AdapterCapability.READ, AdapterCapability.BATCH_READ}
        if self.allow_writes:
            result.add(AdapterCapability.WRITE)
        if self.rf is not None:
            result.add(AdapterCapability.RF_READ)
            if self.allow_rf_writes:
                result.add(AdapterCapability.RF_WRITE)
        return frozenset(result)

    @staticmethod
    def _scalar(value):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("Write value must be finite")
        return value

    def write(self, channel, value):
        self.require(AdapterCapability.WRITE)
        kind, index, plane = self._channels[channel]
        if kind == "bpm":
            raise PermissionError("BPM channels are read only")
        strength = self.groups[kind][index].strength
        factors = {"rad": 1.0, "mrad": 1e-3, "urad": 1e-6, "µrad": 1e-6}
        unit = strength.unit()
        if unit not in factors:
            raise ValueError(f"Unsupported pyAML corrector unit: {unit}")
        target = self._scalar(self._scalar(value) / factors[unit])
        strength.set(target)
        return self.read(channel)

    def _rf_scale(self):
        unit = self.rf.unit()
        factors = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}
        if unit not in factors:
            raise ValueError(f"Unsupported pyAML RF unit: {unit}")
        return factors[unit]

    def get_rf_frequency(self):
        self.require(AdapterCapability.RF_READ)
        value = self._scalar(self.rf.get()) * self._rf_scale()
        if value <= 0:
            raise ValueError("RF frequency must be positive")
        return value

    def set_rf_frequency(self, frequency_hz):
        self.require(AdapterCapability.RF_WRITE)
        value = self._scalar(frequency_hz)
        if value <= 0:
            raise ValueError("RF frequency must be positive")
        self.rf.set(value / self._rf_scale())

    def list_devices(self, kind):
        return self._catalog.get(kind, ())

    def list_channels(self):
        return tuple(self._channels)

    def read(self, channel):
        return self.read_many((channel,))[channel]

    def read_many(self, channels):
        requested = tuple(channels)
        for channel in requested:
            if channel not in self._channels:
                raise KeyError(f"Unknown pyAML channel: {channel}")
        arrays = {}
        for kind in dict.fromkeys(self._channels[channel][0] for channel in requested):
            group = self.groups[kind]
            values = np.asarray((group.positions if kind == "bpm" else group.strengths).get(), dtype=float)
            expected = (len(self._catalog[kind]), 2) if kind == "bpm" else (len(self._catalog[kind]),)
            if values.shape != expected or not np.all(np.isfinite(values)):
                raise ValueError(f"Invalid pyAML {kind} readings: expected finite array {expected}, got {values.shape}")
            accessor = group.positions if kind == "bpm" else group.strengths
            units = np.asarray(accessor.unit())
            factors = {"m": 1.0, "mm": 1e-3, "um": 1e-6, "µm": 1e-6, "nm": 1e-9} if kind == "bpm" else {"rad": 1.0, "mrad": 1e-3, "urad": 1e-6, "µrad": 1e-6}
            try:
                scales = np.array([factors[str(unit)] for unit in units.flat]).reshape(units.shape)
                if kind == "bpm" and scales.shape == (expected[0],):
                    scales = scales[:, None]
                arrays[kind] = values * np.broadcast_to(scales, expected)
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Unsupported pyAML {kind} units: {units}") from exc
        stamp = time()
        result = {}
        for channel in requested:
            kind, index, plane = self._channels[channel]
            value = arrays[kind][index] if plane is None else arrays[kind][index, plane]
            result[channel] = ChannelSample(channel, float(value), stamp)
        return result

    def test_connection(self):
        self.read_many(self.list_channels())
        return {"bpms": len(self._catalog["bpm"]),
                "corrector_readback": "Configured pyAML strengths available",
                "calibration": "pyAML units converted to metres / radians",
                "rf_readback": self.get_rf_frequency() if self.rf is not None else "not configured"}
