"""Capability-gated bridge from configured pyAML arrays to the measurement GUI."""
from __future__ import annotations

from time import time

import numpy as np

from .adapters import AdapterCapability, ChannelSample, WritableAdapter


class PyAMLAdapter(WritableAdapter):
    def __init__(self, config_path, *, mode="live", bpm_group="BPM",
                 hcor_group="HCorr", vcor_group="VCorr", accelerator=None,
                 rf_name="RF", allow_writes=True, allow_rf_writes=True,
                 corrector_values_in_radians=False):
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
        self.corrector_values_in_radians = bool(corrector_values_in_radians)
        self.rf = self.machine.rf.get(rf_name).frequency if rf_name else None
        if self.allow_rf_writes and self.rf is None:
            raise ValueError("Select an RF plant before enabling RF writes")
        bpm_collection, self.bpm_api_path = self._bpm_collection(self.machine)
        self.groups = {
            "bpm": bpm_collection.get(bpm_group),
            "hcor": self.machine.magnets.get(hcor_group),
            "vcor": self.machine.magnets.get(vcor_group),
        }
        for kind, group in self.groups.items():
            if group is None:
                raise ValueError(f"The pyAML configuration has no selected {kind} array")
        self._catalog = {}
        self._channels = {}
        self._corrector_scales_to_rad = {}
        self._corrector_units = {}
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
                    if self.allow_writes and self.corrector_values_in_radians:
                        unit = str(self.machine.magnet.get(name).strength.unit()).strip()
                        self._corrector_scales_to_rad[channel] = self._angular_scale_to_rad(unit)
                        self._corrector_units[channel] = unit
                    devices.append(dict(name=name, setpoint_channel=channel,
                                        readback_channel=channel,
                                        plane="Horizontal" if kind == "hcor" else "Vertical"))
            self._catalog[kind] = tuple(devices)

    @staticmethod
    def _angular_scale_to_rad(unit):
        normalized = str(unit).strip().lower().replace("μ", "u").replace("µ", "u")
        scales = {"rad": 1.0, "radian": 1.0, "radians": 1.0,
                  "mrad": 1e-3, "milliradian": 1e-3, "milliradians": 1e-3,
                  "urad": 1e-6, "microradian": 1e-6, "microradians": 1e-6}
        if normalized not in scales:
            raise ValueError(
                f"pyAML corrector strength unit {unit!r} is not a verified angular unit; "
                "ORM writes require rad, mrad, or urad calibration"
            )
        return scales[normalized]

    @staticmethod
    def _bpm_collection(machine):
        """Return the BPM holder across the pyAML #464 hierarchy change.

        Current pyAML exposes BPM arrays below ``machine.diagnostic.bpms``.
        The legacy fallback keeps released pyAML 0.3.x configurations working.
        Attribute detection is deliberate: lookup errors from a present holder
        must remain visible rather than being mistaken for an older API.
        """
        diagnostic = getattr(machine, "diagnostic", None)
        diagnostic_bpms = getattr(diagnostic, "bpms", None)
        if diagnostic_bpms is not None:
            return diagnostic_bpms, "diagnostic.bpms"
        legacy_bpms = getattr(machine, "bpms", None)
        if legacy_bpms is not None:
            return legacy_bpms, "bpms"
        raise ValueError(
            "The selected pyAML machine exposes neither diagnostic.bpms "
            "(pyAML PR #464) nor the legacy bpms holder"
        )

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
        strength = self._magnet_strength(kind, index)
        value_rad = self._scalar(value)
        scale = self._corrector_scales_to_rad.get(channel,1.0)
        strength.set(value_rad / scale)
        return self.read(channel)

    def _magnet_strength(self, kind, index):
        name = self._catalog[kind][index]["name"]
        return self.machine.magnet.get(name).strength

    def get_rf_frequency(self):
        self.require(AdapterCapability.RF_READ)
        value = self._scalar(self.rf.get())
        if value <= 0:
            raise ValueError("RF frequency must be positive")
        return value

    def set_rf_frequency(self, frequency_hz):
        self.require(AdapterCapability.RF_WRITE)
        value = self._scalar(frequency_hz)
        if value <= 0:
            raise ValueError("RF frequency must be positive")
        self.rf.set(value)

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
            if kind != "bpm":
                continue
            group = self.groups[kind]
            values = np.asarray(group.positions.get(), dtype=float)
            expected = (len(self._catalog[kind]), 2)
            if values.shape != expected or not np.all(np.isfinite(values)):
                raise ValueError(f"Invalid pyAML {kind} readings: expected finite array {expected}, got {values.shape}")
            arrays[kind] = values
        stamp = time()
        result = {}
        for channel in requested:
            kind, index, plane = self._channels[channel]
            if kind == "bpm":
                value = arrays[kind][index, plane]
            else:
                value = float(self._magnet_strength(kind, index).get())
                scale = self._corrector_scales_to_rad.get(channel)
                if scale is not None:
                    value *= scale
                if not np.isfinite(value):
                    raise ValueError(f"Invalid pyAML magnet reading for {self._catalog[kind][index]['name']}: {value}")
            result[channel] = ChannelSample(channel, float(value), stamp)
        return result

    def test_connection(self):
        self.read_many(self.list_channels())
        return {"bpms": len(self._catalog["bpm"]),
                "corrector_readback": "Configured pyAML strengths available",
                "calibration": ("verified angular units converted to SI radians: " +
                                ", ".join(sorted(set(self._corrector_units.values())))
                                if self.allow_writes and self.corrector_values_in_radians else
                                "backend-native values (SI-radian conversion not requested)" if self.allow_writes else
                                "not verified (corrector writes disabled)"),
                "rf_readback": self.get_rf_frequency() if self.rf is not None else "not configured"}
