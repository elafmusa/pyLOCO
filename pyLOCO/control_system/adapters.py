"""Capability-separated control-system adapter interfaces.

Milestone 1 contains no real machine adapter.  The deterministic mock can
simulate writes in memory for tests, but never communicates with hardware.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence


class AdapterCapability(str, Enum):
    READ = "read"
    WRITE = "write"
    BATCH_READ = "batch_read"
    RF_READ = "rf_read"
    RF_WRITE = "rf_write"


class RFAdapterMixin:
    """Optional RF contract used by automatic dispersion acquisition."""

    def get_rf_frequency(self) -> float:
        raise NotImplementedError

    def set_rf_frequency(self, frequency_hz: float) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class ChannelSample:
    channel: str
    value: Any
    timestamp: float
    status: str = "ok"


class ControlSystemAdapter(ABC):
    """Base adapter exposing capabilities without implying write access."""

    @property
    @abstractmethod
    def capabilities(self) -> frozenset[AdapterCapability]:
        raise NotImplementedError

    def require(self, capability: AdapterCapability) -> None:
        if capability not in self.capabilities:
            raise PermissionError(
                f"{type(self).__name__} does not provide {capability.value!r} capability"
            )


class ReadOnlyAdapter(ControlSystemAdapter):
    @abstractmethod
    def read(self, channel: str) -> ChannelSample:
        raise NotImplementedError

    def read_many(self, channels: Iterable[str]) -> dict[str, ChannelSample]:
        self.require(AdapterCapability.READ)
        return {channel: self.read(channel) for channel in channels}

    def list_channels(self) -> tuple[str, ...]:
        """Return discoverable channels, or an empty tuple when unsupported."""
        return ()


class WritableAdapter(ReadOnlyAdapter):
    """Explicit opt-in interface for adapters permitted to write."""

    @abstractmethod
    def write(self, channel: str, value: Any) -> ChannelSample:
        raise NotImplementedError


class MockAdapter(WritableAdapter):
    """Deterministic in-memory adapter for tests and offline development.

    Simulated write support is disabled by default.  Enabling it changes only
    this object's in-memory channel mapping and is never a real machine write.
    """

    def __init__(
        self,
        channels: Mapping[str, Any] | None = None,
        *,
        timestamp: float = 0.0,
        timestamp_step: float = 1.0,
        allow_simulated_writes: bool = False,
        sequences: Mapping[str, Sequence[Any]] | None = None,
        device_catalog: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
        rf_dependent_channels: Mapping[str, tuple[float, float]] | None = None,
        nominal_rf_hz: float | None = None,
        setpoint_dependent_channels: Mapping[str, tuple[float, Mapping[str, float]]] | None = None,
    ) -> None:
        self._channels = dict(channels or {})
        self._timestamp = float(timestamp)
        self._timestamp_step = float(timestamp_step)
        self._allow_simulated_writes = bool(allow_simulated_writes)
        self._sequences = {name: tuple(values) for name, values in (sequences or {}).items()}
        if any(not values for values in self._sequences.values()):
            raise ValueError("Mock channel sequences may not be empty")
        self._sequence_positions = {name: 0 for name in self._sequences}
        self._device_catalog = {
            kind: tuple(dict(device) for device in devices)
            for kind, devices in (device_catalog or {}).items()
        }
        self._rf_dependent_channels = dict(rf_dependent_channels or {})
        self._nominal_rf_hz = None if nominal_rf_hz is None else float(nominal_rf_hz)
        self._simulated_rf_hz = self._nominal_rf_hz
        self._setpoint_dependent_channels = {
            channel: (float(base), {name: float(slope) for name, slope in slopes.items()})
            for channel, (base, slopes) in (setpoint_dependent_channels or {}).items()
        }
        self.history: list[tuple[str, str, Any]] = []

    @property
    def capabilities(self) -> frozenset[AdapterCapability]:
        result = {AdapterCapability.READ, AdapterCapability.BATCH_READ}
        if self._allow_simulated_writes:
            result.add(AdapterCapability.WRITE)
        return frozenset(result)

    def _next_timestamp(self) -> float:
        result = self._timestamp
        self._timestamp += self._timestamp_step
        return result

    def read(self, channel: str) -> ChannelSample:
        self.require(AdapterCapability.READ)
        if channel not in self._channels:
            raise KeyError(f"Unknown mock channel: {channel}")
        if channel in self._setpoint_dependent_channels:
            base, slopes = self._setpoint_dependent_channels[channel]
            position = self._sequence_positions.get(channel, 0)
            noise = self._sequences.get(channel, (0.0,))
            value = base + sum(slope * float(self._channels[name]) for name, slope in slopes.items())
            if channel in self._rf_dependent_channels:
                if self._simulated_rf_hz is None or self._nominal_rf_hz is None:
                    raise RuntimeError("Mock RF-dependent channel has no simulated RF state")
                _rf_base, rf_slope = self._rf_dependent_channels[channel]
                value += float(rf_slope) * (self._simulated_rf_hz - self._nominal_rf_hz)
            value += float(noise[min(position, len(noise) - 1)])
            self._sequence_positions[channel] = position + 1
        elif channel in self._rf_dependent_channels:
            if self._simulated_rf_hz is None or self._nominal_rf_hz is None:
                raise RuntimeError("Mock RF-dependent channel has no simulated RF state")
            base, slope_m_per_hz = self._rf_dependent_channels[channel]
            position = self._sequence_positions.get(channel, 0)
            noise = self._sequences.get(channel, (0.0,))
            value = float(base) + float(slope_m_per_hz) * (
                self._simulated_rf_hz - self._nominal_rf_hz
            ) + float(noise[min(position, len(noise) - 1)])
            self._sequence_positions[channel] = position + 1
        elif channel in self._sequences:
            position = self._sequence_positions[channel]
            values = self._sequences[channel]
            value = values[min(position, len(values) - 1)]
            self._sequence_positions[channel] = position + 1
        else:
            value = self._channels[channel]
        self.history.append(("read", channel, value))
        return ChannelSample(channel, value, self._next_timestamp())

    def set_simulated_rf_state(self, frequency_hz: float) -> None:
        """Select an offline RF state without performing a machine write.

        This is a Mock-only test hook.  It is intentionally separate from the
        writable adapter API and does not advertise RF-write capability.
        """
        if not self._rf_dependent_channels:
            raise RuntimeError("This MockAdapter has no RF-dependent channels")
        self._simulated_rf_hz = float(frequency_hz)
        for channel in self._rf_dependent_channels:
            self._sequence_positions[channel] = 0

    def list_channels(self) -> tuple[str, ...]:
        return tuple(sorted(self._channels))

    def list_devices(self, kind: str) -> tuple[dict[str, Any], ...]:
        """Return deterministic device metadata supplied to this mock."""
        return self._device_catalog.get(kind, ())

    def write(self, channel: str, value: Any) -> ChannelSample:
        self.require(AdapterCapability.WRITE)
        self._channels[channel] = value
        for dependent, (_base, slopes) in self._setpoint_dependent_channels.items():
            if channel in slopes:
                self._sequence_positions[dependent] = 0
        self.history.append(("write", channel, value))
        return ChannelSample(channel, value, self._next_timestamp())

    def set_simulated_writes_enabled(self,enabled: bool) -> None:
        """Explicitly enable or disable in-memory writes; never affects hardware."""
        self._allow_simulated_writes=bool(enabled)
