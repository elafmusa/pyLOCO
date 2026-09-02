"""One backend registry shared by Measure and Correct.

Imports of optional control-system clients are deliberately deferred until a
backend is selected.  GUI code consumes :class:`BackendSession` and never calls
pySC or DOOCS directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from time import time
from typing import Callable, Iterable, Sequence

import numpy as np

from .adapters import AdapterCapability, ChannelSample, WritableAdapter
from .petra import PETRAReadOnlyAdapter
from .pysc_profiles import load_pysc_catalog


@dataclass(frozen=True)
class BackendDescriptor:
    key: str
    label: str
    badge: str
    environment: str
    real_machine: bool
    writes_expected: bool


@dataclass(frozen=True)
class BackendSession:
    descriptor: BackendDescriptor
    adapter: object

    @property
    def badge(self) -> str:
        return self.descriptor.badge


class AbstractInterfaceAdapter(WritableAdapter):
    """Translate a legacy ``pySC AbstractInterface`` into the GUI contract."""

    def __init__(self, interface, bpm_names: Sequence[str], hcor_names: Sequence[str],
                 vcor_names: Sequence[str], *, allow_writes: bool = True,
                 allow_rf_writes: bool = True,
                 backend_metadata: dict | None = None) -> None:
        self.interface = interface
        self.bpm_names = tuple(bpm_names)
        self.horizontal_corrector_names = tuple(hcor_names)
        self.vertical_corrector_names = tuple(vcor_names)
        self.allow_writes = bool(allow_writes)
        self.allow_rf_writes = bool(allow_rf_writes)
        self.backend_metadata = dict(backend_metadata or {})
        self.history: list[tuple[str, str, object]] = []

    @property
    def capabilities(self):
        result = {AdapterCapability.READ, AdapterCapability.BATCH_READ,
                  AdapterCapability.RF_READ}
        if self.allow_writes:
            result.add(AdapterCapability.WRITE)
        if self.allow_rf_writes:
            result.add(AdapterCapability.RF_WRITE)
        return frozenset(result)

    @staticmethod
    def bpm_channel(name: str, plane: str) -> str:
        return f"BPM:{name}:{plane.upper()}"

    @staticmethod
    def magnet_channel(name: str) -> str:
        return f"MAGNET:{name}"

    def list_devices(self, kind: str):
        if kind == "bpm":
            return tuple({"name": name, "x_channel": self.bpm_channel(name, "X"),
                          "y_channel": self.bpm_channel(name, "Y")} for name in self.bpm_names)
        names = self.horizontal_corrector_names if kind == "hcor" else self.vertical_corrector_names if kind == "vcor" else ()
        plane = "Horizontal" if kind == "hcor" else "Vertical"
        return tuple({"name": name, "setpoint_channel": self.magnet_channel(name),
                      "readback_channel": self.magnet_channel(name), "plane": plane}
                     for name in names)

    def read_many(self, channels: Iterable[str]):
        requested = tuple(channels)
        if requested and all(channel.startswith("BPM:") for channel in requested):
            x, y = (np.asarray(values, dtype=float) for values in self.interface.get_orbit())
            if x.shape != y.shape or x.size != len(self.bpm_names):
                raise ValueError("Backend orbit length does not match its BPM device catalog")
            stamp = time(); lookup = {}
            for index, name in enumerate(self.bpm_names):
                for plane, values in (("X", x), ("Y", y)):
                    channel = self.bpm_channel(name, plane)
                    lookup[channel] = ChannelSample(channel, float(values[index]), stamp)
            missing = [channel for channel in requested if channel not in lookup]
            if missing:
                raise KeyError("Unknown BPM channel(s): " + ", ".join(missing))
            self.history.append(("read", "orbit", (x.copy(), y.copy())))
            return {channel: lookup[channel] for channel in requested}
        return super().read_many(requested)

    def read(self, channel: str):
        if channel.startswith("BPM:"):
            return self.read_many((channel,))[channel]
        if not channel.startswith("MAGNET:"):
            raise KeyError(f"Unknown backend channel: {channel}")
        name = channel.split(":", 1)[1]
        value = float(self.interface.get(name)); self.history.append(("read", channel, value))
        return ChannelSample(channel, value, time())

    def write(self, channel: str, value):
        self.require(AdapterCapability.WRITE)
        if not channel.startswith("MAGNET:"):
            raise KeyError(f"Unknown backend channel: {channel}")
        self.interface.set(channel.split(":", 1)[1], float(value))
        result = ChannelSample(channel, float(value), time())
        self.history.append(("write", channel, float(value)))
        return result

    def get_rf_frequency(self) -> float:
        self.require(AdapterCapability.RF_READ)
        return float(self.interface.get_rf_main_frequency())

    def set_rf_frequency(self, frequency_hz: float) -> None:
        self.require(AdapterCapability.RF_WRITE)
        self.interface.set_rf_main_frequency(float(frequency_hz))
        self.history.append(("write", "RF:FREQUENCY", float(frequency_hz)))

    def test_connection(self):
        x, y = self.interface.get_orbit()
        if len(x) != len(self.bpm_names) or len(y) != len(self.bpm_names):
            raise RuntimeError("Connected server orbit does not match the generated BPM catalog")
        rf = self.get_rf_frequency()
        return {"bpms": len(x), "bpm_orbit": "available", "rf_readback": rf,
                "corrector_readback": f"available — {len(self.horizontal_corrector_names)} H / {len(self.vertical_corrector_names)} V",
                "horizontal_correctors": len(self.horizontal_corrector_names),
                "vertical_correctors": len(self.vertical_corrector_names),
                "corrector_unit": self.backend_metadata.get("corrector_control_unit", "rad"),
                "calibration": "backend managed"}


class InterfaceRegistry:
    """Factory and presentation metadata for every supported backend."""

    DESCRIPTORS = (
        BackendDescriptor("mock", "Mock", "MOCK • READ ONLY", "MOCK", False, False),
        BackendDescriptor("pysc", "pySC Server", "DEMO • pySC SERVER", "DEMO", False, True),
        BackendDescriptor("petra", "PETRA III DOOCS", "LIVE • PETRA III DOOCS", "LIVE", True, True),
    )

    def __init__(self, *, mock_factory: Callable[[], object] | None = None,
                 interface_loaders: dict[str, Callable[[], object]] | None = None,
                 repository_root: Path | None = None,
                 pysc_profile: str = "ebs") -> None:
        self.mock_factory = mock_factory
        self.interface_loaders = dict(interface_loaders or {})
        self.root = repository_root or Path(__file__).resolve().parents[2]
        self.pysc_profile = pysc_profile

    def descriptors(self):
        return self.DESCRIPTORS

    def descriptor(self, key: str):
        try: return next(item for item in self.DESCRIPTORS if item.key == key)
        except StopIteration as exc: raise KeyError(f"Unknown backend: {key}") from exc

    def _petra_names(self, filename: str):
        path = self.root / "Examples" / "PETRAIII" / "data" / filename
        return tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())

    def _pysc_catalog(self):
        return load_pysc_catalog(self.pysc_profile, repository_root=self.root)

    def create(self, key: str) -> BackendSession:
        descriptor = self.descriptor(key)
        if key == "mock":
            if self.mock_factory is None:
                raise RuntimeError("A Mock adapter factory must be configured by the application")
            return BackendSession(descriptor, self.mock_factory())
        loader = self.interface_loaders.get(key)
        if key == "pysc":
            catalog = self._pysc_catalog()
            if loader is None:
                interface_class = getattr(import_module("pyLOCO.control_system.pysc_server"), "pySCServerOrbitInterface")
                loader = lambda: interface_class(host=catalog.get("host", "127.0.0.1"),
                                                  port=int(catalog.get("port", 13131)),
                                                  rf_system=catalog.get("rf_system", "main"))
            names = (catalog["bpms"], catalog["horizontal_correctors"], catalog["vertical_correctors"])
        else:
            names = (self._petra_names("BPM_names.txt"), self._petra_names("HCM_names_control.txt"),
                     self._petra_names("VCM_names_control.txt"))
            # The supplied p3_interface has unverified machine-write paths.
            # PETRA discovery therefore uses the independent hard read-only
            # adapter until address semantics and restoration are validated.
            if loader is None:
                return BackendSession(descriptor, PETRAReadOnlyAdapter(*names))
        interface = loader()
        adapter = AbstractInterfaceAdapter(interface, *names,
                                           allow_writes=key == "pysc", allow_rf_writes=key == "pysc",
                                           backend_metadata=catalog.get("metadata", {}) if key == "pysc" else {})
        return BackendSession(descriptor, adapter)
