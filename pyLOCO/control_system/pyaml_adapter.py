"""Capability-gated bridge from configured pyAML arrays to the measurement GUI."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from time import time

import numpy as np

from .adapters import AdapterCapability, ChannelSample, WritableAdapter


class PyAMLAdapter(WritableAdapter):
    def __init__(self, config_path, *, mode="live", bpm_group="BPM",
                 hcor_group="HCorr", vcor_group="VCorr", accelerator=None,
                 rf_name="RF", allow_writes=True, allow_rf_writes=True,
                 corrector_values_in_radians=False, allow_get_only_restoration=False):
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
        self.simulator_identity = self._verified_simulator_identity(accelerator,mode,config_path)
        # Write authorization and restoration evidence are deliberately separate.
        # A verified runtime may attest that strength.get() is a fresh simulator
        # state read; editable configuration labels and caller flags cannot.
        self.allow_get_only_restoration = bool(
            self.simulator_identity and self.simulator_identity.get("strength_get_is_fresh") is True
        )
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
                    strength = self.machine.magnet.get(name).strength
                    unit = str(strength.unit()).strip()
                    self._corrector_units[channel] = unit
                    if self.allow_writes and self.corrector_values_in_radians:
                        self._corrector_scales_to_rad[channel] = self._angular_scale_to_rad(unit)
                    devices.append(dict(name=name, setpoint_channel=channel,
                                        readback_channel=channel,
                                        plane="Horizontal" if kind == "hcor" else "Vertical"))
            self._catalog[kind] = tuple(devices)

    @staticmethod
    def _verified_simulator_identity(accelerator,mode,config_path):
        """Identify a simulator from runtime type, bound to one configuration.

        Editable YAML labels and endpoints are never sufficient.  The fallback
        accepts only the installed pySC2pyaml runtime ControlSystem class exposed
        through Accelerator.controls(), whose get() performs a server read.
        """
        provider=getattr(accelerator,"simulator_identity",None)
        if callable(provider):
            try:identity=provider()
            except Exception:identity=None
            if isinstance(identity,dict) and identity.get("verified") is True:
                backend=str(identity.get("backend","")).strip().lower(); token=str(identity.get("instance_id","")).strip()
                fresh=identity.get("strength_get_is_fresh") is True
                if backend in {"pysc","pysc2pyaml"}:fresh=True
                digest=str(identity.get("configuration_sha256","")).strip().lower()
                if not digest:
                    try:digest=sha256(Path(config_path).expanduser().resolve(strict=True).read_bytes()).hexdigest()
                    except Exception:digest=""
                if backend and token and fresh and len(digest)==64 and all(char in "0123456789abcdef" for char in digest):
                    return {"verified":True,"backend":backend,"instance_id":token,"strength_get_is_fresh":True,"configuration_sha256":digest,"evidence":str(identity.get("evidence","public pyAML runtime attestation with fresh strength.get() capability"))}
        try:
            from pySC2pyaml.controlsystem import pySCControlSystem
            controls=accelerator.controls(); runtime=controls.get(mode)
            path=Path(config_path).expanduser().resolve(strict=True)
            if not isinstance(runtime,pySCControlSystem) or not path.is_file():return None
            digest=sha256(path.read_bytes()).hexdigest()
            token=sha256(f"{digest}:{mode}:{type(runtime).__module__}.{type(runtime).__qualname__}".encode()).hexdigest()[:20]
            return {"verified":True,"backend":"pysc2pyaml","instance_id":token,"configuration_sha256":digest,"strength_get_is_fresh":True,"evidence":"active Accelerator.controls() runtime is pySC2pyaml.controlsystem.pySCControlSystem; strength.get() performs a fresh simulator read; identity bound to configuration SHA-256"}
        except Exception:return None

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

    @staticmethod
    def _diagnostic_value(accessor, method, *, scale=1.0):
        operation=getattr(accessor,method,None)
        if not callable(operation):
            return {"status":"unsupported","value":None,"unit":None,"error":f"pyAML accessor does not expose {method}()"}
        try:
            raw=operation(); raw=getattr(raw,"value",raw); value=float(raw)*float(scale)
            if not np.isfinite(value):raise ValueError(f"non-finite value {value}")
            unit=str(accessor.unit()).strip() if callable(getattr(accessor,"unit",None)) else ""
            return {"status":"available","value":value,"unit":unit,"error":""}
        except Exception as exc:
            return {"status":"failed","value":None,"unit":None,"error":str(exc)}

    def read_corrector_diagnostics(self, name):
        """Read supported corrector state without deriving hardware from strength.

        Current pyAML strength and hardware scalar accessors expose ``get()`` for
        their setpoints.  An RBV is reported only if that accessor independently
        exposes ``readback()``; the setpoint is never duplicated into an RBV cell.
        """
        matches=[(kind,index) for kind in ("hcor","vcor") for index,item in enumerate(self._catalog[kind]) if item["name"]==name]
        if len(matches)!=1:raise KeyError(f"Unknown or ambiguous pyAML corrector: {name}")
        kind,index=matches[0]; magnet=self.machine.magnet.get(name); channel=self._catalog[kind][index]["setpoint_channel"]
        scale=self._corrector_scales_to_rad.get(channel,1.0)
        strength_unit="rad" if channel in self._corrector_scales_to_rad else self._corrector_units.get(channel,"")
        strength_sp=self._diagnostic_value(magnet.strength,"get",scale=scale)
        strength_rbv=self._diagnostic_value(magnet.strength,"readback",scale=scale)
        for item in (strength_sp,strength_rbv):
            if item["status"]=="available":item["unit"]=strength_unit
        try:hardware=magnet.hardware
        except Exception as exc:
            unavailable={"status":"unsupported","value":None,"unit":None,"error":str(exc)}
            hardware_sp=dict(unavailable); hardware_rbv=dict(unavailable)
        else:
            hardware_sp=self._diagnostic_value(hardware,"get")
            hardware_rbv=self._diagnostic_value(hardware,"readback")
        return {"KICK.SP":strength_sp,"KICK.RBV":strength_rbv,"CURRENT.SP":hardware_sp,"CURRENT.RBV":hardware_rbv}

    def corrector_restoration_evidence(self, name):
        magnet=self.machine.magnet.get(name); independent=callable(getattr(magnet.strength,"readback",None))
        if independent:return {"level":"independent","source":"public pyAML magnet.strength.readback()"}
        if self.simulator_identity and self.simulator_identity.get("strength_get_is_fresh") is True:
            return {"level":"adapter_value","source":"fresh public pyAML magnet.strength.get() from an authoritatively verified simulator runtime; independent readback unavailable"}
        return {"level":"unavailable","source":"public pyAML magnet.strength accessor has no readback()"}

    def read_corrector_readback(self, name):
        magnet=self.machine.magnet.get(name); channel=next(item["setpoint_channel"] for kind in ("hcor","vcor") for item in self._catalog[kind] if item["name"]==name); scale=self._corrector_scales_to_rad.get(channel,1.0)
        evidence=self.corrector_restoration_evidence(name)
        if evidence["level"]=="unavailable":raise RuntimeError(evidence["source"])
        method=magnet.strength.readback if evidence["level"]=="independent" else magnet.strength.get
        raw=method(); raw=getattr(raw,"value",raw); return float(raw)*scale,evidence

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
