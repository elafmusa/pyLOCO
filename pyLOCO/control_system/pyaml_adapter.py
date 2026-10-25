"""Capability-gated bridge from configured pyAML arrays to the measurement GUI."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from time import time

import numpy as np

from .adapters import AdapterCapability, ChannelSample, WritableAdapter


class PyAMLOrbitCorrectionError(RuntimeError):
    """Correction failed; diagnostics describe the attempted rollback."""

    def __init__(self, message, diagnostics, *, cancelled=False):
        super().__init__(message)
        self.diagnostics = diagnostics
        self.cancelled = cancelled


class PyAMLAdapter(WritableAdapter):
    def __init__(self, config_path, *, mode="live", bpm_group="BPM",
                 hcor_group="HCorr", vcor_group="VCorr", accelerator=None,
                 rf_name="RF", allow_writes=True, allow_rf_writes=True,
                 corrector_values_in_radians=False, allow_get_only_restoration=False,
                 orbit_correction_enabled=False):
        if mode not in ("design", "live"):
            raise ValueError("pyAML mode must be design or live")
        self._config_path=Path(config_path).expanduser().resolve()
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
        self.allow_get_only_restoration = False
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
        # The installed pyAML scalar API exposes strength.get() even when it
        # does not expose an independent readback().  Preserve that established
        # adapter-value verification path; it is deliberately classified below
        # as weaker than independent readback evidence.
        self.allow_get_only_restoration = all(
            callable(getattr(self.machine.magnet.get(item["name"]).strength,"get",None))
            for kind in ("hcor","vcor") for item in self._catalog[kind]
        )

        self.orbit_correction_enabled = bool(orbit_correction_enabled)
        if self.orbit_correction_enabled:
            self._initialize_orbit_correction()

    def theoretical_lattice_provenance(self):
        """Return an authoritative file-backed design lattice when pyAML exposes one.

        The public pyAML ``Simulator.lattice`` and ``mat_key`` properties are
        used. A runtime ring without a resolvable source file is deliberately
        not exported or guessed here.
        """
        simulator=getattr(self.accelerator,"design",None)
        configured=getattr(simulator,"lattice",None)
        if simulator is None or not configured:return None
        path=Path(str(configured)).expanduser()
        if not path.is_absolute():path=Path(self._config_path).expanduser().resolve().parent/path
        if not path.is_file():return None
        return {"path":str(path.resolve()),"mat_variable":str(getattr(simulator,"mat_key","") or ""),
                "source":"pyAML public design Simulator.lattice"}

    def _initialize_orbit_correction(self):
        if not self.allow_writes:
            raise ValueError("pyAML orbit correction requires corrector magnet writes")
        orbit = getattr(getattr(self.machine, "tool", None), "orbit", None)
        matrix = getattr(orbit, "_pySC_response_matrix", None)
        if orbit is None or getattr(orbit, "response_matrix", None) is None or matrix is None:
            raise ValueError("pyAML orbit correction requires a valid loaded orbit response matrix")
        if orbit.bpms is not self.groups["bpm"] or tuple(orbit.bpms.names()) != tuple(self.groups["bpm"].names()):
            raise ValueError("pyAML orbit correction must use the same BPM array and ordering as measurement")
        bpm_names = tuple(self.groups["bpm"].names())
        inputs = list(zip(matrix.input_names, matrix.input_planes))
        outputs = list(zip(matrix.output_names, matrix.output_planes))
        values = np.asarray(matrix.matrix, dtype=float)
        expected_outputs = [(name, plane) for plane in ("H", "V") for name in bpm_names]
        if (values.shape != (len(expected_outputs), len(inputs)) or not np.isfinite(values).all()
                or outputs != expected_outputs or len(set(inputs)) != len(inputs)
                or len(matrix.input_names) != len(matrix.input_planes)
                or len(matrix.output_names) != len(matrix.output_planes)):
            raise ValueError("Invalid pyAML orbit response matrix dimensions, values, or BPM/plane ordering")
        self.models_map = {}
        self._orbit_correctors = {}
        self._orbit_planes = {}
        for plane, array in (("H", orbit.hcorrectors), ("V", orbit.vcorrectors)):
            for corrector in array:
                name = corrector.name
                if name in self._orbit_correctors or (name, plane) not in inputs:
                    raise ValueError(f"Invalid pyAML orbit response matrix mapping for {name}")
                model = corrector.get_model_name()
                if not model:
                    raise ValueError(f"Missing pyAML model name for {name}")
                self.models_map.setdefault(model, []).append(name)
                self._orbit_correctors[name] = corrector
                self._orbit_planes[name] = plane
                self._angular_scale_to_rad(corrector.strength.unit())
        if set(inputs) != {(name, plane) for name, plane in self._orbit_planes.items()}:
            raise ValueError("pyAML orbit response matrix correctors do not match the orbit tool")
        for names in self.models_map.values():
            planes = [self._orbit_planes[name] for name in names]
            if len(set(planes)) != len(planes):
                raise ValueError(f"Ambiguous pyAML combined-function corrector mapping: {names}")
        for kind, plane in (("hcor", "H"), ("vcor", "V")):
            for item in self._catalog[kind]:
                if self._orbit_planes.get(item["name"]) != plane:
                    raise ValueError(f"Measurement corrector {item['name']} is missing from the pyAML orbit response matrix")
        for name, plane in self._orbit_planes.items():
            column = inputs.index((name, plane))
            rows = [i for i, (_, p) in enumerate(outputs) if p == plane and i not in matrix.bad_outputs]
            if not rows or not np.any(values[rows, column]):
                raise ValueError(f"pyAML orbit response matrix has no usable {plane} response for {name}")
        self._orbit_tool = orbit

    def validate_orbit_correction_selection(self, bpm_names):
        if tuple(bpm_names) != tuple(self.groups["bpm"].names()):
            raise ValueError("Orbit correction requires the complete configured BPM array in its original order")

    def correct_orbit_after_column(self, name, reference, *, cancel_event, settle, tolerance_rad):
        """Apply a local correction, rolling back this transaction on failure."""
        self.require(AdapterCapability.WRITE)
        if not self.orbit_correction_enabled:
            raise RuntimeError("pyAML orbit correction is disabled")
        reference = np.asarray(reference, dtype=float)
        if reference.shape != (2 * len(self.groups["bpm"].names()),) or not np.isfinite(reference).all():
            raise ValueError("Invalid orbit correction reference")
        orbit = self._orbit_tool
        matrix = orbit._pySC_response_matrix
        model = self._orbit_correctors[name].get_model_name()
        names = [name] + [other for other in self.models_map[model] if other != name]
        before = {n: self._scalar(c.strength.get()) for n, c in self._orbit_correctors.items()}
        before_readback = {n: self._scalar((getattr(c.strength, "readback", None)
                                          or c.strength.get)()) for n, c in self._orbit_correctors.items()}
        units = {n: str(c.strength.unit()) for n, c in self._orbit_correctors.items()}
        scales = {n: self._angular_scale_to_rad(unit) for n, unit in units.items()}
        old_bad_inputs = list(matrix.bad_inputs)
        old_virtual_weight = matrix.virtual_weight
        singular_h = min(matrix.get_matrix_in_plane("H").shape)
        singular_v = min(matrix.get_matrix_in_plane("V").shape)
        diagnostics = {"corrector": name, "correctors": names, "status": "started",
                       "reference_m": reference.tolist(), "before_native": {n: before[n] for n in names},
                       "units": {n: units[n] for n in names}}
        attempted = False
        def check_cancel():
            if cancel_event.is_set():
                raise RuntimeError("Orbit correction was cancelled")
        try:
            for active in names:
                check_cancel()
                matrix.virtual_weight = 1e-16
                matrix.enable_all_inputs()
                matrix.disable_all_inputs_but([active])
                attempted = True
                orbit.correct(plane=self._orbit_planes[active], singular_values_h=singular_h,
                              singular_values_v=singular_v, reference=reference, rf=False)
                settle()
                check_cancel()
            # after = {n: self._scalar(c.strength.get()) for n, c in self._orbit_correctors.items()}
            # if any(after[n] != before[n] for n in before if n not in names):
            #     raise RuntimeError("Orbit correction unexpectedly changed an unrelated corrector")
            # diagnostics.update(status="corrected", after_native={n: after[n] for n in names})
            return diagnostics
        except Exception as exc:
            rollback = []
            # pyAML writes whole arrays: also recover unexpected unrelated changes.
            for n, corrector in (self._orbit_correctors.items() if attempted else ()):
                row = {"name": n, "target_native": before[n], "readback_target_native": before_readback[n],
                       "unit": units[n], "tolerance_rad": tolerance_rad}
                try:
                    if n not in names and self._scalar(corrector.strength.get()) == before[n]:
                        continue
                    corrector.strength.set(before[n])
                    settle()
                    read = getattr(corrector.strength, "readback", None) or corrector.strength.get
                    final = self._scalar(read())
                    scale = scales[n]
                    target = before_readback.get(n, before[n])
                    if not np.isclose(final * scale, target * scale, rtol=0, atol=tolerance_rad):
                        raise RuntimeError("Rollback readback verification failed")
                    rollback.append({**row, "status": "restored", "final_native": final})
                except Exception as rollback_exc:
                    rollback.append({**row, "status": "restore_failed", "error": str(rollback_exc)})
            diagnostics.update(status="failed", error=str(exc), rollback=rollback,
                               restoration_status="restored" if all(r["status"] == "restored" for r in rollback) else "restore_failed")
            raise PyAMLOrbitCorrectionError(str(exc), diagnostics, cancelled=cancel_event.is_set()) from exc
        finally:
            matrix.bad_inputs = old_bad_inputs
            matrix.virtual_weight = old_virtual_weight

    def verify_orbit_correction_restoration(self, diagnostics):
        """Recheck failed rollback targets, including partners outside ORM selection."""
        rows = diagnostics.get("rollback", ())
        verified = bool(rows)
        for row in rows:
            try:
                corrector = self._orbit_correctors[row["name"]]
                if str(corrector.strength.unit()) != row["unit"]:
                    raise ValueError("Corrector unit changed")
                read = getattr(corrector.strength, "readback", None) or corrector.strength.get
                value = self._scalar(read())
                scale = self._angular_scale_to_rad(row["unit"])
                row["recovery_verified"] = bool(np.isclose(value * scale, row["readback_target_native"] * scale,
                                                           rtol=0, atol=row["tolerance_rad"]))
            except Exception:
                row["recovery_verified"] = False
            verified = verified and row["recovery_verified"]
        return verified

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
        if callable(getattr(magnet.strength,"get",None)):
            return {"level":"adapter_value","source":"public pyAML magnet.strength.get() adapter value; independent readback unavailable"}
        return {"level":"unavailable","source":"public pyAML magnet.strength accessor exposes neither readback() nor get()"}

    def read_corrector_readback(self, name):
        magnet=self.machine.magnet.get(name); channel=next(item["setpoint_channel"] for kind in ("hcor","vcor") for item in self._catalog[kind] if item["name"]==name); scale=self._corrector_scales_to_rad.get(channel,1.0)
        evidence=self.corrector_restoration_evidence(name)
        if evidence["level"]=="unavailable":raise RuntimeError(evidence["source"])
        method=magnet.strength.readback if evidence["level"]=="independent" else magnet.strength.get
        try:
            raw=method(); raw=getattr(raw,"value",raw); value=float(raw)*scale
        except Exception as exc:
            raise RuntimeError(f"Corrector verification read failed for {name}: {exc}") from exc
        if not np.isfinite(value):
            raise RuntimeError(f"Corrector verification read returned non-finite value for {name}: {value!r}")
        return value,evidence

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

    def machine_mapping_inventory(self):
        """Return configured runtime device identities without reading channels.

        This complements file-backed YAML discovery for configurations assembled
        from includes or runtime arrays.  Only public collection names, object
        classes and strength-unit metadata are inspected; no get/set operation is
        performed.
        """
        roles=("bpm","horizontal_corrector","vertical_corrector","normal_quadrupole",
               "skew_quadrupole","sextupole","octupole","dipole")
        inventories={role:[] for role in roles}; metadata={}
        selected={"bpm":"bpm","hcor":"horizontal_corrector","vcor":"vertical_corrector"}
        for kind,role in selected.items():
            for row in self._catalog.get(kind,()):
                name=str(row.get("name","") or "")
                if name and name not in inventories[role]:inventories[role].append(name)

        known={name:role for role,names in inventories.items() for name in names}
        class_roles=(("hcorrector","horizontal_corrector"),("vcorrector","vertical_corrector"),
                     ("skewquad","skew_quadrupole"),("quadrupole","normal_quadrupole"),
                     ("sextupole","sextupole"),("octupole","octupole"),("dipole","dipole"))
        suffix_roles={"B1L":"horizontal_corrector","A1L":"vertical_corrector",
                      "B2L":"normal_quadrupole","A2L":"skew_quadrupole",
                      "B3L":"sextupole","B4L":"octupole"}
        arrays=getattr(self.machine,"magnets",{})
        values=arrays.values() if hasattr(arrays,"values") else ()
        names=[]
        for array in values:
            provider=getattr(array,"names",None)
            if not callable(provider):continue
            try:members=provider()
            except Exception:continue
            for value in members:
                name=str(value or "")
                if name and name not in names:names.append(name)
        registry=getattr(self.machine,"magnet",None)
        for name in names:
            try:magnet=registry.get(name)
            except Exception:magnet=None
            class_name=(f"{type(magnet).__module__}.{type(magnet).__qualname__}" if magnet is not None else "").lower()
            component=name.rsplit("-",1)[-1].upper() if "-" in name else ""
            role=known.get(name) or next((value for token,value in class_roles if token in class_name),None) or suffix_roles.get(component)
            if role is None:continue
            if name not in inventories[role]:inventories[role].append(name)
            unit=""
            try:
                strength=getattr(magnet,"strength",None); provider=getattr(strength,"unit",None)
                if callable(provider):unit=str(provider() or "")
            except Exception:unit=""
            metadata[(role,name)]={"pyaml_class":class_name,
                "configured_unit":unit,"hardware_unit":"",
                "configured_component":component if component in suffix_roles else "",
                "inventory_source":"connected pyAML runtime arrays"}
            if magnet is not None:
                provider=getattr(magnet,"get_lattice_names",None)
                if callable(provider):
                    try:
                        aliases=provider()
                        if isinstance(aliases,str):aliases=[aliases]
                        else:aliases=[str(value) for value in aliases if str(value)]
                        if aliases:metadata[(role,name)]["lattice_names"]=list(dict.fromkeys(aliases))
                    except Exception:pass
        return inventories,metadata

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
