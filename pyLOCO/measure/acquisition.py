"""Read-only BPM-noise acquisition engine."""
from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from time import monotonic, sleep
from typing import Callable, Sequence

import numpy as np

from pyLOCO.control_system import AdapterCapability, ReadOnlyAdapter, WritableAdapter


class AcquisitionCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class BpmDevice:
    name: str
    x_channel: str
    y_channel: str

    @property
    def identifier(self) -> str:
        return f"{self.x_channel} | {self.y_channel}"


@dataclass(frozen=True)
class CorrectorDevice:
    name: str
    setpoint_channel: str
    readback_channel: str
    plane: str

    @property
    def identifier(self) -> str:
        return f"{self.setpoint_channel} | {self.readback_channel}"


@dataclass(frozen=True)
class ORMResult:
    bpms: tuple[BpmDevice, ...]
    horizontal_correctors: tuple[CorrectorDevice, ...]
    vertical_correctors: tuple[CorrectorDevice, ...]
    response_matrix: np.ndarray
    direction: str
    scaled: bool
    requested_kicks_rad: np.ndarray
    effective_kicks_rad: np.ndarray
    original_setpoints_rad: np.ndarray
    requested_state_a_rad: np.ndarray
    requested_state_b_rad: np.ndarray
    actual_state_a_rad: np.ndarray
    actual_state_b_rad: np.ndarray
    intermediate_restored_setpoints_rad: np.ndarray
    intermediate_restoration_status: tuple[str, ...]
    final_setpoints_rad: np.ndarray
    raw_state_a_m: np.ndarray
    raw_state_b_m: np.ndarray
    raw_reference_before_m: np.ndarray
    raw_reference_intermediate_m: np.ndarray
    raw_reference_final_m: np.ndarray
    timestamps_state_a_s: np.ndarray
    timestamps_state_b_s: np.ndarray
    restoration_status: tuple[str, ...]
    elapsed_seconds: float
    orbit_correction_diagnostics: tuple[dict, ...] = ()

    @property
    def correctors(self) -> tuple[CorrectorDevice, ...]:
        return self.horizontal_correctors + self.vertical_correctors

    @property
    def std_state_a_m(self) -> np.ndarray:
        return np.std(self.raw_state_a_m, axis=1, ddof=0)

    @property
    def std_state_b_m(self) -> np.ndarray:
        return np.std(self.raw_state_b_m, axis=1, ddof=0)


class ORMInterrupted(AcquisitionCancelled):
    def __init__(self, message: str, *, restoration_status: str, restoration_diagnostics=None):
        super().__init__(message); self.restoration_status=restoration_status; self.restoration_diagnostics=restoration_diagnostics or {}


class ORMAcquisitionError(RuntimeError):
    def __init__(self, message: str, *, restoration_status: str, restoration_diagnostics=None):
        super().__init__(message); self.restoration_status=restoration_status; self.restoration_diagnostics=restoration_diagnostics or {}


@dataclass(frozen=True)
class BpmNoiseResult:
    devices: tuple[BpmDevice, ...]
    orbits_x_m: np.ndarray
    orbits_y_m: np.ndarray
    elapsed_seconds: float
    ddof: int = 0

    def __post_init__(self) -> None:
        if self.ddof not in (0, 1):
            raise ValueError("BPM-noise ddof must be 0 or 1")

    @property
    def noise_x_m(self) -> np.ndarray:
        """Per-BPM standard deviation across orbit readings [m]."""
        return np.std(self.orbits_x_m, axis=0, ddof=self.ddof)

    @property
    def noise_y_m(self) -> np.ndarray:
        """Per-BPM standard deviation across orbit readings [m]."""
        return np.std(self.orbits_y_m, axis=0, ddof=self.ddof)

    @property
    def mean_x_m(self) -> np.ndarray:
        return np.mean(self.orbits_x_m, axis=0)

    @property
    def mean_y_m(self) -> np.ndarray:
        return np.mean(self.orbits_y_m, axis=0)


@dataclass(frozen=True)
class DispersionStateResult:
    label: str
    requested_rf_hz: float
    actual_rf_hz: float
    operator_confirmed: bool
    orbits_x_m: np.ndarray
    orbits_y_m: np.ndarray
    timestamps_s: np.ndarray
    bpm_names: tuple[str, ...] = ()
    actual_rf_source: str = "backend_readback"
    rf_readback_verified: bool = True

    @property
    def mean_x_m(self) -> np.ndarray:
        return np.mean(self.orbits_x_m, axis=0)

    @property
    def mean_y_m(self) -> np.ndarray:
        return np.mean(self.orbits_y_m, axis=0)

    @property
    def std_x_m(self) -> np.ndarray:
        return np.std(self.orbits_x_m, axis=0, ddof=0)

    @property
    def std_y_m(self) -> np.ndarray:
        return np.std(self.orbits_y_m, axis=0, ddof=0)


@dataclass(frozen=True)
class DispersionResult:
    devices: tuple[BpmDevice, ...]
    states: tuple[DispersionStateResult, ...]
    direction: str
    nominal_rf_hz: float
    requested_offset_hz: float
    restoration_status: str
    elapsed_seconds: float

    def __post_init__(self) -> None:
        expected=tuple(device.name for device in self.devices)
        for state in self.states:
            if state.orbits_x_m.shape != state.orbits_y_m.shape or state.orbits_x_m.ndim != 2:
                raise ValueError(f"RF state {state.label!r} has inconsistent orbit array shapes")
            if state.orbits_x_m.shape[1] != len(expected):
                raise ValueError(f"RF state {state.label!r} BPM count changed")
            if state.bpm_names and state.bpm_names != expected:
                raise ValueError(f"RF state {state.label!r} BPM order changed")

    def state(self, label: str) -> DispersionStateResult:
        return next(state for state in self.states if state.label == label)

    @property
    def canonical_rf_step_hz(self) -> float:
        if self.direction == "bipolar":
            return 2.0 * self.requested_offset_hz
        return self.requested_offset_hz if self.direction == "positive" else -self.requested_offset_hz

    def _difference(self, plane: str) -> np.ndarray:
        attr = f"mean_{plane}_m"
        reference = getattr(self.state("reference"), attr)
        if self.direction == "bipolar":
            return getattr(self.state("positive"), attr) - getattr(self.state("negative"), attr)
        if self.direction == "positive":
            return getattr(self.state("positive"), attr) - reference
        return getattr(self.state("negative"), attr) - reference

    @property
    def measured_eta_x(self) -> np.ndarray:
        return self._difference("x")

    @property
    def measured_eta_y(self) -> np.ndarray:
        return self._difference("y")

    @property
    def response_x_m_per_hz(self) -> np.ndarray:
        return self.measured_eta_x / self.canonical_rf_step_hz

    @property
    def response_y_m_per_hz(self) -> np.ndarray:
        return self.measured_eta_y / self.canonical_rf_step_hz


class BpmNoiseAcquirer:
    """Acquire orbit samples using only the read-only adapter contract."""

    def __init__(self, adapter: ReadOnlyAdapter, devices: Sequence[BpmDevice], *, ddof: int = 0) -> None:
        adapter.require(AdapterCapability.READ)
        if ddof not in (0, 1):
            raise ValueError("BPM-noise ddof must be 0 or 1")
        self.adapter = adapter
        self.devices = tuple(devices)
        self.ddof = ddof

    def acquire(
        self,
        readings: int,
        delay_seconds: float,
        *,
        cancel_event: Event | None = None,
        progress: Callable[[int, int, float, np.ndarray, np.ndarray], None] | None = None,
        sleeper: Callable[[float], None] = sleep,
        clock: Callable[[], float] = monotonic,
    ) -> BpmNoiseResult:
        if readings < 2:
            raise ValueError("BPM-noise acquisition requires at least two readings")
        if delay_seconds < 0:
            raise ValueError("Delay between readings may not be negative")
        if not self.devices:
            raise ValueError("At least one BPM must be selected")
        cancel = cancel_event or Event()
        x = np.empty((readings, len(self.devices)), dtype=float)
        y = np.empty_like(x)
        start = clock()
        for reading in range(readings):
            if cancel.is_set():
                raise AcquisitionCancelled("BPM-noise acquisition was cancelled")
            samples = self.adapter.read_many(
                channel for device in self.devices for channel in (device.x_channel, device.y_channel)
            )
            for index, device in enumerate(self.devices):
                x[reading, index] = float(samples[device.x_channel].value)
                y[reading, index] = float(samples[device.y_channel].value)
            if not np.isfinite(x[reading]).all() or not np.isfinite(y[reading]).all():
                raise ValueError(f"Non-finite BPM data detected at orbit reading {reading + 1}")
            elapsed = clock() - start
            if progress:
                progress(reading + 1, readings, elapsed, x[reading].copy(), y[reading].copy())
            if reading + 1 < readings:
                sleeper(delay_seconds)
        return BpmNoiseResult(self.devices, x, y, clock() - start, self.ddof)


class DispersionStateAcquirer:
    """Acquire one operator-confirmed RF state using read capability only."""

    def __init__(self, adapter: ReadOnlyAdapter, devices: Sequence[BpmDevice]) -> None:
        adapter.require(AdapterCapability.READ)
        self.adapter = adapter
        self.devices = tuple(devices)

    def acquire(
        self,
        label: str,
        requested_rf_hz: float,
        readings: int,
        delay_seconds: float,
        *,
        actual_rf_hz: float = float("nan"),
        operator_confirmed: bool = True,
        actual_rf_source: str = "backend_readback",
        rf_readback_verified: bool = True,
        cancel_event: Event | None = None,
        progress: Callable[[int, int, float, np.ndarray, np.ndarray], None] | None = None,
        sleeper: Callable[[float], None] = sleep,
        clock: Callable[[], float] = monotonic,
    ) -> DispersionStateResult:
        base = BpmNoiseAcquirer(self.adapter, self.devices)
        start = clock()
        result = base.acquire(
            readings, delay_seconds, cancel_event=cancel_event,
            progress=progress, sleeper=sleeper, clock=clock,
        )
        timestamps = np.linspace(0.0, result.elapsed_seconds, readings)
        return DispersionStateResult(
            label, float(requested_rf_hz), float(actual_rf_hz), bool(operator_confirmed),
            result.orbits_x_m, result.orbits_y_m, timestamps + start,
            tuple(device.name for device in self.devices),
            str(actual_rf_source), bool(rf_readback_verified),
        )


class ORMAcquirer:
    """Deterministic failure-safe ORM acquisition using simulated writes only."""

    def __init__(self, adapter: WritableAdapter, bpms: Sequence[BpmDevice],
                 horizontal_correctors: Sequence[CorrectorDevice],
                 vertical_correctors: Sequence[CorrectorDevice]) -> None:
        adapter.require(AdapterCapability.READ); adapter.require(AdapterCapability.WRITE)
        self.adapter=adapter; self.bpms=tuple(bpms); self.horizontal_correctors=tuple(horizontal_correctors); self.vertical_correctors=tuple(vertical_correctors)
        if not self.bpms or not (self.horizontal_correctors or self.vertical_correctors):
            raise ValueError("ORM acquisition requires BPMs and at least one corrector")

    def _orbit(self, readings, delay, cancel, sleeper, clock, callback, context):
        result=BpmNoiseAcquirer(self.adapter,self.bpms).acquire(
            readings,delay,cancel_event=cancel,sleeper=sleeper,clock=clock,
            progress=lambda current,total,elapsed,x,y: callback and callback({**context,"event":"orbit","reading":current,"readings":total,"x":x,"y":y,"elapsed":elapsed}),
        )
        return np.hstack((result.orbits_x_m,result.orbits_y_m)), np.linspace(0.0,result.elapsed_seconds,readings)

    def acquire(self, requested_kick_h_rad, requested_kick_v_rad, *, direction="bipolar", scaled=False,
                readings=10, delay_seconds=.1, settling_delay_seconds=0.0, cancel_event=None,
                progress=None, sleeper=sleep, clock=monotonic, restoration_tolerance_rad=1e-12) -> ORMResult:
        if direction not in {"bipolar","positive","negative"}: raise ValueError("Unsupported ORM direction")
        if readings < 2 or delay_seconds < 0 or settling_delay_seconds < 0: raise ValueError("Invalid ORM timing configuration")
        from pyLOCO.control_system.pyaml_adapter import PyAMLAdapter, PyAMLOrbitCorrectionError
        correction_enabled = isinstance(self.adapter, PyAMLAdapter) and self.adapter.orbit_correction_enabled
        correction_diagnostics = []
        if correction_enabled:
            self.adapter.validate_orbit_correction_selection([bpm.name for bpm in self.bpms])
        correctors=self.horizontal_correctors+self.vertical_correctors
        kicks=np.concatenate((np.asarray(requested_kick_h_rad,float).ravel(),np.asarray(requested_kick_v_rad,float).ravel()))
        if kicks.shape!=(len(correctors),) or not np.isfinite(kicks).all() or np.any(kicks<=0): raise ValueError("Requested kick arrays must contain one positive finite value per corrector")
        cancel=cancel_event or Event(); nrows=2*len(self.bpms); ncor=len(correctors)
        if not np.isfinite(restoration_tolerance_rad) or restoration_tolerance_rad < 0:raise ValueError("Restoration tolerance must be finite and non-negative")
        arrays={name:np.full(ncor,np.nan) for name in ("original","original_readback","req_a","req_b","act_a","act_b","intermediate","final","effective")}
        raw_a=np.full((ncor,readings,nrows),np.nan); raw_b=np.full_like(raw_a,np.nan); raw_ref_before=np.full_like(raw_a,np.nan); raw_ref_intermediate=np.full_like(raw_a,np.nan); raw_ref_final=np.full_like(raw_a,np.nan); ts_a=np.full((ncor,readings),np.nan); ts_b=np.full_like(ts_a,np.nan)
        matrix=np.full((nrows,ncor),np.nan); intermediate_statuses=["not_applicable"]*ncor; statuses=["not_attempted"]*ncor; start=clock(); touched=set(); restoration_attempts=[0]*ncor

        evidence=[]
        for corrector in correctors:
            resolver=getattr(self.adapter,"corrector_restoration_evidence",None)
            item=resolver(corrector.name) if callable(resolver) else {"level":"independent" if corrector.setpoint_channel!=corrector.readback_channel else "unavailable","source":"distinct adapter setpoint/readback channels" if corrector.setpoint_channel!=corrector.readback_channel else "setpoint and readback channels are identical"}
            if item.get("level")=="unavailable":
                raise ORMAcquisitionError(f"ORM writes blocked for {corrector.name}: independent corrector readback is unavailable. {item.get('source','')}",restoration_status="not_required",restoration_diagnostics={"status":"not_required","verification_level":"unavailable","correctors":[]})
            evidence.append(dict(item))

        def readback(index):
            method=getattr(self.adapter,"read_corrector_readback",None)
            if callable(method):value,_=method(correctors[index].name); return float(value)
            return float(self.adapter.read(correctors[index].readback_channel).value)

        # Snapshot the complete transaction inventory before the first write.
        # Setpoints are the authoritative restore targets; independent readbacks
        # are the verification baselines.
        for index,corrector in enumerate(correctors):
            arrays["original"][index]=float(self.adapter.read(corrector.setpoint_channel).value)
            arrays["original_readback"][index]=readback(index)
            if not np.isfinite(arrays["original"][index]) or not np.isfinite(arrays["original_readback"][index]):
                raise ORMAcquisitionError(f"Cannot snapshot finite original setting/readback for {corrector.name}",restoration_status="not_required",restoration_diagnostics={"status":"not_required","correctors":[]})

        def settle(*, cancellable):
            if not settling_delay_seconds:return
            if cancellable and sleeper is sleep:
                if cancel.wait(settling_delay_seconds):raise AcquisitionCancelled("ORM acquisition was cancelled during corrector settling")
            else:
                sleeper(settling_delay_seconds)
                if cancellable and cancel.is_set():raise AcquisitionCancelled("ORM acquisition was cancelled during corrector settling")

        def restore(indices):
            rows=[]
            for restore_index in sorted(indices):
                corrector=correctors[restore_index]; restoration_attempts[restore_index]+=1; error=""; final=None; command_issued=False
                try:
                    self.adapter.write(corrector.setpoint_channel,float(arrays["original"][restore_index]))
                    command_issued=True
                    settle(cancellable=False)
                    final=readback(restore_index); arrays["final"][restore_index]=final
                    status="restored" if np.isfinite(final) and np.isclose(final,arrays["original_readback"][restore_index],rtol=0,atol=restoration_tolerance_rad) else "verification_failed"
                except Exception as exc:
                    status="restore_failed"; error=str(exc)
                statuses[restore_index]=status
                rows.append({"index":restore_index,"name":corrector.name,"plane":corrector.plane,"setpoint_channel":corrector.setpoint_channel,"readback_channel":corrector.readback_channel,"set_command_issued":command_issued,"verification_level":evidence[restore_index]["level"],"verification_source":evidence[restore_index]["source"],"independent_readback_verified":bool(status=="restored" and evidence[restore_index]["level"]=="independent"),"original_setpoint_rad":float(arrays["original"][restore_index]),"original_readback_rad":float(arrays["original_readback"][restore_index]),"attempted_restoration_rad":float(arrays["original"][restore_index]),"final_readback_rad":final,"tolerance_rad":float(restoration_tolerance_rad),"attempts":restoration_attempts[restore_index],"status":status,"error":error})
            overall="restored" if all(row["status"]=="restored" for row in rows) else "verification_failed" if rows and all(row["status"]!="restore_failed" for row in rows) else "restore_failed"
            levels={row["verification_level"] for row in rows}; level="independent" if levels=={"independent"} else "adapter_value" if levels and levels<={"adapter_value"} else "mixed"
            return overall,{**({"orbit_correction": list(correction_diagnostics)} if correction_enabled else {}), "status":overall,"verification_level":level,"correctors":rows,"all_inventory":[{"index":i,"name":device.name,"plane":device.plane,"original_setpoint_rad":float(arrays["original"][i]),"original_readback_rad":float(arrays["original_readback"][i])} for i,device in enumerate(correctors)]}

        for index,(corrector,dkick) in enumerate(zip(correctors,kicks)):
            if correction_enabled:
                # Completed corrections remain applied, including changes to a
                # partner that may be measured later. Only the active column
                # belongs to the cleanup inventory in this optional path.
                arrays["original"][index] = float(self.adapter.read(corrector.setpoint_channel).value)
                arrays["original_readback"][index] = readback(index)
                if not np.isfinite(arrays["original"][index]) or not np.isfinite(arrays["original_readback"][index]):
                    raise ORMAcquisitionError(f"Invalid current baseline for {corrector.name}", restoration_status="not_required")
            original=float(arrays["original"][index]); original_readback=float(arrays["original_readback"][index])
            if direction=="bipolar": target_a,target_b=original+dkick/2,original-dkick/2; label_a,label_b="+kick","−kick"
            elif direction=="positive": target_a,target_b=original+dkick,original; label_a,label_b="+kick","reference"
            else: target_a,target_b=original-dkick,original; label_a,label_b="−kick","reference"
            arrays["req_a"][index]=target_a; arrays["req_b"][index]=target_b; restoration="not_attempted"
            try:
                if progress:progress({"event":"state","corrector":index+1,"correctors":ncor,"plane":corrector.plane,"device":corrector.name,"state":"reference before kick","elapsed":clock()-start})
                raw_ref_before[index],_=self._orbit(readings,delay_seconds,cancel,sleeper,clock,progress,{"corrector":index+1,"correctors":ncor,"plane":corrector.plane,"device":corrector.name,"requested_kick":dkick,"state":"reference before kick"})
                for slot,target,label,raw,timestamps in (("a",target_a,label_a,raw_a,ts_a),("b",target_b,label_b,raw_b,ts_b)):
                    if cancel.is_set(): raise AcquisitionCancelled("ORM acquisition was cancelled")
                    if progress: progress({"event":"state","corrector":index+1,"correctors":ncor,"plane":corrector.plane,"device":corrector.name,"requested_kick":dkick,"state":label,"elapsed":clock()-start})
                    touched.add(index); self.adapter.write(corrector.setpoint_channel,target); settle(cancellable=True)
                    actual=readback(index); arrays[f"act_{slot}"][index]=actual
                    samples,times=self._orbit(readings,delay_seconds,cancel,sleeper,clock,progress,{"corrector":index+1,"correctors":ncor,"plane":corrector.plane,"device":corrector.name,"requested_kick":dkick,"state":label})
                    raw[index]=samples; timestamps[index]=times+clock()
                    if direction=="bipolar" and slot=="a":
                        if cancel.is_set(): raise AcquisitionCancelled("ORM acquisition was cancelled before intermediate restoration")
                        if progress: progress({"event":"intermediate_restore","corrector":index+1,"correctors":ncor,"plane":corrector.plane,"device":corrector.name,"state":"restore K0 before −kick","elapsed":clock()-start})
                        self.adapter.write(corrector.setpoint_channel,original); settle(cancellable=True)
                        arrays["intermediate"][index]=readback(index)
                        if not np.isclose(arrays["intermediate"][index],original_readback,rtol=0,atol=restoration_tolerance_rad):
                            raise RuntimeError(f"Intermediate restoration verification failed: {arrays['intermediate'][index]!r} != {original_readback!r}")
                        intermediate_statuses[index]="restored"
                        if progress: progress({"event":"intermediate_restored","corrector":index+1,"correctors":ncor,"plane":corrector.plane,"device":corrector.name,"state":"K0 restored and verified","readback":arrays["intermediate"][index],"original":original,"elapsed":clock()-start})
                        raw_ref_intermediate[index],_=self._orbit(readings,delay_seconds,cancel,sleeper,clock,progress,{"corrector":index+1,"correctors":ncor,"plane":corrector.plane,"device":corrector.name,"requested_kick":dkick,"state":"restored reference before −kick"})
                effective=arrays["act_a"][index]-arrays["act_b"][index]; arrays["effective"][index]=effective
                expected_sign=-1.0 if direction=="negative" else 1.0
                if not np.isfinite(effective) or abs(effective)<=1e-18 or np.sign(effective)!=expected_sign:
                    raise RuntimeError(
                        f"Invalid corrector readback separation for {corrector.name}: "
                        f"requested {dkick:+.9g} rad in {direction} mode, measured {effective:+.9g} rad"
                    )
                column=np.mean(raw_a[index],axis=0)-np.mean(raw_b[index],axis=0)
                matrix[:,index]=column/effective if scaled else column
                if progress: progress({"event":"column","corrector":index+1,"correctors":ncor,"column":column,"matrix":matrix.copy(),"elapsed":clock()-start})
            except AcquisitionCancelled as exc:
                restoration,diagnostics=restore(touched)
                raise ORMInterrupted(str(exc),restoration_status=restoration,restoration_diagnostics=diagnostics) from exc
            except Exception as exc:
                restoration,diagnostics=restore(touched)
                raise ORMAcquisitionError(f"ORM acquisition failed for {corrector.name}: {exc}",restoration_status=restoration,restoration_diagnostics=diagnostics) from exc
            else:
                try:
                    restoration,diagnostics=restore((index,))
                    if restoration!="restored":raise RuntimeError(f"Final restoration verification failed: {arrays['final'][index]!r} != {original_readback!r}")
                    if progress:progress({"event":"final_restored","corrector":index+1,"correctors":ncor,"plane":corrector.plane,"device":corrector.name,"state":"final K0 restored and verified","readback":arrays["final"][index],"original":original,"elapsed":clock()-start})
                    if correction_enabled:
                        touched.discard(index)
                        if cancel.is_set():
                            raise AcquisitionCancelled("ORM acquisition was cancelled before orbit correction")
                        if progress:
                            progress({"event": "state", "corrector": index+1, "correctors": ncor,
                                      "plane": corrector.plane, "device": corrector.name,
                                      "state": "correcting orbit after nominal restoration", "elapsed": clock()-start})
                        correction = self.adapter.correct_orbit_after_column(
                            corrector.name, raw_ref_before[index].mean(axis=0), cancel_event=cancel,
                            settle=lambda: settle(cancellable=False), tolerance_rad=restoration_tolerance_rad)
                        correction_diagnostics.append(correction)
                    raw_ref_final[index],_=self._orbit(readings,delay_seconds,cancel,sleeper,clock,progress,{"corrector":index+1,"correctors":ncor,"plane":corrector.plane,"device":corrector.name,"requested_kick":dkick,"state":"post-correction reference" if correction_enabled else "final restored reference"})
                except PyAMLOrbitCorrectionError as exc:
                    diagnostics = {"orbit_correction": [*correction_diagnostics, exc.diagnostics],
                                   "status": exc.diagnostics["restoration_status"]}
                    error_type = ORMInterrupted if exc.cancelled else ORMAcquisitionError
                    raise error_type(f"Orbit correction failed for {corrector.name}: {exc}",
                                     restoration_status=diagnostics["status"], restoration_diagnostics=diagnostics) from exc
                except AcquisitionCancelled as exc:
                    restoration,diagnostics=restore(touched)
                    raise ORMInterrupted(str(exc),restoration_status=restoration,restoration_diagnostics=diagnostics) from exc
                except Exception as exc:
                    restoration,diagnostics=restore(touched)
                    raise ORMAcquisitionError(f"Restoration failed for {corrector.name}: {exc}",restoration_status=restoration,restoration_diagnostics=diagnostics) from exc
        return ORMResult(self.bpms,self.horizontal_correctors,self.vertical_correctors,matrix,direction,bool(scaled),kicks,arrays["effective"],arrays["original"],arrays["req_a"],arrays["req_b"],arrays["act_a"],arrays["act_b"],arrays["intermediate"],tuple(intermediate_statuses),arrays["final"],raw_a,raw_b,raw_ref_before,raw_ref_intermediate,raw_ref_final,ts_a,ts_b,tuple(statuses),clock()-start,tuple(correction_diagnostics))
