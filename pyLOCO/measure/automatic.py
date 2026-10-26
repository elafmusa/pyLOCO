"""Transactional measurement workflows that temporarily change machine state."""
from __future__ import annotations

from threading import Event
from time import monotonic, sleep

import numpy as np

from pyLOCO.control_system import AdapterCapability
from .acquisition import AcquisitionCancelled, DispersionResult, DispersionStateAcquirer


class AutomaticDispersionAcquirer:
    def __init__(self, adapter, devices):
        adapter.require(AdapterCapability.RF_READ); adapter.require(AdapterCapability.RF_WRITE)
        self.adapter = adapter; self.devices = tuple(devices)

    def acquire(self, rf_step_hz, readings, delay_seconds, *, direction="bipolar",
                settling_delay_seconds=0.0, cancel_event=None, progress=None,
                verify_restored_orbit=True, status=None,
                sleeper=sleep, clock=monotonic, restoration_tolerance_hz=0.1):
        if rf_step_hz <= 0: raise ValueError("RF step must be positive")
        if direction not in {"bipolar", "positive", "negative"}: raise ValueError("Unsupported dispersion direction")
        if not np.isfinite(restoration_tolerance_hz) or restoration_tolerance_hz < 0:
            raise ValueError("RF restoration tolerance must be finite and non-negative")
        cancel = cancel_event or Event(); original = float(self.adapter.get_rf_frequency())
        offsets = {"bipolar": (("reference", 0.0), ("positive", rf_step_hz), ("negative", -rf_step_hz)),
                   "positive": (("reference", 0.0), ("positive", rf_step_hz)),
                   "negative": (("reference", 0.0), ("negative", -rf_step_hz))}[direction]
        states = []; start = clock(); restoration = "not_attempted"; final_readback = None
        acquisition_error = None
        try:
            for index, (label, offset) in enumerate(offsets):
                if cancel.is_set(): raise AcquisitionCancelled("Dispersion acquisition was cancelled")
                requested = original + offset
                if status: status("acquiring", {"index":index,"count":len(offsets),"label":label,"offset_hz":offset,"rf_hz":requested})
                self.adapter.set_rf_frequency(requested)
                if settling_delay_seconds and cancel.wait(settling_delay_seconds):
                    raise AcquisitionCancelled("Dispersion acquisition was cancelled during RF settling")
                actual = float(self.adapter.get_rf_frequency())
                state = DispersionStateAcquirer(self.adapter, self.devices).acquire(
                    label,
                    requested, readings, delay_seconds, actual_rf_hz=actual,
                    cancel_event=cancel, progress=(lambda *args, state_index=index: progress(state_index, *args) if progress else None),
                    sleeper=sleeper, clock=clock)
                states.append(state)
        except Exception as exc:
            acquisition_error = exc
            raise
        finally:
            try:
                if status: status("restoring", {"offset_hz":0.0,"rf_hz":original})
                self.adapter.set_rf_frequency(original)
                # Some control-system and simulator adapters acknowledge the
                # write before their public read accessor exposes the new RF.
                # Apply the same settling interval used for measurement states
                # before deciding whether restoration succeeded.
                if settling_delay_seconds:
                    sleeper(settling_delay_seconds)
                final_readback = float(self.adapter.get_rf_frequency())
                restoration = "restored" if abs(final_readback - original) <= restoration_tolerance_hz else "verification_failed"
            except Exception:
                restoration = "restore_failed"
            difference = None if final_readback is None else abs(final_readback - original)
            diagnostics = {
                "original_rf_hz": original, "attempted_restoration_hz": original,
                "final_readback_hz": final_readback,
                "absolute_difference_hz": difference,
                "tolerance_hz": float(restoration_tolerance_hz), "status": restoration,
            }
            if acquisition_error is not None:
                acquisition_error.restoration_status = restoration
                acquisition_error.restoration_diagnostics = diagnostics
            if restoration != "restored":
                if acquisition_error is None:
                    error = RuntimeError(f"RF restoration failed ({restoration})")
                    error.restoration_status = restoration
                    error.restoration_diagnostics = diagnostics
                    raise error
        if verify_restored_orbit:
            if status: status("verifying_orbit", {"offset_hz":0.0,"rf_hz":original})
            state = DispersionStateAcquirer(self.adapter, self.devices).acquire(
                "reference_after", original, readings, delay_seconds,
                actual_rf_hz=float(self.adapter.get_rf_frequency()), cancel_event=cancel,
                progress=(lambda *args: progress(len(offsets), *args) if progress else None),
                sleeper=sleeper, clock=clock,
            )
            states.append(state)
        return DispersionResult(self.devices, tuple(states), direction, original,
                                float(rf_step_hz), restoration, clock() - start)
