"""Capability-gated correction preview/read/apply service."""
from __future__ import annotations

from dataclasses import dataclass

from pyLOCO.control_system import AdapterCapability


@dataclass(frozen=True)
class CorrectionChange:
    name: str
    channel: str
    current: float
    proposed: float
    readback: float | None = None
    status: str = "preview"


class CorrectionApplicationService:
    def __init__(self, adapter): self.adapter = adapter

    def preview(self, review):
        result = []
        for item in review.items:
            if not item.included: continue
            name = item.control_name or item.name; channel = f"MAGNET:{name}"
            current = float(self.adapter.read(channel).value)
            result.append(CorrectionChange(name, channel, current, current + float(item.final_delta)))
        return tuple(result)

    def apply(self, changes, *, confirmed: bool):
        if not confirmed: raise PermissionError("Apply requires explicit operator confirmation")
        self.adapter.require(AdapterCapability.WRITE)
        completed = []
        for change in changes:
            try:
                self.adapter.write(change.channel, change.proposed)
                readback = float(self.adapter.read(change.channel).value)
                status = "success" if abs(readback-change.proposed) <= max(1e-12, abs(change.proposed)*1e-9) else "readback_mismatch"
                completed.append(CorrectionChange(change.name,change.channel,change.current,change.proposed,readback,status))
            except Exception:
                for prior in reversed(completed):
                    try:self.adapter.write(prior.channel, prior.current)
                    except Exception:pass
                raise
        return tuple(completed)
