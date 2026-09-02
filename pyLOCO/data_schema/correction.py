"""Correction-plan schema.  No machine application logic lives here."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import math

from ._json import read_json, write_json
from .measurement import SCHEMA_VERSION

CORRECTION_FILE_TYPE = "pyloco.correction_plan"
CORRECTION_TYPES = frozenset({"normal_quadrupole", "skew_quadrupole", "quadrupole_tilt"})


@dataclass(frozen=True)
class CorrectionRecord:
    correction_type: str
    name: str
    lattice_ordinal: int | None
    unit: str
    initial_value: float
    raw_fitted_delta: float
    recommended_machine_delta: float
    individual_scale: float = 1.0
    final_applied_delta: float | None = None
    family: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def final_delta(self, global_scale: float) -> float:
        return self.recommended_machine_delta * global_scale * self.individual_scale

    def validate(self, global_scale: float) -> None:
        if self.correction_type not in CORRECTION_TYPES:
            raise ValueError(f"Unsupported correction_type: {self.correction_type}")
        if not self.name.strip() or not self.unit.strip():
            raise ValueError("Correction records require name and unit")
        numeric = (
            self.initial_value, self.raw_fitted_delta, self.recommended_machine_delta,
            self.individual_scale, global_scale,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError(f"Correction record {self.name!r} contains a non-finite value")
        if self.final_applied_delta is not None:
            if not math.isfinite(float(self.final_applied_delta)):
                raise ValueError(f"Correction record {self.name!r} has non-finite final_applied_delta")
            expected = self.final_delta(global_scale)
            if not math.isclose(self.final_applied_delta, expected, rel_tol=1e-12, abs_tol=1e-15):
                raise ValueError(
                    f"final_applied_delta for {self.name!r} does not equal "
                    "recommended_machine_delta * global_scale * individual_scale"
                )


@dataclass(frozen=True)
class CorrectionPlan:
    plan_id: str
    source_result: str
    records: tuple[CorrectionRecord, ...]
    global_scale: float = 1.0
    application_state: str = "dry_run"
    fraction_comparison: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    file_type: str = CORRECTION_FILE_TYPE

    def records_by_type(self, correction_type: str) -> tuple[CorrectionRecord, ...]:
        if correction_type not in CORRECTION_TYPES:
            raise ValueError(f"Unsupported correction_type: {correction_type}")
        return tuple(record for record in self.records if record.correction_type == correction_type)

    def validate(self) -> None:
        if self.file_type != CORRECTION_FILE_TYPE or self.schema_version != SCHEMA_VERSION:
            raise ValueError("Invalid or unsupported correction-plan schema")
        if self.application_state not in {"dry_run", "approved", "applied", "rolled_back"}:
            raise ValueError(f"Unsupported correction application_state: {self.application_state}")
        if not self.plan_id.strip() or not self.source_result.strip():
            raise ValueError("Correction plan requires plan_id and source_result")
        if not math.isfinite(float(self.global_scale)):
            raise ValueError("Correction plan global_scale must be finite")
        identities: set[tuple[str, str, int | None]] = set()
        for record in self.records:
            record.validate(self.global_scale)
            identity = (record.correction_type, record.name, record.lattice_ordinal)
            if identity in identities:
                raise ValueError(f"Duplicate correction record: {identity}")
            identities.add(identity)
        fractions = [float(entry.get("fraction", -1)) for entry in self.fraction_comparison]
        if fractions and (fractions != sorted(set(fractions)) or any(not 0 < value <= 1 for value in fractions)):
            raise ValueError("Correction fractions must be unique, increasing values in (0, 1]")
        for entry in self.fraction_comparison:
            for required in ("fraction", "max_abs_delta_k_over_k_percent", "max_abs_delta_i_ampere", "current_limit_violations"):
                if required not in entry:
                    raise ValueError(f"Correction fraction entry is missing {required}")


def save_correction_plan(path: str | Path, plan: CorrectionPlan) -> Path:
    plan.validate()
    return write_json(path, asdict(plan))


def load_correction_plan(path: str | Path) -> CorrectionPlan:
    data = read_json(path)
    records = tuple(CorrectionRecord(**entry) for entry in data.get("records", ()))
    plan = CorrectionPlan(
        plan_id=str(data.get("plan_id", "")),
        source_result=str(data.get("source_result", "")),
        records=records,
        global_scale=float(data.get("global_scale", 1.0)),
        application_state=str(data.get("application_state", "")),
        fraction_comparison=tuple(data.get("fraction_comparison", ())),
        metadata=dict(data.get("metadata", {})),
        schema_version=str(data.get("schema_version", "")),
        file_type=str(data.get("file_type", "")),
    )
    plan.validate()
    return plan
