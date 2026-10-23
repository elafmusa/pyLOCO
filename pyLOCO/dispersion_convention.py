"""Canonicalize persisted bipolar RF-response measurements without rewriting them."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


CANONICAL_SIGN = "positive_minus_negative"
LEGACY_SIGN = "negative_minus_positive"


@dataclass(frozen=True)
class DispersionConvention:
    rf_step_hz: float
    array_scale: float
    source_sign: str


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return str(value or "").strip().lower().replace(" ", "_")


def canonical_dispersion_convention(attrs: Mapping[str, Any]) -> DispersionConvention:
    """Return positive total separation and the required in-memory array scale.

    Bipolar files must explicitly and consistently identify their subtraction
    direction.  This prevents a plausible-looking but sign-inverted RF column.
    """
    bidirectional = bool(attrs.get("bidirectional", False))
    raw_step = attrs.get("rf_step_hz", attrs.get("rf_signed_step_hz"))
    if raw_step is None:
        raise ValueError("Dispersion metadata does not contain rf_step_hz.")
    step = float(np.asarray(raw_step).ravel()[0])
    if not np.isfinite(step) or step == 0:
        raise ValueError("Dispersion rf_step_hz must be finite and non-zero.")
    if not bidirectional:
        return DispersionConvention(step, 1.0, "one_sided")

    sign = _text(attrs.get("rf_difference_sign_convention"))
    definition = _text(attrs.get("measured_eta_definition"))
    if not sign:
        if "negative_-_mean_orbit_positive" in definition or "negative_-_positive" in definition:
            sign = LEGACY_SIGN
        elif "positive_-_mean_orbit_negative" in definition or "positive_-_negative" in definition:
            sign = CANONICAL_SIGN
    expected = LEGACY_SIGN if step < 0 else CANONICAL_SIGN
    if sign not in {LEGACY_SIGN, CANONICAL_SIGN}:
        raise ValueError(
            "Bipolar dispersion subtraction direction is missing or ambiguous; expected explicit "
            "rf_difference_sign_convention metadata."
        )
    if sign != expected:
        raise ValueError(
            f"Inconsistent bipolar dispersion metadata: rf_step_hz={step:g} Hz implies {expected}, "
            f"but the stored subtraction convention is {sign}."
        )
    return DispersionConvention(abs(step), -1.0 if sign == LEGACY_SIGN else 1.0, sign)
