"""Physics conversions kept separate from transactional RF acquisition."""
from __future__ import annotations

import numpy as np


MOMENTUM_RELATION = "delta_p_over_p = -(f - f0) / (eta * f0), eta = alpha_c - 1/gamma^2, first-order"


def relative_momentum_deviation(frequency_hz, nominal_hz: float, slip_factor_eta: float):
    if not np.isfinite(nominal_hz) or nominal_hz <= 0:
        raise ValueError("Nominal RF frequency must be positive and finite")
    if not np.isfinite(slip_factor_eta) or slip_factor_eta == 0:
        raise ValueError("Slip factor eta must be finite and non-zero")
    return -(np.asarray(frequency_hz, dtype=float)-nominal_hz)/(slip_factor_eta*nominal_hz)


def physical_dispersion(orbit_positive_m, orbit_negative_m, delta_positive: float, delta_negative: float):
    """Return dispersion using matching positive-minus-negative RF states."""
    delta_span=float(delta_positive-delta_negative)
    if not np.isfinite(delta_span) or delta_span == 0:
        raise ValueError("The recorded RF states do not provide a non-zero momentum separation")
    return ((np.asarray(orbit_positive_m)-np.asarray(orbit_negative_m))/delta_span,
            delta_span)
