"""User-facing YAML configuration helpers.

The numerical API deliberately remains unchanged.  This module translates the
readable example schema into the existing ``fit_list`` and ``ConstraintConfig``
objects, so scripts and the GUI do not need to make fit-policy decisions.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .config import BLOCK_ORDER, ConstraintConfig


PARAMETER_GROUPS: dict[str, tuple[str, ...]] = {
    "quadrupoles": ("quads",),
    "skew_quadrupoles": ("skew_quads",),
    "quadrupole_tilt": ("quads_tilt",),
    "bpm_gains": ("hbpm_gain", "vbpm_gain"),
    "bpm_coupling": ("hbpm_coupling", "vbpm_coupling"),
    "corrector_calibration": ("hcor_cal", "vcor_cal"),
    "corrector_coupling": ("hcor_coupling", "vcor_coupling"),
    "hcm_energy_shift": ("HCMEnergyShift",),
    "vcm_energy_shift": ("VCMEnergyShift",),
    # Backward-compatible convenience group used by the first YAML cleanup.
    "corrector_energy_shift": ("HCMEnergyShift", "VCMEnergyShift"),
    "rf_frequency_shift": ("delta_rf",),
}


def selected_fit_parameters(config: Mapping[str, Any]) -> list[str]:
    """Return backend block names from ``fit_parameters``.

    A legacy ``loco.fit_list`` (and the older PETRA list variants) is accepted
    when the new section is absent.  Constraints are intentionally ignored.
    """
    groups = config.get("fit_parameters")
    if groups is not None:
        unknown = sorted(set(groups) - set(PARAMETER_GROUPS))
        if unknown:
            raise ValueError("Unknown fit_parameters group(s): " + ", ".join(unknown))
        enabled: set[str] = set()
        for name, blocks in PARAMETER_GROUPS.items():
            value = groups.get(name, {}) or {}
            switch = value if isinstance(value, bool) else value.get("enable", False)
            if bool(switch):
                enabled.update(blocks)
        return [name for name in BLOCK_ORDER if name in enabled]

    loco = config.get("loco", {}) or {}
    legacy = loco.get("fit_list")
    if legacy is None:
        for name in ("standard_fit_list", "coupling_fit_list", "constrained_fit_list"):
            if name in loco:
                legacy = loco[name]
                break
    if legacy is None:
        raise ValueError("Add fit_parameters to the YAML configuration")
    unknown = sorted(set(legacy) - set(BLOCK_ORDER))
    if unknown:
        raise ValueError("Unknown fit block(s): " + ", ".join(unknown))
    return [name for name in BLOCK_ORDER if name in set(legacy)]

def build_constraints(
    config: Mapping[str, Any],
    *,
    quad_nominal: Sequence[float],
    n_skew: int,
    n_hbpm: int = 0,
    n_vbpm: int = 0,
    n_hcor: int = 0,
    n_vcor: int = 0,
) -> ConstraintConfig | None:
    """Build constraints independently of parameter selection."""

    raw = config.get("constraints", {}) or {}

    if not bool(raw.get("enable", False)):
        return None

    # ------------------------------------------------------------
    # Read constraint sections
    # ------------------------------------------------------------
    q = raw.get("quadrupoles", {}) or {}
    s = raw.get("skew_quadrupoles", {}) or {}

    hg = raw.get("hbpm_gain", {}) or {}
    vg = raw.get("vbpm_gain", {}) or {}
    hc = raw.get("hcor_cal", {}) or {}
    vc = raw.get("vcor_cal", {}) or {}

    # ------------------------------------------------------------
    # Quadrupoles
    # ------------------------------------------------------------
    nominal = np.asarray(quad_nominal, dtype=float).ravel()
    n_quad = nominal.size

    relative = q.get("relative_sigma")
    absolute = q.get("sigma")

    if relative is not None and absolute is not None:
        raise ValueError(
            "Set only one of constraints.quadrupoles.sigma "
            "or relative_sigma"
        )

    if relative is not None:
        quad_sigma: float | np.ndarray = np.maximum(
            np.abs(nominal) * float(relative),
            float(q.get("minimum_sigma", 1e-12)),
        )
    else:
        quad_sigma = _scalar_or_vector(
            q.get("sigma", 0.01),
            n_quad,
            "quadrupole sigma",
        )

    quad_weights = _weights(
        q,
        n_quad,
        "quadrupole family",
    )

    quad_mask = _mask(
        q.get("mask"),
        n_quad,
        "quadrupole",
    )

    # ------------------------------------------------------------
    # Skew quadrupoles
    # ------------------------------------------------------------
    skew_sigma = _scalar_or_vector(
        s.get("sigma", 0.001),
        n_skew,
        "skew sigma",
    )

    skew_weights = _weights(
        s,
        n_skew,
        "skew quadrupole",
    )

    skew_mask = _mask(
        s.get("mask"),
        n_skew,
        "skew quadrupole",
    )

    # ------------------------------------------------------------
    # Horizontal BPM gain
    #
    # Only activate if the YAML contains:
    #
    #   constraints:
    #     hbpm_gain:
    #       ...
    # ------------------------------------------------------------
    if "hbpm_gain" in raw:
        hbpm_gain_sigma = _scalar_or_vector(
            hg.get("sigma", 0.05),
            n_hbpm,
            "horizontal BPM gain sigma",
        )

        hbpm_gain_weights = _weights(
            hg,
            n_hbpm,
            "horizontal BPM gain",
        )

        hbpm_gain_mask = _mask(
            hg.get("mask"),
            n_hbpm,
            "horizontal BPM gain",
        )
    else:
        hbpm_gain_sigma = 0.05
        hbpm_gain_weights = None
        hbpm_gain_mask = None

    # ------------------------------------------------------------
    # Vertical BPM gain
    # ------------------------------------------------------------
    if "vbpm_gain" in raw:
        vbpm_gain_sigma = _scalar_or_vector(
            vg.get("sigma", 0.05),
            n_vbpm,
            "vertical BPM gain sigma",
        )

        vbpm_gain_weights = _weights(
            vg,
            n_vbpm,
            "vertical BPM gain",
        )

        vbpm_gain_mask = _mask(
            vg.get("mask"),
            n_vbpm,
            "vertical BPM gain",
        )
    else:
        vbpm_gain_sigma = 0.05
        vbpm_gain_weights = None
        vbpm_gain_mask = None

    # ------------------------------------------------------------
    # Horizontal corrector calibration
    #
    # hcor_cal is stored in radians.
    # For CMstep = 100 urad:
    #     5% -> sigma = 5e-6 rad
    # ------------------------------------------------------------
    if "hcor_cal" in raw:
        hcor_cal_sigma = _scalar_or_vector(
            hc.get("sigma", 5.0e-6),
            n_hcor,
            "horizontal corrector calibration sigma",
        )

        hcor_cal_weights = _weights(
            hc,
            n_hcor,
            "horizontal corrector calibration",
        )

        hcor_cal_mask = _mask(
            hc.get("mask"),
            n_hcor,
            "horizontal corrector calibration",
        )
    else:
        hcor_cal_sigma = 5.0e-6
        hcor_cal_weights = None
        hcor_cal_mask = None

    # ------------------------------------------------------------
    # Vertical corrector calibration
    # ------------------------------------------------------------
    if "vcor_cal" in raw:
        vcor_cal_sigma = _scalar_or_vector(
            vc.get("sigma", 5.0e-6),
            n_vcor,
            "vertical corrector calibration sigma",
        )

        vcor_cal_weights = _weights(
            vc,
            n_vcor,
            "vertical corrector calibration",
        )

        vcor_cal_mask = _mask(
            vc.get("mask"),
            n_vcor,
            "vertical corrector calibration",
        )
    else:
        vcor_cal_sigma = 5.0e-6
        vcor_cal_weights = None
        vcor_cal_mask = None

    # ------------------------------------------------------------
    # Construct backend ConstraintConfig
    # ------------------------------------------------------------
    return ConstraintConfig(
        enable=True,

        # Quadrupoles
        quad_sigma=quad_sigma,
        quad_weights=quad_weights,
        quad_mask=quad_mask,

        # Skew quadrupoles
        skew_sigma=skew_sigma,
        skew_weights=skew_weights,
        skew_mask=skew_mask,

        # BPM gains
        hbpm_gain_sigma=hbpm_gain_sigma,
        hbpm_gain_weights=hbpm_gain_weights,
        hbpm_gain_mask=hbpm_gain_mask,

        vbpm_gain_sigma=vbpm_gain_sigma,
        vbpm_gain_weights=vbpm_gain_weights,
        vbpm_gain_mask=vbpm_gain_mask,

        # Corrector calibration
        hcor_cal_sigma=hcor_cal_sigma,
        hcor_cal_weights=hcor_cal_weights,
        hcor_cal_mask=hcor_cal_mask,

        vcor_cal_sigma=vcor_cal_sigma,
        vcor_cal_weights=vcor_cal_weights,
        vcor_cal_mask=vcor_cal_mask,
    )

def _weights(section: Mapping[str, Any], size: int, label: str) -> np.ndarray:
    explicit = section.get("weights")
    result = (np.full(size, float(section.get("default_weight", 1.0)))
              if explicit is None else _vector(explicit, size, f"{label} weights"))
    selected, weight = section.get("selected_families"), section.get("selected_weight")
    if (selected is None) != (weight is None):
        raise ValueError(f"constraints.{label.replace(' ', '_')} selected_families and selected_weight must be set together")
    if selected is not None:
        indices = np.asarray(selected, dtype=int).ravel()
        if indices.size != len(set(indices.tolist())):
            raise ValueError(f"Selected {label} list contains duplicates")
        if np.any((indices < 0) | (indices >= size)):
            bad = int(indices[(indices < 0) | (indices >= size)][0])
            raise ValueError(f"Selected {label} {bad} is outside 0..{size - 1}")
        result[indices] = float(weight)
    for index, value in (section.get("weighted_families", {}) or {}).items():
        index = int(index)
        if index < 0 or index >= size:
            raise ValueError(f"Weighted {label} {index} is outside 0..{size - 1}")
        result[index] = float(value)
    return result


def _vector(value: Any, size: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=float).ravel()
    if result.size != size:
        raise ValueError(f"The {label} vector has {result.size} entries; expected {size}")
    return result


def _scalar_or_vector(value: Any, size: int, label: str) -> float | np.ndarray:
    result = np.asarray(value, dtype=float).ravel()
    return float(result[0]) if result.size == 1 else _vector(result, size, label)


def _mask(value: Any, size: int, label: str) -> np.ndarray | None:
    if value is None:
        return None
    result = np.asarray(value, dtype=bool).ravel()
    if result.size != size:
        raise ValueError(f"The {label} constraint mask has {result.size} entries; expected {size}")
    return result
