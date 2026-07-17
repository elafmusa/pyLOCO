"""Default configuration objects for the pyLOCO numerical backend.

These defaults let backend modules import and run without requiring a process-wide
``pyloco_config`` module.  Scripts and the GUI can still pass explicit config
objects with matching attributes to override these values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence



def _cfg_get(cfg: object | dict[str, Any] | None, name: str, current: Any) -> Any:
    """Read a config attribute/key, falling back to *current* when absent."""

    if cfg is None:
        return current
    if isinstance(cfg, dict):
        return cfg.get(name, current)
    return getattr(cfg, name, current)


BLOCK_ORDER: tuple[str, ...] = (
    "hbpm_gain",
    "hbpm_coupling",
    "vbpm_coupling",
    "vbpm_gain",
    "hcor_cal",
    "vcor_cal",
    "hcor_coupling",
    "vcor_coupling",
    "HCMEnergyShift",
    "VCMEnergyShift",
    "delta_rf",
    "quads",
    "skew_quads",
    "quads_tilt",
)

DEFAULT_INIT_POLICY: dict[str, str] = {
    "hbpm_gain": "ones",
    "hbpm_coupling": "zeros",
    "vbpm_coupling": "zeros",
    "vbpm_gain": "ones",
    "hcor_cal": "cmstep:h",
    "vcor_cal": "cmstep:v",
    "hcor_coupling": "zeros",
    "vcor_coupling": "zeros",
    "HCMEnergyShift": "zeros",
    "VCMEnergyShift": "zeros",
    "delta_rf": "rfstep",
    "quads": "quads:attr",
    "skew_quads": "zeros",
    "quads_tilt": "tilts:zeros",
}


@dataclass(slots=True)
class LOCOOptions:
    algorithm: str = "lm"
    nIter: int = 1
    nLMIter: int = 10
    Starting_Lambda: float = 1e-3
    max_lm_lambda: float = 15.0
    scaled: bool = True
    svd_selection_method: str = "threshold"
    svd_threshold: float = 1e-7
    cut_: int = 397
    show_svd_plot: bool = True
    fit_list: Sequence[str] = (
        "quads",
        "hbpm_gain",
        "vbpm_gain",
        "hcor_cal",
        "vcor_cal",
        "HCMEnergyShift",
    )
    outlier_rejection: bool = True
    sigma_outlier: float = 10.0
    apply_normalization: bool = True
    normalization_mode: str = "component"
    includeDispersion: bool = False
    hor_dispersion_weight: float = 1.0
    ver_dispersion_weight: float = 1.0
    plot_fit_parameters: bool = False
    auto_correct_delta: bool = True
    fixedpathlength: bool = False
    individuals: bool = True
    remove_coupling_: bool = True


@dataclass(slots=True)
class RMConfig:
    bpm_ords: Optional[Sequence[int]] = None
    cm_ords: Optional[tuple[Sequence[int], Sequence[int]]] = None
    cav_ords: Optional[Sequence[int]] = None
    dkick: Any = 1e-5
    bidirectional: bool = True
    includeDispersion: bool = False
    rfStep: float = -3000.0
    delta_coupling: float = 1e-6
    coupling_orm: bool = False
    calculator: str = "Linear"
    NewVectorizedMethod: bool = True
    fixedpathlength: bool = False
    log_info: bool = False
    HCMCoupling: Any = None
    VCMCoupling: Any = None
    Frequency: Optional[float] = None
    HarmNumber: Optional[int] = None
    RFAttr: str = "Frequency"


@dataclass(slots=True)
class FitInitConfig:
    fit_list: Optional[Sequence[str]] = None
    block_order: Sequence[str] = BLOCK_ORDER
    init_policy: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_INIT_POLICY))
    CMstep: Any = (1e-5, 1e-5)
    rfStep: float = -3000.0
    individuals: bool = True
    init: Optional[dict[str, Any]] = None
    quads_attr: str = "PolynomB"
    quads_attr_index: Optional[int] = 1
    skew_attr: str = "PolynomA"
    skew_attr_index: Optional[int] = 1
    quads_tilt_attr_R1: str = "R1"
    quads_tilt_attr_R2: str = "R2"
    quads_tilt_method: str = "set"


@dataclass(slots=True)
class FixedParameters:
    Frequency: float = 499664399.4230182
    HarmNumber: int = 3840
    rfstep: float = -3000.0
    dk: Any = None
    delta_skew: float = 1e-3
    delta_q_tilt: float = 1e-6


loco_options = LOCOOptions()
fixed_parameters = FixedParameters()


def get_mcf(ring):
    """Default momentum compaction factor resolver used by the backend."""

    import at

    return at.get_mcf(ring)
