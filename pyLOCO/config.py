"""Internal configuration defaults for pyLOCO.

This module replaces the historical requirement for an importable
``pyloco_config.py``.  Example/user config files can still be loaded through
``pyLOCO.helpers.load_config`` for backwards compatibility, but the backend and
GUI can now run with these built-in defaults.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
import numpy as np
from dataclasses import dataclass
from typing import Optional


def _cfg_get(cfg: Any, name: str, current: Any) -> Any:
    """Read a config value from a dataclass/object/dict with a fallback."""

    if cfg is None:
        return current
    if isinstance(cfg, dict):
        return cfg.get(name, current)
    return getattr(cfg, name, current)


BLOCK_ORDER: Tuple[str, ...] = (
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

DEFAULT_INIT_POLICY: Dict[str, str] = {
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
    "quads": "quads:K",
    "skew_quads": "zeros",
    "quads_tilt": "tilts:zeros",
}


@dataclass
class LOCOOptions:
    algorithm: str = "lm"
    nIter: int = 1
    nLMIter: int = 10
    Starting_Lambda: float = 1e-3
    max_lm_lambda: float = 15
    scaled: bool = True
    svd_selection_method: str = "threshold"
    svd_threshold: float = 1e-7
    cut_: int = 397
    show_svd_plot: bool = True
    fit_list: Sequence[str] = ("quads", "hbpm_gain", "vbpm_gain", "hcor_cal", "vcor_cal", "HCMEnergyShift")
    outlier_rejection: bool = True
    sigma_outlier: float = 10
    apply_normalization: bool = False
    normalization_mode: str = "component"
    includeDispersion: bool = False
    hor_dispersion_weight: float = 1.0
    ver_dispersion_weight: float = 1.0
    plot_fit_parameters: bool = False
    auto_correct_delta: bool = True
    fixedpathlength: bool = False
    individuals: bool = True
    remove_coupling_: bool = True


loco_options = LOCOOptions()


@dataclass
class RMConfig:
    bpm_ords: Optional[Sequence[int]] = None
    cm_ords: Optional[Tuple[Sequence[int], Sequence[int]]] = None
    cav_ords: Optional[Sequence[int]] = None
    dkick: Union[float, tuple, list, Any] = 1e-5
    bidirectional: bool = True
    includeDispersion: bool = False
    rfStep: float = -3000.0
    delta_coupling: float = 1e-6
    coupling_orm: bool = False
    calculator: str = "Linear"
    NewVectorizedMethod: bool = True
    fixedpathlength: bool = False
    log_info: bool = False
    HCMCoupling: Optional[Union[Any, list, float]] = None
    VCMCoupling: Optional[Union[Any, list, float]] = None
    Frequency: Optional[float] = None
    HarmNumber: Optional[int] = None
    RFAttr: str = "Frequency"


@dataclass
class FitInitConfig:
    fit_list: Optional[Sequence[str]] = None
    block_order: Sequence[str] = BLOCK_ORDER
    init_policy: Optional[Dict[str, str]] = None
    CMstep: Union[tuple, list, Any] = (1e-5, 1e-5)
    rfStep: float = -3000.0
    individuals: bool = True
    init: Optional[Dict[str, Any]] = None
    quads_attr: str = "PolynomB"
    quads_attr_index: Optional[int] = 1
    skew_attr: str = "PolynomA"
    skew_attr_index: Optional[int] = 1
    quads_tilt_attr_R1: str = "R1"
    quads_tilt_attr_R2: str = "R2"
    quads_tilt_method: str = "set"

    def __post_init__(self) -> None:
        if self.init_policy is None:
            self.init_policy = dict(DEFAULT_INIT_POLICY)


@dataclass
class LOCOAPI:
    get_mcf: Optional[Callable[[Any], float]] = None

    def resolve_get_mcf(self) -> Callable[[Any], float]:
        fn = self.get_mcf or _default_get_mcf
        if not callable(fn):
            raise TypeError("get_mcf must be a callable(ring)->float")
        return fn


def _default_get_mcf(ring: Any) -> float:
    import at

    return at.get_mcf(ring)


BACKEND = LOCOAPI()


def get_mcf(ring: Any) -> float:
    return BACKEND.resolve_get_mcf()(ring)


@dataclass
class FixedParameters:
    Frequency: float = 499664399.4230182
    HarmNumber: int = 3840
    rfstep: float = -3000.0
    dk: Any = None
    delta_skew: float = 1e-3
    delta_q_tilt: float = 1e-6


@dataclass
class ConstraintConfig:
    enable: bool = True

    quad_sigma: float | np.ndarray = 0.01
    quad_weights: Optional[np.ndarray] = None
    quad_mask: Optional[np.ndarray] = None

    skew_sigma: float | np.ndarray = 0.001
    skew_weights: Optional[np.ndarray] = None
    skew_mask: Optional[np.ndarray] = None



fixed_parameters = FixedParameters()


# Stable references to the built-in classes.  These let GUI/backend code reset
# to internal defaults even after a user config has been loaded for scripts.
INTERNAL_LOCOOptions = LOCOOptions
INTERNAL_RMConfig = RMConfig
INTERNAL_FitInitConfig = FitInitConfig
INTERNAL_FixedParameters = FixedParameters
