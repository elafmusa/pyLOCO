# pyloco_config.py
import logging
from dataclasses import dataclass
import numpy as np
from typing import Optional, Callable, Any, Union, Sequence, Tuple, Dict
#import at
#ring = at.load_lattice('new_lattice0')

LOGGER = logging.getLogger(__name__)

def _cfg_get(cfg, name, current):
    if cfg is None:
        return current
    if isinstance(cfg, dict):
        return cfg.get(name, current)
    return getattr(cfg, name, current)

# block order matches pyloco module (for the moment) do not edit it.
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
    "hbpm_gain":      "ones",
    "hbpm_coupling":  "zeros",
    "vbpm_coupling":  "zeros",
    "vbpm_gain":      "ones",
    "hcor_cal":       "cmstep:h",
    "vcor_cal":       "cmstep:v",
    "hcor_coupling":  "zeros",
    "vcor_coupling":  "zeros",
    "HCMEnergyShift": "zeros",
    "VCMEnergyShift": "zeros",
    "delta_rf":       "rfstep",
    "quads":          "quads:K",
    "skew_quads":     "zeros",
    "quads_tilt":     "tilts:zeros",
}

@dataclass
class RMConfig:
    bpm_ords: Optional[Sequence[int]] = None
    cm_ords: Optional[Tuple[Sequence[int], Sequence[int]]] = None
    cav_ords: Optional[Sequence[int]] = None
    dkick: Union[float, tuple, list, np.ndarray] = 1e-5
    bidirectional: bool = True
    includeDispersion: bool = False
    rfStep: float = -3000
    delta_coupling: float = 1e-6
    coupling_orm: bool = False
    calculator: str = "Linear"
    NewVectorizedMethod: bool = True
    fixedpathlength: bool = False
    log_info: bool = False
    HCMCoupling: Optional[Union[np.ndarray, list, float]] = None
    VCMCoupling: Optional[Union[np.ndarray, list, float]] = None
    Frequency: Optional[float] = None
    HarmNumber: Optional[int] = None
    RFAttr: str = "Frequency"

@dataclass
class FitInitConfig:
    fit_list: Optional[Sequence[str]] = None
    block_order: Sequence[str] = BLOCK_ORDER
    init_policy: Dict[str, str] = None
    CMstep: Union[tuple, list, np.ndarray] = (1e-5, 1e-5)
    rfStep: float = -3000
    individuals: bool = True
    init: Optional[Dict[str, Any]] = None

    # what to read/write for each block
    quads_attr: str = "PolynomB"   # default quad strength holder
    quads_attr_index: Optional[int] = 1
    skew_attr: str = "PolynomB"    # <- you want skew to use B? set it here.
    skew_attr_index: Optional[int] = 1

    quads_tilt_attr_R1: str = "R1"
    quads_tilt_attr_R2: str = "R2"
    quads_tilt_method: str = "set"

    def __post_init__(self):
        if self.init_policy is None:
            self.init_policy = dict(DEFAULT_INIT_POLICY)


def _default_get_mcf(ring):
    import at
    return at.get_mcf(ring)


from dataclasses import dataclass
from typing import Optional, Callable, Any

def _default_get_mcf(ring):
    import at
    return at.get_mcf(ring)

@dataclass
class LOCOAPI:
    get_mcf: Optional[Callable[[Any], float]] = None

    def resolve_get_mcf(self) -> Callable[[Any], float]:
        #fn = self.get_mcf or _default_get_mcf   # to be chosen by the user
        fn = _default_get_mcf
        if not callable(fn):
            raise TypeError("get_mcf must be a callable(ring)->float")
        return fn

#  replaced by the user's function to get mcf
def my_get_mcf(ring):
    return 1e-3

BACKEND = LOCOAPI(get_mcf=my_get_mcf)

def get_mcf(ring):
    return BACKEND.resolve_get_mcf()(ring)

@dataclass
class FixedParameters:
    Frequency: float = 499664399.4230182
    HarmNumber: int = 3840
    rfstep: float = -3000.0
    dk: [float] = None
    delta_skew: float = 1e-3
    delta_q_tilt: float = 1e-6

fixed_parameters = FixedParameters()

